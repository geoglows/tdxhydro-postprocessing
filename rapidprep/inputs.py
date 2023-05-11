import json
import logging
import os
from itertools import chain
from multiprocessing import Pool

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Geod
from .correct_network import identify_0_length

# set up logging
logger = logging.getLogger(__name__)


def _calculate_geodesic_length(line) -> float:
    """
    Input is shapely geometry, should be all shapely LineString objects
    """
    length = Geod(ellps='WGS84').geometry_length(line) / 1000  # To convert to km

    # This is for the outliers that have 0 length
    if length < 0.0000001:
        length = 0.01
    return length


# todo this should be directly from the master table -> why is it different?
def _make_rapid_connect_row(stream_id, streams_gdf, id_field, ds_id_field):
    upstreams = streams_gdf.loc[streams_gdf[ds_id_field] == stream_id, id_field].values
    return {
        'HydroID': stream_id,
        'NextDownID': streams_gdf.loc[streams_gdf[id_field] == stream_id, ds_id_field].values[0],
        'CountUpstreamID': len(upstreams),
        **{f'UpstreamID{i + 1}': upstreams[i] for i in range(len(upstreams))}
    }


def _combine_routing_rows(streams_df: pd.DataFrame, id_to_preserve: str, ids_to_merge: list):
    target_row = streams_df.loc[streams_df['LINKNO'] == int(id_to_preserve)]
    if target_row.empty:
        return None
    # todo only sum muskingum parameters if they are part of the branches being kept
    musk_k, musk_kfac = streams_df.loc[streams_df['LINKNO'].isin(ids_to_merge[1:]), ['musk_k', 'musk_kfac']].mean()
    return pd.DataFrame({
        'LINKNO': int(id_to_preserve),
        'DSLINKNO': int(target_row['DSLINKNO'].values[0]),
        'strmOrder': int(target_row['strmOrder'].values[0]),
        'musk_k': float(musk_k),
        'musk_kfac': float(musk_kfac),
        'musk_x': float(target_row['musk_x'].values[0]),
        'lat': float(target_row['lat'].values[0]),
        'lon': float(target_row['lon'].values[0]),
        'z': int(target_row['z'].values[0]),
    }, index=[0])


def rapid_master_files(streams_gpkg: str,
                       save_dir: str,
                       id_field: str = 'LINKNO',
                       ds_id_field: str = 'DSLINKNO',
                       length_field: str = 'Length',
                       default_k: float = 0.35,
                       default_x: float = .3,
                       n_workers: int or None = 1) -> None:
    """
    Create RAPID master files from a stream network

    Saves the following files to the save_dir:
        - rapid_connect_master.parquet
        - rapid_inputs_master.parquet
    Args:
        streams_gpkg:
        save_dir:
        id_field:
        ds_id_field:
        length_field:
        default_k:
        default_x:
        n_workers:

    Returns:

    """
    # Create rapid preprocessing files
    logger.info('Creating RAPID Master files')
    streams_gdf = gpd.read_file(streams_gpkg)

    if not streams_gdf.crs == 'epsg:4326':
        streams_gdf = streams_gdf.to_crs('epsg:4326')

    with Pool(n_workers) as p:
        logger.info('\tCalculating lengths')
        streams_gdf["length_geod"] = p.map(_calculate_geodesic_length, streams_gdf.geometry.values)

        logger.info('\tCalculating lat, lon, z, k, x')
        streams_gdf['lat'] = streams_gdf.geometry.apply(lambda geom: geom.xy[1][0]).values
        streams_gdf['lon'] = streams_gdf.geometry.apply(lambda geom: geom.xy[0][0]).values
        streams_gdf['z'] = 0

        streams_gdf["musk_k"] = streams_gdf["length_geod"] * 3600 * default_k
        streams_gdf["musk_kfac"] = streams_gdf["musk_k"].values.flatten()
        streams_gdf["musk_x"] = default_x
        streams_gdf["musk_xfac"] = streams_gdf["musk_x"].values.flatten()

        logger.info('\tCalculating RAPID connect file')
        rapid_connect = p.starmap(_make_rapid_connect_row,
                                  [[x, streams_gdf, id_field, ds_id_field] for x in streams_gdf[id_field].values])
        rapid_connect = (
            pd
            .DataFrame(rapid_connect)
            .fillna(0)
            .astype(int)
        )

    # Fix 0 length segments
    logger.info('\tLooking for 0 length segments')
    if 0 in streams_gdf[length_field].values:
        zero_length_fixes_df = identify_0_length(streams_gdf, id_field, ds_id_field, length_field)
        zero_length_fixes_df.to_csv(os.path.join(save_dir, 'mod_zero_length_streams.csv'), index=False)

    streams_gdf = streams_gdf.drop(columns=['geometry'])

    logger.info('\tSorting streams topologically')
    sorted_order = sort_topologically(streams_gdf, id_field=id_field, ds_id_field=ds_id_field)

    streams_df = (
        streams_gdf
        .set_index(pd.Index(streams_gdf[id_field]))
        .reindex(sorted_order)
        .reset_index(drop=True)
    )

    # adjust rapid connect to match
    rapid_connect = rapid_connect.loc[rapid_connect.iloc[:, 0].isin(streams_df[id_field].values)]
    rapid_connect = (
        rapid_connect
        .set_index(pd.Index(rapid_connect.iloc[:, 0]))
        .reindex(sorted_order)
        .reset_index(drop=True)
        .astype(int)
    )

    logger.info('\tWriting RAPID master parquets')
    streams_gdf.to_parquet(os.path.join(save_dir, "rapid_inputs_master.parquet"))
    rapid_connect.to_parquet(os.path.join(save_dir, "rapid_connect_master.parquet"))

    return


def rapid_input_csvs(save_dir: str,
                     streams_df: pd.DataFrame = None,
                     rapcon_df: pd.DataFrame = None,
                     id_field: str = 'LINKNO',
                     ds_id_field: str = 'DSLINKNO', ) -> None:
    """
    Create RAPID input csvs from a stream network dataframe

    Produces the following files:
        - rapid_connect.csv
        - riv_bas_id.csv
        - k.csv
        - x.csv
        - comid_lat_lon_z.csv
        - kfac.csv
        - xfac.csv

    Args:
        save_dir:
        streams_df:
        rapcon_df:
        id_field:
        ds_id_field:

    Returns:

    """
    logger.info('Creating RAPID input csvs')
    if streams_df is None:
        streams_df = pd.read_parquet(os.path.join(save_dir, 'rapid_inputs_master.parquet'))
    if rapcon_df is None:
        rapcon_df = pd.read_parquet(os.path.join(save_dir, 'rapid_connect_master.parquet')).astype(int)

    logger.info('\tSorting IDs topologically')

    # todo delete this code block - add function to sort master files if they are not already sorted
    sorted_order = sort_topologically(streams_df, id_field=id_field, ds_id_field=ds_id_field)
    streams_df = (
        streams_df
        .set_index(pd.Index(streams_df[id_field]))
        .reindex(sorted_order)
        .reset_index(drop=True)
    )
    rapcon_df = rapcon_df.loc[rapcon_df.iloc[:, 0].isin(streams_df[id_field].values)]
    rapcon_df = (
        rapcon_df
        .set_index(pd.Index(rapcon_df.iloc[:, 0]))
        .reindex(sorted_order)
        .reset_index(drop=True)
    )

    streams_df[id_field].to_csv(os.path.join(save_dir, "riv_bas_id.csv"), index=False, header=False)
    rapcon_df.to_csv(os.path.join(save_dir, "rapid_connect.csv"), index=False, header=False)

    logger.info('\tWriting csvs')
    streams_df["musk_k"].to_csv(os.path.join(save_dir, "k.csv"), index=False, header=False)
    streams_df["musk_x"].to_csv(os.path.join(save_dir, "x.csv"), index=False, header=False)
    streams_df["musk_kfac"].to_csv(os.path.join(save_dir, "kfac.csv"), index=False, header=False)
    streams_df["musk_xfac"].to_csv(os.path.join(save_dir, "xfac.csv"), index=False, header=False)
    streams_df[[id_field, 'lat', 'lon', 'z']].to_csv(os.path.join(save_dir, "comid_lat_lon_z.csv"), index=False)

    return


def sort_topologically(df, id_field='LINKNO', ds_id_field='DSLINKNO'):
    df = df[[id_field, ds_id_field]].astype(int)
    df[df == -1] = np.nan
    digraph = nx.DiGraph()
    nodes = set(df[id_field].dropna()).union(set(df[ds_id_field].dropna()))
    for node in nodes:
        digraph.add_node(node)
    for i, row in df.iterrows():
        if not pd.isna(row[ds_id_field]):
            digraph.add_edge(row[id_field], row[ds_id_field])

    # perform topological sorting on the graph
    sorted_nodes = np.array(list(nx.topological_sort(digraph))).astype(int)
    return sorted_nodes
