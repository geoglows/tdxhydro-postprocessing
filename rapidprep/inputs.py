import logging
import os

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Geod

from .correct_network import correct_0_length_streams
from .correct_network import identify_0_length
from .correct_network import find_headwater_streams_to_dissolve

# set up logging
logger = logging.getLogger(__name__)


def _calculate_geodesic_length(line) -> float:
    """
    Input is shapely geometry, should be all shapely LineString objects

    returns length in meters
    """
    length = Geod(ellps='WGS84').geometry_length(line)

    # This is for the outliers that have 0 length
    if length < 0.0000001:
        length = 0.01
    return length


def rapid_master_files(streams_gpq: str,
                       save_dir: str,
                       id_field: str = 'LINKNO',
                       ds_id_field: str = 'DSLINKNO',
                       length_field: str = 'Length',
                       default_k: float = 0.35,
                       default_x: float = .3, ) -> None:
    """
    Create RAPID master files from a stream network

    Saves the following files to the save_dir:
        - rapid_inputs_master.parquet
    Args:
        streams_gpq:
        save_dir:
        id_field:
        ds_id_field:
        length_field:
        default_k:
        default_x:

    Returns:

    """
    sgdf = gpd.read_parquet(streams_gpq)

    # enforce data types
    sgdf['LINKNO'] = sgdf['LINKNO'].astype(int)
    sgdf['DSLINKNO'] = sgdf['DSLINKNO'].astype(int)
    sgdf['USLINKNO1'] = sgdf['USLINKNO1'].astype(int)
    sgdf['USLINKNO2'] = sgdf['USLINKNO2'].astype(int)
    sgdf['strmOrder'] = sgdf['strmOrder'].astype(int)
    sgdf['Length'] = sgdf['Length'].astype(float)
    sgdf['lat'] = sgdf['lat'].astype(float)
    sgdf['lon'] = sgdf['lon'].astype(float)
    sgdf['z'] = sgdf['z'].astype(int)

    # length is in m, convert to km, then multiply by 3600 to get km/hr, then multiply by default k (velocity)
    sgdf["musk_k"] = sgdf['geometry'].apply(_calculate_geodesic_length) / 1000 * 3600 * default_k
    sgdf["musk_kfac"] = sgdf["musk_k"].values.flatten()
    sgdf["musk_x"] = default_x
    sgdf["musk_xfac"] = default_x
    sgdf['musk_k'] = sgdf['musk_k'].round(3)
    sgdf['musk_kfac'] = sgdf['musk_kfac'].round(3)
    sgdf['musk_x'] = sgdf['musk_x'].round(3)
    sgdf['musk_xfac'] = sgdf['musk_xfac'].round(3)

    # Fix 0 length segments
    logger.info('\tLooking for 0 length segments')
    if 0 in sgdf[length_field].values:
        zero_length_fixes_df = identify_0_length(sgdf, id_field, ds_id_field, length_field)
        zero_length_fixes_df.to_csv(os.path.join(save_dir, 'mod_zero_length_streams.csv'), index=False)
        sgdf = correct_0_length_streams(sgdf, zero_length_fixes_df, id_field)

    # Fix basins with ID of 0
    if 0 in sgdf[id_field].values:
        logger.info('\tFixing basins with ID of 0')
        pd.DataFrame({
            id_field: [0, ],
            'centroid_x': sgdf[sgdf[id_field] == 0].centroid.x.values[0],
            'centroid_y': sgdf[sgdf[id_field] == 0].centroid.y.values[0]
        }).to_csv(os.path.join(save_dir, 'mod_basin_zero_centroid.csv'), index=False)

    logger.info('\tSorting streams topologically')
    G = create_directed_graphs(sgdf, save_dir, id_field=id_field, ds_id_field=ds_id_field)
    sorted_order = sort_topologically(G)
    sgdf = (
        sgdf
        .set_index(pd.Index(sgdf[id_field]))
        .reindex(sorted_order)
        .reset_index(drop=True)
    )

    # Drop trees with small total length/area
    logger.info('\tFinding and removing small trees')
    small_tree_outlet_ids = sgdf.loc[np.logical_and(
        sgdf[ds_id_field] == -1,
        sgdf['DSContArea'] < 100_000_000
    ), id_field].values
    small_tree_segments = [nx.ancestors(G, x) for x in small_tree_outlet_ids]
    small_tree_segments = set().union(*small_tree_segments).union(small_tree_outlet_ids)
    (
        pd
        .DataFrame(small_tree_segments, columns=['drop'])
        .to_csv(os.path.join(save_dir, 'mod_drop_small_trees.csv'), index=False)
    )
    sgdf = sgdf.loc[~sgdf[id_field].isin(small_tree_segments)]

    # label watersheds by terminal node
    logger.info('\tLabeling watersheds by terminal node')
    for term_node in sgdf[sgdf[ds_id_field] == -1][id_field].values:
        sgdf.loc[sgdf[id_field].isin(list(nx.ancestors(G, term_node)) + [term_node, ]), 'TerminalNode'] = term_node
    sgdf['TerminalNode'] = sgdf['TerminalNode'].astype(int)

    # if calculate_rapid_connectivity:
    logger.info('\tCalculating RAPID connect columns')
    us_ids = sgdf[id_field].apply(lambda x: list(G.predecessors(x)))
    count_us = us_ids.apply(lambda x: len(x))
    max_num_upstream = np.max(count_us)
    us_ids = us_ids.apply(lambda x: x + [-1, ] * (max_num_upstream - len(x)))
    us_ids = pd.DataFrame(us_ids.tolist(), index=us_ids.index)
    upstream_columns = [f'USLINKNO{i}' for i in range(1, us_ids.shape[1] + 1)]
    us_ids.columns = upstream_columns
    us_ids['CountUS'] = count_us
    sgdf = sgdf.drop(columns=[x for x in sgdf.columns if x.startswith('USLINKNO')]).join(us_ids)
    sgdf[upstream_columns] = sgdf[upstream_columns].fillna(-1).astype(int)

    dtypes = sgdf.dtypes.to_dict()
    dtypes.pop('geometry')
    sgdf = (
        sgdf
        .drop(columns=['geometry'])
        .dropna()
        .astype(dtypes)
    )

    logger.info('\tWriting RAPID master parquets')
    sgdf.to_parquet(os.path.join(save_dir, "rapid_inputs_master.parquet"))

    return


def dissolve_headwater_table(save_dir: str):
    logger.info('Slimming streams table')
    streams_df = pd.read_parquet(os.path.join(save_dir, "rapid_inputs_master.parquet"))

    logger.info('\tDissolving headwater streams in inputs master')
    o2_to_dissolve = find_headwater_streams_to_dissolve(streams_df)

    # streams_df = streams_df[~streams_df['LINKNO'].isin(o2_to_dissolve.values[:, 1:].flatten())].reset_index(drop=True)
    # streams_df.loc[streams_df['LINKNO'].isin(o2_to_dissolve.values[:, 0]), ['USLINKNO1', 'USLINKNO2', 'USLINKNO3']] = -1

    for streams_to_merge in o2_to_dissolve.values:
        streams_df.loc[streams_df['LINKNO'].isin(streams_to_merge), 'LINKNO'] = streams_to_merge[0]
    agg_rules = {
        # 'LINKNO': 'last',
        'DSLINKNO': 'last',
        'DSNODEID': 'last',
        'strmOrder': 'last',
        # 'Length': lambda x: x.iloc[-1] + x.iloc[:-1].max() if len(x) > 1 else x.iloc[0],
        'Length': lambda x: x.sum() if len(x) > 1 else x.iloc[0],
        'Magnitude': 'last',
        'DSContArea': 'last',
        'strmDrop': lambda x: x.iloc[-1] + x.iloc[:-1].max() if len(x) > 1 else x.iloc[0],
        'Slope': lambda x: -1,
        'StraightL': lambda x: -1,
        'USContArea': lambda x: x.iloc[:-1].sum() if len(x) > 1 else x.iloc[0],
        'WSNO': 'last',
        'DOUTEND': 'last',
        'DOUTSTART': lambda x: x.iloc[:-1].max() if len(x) > 1 else x.iloc[0],
        'DOUTMID': lambda x: x.mean() if len(x) > 1 else x.iloc[0],
        'lat': 'last',
        'lon': 'last',
        'z': 'last',
        'musk_k': lambda x: x.iloc[-1] + x.iloc[:-1].max() if len(x) > 1 else x.iloc[0],
        'musk_x': 'last',
        'musk_kfac': lambda x: x.iloc[-1] + x.iloc[:-1].max() if len(x) > 1 else x.iloc[0],
        'musk_xfac': 'last',
        'CountUS': lambda x: 0,
    }
    agg_rules.update({
        col: (lambda x: -1) for col in sorted(streams_df.columns) if col.startswith('USLINKNO')
    })
    streams_df = streams_df.groupby('LINKNO').agg(agg_rules).reset_index()

    streams_df.to_parquet(os.path.join(save_dir, "rapid_inputs_slim.parquet"))
    return


def rapid_input_csvs(save_dir: str,
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
        id_field:
        ds_id_field:

    Returns:

    """
    logger.info('Creating RAPID input csvs')

    streams_df = pd.read_parquet(os.path.join(save_dir, "rapid_inputs_slim.parquet"))

    logger.info('\tWriting Rapid Connect CSV')
    # upstream_columns = sorted([x for x in streams_df.columns if 'USLINKNO' in x])
    # rapcon_columns = [id_field, ds_id_field, 'CountUS', ] + upstream_columns
    # rapcon_df = streams_df[rapcon_columns].copy()
    # rapcon_df[upstream_columns] = rapcon_df[upstream_columns].replace(-1, 0)
    #
    # (
    #     rapcon_df
    #     .fillna(0)
    #     .astype(int)
    #     .to_csv(os.path.join(save_dir, "rapid_connect.csv"), index=False, header=False)
    # )

    downstream_field = ds_id_field
    list_all = []
    max_count_Upstream = 0
    for hydroid in streams_df[id_field].values:
        # find the HydroID of the upstreams
        list_upstreamID = streams_df.loc[streams_df[downstream_field] == hydroid, id_field].values
        # count the total number of the upstreams
        count_upstream = len(list_upstreamID)
        if count_upstream > max_count_Upstream:
            max_count_Upstream = count_upstream
        nextDownID = streams_df.loc[streams_df[id_field] == hydroid, downstream_field].values[0]

        row_dict = {'HydroID': hydroid, 'NextDownID': nextDownID, 'CountUpstreamID': count_upstream}
        for i in range(count_upstream):
            row_dict[f'UpstreamID{i + 1}'] = list_upstreamID[i]
        list_all.append(row_dict)

        # Fill in NaN values for any missing upstream IDs
    for i in range(max_count_Upstream):
        col_name = f'UpstreamID{i + 1}'
        for row in list_all:
            if col_name not in row:
                row[col_name] = 0

    df = pd.DataFrame(list_all)
    df.to_csv(os.path.join(save_dir, 'rapid_connect.csv'), index=False, header=None)

    logger.info('\tWriting RAPID Input CSVS')
    streams_df[id_field].to_csv(os.path.join(save_dir, "riv_bas_id.csv"), index=False, header=False)
    streams_df["musk_k"].to_csv(os.path.join(save_dir, "k.csv"), index=False, header=False)
    streams_df["musk_x"].to_csv(os.path.join(save_dir, "x.csv"), index=False, header=False)
    streams_df["musk_kfac"].to_csv(os.path.join(save_dir, "kfac.csv"), index=False, header=False)
    streams_df["musk_xfac"].to_csv(os.path.join(save_dir, "xfac.csv"), index=False, header=False)
    streams_df[[id_field, 'lat', 'lon', 'z']].to_csv(os.path.join(save_dir, "comid_lat_lon_z.csv"), index=False)

    return


def sort_topologically(digraph_from_headwaters: nx.DiGraph) -> np.array:
    return np.array(list(nx.topological_sort(digraph_from_headwaters))).astype(int)


def create_directed_graphs(df: pd.DataFrame, save_dir: str, id_field='LINKNO', ds_id_field='DSLINKNO'):
    logger.info('\tCreating DiGraphs')
    G = nx.DiGraph()

    for node in df[id_field].values:
        G.add_node(node)
    for i, row in df.iterrows():
        if row[ds_id_field] != -1:
            G.add_edge(row[id_field], row[ds_id_field])

    logger.info('\tWriting DiGraphs')
    region_name = os.path.basename(save_dir)
    nx.write_gexf(G, os.path.join(save_dir, f'digraph_from_headwaters_{region_name}.gexf'))
    return G
