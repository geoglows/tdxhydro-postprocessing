import glob
import json
import logging
import os
import types

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .network import correct_0_length_streams
from .network import create_directed_graphs
from .network import find_branches_to_prune
from .network import find_headwater_branches_to_dissolve
from .network import identify_0_length
from .network import sort_topologically

# set up logging
logger = logging.getLogger(__name__)

__all__ = [
    'rapid_master_files',
    'dissolve_branches',
    'prune_branches',
    'rapid_input_csvs',
    'concat_tdxregions',
    'vpu_files_from_masters',
]


def rapid_master_files(streams_gpq: str,
                       save_dir: str,
                       id_field: str = 'LINKNO',
                       ds_id_field: str = 'DSLINKNO',
                       length_field: str = 'Length',
                       default_velocity_factor: float = None,
                       default_x: float = .25,
                       drop_small_watersheds: bool = True,
                       dissolve_headwaters: bool = True,
                       prune_branches_from_main_stems: bool = True,
                       cache_geometry: bool = True,
                       min_drainage_area_m2: float = 200_000_000,
                       min_headwater_stream_order: int = 3,
                       min_velocity_factor: float = 0.4, ) -> None:
    """
    Create RAPID master files from a stream network

    Saves the following files to the save_dir:
        - rapid_inputs_master.parquet
        - {region_num}_dissolved_network.gpkg
        - mod_zero_length_streams.csv (if any 0 length streams are found)
        - mod_basin_zero_centroid.csv (if any basins have an ID of 0 and geometry is not available)
        - mod_drop_small_streams.csv (if drop_small_watersheds is True)
        - mod_dissolved_headwaters.csv (if dissolve_headwaters is True)
        - mod_pruned_branches.csv (if prune_branches_from_main_stems is True)


    Args:
        streams_gpq: str, path to the streams geoparquet
        save_dir: str, path to the directory to save the master files
        id_field: str, field name for the link id
        ds_id_field: str, field name for the downstream link id
        length_field: str, field name for the length of the stream segment
        default_velocity_factor: float, default velocity factor (k) for Muskingum routing
        default_x: float, default attenuation factor (x) for Muskingum routing

        drop_small_watersheds: bool, drop small watersheds
        dissolve_headwaters: bool, dissolve headwater branches
        prune_branches_from_main_stems: bool, prune branches from main stems
        cache_geometry: bool, save the dissolved geometry as a geoparquet
        min_drainage_area_m2: float, minimum drainage area in m2 to keep a watershed
        min_headwater_stream_order: int, minimum stream order to keep a headwater branch

    Returns:
        None
    """
    sgdf = gpd.read_parquet(streams_gpq)

    # length is in m, divide by estimated m/s to get k in seconds
    logger.info('\tCalculating Muskingum k and x')
    sgdf['velocity_factor'] = np.exp(0.16842 * np.log(sgdf['DSContArea']) - 4.68).round(3) \
        if default_velocity_factor is None else default_velocity_factor
    sgdf['velocity_factor'] = sgdf['velocity_factor'].clip(lower=min_velocity_factor)
    sgdf['musk_k'] = sgdf['LengthGeodesicMeters'] / sgdf['velocity_factor']
    sgdf["musk_x"] = default_x

    logger.info('\tRemoving 0 length segments')
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

    logger.info('\tCreating Directed Graph')
    G = create_directed_graphs(sgdf, id_field, ds_id_field=ds_id_field)
    sorted_order = sort_topologically(G)
    sgdf = (
        sgdf
        .set_index(pd.Index(sgdf[id_field]))
        .reindex(sorted_order)
        .reset_index(drop=True)
        .dropna(axis=0, subset=[id_field])
        .astype(sgdf.dtypes.to_dict())
        .set_index(pd.Index(range(1, len(sgdf) + 1)).rename('TopologicalOrder'))
        .reset_index()
        .dropna()
        .astype(sgdf.dtypes.to_dict())
    )

    # Drop trees with small total length/area
    if drop_small_watersheds:
        logger.info('\tFinding and removing small trees')
        small_tree_outlet_ids = sgdf.loc[np.logical_and(
            sgdf[ds_id_field] == -1,
            sgdf['DSContArea'] < min_drainage_area_m2
        ), id_field].values
        small_tree_segments = [nx.ancestors(G, x) for x in small_tree_outlet_ids]
        small_tree_segments = set().union(*small_tree_segments).union(small_tree_outlet_ids)
        (
            pd
            .DataFrame(small_tree_segments, columns=['drop'])
            .to_csv(os.path.join(save_dir, 'mod_drop_small_trees.csv'), index=False)
        )
        sgdf = sgdf.loc[~sgdf[id_field].isin(small_tree_segments)]

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

    if dissolve_headwaters:
        logger.info('\tFinding headwater streams to dissolve')
        geometry_diss = lambda x: gpd.GeoSeries(x).unary_union if cache_geometry else 'last'
        headwater_dissolve_dfs = []
        for strmorder in range(2, min_headwater_stream_order + 1):
            branches = find_headwater_branches_to_dissolve(sgdf, G, strmorder)
            if strmorder == 2:
                sgdf = dissolve_branches(sgdf, branches, geometry_diss=geometry_diss, k_agg_func=_k_agg_order_2)
            elif strmorder == 3:
                sgdf = dissolve_branches(sgdf, branches, geometry_diss=geometry_diss, k_agg_func=_k_agg_order_3)
            headwater_dissolve_dfs.append(branches)
        (
            pd
            .concat(headwater_dissolve_dfs)
            .fillna(-1)
            .astype(int)
            .to_csv(os.path.join(save_dir, 'mod_dissolve_headwater.csv'), index=False)
        )
        headwater_dissolve_dfs = []

    if prune_branches_from_main_stems:
        logger.info('\tFinding branches to prune')
        streams_to_prune = find_branches_to_prune(sgdf, G)
        streams_to_prune.to_csv(os.path.join(save_dir, 'mod_prune_streams.csv'), index=False)
        sgdf = prune_branches(sgdf, streams_to_prune)

    logger.info('\tLabeling watersheds by terminal node')
    for term_node in sgdf[sgdf[ds_id_field] == -1][id_field].values:
        sgdf.loc[sgdf[id_field].isin(list(nx.ancestors(G, term_node)) + [term_node, ]), 'TerminalNode'] = term_node
    sgdf['TerminalNode'] = sgdf['TerminalNode'].astype(int)

    if cache_geometry:
        logger.info('\tWriting altered geometry to geopackage')
        region_number = sgdf['TDXHydroRegion'].values[0]
        gpd.GeoDataFrame(sgdf)[['TDXHydroLinkNo', 'geometry']].to_parquet(
            os.path.join(save_dir, f"{region_number}_altered_network.geoparquet"))

    logger.info('\tWriting RAPID master parquet')
    sgdf.drop(columns=['geometry', ]).to_parquet(os.path.join(save_dir, "rapid_inputs_master.parquet"))
    return


def dissolve_branches(sgdf: pd.DataFrame,
                      head_to_dissolve: pd.DataFrame,
                      geometry_diss: types.FunctionType or str = 'last',
                      k_agg_func: types.FunctionType or str = 'last', ) -> pd.DataFrame:
    """
    Use pandas groupby to "dissolve" streams in the table by combining rows and handle metadata correctly

    Args:
        sgdf: streams geodataframe with all the metadata columns from the source files
        head_to_dissolve: dataframe with the values of streams to be dissolved and the ID they are dissolved to
        geometry_diss: a string or function to use in groupby to handle combining the geometry column
        k_agg_func: a string or function to use in groupby to handle combining the k_agg column

    Returns:
        a copy of the streams geodataframe with rows dissolved
    """
    logger.info('\tDissolving headwater streams in inputs master')
    for streams_to_merge in head_to_dissolve.values:
        sgdf.loc[sgdf['LINKNO'].isin(streams_to_merge), 'LINKNO'] = streams_to_merge[0]
    agg_rules = {
        # 'LINKNO': 'last',
        'DSLINKNO': 'last',
        'DSNODEID': 'last',
        'strmOrder': 'last',
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
        'LengthGeodesicMeters': 'last',
        'lat': 'last',
        'lon': 'last',
        'z': 'last',
        'TDXHydroRegion': 'last',
        'TDXHydroLinkNo': 'last',
        'TopologicalOrder': 'last',
        'geometry': geometry_diss,
    }

    if all([x in sgdf.columns for x in ['musk_k', 'musk_x', 'velocity_factor', 'CountUS', ]]):
        agg_rules.update({
            'musk_k': k_agg_func,
            'musk_x': 'last',
            'velocity_factor': 'last',
            'CountUS': lambda x: 0 if len(x) else x,
        })
    agg_rules.update({
        col: (lambda x: -1 if len(x) > 1 else x) for col in sorted(sgdf.columns) if col.startswith('USLINKNO')
    })
    return sgdf.groupby('LINKNO').agg(agg_rules).reset_index().sort_values('TopologicalOrder')


def prune_branches(sdf: pd.DataFrame, streams_to_prune: pd.DataFrame) -> pd.DataFrame:
    return sdf[~sdf['LINKNO'].isin(streams_to_prune.iloc[:, 1].values.flatten())]


def rapid_input_csvs(sdf: pd.DataFrame,
                     save_dir: str,
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

    Args:
        sdf: stream network dataframe
        save_dir: directory to save the regions outputs
        id_field: the field in the dataframe that contains the unique ID for each stream
        ds_id_field: the field in the dataframe that contains the unique ID for the downstream stream

    Returns:

    """
    logger.info('Creating RAPID input csvs')

    downstream_field = ds_id_field
    rapid_connect = []
    max_count_upstream = 0
    for hydroid in sdf[id_field].values:
        # find the HydroID of the upstreams
        list_upstream_ids = sdf.loc[sdf[downstream_field] == hydroid, id_field].values
        # count the total number of the upstreams
        count_upstream = len(list_upstream_ids)
        if count_upstream > max_count_upstream:
            max_count_upstream = count_upstream
        next_down_id = sdf.loc[sdf[id_field] == hydroid, downstream_field].values[0]

        row_dict = {'HydroID': hydroid, 'NextDownID': next_down_id, 'CountUpstreamID': count_upstream}
        for i in range(count_upstream):
            row_dict[f'UpstreamID{i + 1}'] = list_upstream_ids[i]
        rapid_connect.append(row_dict)

        # Fill in NaN values for any missing upstream IDs
    for i in range(max_count_upstream):
        col_name = f'UpstreamID{i + 1}'
        for row in rapid_connect:
            if col_name not in row:
                row[col_name] = 0

    logger.info('\tWriting Rapid Connect CSV')
    df = pd.DataFrame(rapid_connect)
    upstream_columns = [x for x in df.columns if x.startswith('UpstreamID')]
    header_number = _get_tdxhydro_header_number(sdf['TDXHydroRegion'].values.flatten()[0])
    df[df[['HydroID', 'NextDownID', *upstream_columns]] > 0] += int(header_number * 10_000_000)
    df.to_csv(os.path.join(save_dir, 'rapid_connect.csv'), index=False, header=None)

    logger.info('\tWriting RAPID Input CSVS')
    sdf['TDXHydroLinkNo'].to_csv(os.path.join(save_dir, "riv_bas_id.csv"), index=False, header=False)
    sdf["musk_k"].to_csv(os.path.join(save_dir, "k.csv"), index=False, header=False)
    sdf["musk_x"].to_csv(os.path.join(save_dir, "x.csv"), index=False, header=False)
    sdf[['TDXHydroLinkNo', 'lat', 'lon', 'z']].to_csv(os.path.join(save_dir, "comid_lat_lon_z.csv"), index=False)
    return


def concat_tdxregions(tdxinputs_dir: str, vpu_dir: str, vpu_table: str) -> None:
    mdf = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(tdxinputs_dir, '*', 'rapid_inputs*.parquet'))])

    # relabel the terminal nodes as globally unique IDs
    mdf['TerminalNode'] = (
            mdf['TDXHydroLinkNo'].astype(str).str[:2].astype(int) * 10_000_000 + mdf['TerminalNode']
    ).astype(int)

    vpu_df = pd.read_csv(vpu_table)
    mdf = mdf.merge(vpu_df, on='TerminalNode', how='left')

    if not mdf[mdf['VPUCode'].isna()].empty:
        raise RuntimeError('Some terminal nodes are not in the VPU table and must be fixed before continuing.')
    mdf['VPUCode'] = mdf['VPUCode'].astype(int)

    mdf.to_parquet(os.path.join(vpu_dir, 'geoglows-v2-master-table.parquet'))
    return


def vpu_files_from_masters(vpu_df: pd.DataFrame,
                           vpu_dir: str,
                           tdxinputs_directory: str,
                           make_gpkg: bool,
                           gpkg_dir: str, ) -> None:
    tdx_region = vpu_df['TDXHydroRegion'].values[0]
    vpu = vpu_df['VPUCode'].values[0]

    # make the rapid input files
    rapid_input_csvs(vpu_df, vpu_dir)

    # subset the weight tables
    logging.info('Subsetting weight tables')
    weight_tables = glob.glob(os.path.join(tdxinputs_directory, tdx_region, f'weight*.csv'))
    weight_tables = [x for x in weight_tables if '_full.csv' not in x]
    for weight_table in weight_tables:
        a = pd.read_csv(weight_table)
        a = a[a.iloc[:, 0].astype(int).isin(vpu_df['TDXHydroLinkNo'].values)]
        a.to_csv(os.path.join(vpu_dir, os.path.basename(weight_table)), index=False)

    if not make_gpkg:
        return
    logging.info('Making gpkg')
    altered_network = os.path.join(tdxinputs_directory, tdx_region, f'{tdx_region}_altered_network.geoparquet')
    vpu_network = os.path.join(gpkg_dir, f'streams_{vpu}.gpkg')
    if os.path.exists(altered_network):
        (
            gpd
            .read_parquet(altered_network)
            .merge(vpu_df, on='TDXHydroLinkNo', how='inner')
            .to_file(vpu_network, driver='GPKG')
        )
    return


def _k_agg_order_3(x: pd.Series) -> np.ndarray:
    return x.mean() * 3.5 if len(x) > 1 else x.iloc[0]


def _k_agg_order_2(x: pd.Series) -> np.ndarray:
    return x.iloc[-1] + x.iloc[:-1].max() if len(x) > 1 else x.iloc[0]


def _geom_diss(x: pd.Series or gpd.GeoSeries):
    return gpd.GeoSeries(x).unary_union


def _get_tdxhydro_header_number(region_number: int) -> int:
    with open(os.path.join(os.path.dirname(__file__), 'network_data', 'tdx_header_numbers.json')) as f:
        header_numbers = json.load(f)
    return int(header_numbers[str(region_number)])
