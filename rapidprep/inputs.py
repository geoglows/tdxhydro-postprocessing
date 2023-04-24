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

# set up logging
logger = logging.getLogger(__name__)


def _calculate_geodesic_length(line) -> float:
    """
    Input is shapely geometry, should be all shapely LineString objects
    """
    length = Geod(ellps='WGS84').geometry_length(line) / 1000  # To convert to km

    # This is for the outliers that have 0 length
    if length < 0.0000001:
        length = 0.001
    return length


def _make_rapid_connect_row(stream_id, streams_gdf):
    id_field = 'LINKNO'
    ds_id_field = 'DSLINKNO'
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
                       default_k: float = 0.35,
                       default_x: float = 3,
                       n_workers: int or None = 1) -> None:
    # Create rapid preprocessing files
    logger.info('Creating RAPID Master files')
    streams_gdf = gpd.read_file(streams_gpkg)

    with Pool(n_workers) as p:
        logger.info('\tCalculating lengths')
        streams_gdf["length_geod"] = p.map(_calculate_geodesic_length, streams_gdf.geometry.values)

        streams_gdf['lat'] = streams_gdf.geometry.apply(lambda geom: geom.xy[1][0]).values
        streams_gdf['lon'] = streams_gdf.geometry.apply(lambda geom: geom.xy[0][0]).values

        streams_gdf["musk_kfac"] = streams_gdf["length_geod"] * 3600
        streams_gdf["musk_k"] = streams_gdf["musk_kfac"] * default_k
        streams_gdf["musk_x"] = default_x * 0.1
        streams_gdf['z'] = 0

        streams_gdf = streams_gdf.drop(columns=['geometry'])

        logger.info('\tCalculating RAPID connect file')
        rapid_connect = p.starmap(_make_rapid_connect_row, [[x, streams_gdf] for x in streams_gdf[id_field].values])

    logger.info('\tWriting RAPID master parquets')
    (
        pd
        .DataFrame(rapid_connect)
        .fillna(0)
        .astype(int)
        .to_parquet(os.path.join(save_dir, "rapid_connect_master.parquet"))
    )
    streams_gdf.to_parquet(os.path.join(save_dir, "rapid_inputs_master.parquet"))
    return


def rapid_input_csvs(save_dir: str,
                     id_field: str = 'LINKNO',
                     n_processes: int or None = 1):
    logger.info('Creating RAPID input csvs')
    logger.info('\tReading master files')
    streams_df = pd.read_parquet(os.path.join(save_dir, 'rapid_inputs_master.parquet'))
    rapcon_df = pd.read_parquet(os.path.join(save_dir, 'rapid_connect_master.parquet')).astype(int)

    logger.info('\tReading stream modification files')
    with open(os.path.join(save_dir, 'mod_dissolve_headwaters.json'), 'r') as f:
        diss_headwaters = json.load(f)
        all_merged_headwater = set(chain.from_iterable([diss_headwaters[rivid] for rivid in diss_headwaters.keys()]))
    with open(os.path.join(save_dir, 'mod_prune_shoots.json'), 'r') as f:
        pruned_shoots = json.load(f)
        pruned_shoots = set([ids[-1] for _, ids in pruned_shoots.items()])
    small_trees = pd.read_csv(os.path.join(save_dir, 'mod_drop_small_trees.csv')).values.flatten()

    logger.info('\tDropping small trees')
    streams_df = streams_df.loc[~streams_df[id_field].isin(small_trees)]

    logger.info('\tDropping pruned shoots')
    streams_df = streams_df.loc[~streams_df[id_field].isin(pruned_shoots)]

    with Pool(n_processes) as p:
        # Apply corrections based on the stream modification files
        logger.info('\tMerging head water stream segment rows')
        corrected_headwater_rows = p.starmap(
            _combine_routing_rows,
            [[streams_df, id_to_keep, ids_to_merge] for id_to_keep, ids_to_merge in diss_headwaters.items()]
        )
        logger.info('\tApplying corrected rows to gdf')
        streams_df = pd.concat([
            streams_df.loc[~streams_df[id_field].isin(all_merged_headwater)],
            *corrected_headwater_rows
        ])

    logger.info('\tSorting IDs topologically')
    sorted_order = sort_topologically(streams_df)
    streams_df = (
        streams_df
        .set_index(pd.Index(streams_df[id_field]))
        .loc[sorted_order]
        .reset_index(drop=True)
    )
    streams_df[id_field].to_csv(os.path.join(save_dir, "riv_bas_id.csv"), index=False, header=False)

    # adjust rapid connect to match
    rapcon_df = rapcon_df.loc[rapcon_df.iloc[:, 0].isin(streams_df[id_field].values)]
    rapcon_df = (
        rapcon_df
        .set_index(pd.Index(rapcon_df.iloc[:, 0]))
        .loc[sorted_order]
        .reset_index(drop=True)
    )
    rapcon_df.to_csv(os.path.join(save_dir, "rapid_connect.csv"), index=False, header=False)

    logger.info('\tWriting csvs')
    streams_df["musk_kfac"].to_csv(os.path.join(save_dir, "kfac.csv"), index=False, header=False)
    streams_df["musk_k"].to_csv(os.path.join(save_dir, "k.csv"), index=False, header=False)
    streams_df["musk_x"].to_csv(os.path.join(save_dir, "x.csv"), index=False, header=False)
    streams_df[[id_field, 'lat', 'lon', 'z']].to_csv(os.path.join(save_dir, "comid_lat_lon_z.csv"), index=False)

    return


def sort_topologically(df):
    df = df[['LINKNO', 'DSLINKNO']].astype(int)
    df[df == -1] = np.nan
    digraph = nx.DiGraph()
    nodes = set(df['LINKNO'].dropna()).union(set(df['DSLINKNO'].dropna()))
    for node in nodes:
        digraph.add_node(node)
    for i, row in df.iterrows():
        if not pd.isna(row['DSLINKNO']):
            digraph.add_edge(row['LINKNO'], row['DSLINKNO'])

    # perform topological sorting on the graph
    sorted_nodes = np.array(list(nx.topological_sort(digraph))).astype(int)
    return sorted_nodes


# def make_hash_table(riv_bas_id_list):
#     hash_table = {}
#     for river_basin_id_index in range(len(riv_bas_id_list)):
#         hash_table[riv_bas_id_list[river_basin_id_index]] = river_basin_id_index
#     return hash_table
#
#
# def check_sorting(rapid_connect_id_list, rapid_connect_ds_list, hash_table, num_rapid_con_ids, num_river_basin_ids):
#     move_to_end = []
#     for index_rapid_connect in range(num_rapid_con_ids):
#         if rapid_connect_id_list[index_rapid_connect] not in hash_table:
#             continue
#         idx_id_in_rapid_connect = hash_table[rapid_connect_id_list[index_rapid_connect]]
#         if rapid_connect_ds_list[index_rapid_connect] in hash_table:
#             idx_ds_in_rapid_connect = hash_table[rapid_connect_ds_list[index_rapid_connect]]
#         else:
#             idx_ds_in_rapid_connect = num_river_basin_ids
#         if idx_id_in_rapid_connect > idx_ds_in_rapid_connect:
#             move_to_end.append(rapid_connect_id_list[idx_id_in_rapid_connect])
#
#     return move_to_end
#
#
# def sort_rivers(directory):
#     n_iter = 0
#     all_ids_to_move = []
#     rapid_master_file = os.path.join(directory, "rapid_inputs_master.parquet")
#     rapid_connect_master_file = os.path.join(directory, "rapid_connect_master.parquet")
#     if not os.path.exists(rapid_master_file) or not os.path.exists(rapid_connect_master_file):
#         return None
#     rapid_df = pd.read_parquet(rapid_master_file)
#     rapid_connect_df = pd.read_parquet(rapid_connect_master_file)
#     try:
#         while True:
#             riv_bas_id_list = rapid_df.values.flatten()
#             rapid_connect_id_list = rapid_connect_df.iloc[:, 0].values.flatten()
#             rapid_connect_ds_list = rapid_connect_df.iloc[:, 1].values.flatten()
#             num_river_basin_ids = len(riv_bas_id_list)
#             num_rapid_con_ids = len(rapid_connect_id_list)
#             hash_table = make_hash_table(riv_bas_id_list)
#             ids_to_move = check_sorting(rapid_connect_id_list, rapid_connect_ds_list, hash_table, num_rapid_con_ids,
#                                         num_river_basin_ids)
#             ids_to_move = np.array(ids_to_move).astype(int)
#             if len(ids_to_move) == 0:
#                 break
#
#             all_ids_to_move.append(ids_to_move)
#             selector = pd.Series(rapid_df.values.flatten()).isin(ids_to_move)
#             rapid_df = pd.concat([
#                 rapid_df[selector],
#                 rapid_df[~selector],
#             ]).reset_index(drop=True)
#
#             # selector = pd.Series(rapid_connect_df.iloc[:, 0].values.flatten()).isin(ids_to_move)
#             # rapid_connect_df = pd.concat([
#             #     rapid_connect_df[selector],
#             #     rapid_connect_df[~selector],
#             # ]).reset_index(drop=True)
#
#             n_iter += 1
#             if n_iter > 5_000:
#                 print('Too many iterations in folder: ' + directory)
#                 break
#
#         logger.info('Done sorting rivers in folder: ' + directory)
#         logger.info('Number of iterations: ' + str(n_iter))
#     except Exception as e:
#         logger.error('Error sorting rivers in folder: ' + directory)
#         logger.error(e)
#         return
#
#     return rapid_df
