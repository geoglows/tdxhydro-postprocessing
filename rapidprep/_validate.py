import glob
import logging
import os
from multiprocessing import Pool

import geopandas as gpd
import numpy as np
import pandas as pd


NETWORK_TRACE_FILES = (
    'adjoint_tree.json',
    'mod_zero_length_streams.csv',
)
RAPID_MASTER_FILES = (
    'rapid_connect_master.parquet',
    'rapid_inputs_master.parquet'
)
MODIFICATION_FILES = [
    'adjoint_tree.json',
    'mod_zero_length_streams.csv',
    'mod_drop_small_trees.csv',
    'mod_dissolve_headwaters.json',
    'mod_prune_shoots.json',
]
RAPID_FILES = [
    'rapid_connect.csv',
    'riv_bas_id.csv',
    'comid_lat_lon_z.csv',
    'k.csv',
    'kfac.csv',
    'x.csv',
]
GEOPACKAGES = [
    'TDX_streamnet_*_model.gpkg',
    'TDX_streamnet_*_vis.gpkg',
    'TDX_streamreach_basins_*_corrected.gpkg'
]

logger = logging.getLogger(__name__)


def is_valid_result(directory: str):
    """
    Validate that the directory contains the necessary files for RAPID.

    Args:
        directory (str): Path to the directory to validate
    """
    logger.info(f'Validating {directory}')

    has_network_trace_files = all([os.path.exists(os.path.join(directory, f)) for f in NETWORK_TRACE_FILES])
    has_rapid_master_files = all([os.path.exists(os.path.join(directory, f)) for f in RAPID_MASTER_FILES])
    has_mod_files = all([os.path.exists(os.path.join(directory, f)) for f in MODIFICATION_FILES])
    has_rapid_files = all([os.path.exists(os.path.join(directory, f)) for f in RAPID_FILES])
    # has_geopackages = all([len(glob.glob(os.path.join(directory, f))) > 0 for f in GEOPACKAGES])
    has_geopackages = True
    weight_tables = glob.glob(os.path.join(directory, 'weight_*.csv'))

    if all([has_network_trace_files, has_rapid_master_files, has_mod_files, has_rapid_files, has_geopackages]):
        logger.info('All expected files found in this directory')
        logger.info(f'  Found {len(weight_tables)} weight tables')
        logger.info('')
        return True

    if not has_network_trace_files:
        logger.info('  ERROR: Missing network trace files')
    if not has_rapid_master_files:
        logger.info('  ERROR: Missing RAPID master files')
    if not has_mod_files:
        logger.info('  ERROR: Missing modification files')
    if not has_rapid_files:
        logger.info('  ERROR: Missing RAPID files')
    if not has_geopackages:
        logger.info('  ERROR: Missing geopackages')
    logger.info('')
    return False


def has_base_files(directory: str):
    has_rapid_master_files = all([os.path.exists(os.path.join(directory, f)) for f in RAPID_MASTER_FILES])
    has_network_trace_files = all([os.path.exists(os.path.join(directory, f)) for f in NETWORK_TRACE_FILES])
    has_weight_tables = len(glob.glob(os.path.join(directory, 'weight_*_full.csv'))) >= 3
    return has_rapid_master_files and has_network_trace_files and has_weight_tables


def _get_gdf_rows(gpkg):
    return os.path.basename(gpkg).split('_')[2], gpd.read_file(gpkg, ignore_geometry=True).shape[0]


def count_streams(output_path: str, n_processes: int):
    streams_paths = sorted(glob.glob(os.path.join(output_path, '*/TDX_streamnet*model.gpkg')))
    with Pool(n_processes) as p:
        results = p.map(count_streams, streams_paths)
    results = np.array(results)
    pd.DataFrame(results, columns=['region', 'count']).to_csv('network_data/stream_counts.csv', index=False)

    merged_df = (
        pd
        .read_csv('network_data/stream_counts_source.csv')
        .merge(pd.read_csv('network_data/stream_counts_headdis.csv'),
               how='outer', left_on='region', right_on='region')
        .merge(pd.read_csv('network_data/stream_counts_headdis_drop75k.csv'),
               how='outer', left_on='region', right_on='region')
        .sort_values(by='region')
    )
    merged_df['percent_remove_dis'] = (merged_df['count'] - merged_df['count_dis']) / merged_df['count']
    merged_df['percent_remove_disdrp'] = (merged_df['count'] - merged_df['count_dis_drp']) / merged_df['count']
    merged_df['percent_remove_dis'] = merged_df['percent_remove_dis'].round(4)
    merged_df['percent_remove_disdrp'] = merged_df['percent_remove_disdrp'].round(4)
    merged_df.to_csv('network_data/stream_counts_merged.csv', index=False)


def count_rivers_in_generated_files(input_dir: str):
    n_comid_lat_lon_z = pd.read_csv(os.path.join(input_dir, 'comid_lat_lon_z.csv')).shape[0]
    n_rapidconnect = pd.read_csv(os.path.join(input_dir, 'rapid_connect.csv'), header=None).shape[0]
    n_rivbasid = pd.read_csv(os.path.join(input_dir, 'riv_bas_id.csv'), header=None).shape[0]
    n_k = pd.read_csv(os.path.join(input_dir, 'k.csv'), header=None).shape[0]
    n_x = pd.read_csv(os.path.join(input_dir, 'x.csv'), header=None).shape[0]
    n_weights = []
    for f in sorted(glob.glob(os.path.join(input_dir, 'weight_*.csv'))):
        df = pd.read_csv(f)
        n_weights.append((os.path.basename(f), df.iloc[:, 0].unique().shape[0]))
    # print(f'comid_lat_lon_z: {n_comid_lat_lon_z}')
    # print(f'rapid_connect: {n_rapidconnect}')
    # print(f'riv_bas_id: {n_rivbasid}')
    # print(f'k: {n_k}')
    # print(f'x: {n_x}')
    # for f, n in n_weights:
    #     if 'full' in f:
    #         continue
    #     print(f'{f}: {n}')

    all_nums = [n_comid_lat_lon_z, n_rapidconnect, n_rivbasid, n_k, n_x] + [n for f, n in n_weights if 'full' not in f]
    print(os.path.basename(input_dir))
    print(f'All nums: {all_nums}')
    print(f'All match: {all([n == all_nums[0] for n in all_nums])}')
    print('')
