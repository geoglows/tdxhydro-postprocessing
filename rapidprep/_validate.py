import glob
import logging
import os
from multiprocessing import Pool

import geopandas as gpd
import numpy as np
import pandas as pd

REQUIRED_MODIFICATION_FILES = [
    'adjoint_tree.json',
    'mod_zero_length_streams.csv',
    'mod_drop_small_trees.csv',
    'mod_dissolve_headwaters.json',
    'mod_prune_shoots.json',
]

REQUIRED_RAPID_FILES = [
    'rapid_connect.csv',
    'riv_bas_id.csv',
    'comid_lat_lon_z.csv',
    'k.csv',
    'kfac.csv',
    'x.csv',
]

REQUIRED_GEOPACKAGES = [
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
    # Look for RAPID files
    missing_rapid_files = [f for f in REQUIRED_RAPID_FILES if not os.path.isfile(os.path.join(directory, f))]

    # look for weight tables
    weight_tables = glob.glob(os.path.join(directory, 'weight_*.csv'))

    # look for dissolved support files
    missing_network_files = [f for f in REQUIRED_MODIFICATION_FILES if not os.path.isfile(os.path.join(directory, f))]

    # look for geopackages
    missing_geopackages = [f for f in REQUIRED_GEOPACKAGES if len(glob.glob(os.path.join(directory, f))) == 0]
    count_geopackages = list(glob.glob(os.path.join(directory, '*.gpkg')))

    # summarize findings
    logger.info(f'Validating directory: {directory}')
    if all([
        len(missing_rapid_files) == 0,
        len(weight_tables) >= 3,
        len(missing_network_files) == 0,
    ]):
        logger.info('All expected files found in this directory')
        logger.info(f'  Found {len(weight_tables)} weight tables')
        logger.info(f'  Found {len(count_geopackages)} geopackages')
        logger.info('')
        return True

    if len(missing_rapid_files) != 0:
        logger.info('\tMissing RAPID files:')
        for file in missing_rapid_files:
            logger.info(f'\t{file}')

    logger.info(f'\tFound {len(weight_tables)} weight tables')
    for table in weight_tables:
        logger.info(f'\t{table}')

    if len(missing_network_files) != 0:
        logger.info('\tMissing modification files:')
        for file in missing_network_files:
            logger.info(f'\t{file}')

    if len(missing_geopackages) != 0:
        logger.info('\tMissing geopackages:')
        for file in missing_geopackages:
            logger.info(f'\t{file}')

    logger.info('')
    return False


def has_base_files(directory: str):
    if not all([
        os.path.exists(os.path.join(directory, 'adjoint_tree.json')),
        os.path.exists(os.path.join(directory, 'mod_zero_length_streams.csv')),
    ]) and len(glob.glob(os.path.join(directory, 'weight_*_full.csv'))) >= 0:
        return False


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
