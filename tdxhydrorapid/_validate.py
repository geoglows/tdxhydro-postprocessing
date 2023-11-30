import glob
import logging
import os

import pandas as pd

__all__ = [
    'check_outputs_are_valid',
    'tdxhydro_corrections_consistent',
    'RAPID_FILES',
]

RAPID_FILES = [
    'rapid_connect.csv',
    'riv_bas_id.csv',
    'comid_lat_lon_z.csv',
    'k.csv',
    'x.csv',
]

logger = logging.getLogger(__name__)


def tdxhydro_corrections_consistent(input_dir: str) -> bool:
    n_streams = pd.read_parquet(os.path.join(input_dir, 'rapid_inputs_master.parquet')).shape[0]
    weights = sorted(glob.glob(os.path.join(input_dir, 'weight_*.csv')))
    weights = [x for x in weights if 'full' not in x]
    n_weights = [pd.read_csv(x).iloc[:, 0].unique().shape[0] for x in weights]
    logger.info(f'Validating {os.path.basename(input_dir)}')
    logger.info(f'\tNumber of streams: {n_streams}')
    for w, n in zip(weights, n_weights):
        logger.info(f'\t{os.path.basename(w)}: {n}')
    if all([n == n_streams for n in n_weights]):
        logger.info(f'\t---Inputs Match---')
        return True
    logger.error(f'\tInputs Do Not Match')
    return False


def check_outputs_are_valid(input_dir: str) -> bool:
    n_comid_lat_lon_z = pd.read_csv(os.path.join(input_dir, 'comid_lat_lon_z.csv')).shape[0]
    n_rapidconnect = pd.read_csv(os.path.join(input_dir, 'rapid_connect.csv'), header=None).shape[0]
    n_rivbasid = pd.read_csv(os.path.join(input_dir, 'riv_bas_id.csv'), header=None).shape[0]
    n_k = pd.read_csv(os.path.join(input_dir, 'k.csv'), header=None).shape[0]
    n_x = pd.read_csv(os.path.join(input_dir, 'x.csv'), header=None).shape[0]
    n_weights = []
    for f in sorted(glob.glob(os.path.join(input_dir, 'weight_*.csv'))):
        df = pd.read_csv(f)
        n_weights.append((os.path.basename(f), df.iloc[:, 0].unique().shape[0]))

    logger.info('Checking for consistent numbers of basins in generated files')
    all_nums = [n_comid_lat_lon_z, n_rapidconnect, n_rivbasid, n_k, n_x] + [n for f, n in n_weights if 'full' not in f]
    all_match = all([n == all_nums[0] for n in all_nums])
    logger.info(f'\t{os.path.basename(input_dir)}')
    logger.info(f'\tAll nums: {all_nums}')
    logger.info(f'\tAll match: {all_match}')
    logger.info(f'\tcomid_lat_lon_z: {n_comid_lat_lon_z}')
    logger.info(f'\trapid_connect: {n_rapidconnect}')
    logger.info(f'\triv_bas_id: {n_rivbasid}')
    logger.info(f'\tk: {n_k}')
    logger.info(f'\tx: {n_x}')
    for f, n in n_weights:
        if 'full' in f:
            continue
        logger.info(f'\t{f}: {n}')

    if all_match:
        logger.info('\tGenerated files appear consistent')
    else:
        logger.error('\tERROR: Generated files are not consistent')
    return all_match


def has_slimmed_weight_tables(save_dir: str) -> bool:
    full_wts = sorted(glob.glob(os.path.join(save_dir, 'weight_*_full.csv')))
    has_slim_tables = [os.path.exists(f.replace('_full.csv', '.csv')) for f in full_wts]
    return all(has_slim_tables)
