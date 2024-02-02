import logging
import os
import shutil
import sys

import pandas as pd

import tdxhydrorapid as rp

tdx_inputs_dir = '/Volumes/T9Hales4TB/geoglows2/tdxhydro-inputs'
final_output_dir = '/Volumes/T9Hales4TB/geoglows2/'
vpu_inputs_dir = os.path.join(final_output_dir, 'inputs')
gpkg_dir = os.path.join(final_output_dir, 'streams')
vpu_assignment_table = './tdxhydrorapid/network_data/vpu_table.csv'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

os.makedirs(vpu_inputs_dir, exist_ok=True)
os.makedirs(gpkg_dir, exist_ok=True)

logging.info('Creating Model Master Table')
master_table_path = os.path.join(os.path.dirname(vpu_inputs_dir), 'geoglows-v2-master-table.parquet')
if not os.path.exists(master_table_path):
    rp.inputs.concat_tdxregions(tdx_inputs_dir, vpu_assignment_table, master_table_path)
mdf = pd.read_parquet(master_table_path)
logging.info(f'Total streams: {len(mdf)}')

for vpu in sorted(mdf['VPUCode'].unique()):
    logging.info(vpu)
    vpu_df = mdf.loc[mdf['VPUCode'] == vpu]
    tdx_region = str(vpu_df['TDXHydroRegion'].values[0])

    vpu_dir = os.path.join(vpu_inputs_dir, str(vpu))
    if os.path.exists(vpu_dir):
        if rp.check_outputs_are_valid(vpu_dir):
            continue
        else:
            shutil.rmtree(vpu_dir)

    os.makedirs(vpu_dir, exist_ok=True)
    try:
        rp.inputs.vpu_files_from_masters(vpu_df,
                                         vpu_dir,
                                         tdxinputs_directory=tdx_inputs_dir,
                                         make_gpkg=True,
                                         gpkg_dir=gpkg_dir, )
    except Exception as e:
        logging.info(vpu)
        logging.info(tdx_region)
        logging.info(e)
        shutil.rmtree(vpu_dir)
        continue

    if not rp.check_outputs_are_valid(vpu_dir):
        shutil.rmtree(vpu_dir)
        continue
