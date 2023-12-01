import glob
import logging
import os
import shutil
import sys

import geopandas as gpd
import pandas as pd

import tdxhydrorapid as rp

tdx_inputs_dir = '/Volumes/EB406_T7_2/geoglows2/tdxhydro-inputs'
final_output_dir = '/Volumes/EB406_T7_2/geoglows2/'
vpu_inputs_dir = os.path.join(final_output_dir, 'inputs')
gpkg_dir = os.path.join(final_output_dir, 'streams')
vpu_table = './tdxhydrorapid/network_data/vpu_table.csv'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

os.makedirs(vpu_inputs_dir, exist_ok=True)
os.makedirs(gpkg_dir, exist_ok=True)

logging.info('Creating Model Master Table')
if not os.path.exists(os.path.join(vpu_inputs_dir, 'geoglows-v2-master-table.parquet')):
    rp.inputs.concat_tdxregions(tdx_inputs_dir, vpu_inputs_dir, vpu_table)
mdf = pd.read_parquet(os.path.join(vpu_inputs_dir, 'geoglows-v2-master-table.parquet'))
logging.info(f'Total streams: {len(mdf)}')

if not os.path.exists(os.path.join(vpu_inputs_dir, 'streams_simplified.geoparquet')):
    logging.info('Concat global simplified streams')
    mgdf = pd.concat([gpd.read_parquet(f) for f in glob.glob(os.path.join(tdx_inputs_dir, '*', '*.geoparquet'))])
    logging.info('Simplifying geometry')
    mgdf['geometry'] = mgdf['geometry'].simplify(0.01, preserve_topology=False)
    logging.info('Adding attributes')
    mgdf = mgdf.merge(mdf[['VPUCode', 'TDXHydroLinkNo']], on='TDXHydroLinkNo')
    logging.info('Writing to file')
    mgdf.to_parquet(os.path.join(vpu_inputs_dir, 'streams_simplified.geoparquet'))
    mgdf = None

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
