import glob
import logging
import os
import shutil

import geopandas as gpd
import pandas as pd

import tdxhydrorapid as rp

tdx_inputs_dir = '/Volumes/EB406_T7_2/TDXHydroRapid_V15'
vpu_inputs_dir = '/Volumes/EB406_T7_2/VPUInputs/'
vpu_table = './vpu_table_for_revisions.csv'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='log.log',
    filemode='w'
)

if not os.path.exists(vpu_inputs_dir):
    os.mkdir(vpu_inputs_dir)

print('making master table')
if not os.path.exists(os.path.join(vpu_inputs_dir, 'master_table.parquet')):
    rp.inputs.concat_tdxregions(tdx_inputs_dir, vpu_inputs_dir, vpu_table)
mdf = pd.read_parquet(os.path.join(vpu_inputs_dir, 'master_table.parquet'))

if not os.path.exists(os.path.join(vpu_inputs_dir, 'global_streams_simplified.gpkg')):
    print('Concat global simplified streams')
    mgdf = pd.concat([gpd.read_parquet(f) for f in glob.glob(os.path.join(tdx_inputs_dir, '*', '*.geoparquet'))])
    print('Simplifying geometry')
    mgdf['geometry'] = mgdf['geometry'].apply(lambda x: x.simplify(0.005, preserve_topology=False))
    print('Adding attributes')
    mgdf = mgdf.merge(mdf[['VPUCode', 'TDXHydroLinkNo']], on='TDXHydroLinkNo')
    print('Writing to file')
    mgdf.to_parquet(os.path.join(vpu_inputs_dir, 'global_streams_simplified.geoparquet'))
    mgdf = None

for vpu in sorted(mdf['VPUCode'].unique()):
    print(vpu)
    vpu_df = mdf.loc[mdf['VPUCode'] == vpu]
    tdx_region = str(vpu_df['TDXHydroRegion'].values[0])

    vpu_dir = os.path.join(vpu_inputs_dir, str(vpu))
    if os.path.exists(vpu_dir):
        rp.check_outputs_are_valid(vpu_dir)
        continue

    try:
        os.makedirs(vpu_dir, exist_ok=True)
        rp.inputs.vpu_files_from_masters(vpu_df, vpu_dir, tdx_inputs_dir)
    except Exception as e:
        print(vpu)
        print(tdx_region)
        print(e)
        shutil.rmtree(vpu_dir)
        continue

    rp.check_outputs_are_valid(vpu_dir)
