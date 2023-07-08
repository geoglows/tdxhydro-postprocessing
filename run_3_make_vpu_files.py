import glob
import os

import pandas as pd

import tdxhydrorapid as rp

inputs_directory = '/Volumes/EB406_T7_2/TDXHydroRapid_V15'
final_inputs_directory = '/Volumes/EB406_T7_2/GEOGLOWS2/'
if not os.path.exists(final_inputs_directory):
    os.mkdir(final_inputs_directory)

mdf = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(inputs_directory, '*', 'rapid_inputs*.parquet'))])
mdf['TerminalNode'] = (
        mdf['TDXHydroLinkNo'].astype(str).str[:2].astype(int) * 10_000_000 + mdf['TerminalNode']
).astype(int)
# vpu_table = pd.read_parquet('./tdxhydro_vpu_table.parquet')
vpu_table = pd.read_csv('./vpu_table_for_revisions.csv')
mdf = mdf.merge(vpu_table, on='TerminalNode', how='left')

if not mdf[mdf['VPUCode'].isna()].empty:
    raise RuntimeError('Some terminal nodes are not in the VPU table and must be fixed before continuing.')

mdf.to_parquet(os.path.join(final_inputs_directory, 'master_table.parquet'))
rp.inputs.vpu_files_from_masters(mdf, final_inputs_directory, inputs_directory)
