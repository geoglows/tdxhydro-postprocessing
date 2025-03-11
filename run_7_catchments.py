import glob
import logging
import os
import sys

import geopandas as gpd
from natsort import natsorted
import pandas as pd
import tdxhydrorapid as rp

tdx_geoparquet_dir = r"D:\geoglows_v3\parquets"
tdx_inputs_dir = r"C:\Users\lrr43\Documents\rfs-v2\hydrography-tdxhydro-sources\alterations"
hydrography_dir = r"C:\Users\lrr43\Documents\rfs-v2\hydrography"
master_table_path = r"C:\Users\lrr43\Documents\rfs-v2\geoglows-v2-master-table.parquet"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

def save(gdf: gpd.GeoDataFrame, 
         mdf: pd.DataFrame, 
         lake_gdf: gpd.GeoDataFrame, 
         vpu: int, 
         hydrography_dir: str):
    
    vpu_dir = os.path.join(hydrography_dir, f"vpu={vpu}")
    os.makedirs(vpu_dir, exist_ok=True)
    vpu_numbers = mdf.loc[mdf['VPUCode'] == vpu, 'LINKNO'].unique()
    (
        gdf
        .loc[gdf['LINKNO'].isin(vpu_numbers)]
        .to_file(os.path.join(vpu_dir, f'catchments_{vpu}.spatialite'), driver='SQLite', spatialite=True)
    )

    lake_subset: gpd.GeoDataFrame = lake_gdf.loc[lake_gdf['LINKNO'].isin(vpu_numbers)]
    if not lake_gdf.empty:
        lake_subset.to_file(os.path.join(vpu_dir, f'lakes_{vpu}.gpkg'))

os.makedirs(hydrography_dir, exist_ok=True)
mdf = pd.read_parquet(master_table_path)
for tdx in natsorted(glob.glob(os.path.join(tdx_inputs_dir, '*'))):
    tdxnumber = os.path.basename(tdx)
    logging.info(tdxnumber)
    catchments_gpq = os.path.join(tdx_geoparquet_dir, f'TDX_streamreach_basins_{tdxnumber}_01.parquet')
    gdf = gpd.read_parquet(catchments_gpq)
    gdf, lake_gdf = (
        rp
        .network
        .dissolve_catchments(tdx, gdf)
    )

    logging.info("\tSubsetting catchments")
    for vpu in natsorted(mdf.loc[mdf['TDXHydroRegion'] == tdxnumber, 'VPUCode'].unique()):
        save(gdf, mdf, lake_gdf, vpu, hydrography_dir)

    gdf = None
    
