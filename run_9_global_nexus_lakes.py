import os
import glob
import geopandas as gpd
import pandas as pd


hydrography_dir = r"D:\geoglows_v3\geoglows_v3\hydrography"
global_dir = r"D:\geoglows_v3\geoglows_v3\hydroprahy_global"


if __name__ == '__main__':
   # Look into subdirs of hydrography_dir
    lakes = glob.glob(os.path.join(hydrography_dir, '*', 'lakes_*.gpkg'))
    nexus = glob.glob(os.path.join(hydrography_dir, '*', 'nexus_*.gpkg'))

    # Read all the files
    lakes_gdf: gpd.GeoDataFrame = pd.concat([gpd.read_file(l) for l in lakes])
    nexus_gdf: gpd.GeoDataFrame = pd.concat([gpd.read_file(s) for s in nexus])

    # Write the files
    lakes_gdf.to_file(os.path.join(global_dir, 'global_lakes.gpkg'), driver='GPKG')
    nexus_gdf.to_file(os.path.join(global_dir, 'global_nexus.gpkg'), driver='GPKG')