import geopandas as gpd
import glob
import os
import json
import numpy as np
from pyproj import Geod

gpkg_dir = '/Users/rchales/Data/TDXHydro*'
gpq_dir = '/Volumes/EB406_T7_2/TDXHydroGeoParquet'


def _calculate_geodesic_length(line) -> float:
    """
    Input is shapely geometry, should be all shapely LineString objects

    returns length in meters
    """
    length = Geod(ellps='WGS84').geometry_length(line)

    # This is for the outliers that have 0 length
    if length < 0.0000001:
        length = 0.01
    return length


# add globally unique ID numbers
with open(os.path.join(os.path.dirname(__file__), 'tdxhydrorapid', 'network_data', 'tdx_header_numbers.json')) as f:
    tdx_header_numbers = json.load(f)

for gpkg in sorted(glob.glob(os.path.join(gpkg_dir, 'TDX*.gpkg'))):
    region_number = os.path.basename(gpkg).split('_')[-2]
    tdx_header_number = int(tdx_header_numbers[str(region_number)])

    out_file_name = os.path.join(gpq_dir, os.path.basename(gpkg).replace('.gpkg', '.parquet'))
    if os.path.exists(out_file_name):
        continue

    gdf = gpd.read_file(gpkg)

    if 'streamnet' in os.path.basename(gpkg):
        gdf[['lon', 'lat']] = np.vstack(gdf.geometry.apply(lambda x: x.coords[0]).to_numpy())
        gdf['z'] = 0
        gdf['LINKNO'] = gdf['LINKNO'].astype(int)
        gdf['DSLINKNO'] = gdf['DSLINKNO'].astype(int)
        gdf['USLINKNO1'] = gdf['USLINKNO1'].astype(int)
        gdf['USLINKNO2'] = gdf['USLINKNO2'].astype(int)
        gdf['strmOrder'] = gdf['strmOrder'].astype(int)
        gdf['Length'] = gdf['Length'].astype(float)
        gdf['lat'] = gdf['lat'].astype(float)
        gdf['lon'] = gdf['lon'].astype(float)
        gdf['z'] = gdf['z'].astype(int)
        gdf['LengthGeodesicMeters'] = gdf['geometry'].apply(_calculate_geodesic_length)

        gdf['TDXHydroRegion'] = region_number
        gdf['TDXHydroLinkNo'] = tdx_header_number * 10_000_000 + gdf['LINKNO']

    else:
        gdf['TDXHydroLinkNo'] = tdx_header_number * 10_000_000 + gdf['streamID']

    gdf.to_parquet(out_file_name)
