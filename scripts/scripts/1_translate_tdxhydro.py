import glob
import json
import logging
import os
import sys

import geopandas as gpd
from pyproj import Geod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydrography.schema as schema

gpkg_dir = 'test/gpkgs'
gpq_dir = '/Volumes/EB406_T7_3/geoglows_v3/parquets'
save_dir = 'test/'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)


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


if __name__ == '__main__':
    logging.info('Converting TDX-Hydro GPKG to Geoparquet')
    # add globally unique ID numbers
    with open(os.path.join(os.path.dirname(__file__), '../../tdxhydrorapid', 'network_data', 'tdx_header_numbers.json')) as f:
        tdx_header_numbers = json.load(f)

    os.makedirs(gpq_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    for gpkg in sorted(glob.glob(os.path.join(gpkg_dir, 'TDX*.gpkg'))):
        region_number = os.path.basename(gpkg).split('_')[-2]
        tdx_header_number = int(tdx_header_numbers[str(region_number)])
        logging.info(gpkg)

        out_file_name = os.path.join(gpq_dir, os.path.basename(gpkg).replace('.gpkg', '.parquet'))
        if os.path.exists(out_file_name):
            continue

        gdf = gpd.read_file(gpkg)

        if 'streamnet' in os.path.basename(gpkg):
            gdf[schema.tdx_link_field] = gdf[schema.tdx_link_field].astype(int) + (tdx_header_number * 10_000_000)
            gdf[schema.tdx_ds_link_field] = gdf[schema.tdx_ds_link_field].astype(int)
            gdf.loc[gdf[schema.tdx_ds_link_field] != -1, schema.tdx_ds_link_field] = gdf[schema.tdx_ds_link_field] + (tdx_header_number * 10_000_000)
            gdf[schema.tdx_strm_order_field] = gdf[schema.tdx_strm_order_field].astype(int)
            gdf[schema.tdx_geodesic_length_field] = gdf[schema.geometry].apply(_calculate_geodesic_length)
            gdf[schema.tdx_region_field] = region_number

            gdf = gdf[schema.tdx_streamnet_output_columns]

        else:
            gdf[schema.tdx_link_field] = gdf[schema.basin_stream_id_field].astype(int) + (tdx_header_number * 10_000_000)
            gdf = gdf.drop(columns=[schema.basin_stream_id_field])

        gdf.to_parquet(out_file_name)
