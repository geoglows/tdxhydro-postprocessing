import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
from pyproj import Geod

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hydrography.schema as schema

gpkg_dir = Path('test/gpkgs')
gpq_dir = Path('/Volumes/EB406_T7_3/geoglows_v3/parquets')
save_dir = Path('test/')

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
    with open(Path(__file__).parent / '..' / 'network_data' / 'tdxhydro_splits' / 'tdx_header_numbers.json') as f:
        tdx_header_numbers = json.load(f)

    gpq_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    for gpkg in sorted(gpkg_dir.glob('TDX*.gpkg')):
        region_number = gpkg.name.split('_')[-2]
        tdx_header_number = int(tdx_header_numbers[str(region_number)])
        logging.info(gpkg)

        out_file_name = gpq_dir / gpkg.name.replace('.gpkg', '.parquet')
        if out_file_name.exists():
            continue

        gdf = gpd.read_file(gpkg)

        if 'streamnet' in gpkg.name:
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
