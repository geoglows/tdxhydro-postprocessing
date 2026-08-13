import json
import logging
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Geod

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography.parquet as parquet
import hydrography.paths as paths
import hydrography.schema as schema
from hydrography.streams import add_outlet_coordinates

# the source gpkgs are not produced by this pipeline, so they are not under the data root
gpkg_dir = Path(os.environ.get('TDXHYDRO_GPKG_DIR') or 'test/gpkgs')
# the raw geoparquet every later step reads - see hydrography/paths.py and $RFS_DATA_ROOT
gpq_dir = paths.tdx_root

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)


wgs84 = Geod(ellps='WGS84')


def _calculate_geodesic_lengths(geoms) -> np.ndarray:
    """
    Geodesic length in metres of every LineString in ``geoms``, on the WGS84 ellipsoid.

    One ``Geod.inv`` call over every segment in the region at once rather than one
    ``geometry_length`` call per reach. The inverse geodesic is the whole cost - a region is ~600k
    vertices and pyproj solves them in a single C loop - so the only work here is flattening the
    lines into that one array and summing the segments back per line. Measured on 20k lines /
    606k vertices: 0.318 s per-geometry against 0.156 s here, and the results are bit-identical.

    LineString-only, which the source is: ``get_coordinates`` flattens a MultiLineString's parts
    into one run, so a multi-part input would gain a phantom segment bridging its parts. The
    ``add_outlet_coordinates`` call below asserts the same thing on the same geometry.
    """
    counts = shapely.get_num_coordinates(geoms)
    coords = shapely.get_coordinates(geoms)
    # every vertex starts a segment except the last one of each line - dropping those leaves each
    # remaining vertex paired with its successor inside the same line
    starts = np.ones(len(coords), dtype=bool)
    starts[np.cumsum(counts)[counts > 0] - 1] = False
    starts = np.flatnonzero(starts)
    a = coords[starts]
    b = coords[starts + 1]
    _, _, segments = wgs84.inv(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
    # bincount rather than add.reduceat: it sums an empty group to 0 instead of reading past it
    lengths = np.bincount(np.repeat(np.arange(len(geoms)), np.maximum(counts - 1, 0)),
                          weights=segments, minlength=len(geoms))

    # This is for the outliers that have 0 length
    return np.where(lengths < 0.0000001, 0.01, lengths)


if __name__ == '__main__':
    logging.info('Converting TDX-Hydro GPKG to Geoparquet')
    # add globally unique ID numbers
    with open(paths.network_data_root / 'tdxhydro_splits' / 'tdx_header_numbers.json') as f:
        tdx_header_numbers = json.load(f)

    gpq_dir.mkdir(parents=True, exist_ok=True)

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
            gdf[schema.tdx_geodesic_length_field] = _calculate_geodesic_lengths(gdf[schema.geometry].values)
            gdf[schema.tdx_region_field] = region_number
            # coordinate 0 of each line is the reach outlet - see add_outlet_coordinates
            gdf = add_outlet_coordinates(gdf)

            gdf = gdf[schema.tdx_standardized_columns]

        else:
            gdf[schema.tdx_link_field] = gdf[schema.basin_stream_id_field].astype(int) + (tdx_header_number * 10_000_000)
            gdf = gdf.drop(columns=[schema.basin_stream_id_field])

        # geoarrow + zstd, and deliberately not the BYTE_STREAM_SPLIT the published files use:
        # this is the one product written before the 1 m snap, and the encoding needs the snap to
        # pay. See hydrography/parquet.py. recompress_tdxhydro.py brings an already-converted tree
        # up to this without going back to the gpkgs.
        parquet.write_source_geoparquet(gdf, out_file_name)
