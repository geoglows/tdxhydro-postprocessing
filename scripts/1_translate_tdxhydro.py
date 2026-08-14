"""
Convert the source TDX-Hydro GPKGs to geoparquet and stamp the globally unique reach ids.

**This step owns the global id scheme.** Every downstream step - the one-time basin generation and
every per-release step alike - reads ids as written here and never derives them: LINKNO and
DSLINKNO go out already offset by the region's header number (tdx_header_numbers.json is consulted
nowhere else for ids), so a reach id is globally unique the moment the file exists. Trees
converted before this convention carry the stamped id in a TDXHydroLinkNo column beside a local
LINKNO instead; downstream readers accept either vintage by preferring TDXHydroLinkNo and mapping
DSLINKNO through the file's own local-to-global pairing.
"""
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from pyproj import Geod

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography.console as console
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


def convert(gpkg: Path, tdx_header_number: int, region_number: str) -> str:
    """One GPKG to geoparquet, ids stamped. Written aside and renamed so an interrupted run can
    never leave a truncated file the skip check would trust."""
    out_file_name = gpq_dir / gpkg.name.replace('.gpkg', '.parquet')
    if out_file_name.exists():
        return f'{gpkg.name}: already converted, skipped'
    started = time.time()

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
    partial = out_file_name.with_name(f'{out_file_name.name}.partial')
    parquet.write_source_geoparquet(gdf, partial)
    partial.replace(out_file_name)
    return f'{gpkg.name}: {len(gdf):,} rows, {time.time() - started:.0f}s -> {out_file_name.name}'


if __name__ == '__main__':
    console.banner('Translate TDX-Hydro to geoparquet')
    logging.info('Converting TDX-Hydro GPKG to Geoparquet')
    # add globally unique ID numbers
    with open(paths.network_data_root / 'tdxhydro_splits' / 'tdx_header_numbers.json') as f:
        tdx_header_numbers = json.load(f)

    gpkgs = sorted(gpkg_dir.glob('TDX*.gpkg'), key=lambda p: p.stat().st_size, reverse=True)

    # Early exit if all outputs already exist
    expected_outputs = [gpq_dir / gpkg.name.replace('.gpkg', '.parquet') for gpkg in gpkgs]
    if all(out.exists() for out in expected_outputs):
        logging.info(f'All {len(expected_outputs)} output files already exist, skipping')
        sys.exit(0)

    gpq_dir.mkdir(parents=True, exist_ok=True)

    # Several files convert at once - each worker owns one GPKG end to end. The dial is memory,
    # not cores: a basins file is ~5 GB on disk and tens of GB as a GeoDataFrame, and the
    # largest-first order below deliberately runs the biggest files together so the tail is small
    # files draining fast. Six workers suits the 512 GB machine this runs on; override with
    # $TRANSLATE_JOBS for anything smaller.
    workers = max(1, int(os.environ.get('TRANSLATE_JOBS', 6)))
    jobs = [(g, int(tdx_header_numbers[str(g.name.split('_')[-2])]), g.name.split('_')[-2])
            for g in gpkgs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for message in pool.map(convert, *zip(*jobs)):
            logging.info(message)
