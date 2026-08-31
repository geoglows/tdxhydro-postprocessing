"""Convert the source TDX-Hydro GPKGs to geoparquet and stamp the globally unique reach ids."""
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Geod

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography.console as console
import hydrography.coverage as coverage
import hydrography.parquet as parquet
import hydrography.paths as paths
import hydrography.schema as schema
from hydrography.streams import add_outlet_coordinates

gpkg_dir = Path(os.environ.get('TDXHYDRO_GPKG_DIR') or 'test/gpkgs')
gpq_dir = paths.tdx_root

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)


wgs84 = Geod(ellps='WGS84')


def _calculate_geodesic_lengths(geoms) -> np.ndarray:
    counts = shapely.get_num_coordinates(geoms)
    coords = shapely.get_coordinates(geoms)
    starts = np.ones(len(coords), dtype=bool)
    starts[np.cumsum(counts)[counts > 0] - 1] = False
    starts = np.flatnonzero(starts)
    a = coords[starts]
    b = coords[starts + 1]
    _, _, segments = wgs84.inv(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
    lengths = np.bincount(np.repeat(np.arange(len(geoms)), np.maximum(counts - 1, 0)),
                          weights=segments, minlength=len(geoms))

    return np.where(lengths < 0.0000001, 0.01, lengths)


def duplicated_outlets() -> np.ndarray:
    table = paths.network_data_root / 'tdxhydro_splits' / 'duplicated_watersheds.csv'
    if not table.exists():
        return np.empty(0, dtype=np.int64)
    return pd.read_csv(table)['drop'].to_numpy().astype(np.int64)


def upstream_mask(outlets: np.ndarray, link: np.ndarray, ds_link: np.ndarray) -> np.ndarray:
    mask = np.isin(link, outlets)
    frontier = link[mask]
    while len(frontier):
        above = np.isin(ds_link, frontier) & ~mask
        if not above.any():
            break
        mask |= above
        frontier = link[above]
    return mask


def convert(gpkg: Path, tdx_header_number: int, region_number: str) -> str:
    out_file_name = gpq_dir / gpkg.name.replace('.gpkg', '.parquet')
    if out_file_name.exists():
        return f'{gpkg.name}: already converted, skipped'
    started = time.time()
    removed = ''

    gdf = gpd.read_file(gpkg)

    if 'streamnet' in gpkg.name:
        gdf[schema.tdx_link_field] = gdf[schema.tdx_link_field].astype(int) + (tdx_header_number * 10_000_000)
        gdf[schema.tdx_ds_link_field] = gdf[schema.tdx_ds_link_field].astype(int)
        gdf.loc[gdf[schema.tdx_ds_link_field] != -1, schema.tdx_ds_link_field] = gdf[schema.tdx_ds_link_field] + (tdx_header_number * 10_000_000)
        gdf[schema.tdx_strm_order_field] = gdf[schema.tdx_strm_order_field].astype(int)
        gdf[schema.tdx_geodesic_length_field] = _calculate_geodesic_lengths(gdf[schema.geometry].values)
        gdf[schema.tdx_region_field] = region_number
        gdf = add_outlet_coordinates(gdf)

        duplicated = upstream_mask(duplicated_outlets(), gdf[schema.tdx_link_field].to_numpy(),
                                   gdf[schema.tdx_ds_link_field].to_numpy())
        if duplicated.any():
            gdf = gdf[~duplicated]
            removed = f', {int(duplicated.sum()):,} duplicated reaches removed'

        gdf = gdf[schema.tdx_standardized_columns]

    else:
        gdf[schema.tdx_link_field] = gdf[schema.basin_stream_id_field].astype(int) + (tdx_header_number * 10_000_000)
        gdf = gdf.drop(columns=[schema.basin_stream_id_field])

    partial = out_file_name.with_name(f'{out_file_name.name}.partial')
    parquet.write_source_geoparquet(gdf, partial)
    partial.replace(out_file_name)
    coverage.marker_for(out_file_name).unlink(missing_ok=True)
    return (f'{gpkg.name}: {len(gdf):,} rows{removed}, {time.time() - started:.0f}s '
            f'-> {out_file_name.name}')


def clean_coverage(basins_file: Path, workers: int = 1) -> str:
    if coverage.is_clean(basins_file):
        return f'{basins_file.name}: coverage already clean, skipped'
    record = coverage.clean_in_place(basins_file, workers)
    grew = record['vertices_after'] - record['vertices_before']
    tiles = record.get('tiles', 1)
    how = f'in {tiles} tiles, ' if tiles > 1 else ''
    return (f'{basins_file.name}: coverage cleaned, {how}{record["rows"]:,} basins, '
            f'{record["vertices_before"]:,} -> {record["vertices_after"]:,} vertices '
            f'({grew / max(record["vertices_before"], 1):+.2%}), {record["seconds"]:.0f}s')


if __name__ == '__main__':
    console.banner('Translate TDX-Hydro to geoparquet')
    gpq_dir.mkdir(parents=True, exist_ok=True)

    gpkgs = sorted(gpkg_dir.glob('TDX*.gpkg'), key=lambda p: p.stat().st_size, reverse=True)
    if not gpkgs:
        sys.exit(f'no TDX*.gpkg under {gpkg_dir}')
    wanted = set(sys.argv[1:])

    expected_outputs = [gpq_dir / gpkg.name.replace('.gpkg', '.parquet') for gpkg in gpkgs]
    expected_markers = [coverage.marker_for(p) for p in expected_outputs
                        if 'streamreach_basins' in p.name
                        and (not wanted or p.name.split('_')[-2] in wanted)]
    if all(path.exists() for path in expected_outputs + expected_markers):
        # exit 0, not 1: pipeline.sh runs under `set -e`, so a step with nothing to do must report
        # success or it aborts the whole run. sys.exit(<string>) prints to stderr and exits 1.
        print('all outputs exist, nothing to do')
        sys.exit(0)

    with open(paths.network_data_root / 'tdxhydro_splits' / 'tdx_header_numbers.json') as f:
        tdx_header_numbers = json.load(f)

    if all(out.exists() for out in expected_outputs):
        print(f'all {len(expected_outputs)} converted files exist, nothing to translate')
    else:
        workers = max(1, int(os.environ.get('TRANSLATE_JOBS', 6)))
        jobs = [(g, int(tdx_header_numbers[str(g.name.split('_')[-2])]), g.name.split('_')[-2])
                for g in gpkgs]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for message in pool.map(convert, *zip(*jobs)):
                logging.info(message)
                print(message)

    console.banner('Node the source basin coverage')
    basins = sorted(gpq_dir.glob('TDX_streamreach_basins_*.parquet'), key=lambda p: p.stat().st_size)
    if wanted:
        basins = [p for p in basins if p.name.split('_')[-2] in wanted]
        print(f'{len(basins)} of the tree\'s basins files selected by argument')
    reason = coverage.unavailable()
    if not basins:
        print(f'no basins files under {gpq_dir}, nothing to clean')
    elif reason:
        logging.warning(f'coverage clean skipped: {reason}')
        print(f'WARNING: coverage clean skipped for all {len(basins)} basins files.\n  {reason}\n'
              f'  The pipeline still runs on an un-noded coverage, as it did before this step '
              f'existed - see hydrography/coverage.py.')
    else:
        pending = [p for p in basins if not coverage.is_clean(p)]
        clean_workers = max(1, int(os.environ.get('CLEAN_JOBS', 2)))
        parallel, serial = coverage.schedule(pending, clean_workers)
        budget = coverage.tile_vertices(clean_workers) * coverage.bytes_per_vertex
        print(f'{len(basins)} basins files, {len(pending)} to clean: {len(parallel)} whole, '
              f'{clean_workers} at a time, {len(serial)} tiled to '
              f'~{budget / 1e9:.0f} GB a tile and run alone')
        started = time.time()
        with ProcessPoolExecutor(max_workers=clean_workers, max_tasks_per_child=1) as pool:
            for message in pool.map(clean_coverage, parallel):
                logging.info(message)
                print(message)
        for path in serial:
            tiles = max(1, math.ceil(coverage.estimated_peak(path) / max(budget, 1)))
            logging.info(f'{path.name}: cleaning alone in ~{tiles} tiles, '
                         f'~{coverage.estimated_peak(path) / 1e9:.0f} GB whole')
            with ProcessPoolExecutor(max_workers=1) as pool:
                message = pool.submit(clean_coverage, path, clean_workers).result()
            logging.info(message)
            print(message)
        if pending:
            print(f'{len(pending)} coverages cleaned in {time.time() - started:.0f}s')
