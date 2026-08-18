"""Build one region's leaf catchment polygons, one per surviving reach, in published row order."""
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyproj
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

region_root = hy.paths.region_root
tdx_root = hy.paths.tdx_root
logs_root = hy.paths.logs_root

chunk_threads = int(os.environ.get('CATCHMENT_CHUNK_THREADS', 2))
catchment_jobs = max(1, int(os.environ.get('CATCHMENT_JOBS', 1)))
dissolve_threads = max(1, (os.cpu_count() or 8) // (chunk_threads * catchment_jobs))
TOLERANCE_METERS = 20.0
CHUNK_SIZE = int(os.environ.get('CATCHMENT_CHUNK_SIZE', 20_000))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def build_basin_edits(mods_dir: Path) -> tuple[dict, set]:
    redirect: dict = {}
    deleted: set = set()

    for outlet, edit in _load_json(mods_dir / 'lake_edits.json').items():
        outlet = int(outlet)
        for d in edit.get('delete', []):
            redirect[int(d)] = outlet

    zero_lengths = _load_json(mods_dir / 'zero_length_streams.json')
    for case in ('case1', 'case2', 'case3'):
        deleted.update(int(i) for i in zero_lengths.get(case, {}).get('ids', []))

    for fname in ('headwater_dissolves.json', 'branches_to_prune.json', 'short_consolidations.json'):
        for keeper, members in _load_json(mods_dir / fname).items():
            keeper = int(keeper)
            for member in members:
                redirect[int(member)] = keeper

    return redirect, deleted


def resolve_keeper(rid: int, redirect: dict) -> int:
    seen = set()
    while rid in redirect and rid not in seen:
        seen.add(rid)
        rid = redirect[rid]
    return rid


def union_threaded(wkb: np.ndarray, offsets: np.ndarray, workers: int,
                   groups_per_task: int = 300) -> tuple:
    groups = len(offsets) - 1

    def union_chunk(start: int) -> tuple:
        stop = min(start + groups_per_task, groups)
        base, end = offsets[start], offsets[stop]
        flat = shapely.from_wkb(wkb[base:end])
        counted = int(shapely.get_num_coordinates(flat).sum())
        bounds = offsets[start:stop + 1] - base
        return counted, [flat[bounds[i]] if bounds[i + 1] - bounds[i] == 1
                         else shapely.union_all(flat[bounds[i]:bounds[i + 1]])
                         for i in range(stop - start)]

    geometries = []
    vertices = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for counted, part in pool.map(union_chunk, range(0, groups, groups_per_task)):
            vertices += counted
            geometries.extend(part)
    return geometries, vertices


def source_crs(basins_src: Path):
    metadata = pq.ParquetFile(basins_src).schema_arrow.metadata or {}
    geo = json.loads(metadata.get(b'geo', b'{}'))
    column = geo.get('primary_column', hy.schema.geometry)
    crs = geo.get('columns', {}).get(column, {}).get('crs', 'missing')
    if crs == 'missing':
        raise RuntimeError(f'{basins_src.name} carries no GeoParquet CRS metadata')
    return pyproj.CRS.from_json_dict(crs) if crs is not None else pyproj.CRS.from_user_input('OGC:CRS84')


def read_source_wkb(basins_src: Path, wanted: np.ndarray, batch_size: int = 20_000) -> np.ndarray:
    wkb = np.empty(len(wanted), dtype=object)
    parquet = pq.ParquetFile(basins_src)
    geoarrow = parquet.schema_arrow.field(hy.schema.geometry).metadata is not None
    row = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=[hy.schema.geometry]):
        stop = row + batch.num_rows
        keep = wanted[row:stop]
        if keep.any():
            column = batch.column(hy.schema.geometry)
            if geoarrow:
                block = shapely.to_wkb(gpd.GeoDataFrame.from_arrow(batch).geometry.values)
            else:
                block = column.to_numpy(zero_copy_only=False)
            wkb[row:stop] = np.where(keep, block, None)
        row = stop
    if row != len(wanted):
        raise RuntimeError(f'{basins_src.name} has {row:,} rows, expected {len(wanted):,}')
    return wkb


def simplify_chunk(geometries: np.ndarray, start: int) -> np.ndarray:
    clean = shapely.coverage_simplify(geometries, TOLERANCE_METERS, simplify_boundary=False)
    empty = int(shapely.is_empty(clean).sum())
    if empty:
        raise RuntimeError(f'chunk starting at row {start:,} simplified {empty} catchment(s) '
                           f'out of existence')
    return clean


def build_leaf_catchments(region_number: int, order: pd.DataFrame) -> gpd.GeoDataFrame:
    outputs_dir = region_root / f'{region_number}'
    mods_dir = outputs_dir / 'mods'

    basins_src = tdx_root / f'TDX_streamreach_basins_{region_number}_01.parquet'
    id_column = hy.schema.tdx_link_no_field \
        if hy.schema.tdx_link_no_field in pq.read_schema(basins_src).names \
        else hy.schema.tdx_link_field
    source = pq.read_table(basins_src, columns=[id_column])
    source_ids = source.column(0).to_numpy().astype(np.int64)
    del source
    logging.info(f'{len(source_ids):,} source basins in {basins_src.name}')

    redirect, deleted = build_basin_edits(mods_dir)
    keeper_map = {rid: resolve_keeper(rid, redirect) for rid in np.unique(source_ids)}
    keeper = pd.Series(source_ids).map(keeper_map).to_numpy()

    river_ids = order[hy.schema.river_id].to_numpy()
    position = pd.Series(np.arange(len(river_ids), dtype=np.int64), index=river_ids)
    place = position.reindex(keeper).to_numpy(dtype=float, copy=True)
    place[np.isin(keeper, list(deleted))] = np.nan
    wanted = ~np.isnan(place)
    logging.info(f'{int((~wanted).sum()):,} source basins dropped (deleted or not in the metadata)')

    counts = np.bincount(place[wanted].astype(np.int64), minlength=len(river_ids))
    if (counts == 0).any():
        absent = river_ids[counts == 0]
        raise RuntimeError(f'{len(absent):,} reach(es) have no catchment, e.g. {absent[:5].tolist()}')

    by_place = np.argsort(place[wanted], kind='stable')
    rows = np.flatnonzero(wanted)[by_place]
    bounds = np.r_[0, np.cumsum(counts)]

    started = time.time()
    crs = source_crs(basins_src)
    wkb = read_source_wkb(basins_src, wanted)
    logging.info(f'source geometry held as WKB in {time.time() - started:.0f}s; '
                 f'{len(river_ids):,} catchments to build in chunks of {CHUNK_SIZE:,}')

    out = np.empty(len(river_ids), dtype=object)

    def build_chunk(start: int) -> tuple:
        stop = min(start + CHUNK_SIZE, len(river_ids))
        block = rows[bounds[start]:bounds[stop]]
        source = wkb[block]
        wkb[block] = None
        offsets = bounds[start:stop + 1] - bounds[start]

        merged, before = union_threaded(source, offsets, dissolve_threads)
        del source
        chunk = hy.projection.to_web_mercator(gpd.GeoDataFrame(geometry=merged, crs=crs),
                                              round_meters=None)
        del merged
        clean = simplify_chunk(chunk[hy.schema.geometry].values, start)
        del chunk
        snapped = hy.projection.snap_to_grid(clean)
        broken = ~shapely.is_valid(snapped)
        if broken.any():
            snapped[broken] = shapely.set_precision(clean[broken],
                                                    hy.projection.precision_meters)
        clean, held = hy.geometry.repair(snapped, fallback=snapped)
        del snapped
        after = int(shapely.get_num_coordinates(clean).sum())
        out[start:stop] = shapely.to_wkb(clean)
        return before, after, held

    starts = range(0, len(river_ids), CHUNK_SIZE)
    with ThreadPoolExecutor(max_workers=chunk_threads) as pool:
        counted = list(pool.map(build_chunk, starts))
    before = sum(c[0] for c in counted)
    after = sum(c[1] for c in counted)
    held = sum(c[2] for c in counted)

    logging.info(f'coverage pass at {TOLERANCE_METERS:g} m in chunks of {CHUNK_SIZE:,}, '
                 f'{chunk_threads} at a time: '
                 f'{before:,} -> {after:,} vertices ({100 * after / before:.2f}%), '
                 f'{time.time() - started:.0f}s')
    if held:
        logging.info(f'{held:,} catchment(s) kept their unsnapped geometry: rounding onto the '
                     f'{hy.projection.precision_meters:g} m grid broke them and the repair '
                     f'left no area')

    catchments = gpd.GeoDataFrame(
        {hy.schema.river_id: river_ids},
        geometry=shapely.from_wkb(out),
        crs=f'EPSG:{hy.projection.web_mercator_epsg}',
    )
    catchments.insert(1, hy.schema.river_index, order[hy.schema.river_index].to_numpy())
    return hy.schema.enforce_int32(catchments)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit('usage: 4_create_catchments.py <region_number>')
    region_number = int(args[0])

    outputs_dir = region_root / f'{region_number}'
    mods_dir = outputs_dir / 'mods'

    catchments_output = outputs_dir / f'catchments_{region_number}.geo.parquet'
    if catchments_output.exists():
        print(f'region {region_number}: catchments exist, skipping')
        sys.stdout.flush()
        os._exit(0)

    mods_dir.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logs_root / f'create_catchments_{region_number}.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    metadata = pd.read_parquet(
        outputs_dir / f'metadata_{region_number}.parquet',
        columns=[hy.schema.river_id, hy.schema.river_index],
    )
    logging.info(f'{len(metadata):,} reaches in the region-local order')

    catchments = build_leaf_catchments(region_number, metadata)
    hy.parquet.write_geoparquet(catchments, catchments_output)
    logging.info(f'Catchments written to {catchments_output}')
    print(f'region {region_number}: {len(catchments):,} catchments -> {catchments_output.name}')
