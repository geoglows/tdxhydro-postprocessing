#!/usr/bin/env python
"""Generate the global basin product once, deterministically, from the raw TDX-Hydro inputs."""
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

LEVEL_ZOOMS = hy.basins.LEVEL_ZOOMS
zoom_tolerance = hy.basins.zoom_tolerance
LEVELS = sorted(LEVEL_ZOOMS)

LEVEL_GROWTH = 4.0

UNION_THREADS = max(1, os.cpu_count() or 8)

out_dir = hy.paths.global_basins_root
parts_dir = out_dir / 'parts'
codes_output = out_dir / 'pfaf_codes.parquet'
outlets_output = out_dir / 'basin_outlets.parquet'

RAW_COLUMNS = ['LINKNO', 'DSLINKNO', 'strmOrder', 'USContArea', 'DSContArea', 'lon', 'lat']

WGS84, MERCATOR = 'EPSG:4326', 'EPSG:3857'


def level_output(level: int, region: str = None) -> Path:
    if region is None:
        return out_dir / f'basins_level{level}.geo.parquet'
    return parts_dir / region / f'basins_level{level}_{region}.geo.parquet'


def codes_part(region: str) -> Path:
    return parts_dir / region / f'pfaf_codes_{region}.parquet'


def outlets_part(region: str) -> Path:
    return parts_dir / region / f'basin_outlets_{region}.parquet'


def write_codes_parts(region: str, raw: pd.DataFrame, codes: pd.Series,
                      outlets: pd.DataFrame) -> None:
    frames = (
        (pd.DataFrame({
            hy.schema.tdx_link_no_field: raw[hy.schema.river_id].to_numpy(),
            hy.schema.tdx_region_field: region,
            'pfafCode': codes.to_numpy(),
        }), codes_part(region)),
        (outlets.assign(**{hy.schema.tdx_region_field: region})[
            [hy.schema.tdx_link_no_field, hy.schema.tdx_region_field,
             *(f'level{lv}' for lv in LEVELS)]], outlets_part(region)),
    )
    for frame, path in frames:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f'{path.name}.partial')
        hy.parquet.write_parquet(frame, partial, index=False)
        partial.replace(path)


def global_ids(frame: pd.DataFrame) -> np.ndarray:
    column = hy.schema.tdx_link_no_field if hy.schema.tdx_link_no_field in frame.columns \
        else hy.schema.tdx_link_field
    return frame[column].to_numpy().astype(np.int64)


def load_raw_region(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path, columns=RAW_COLUMNS + [hy.schema.tdx_link_no_field])
    except Exception:
        df = pd.read_parquet(path, columns=RAW_COLUMNS)
    ids = global_ids(df)
    local = df[hy.schema.tdx_link_field].to_numpy().astype(np.int64)
    ds_local = df[hy.schema.tdx_ds_link_field].to_numpy().astype(np.int64)
    to_global = pd.Series(ids, index=local)
    next_ids = to_global.reindex(ds_local).to_numpy()
    next_ids = np.where(ds_local == -1, -1, np.nan_to_num(next_ids, nan=-1)).astype(np.int64)

    raw = pd.DataFrame({
        hy.schema.river_id: ids,
        hy.schema.next_river_id: next_ids,
        hy.schema.strahler_order: df['strmOrder'].to_numpy().astype(np.int64),
        hy.schema.tdx_ds_area_field: df['DSContArea'].to_numpy().astype(np.float64),
        hy.schema.area: (df['DSContArea'] - df['USContArea']).to_numpy().astype(np.float64),
        hy.schema.lon_field: df['lon'].to_numpy().astype(np.float64),
        hy.schema.lat_field: df['lat'].to_numpy().astype(np.float64),
    })
    raw = raw.sort_values([hy.schema.strahler_order, hy.schema.tdx_ds_area_field,
                           hy.schema.river_id], kind='stable', ignore_index=True)

    river = raw[hy.schema.river_id].to_numpy()
    parent = hy.topology.parent_rows(river, raw[hy.schema.next_river_id].to_numpy())
    outlet = np.empty(len(raw), dtype=np.int64)
    for i in range(len(raw) - 1, -1, -1):
        outlet[i] = river[i] if parent[i] < 0 else outlet[parent[i]]
    raw[hy.schema.last_river_id] = outlet
    return raw


def outlet_flags(raw: pd.DataFrame, codes: pd.Series) -> pd.DataFrame:
    code = codes.to_numpy().astype(np.int64)
    river = raw[hy.schema.river_id].to_numpy()
    parent = hy.topology.parent_rows(river, raw[hy.schema.next_river_id].to_numpy())
    ends = parent < 0
    downstream = np.where(ends, 0, parent)

    flags = {}
    for k, level in enumerate(LEVELS, start=1):
        prefix = code // 10 ** (len(LEVELS) - k)
        flags[f'level{level}'] = ends | (prefix[downstream] != prefix)
    table = pd.DataFrame({hy.schema.tdx_link_no_field: river, **flags})
    return table[table[f'level{LEVELS[-1]}']].reset_index(drop=True)


def dissolve_by(geometries: np.ndarray, group: np.ndarray, workers: int = None,
                coverage: bool = True) -> tuple:
    order = np.argsort(group, kind='stable')
    ordered = group[order]
    edges = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1], True])
    keys = ordered[edges[:-1]]
    blocks = [order[start:end] for start, end in zip(edges[:-1], edges[1:])]

    merged = np.empty(len(blocks), dtype=object)
    retry = []

    def union(index: int) -> None:
        rows = blocks[index]
        if len(rows) == 1:
            merged[index] = geometries[rows[0]]
            return
        members = geometries[rows]
        try:
            if not coverage:
                merged[index] = shapely.union_all(members)
                return
            expected = float(shapely.area(members).sum())
            candidate = shapely.coverage_union_all(members)
            if abs(candidate.area - expected) <= hy.geometry.coverage_area_tolerance \
                    * max(expected, 1.0):
                merged[index] = candidate
            else:
                retry.append(index)
        except shapely.errors.GEOSException:
            retry.append(index)

    with ThreadPoolExecutor(max_workers=workers or UNION_THREADS) as pool:
        list(pool.map(union, range(len(blocks))))
    for index in retry:
        members, _ = hy.geometry.repair(geometries[blocks[index]])
        try:
            merged[index] = hy.geometry.hierarchical_union(list(members), workers=workers)
        except shapely.errors.GEOSException:
            merged[index] = shapely.union_all(members, grid_size=1.0)
    return keys, merged, len(retry)


simplify_coverage = hy.geometry.simplify_coverage
snap = hy.geometry.snap


def close_holes(geometries: np.ndarray) -> tuple:
    """Close each basin's interior rings, leaving open only the ground another basin stands in.

    A band is a partition of the region, so a hole in one of its basins is one of two things. Most
    are ground the partition never claimed - a watershed this release dropped, a no-runoff basin,
    an endorheic sink - and a basin drawn with those punched out of it reads as shrapnel. The rest
    are a *neighbouring basin* this one happens to enclose, and closing that kind is how a basin
    comes to claim ground another already claims.

    **This used to close every ring and that was the overlap the map showed.** Measured on
    1020011530, whose level-4 basin 89 wraps around basin 55: closing rings blind put 7,441 km2 of
    basin 55's ground - level-8 codes 550730 and its siblings - inside basin 89 as well, which the
    renderer paints twice. Over the whole region, per level, overlap between basins before the fill
    against after it:

        level 8     0.4 km2  ->     17.1 km2
        level 7     1.5 km2  ->    495.2 km2
        level 6     5.0 km2  ->  3,400.2 km2
        level 5     3.7 km2  ->  6,874.1 km2
        level 4     4.2 km2  -> 17,330.2 km2
        level 3     8.4 km2  -> 98,465.5 km2

    The dissolve is not what breaks: the coverage handed to this function is clean to within a few
    km2 at every level, and the whole defect was here. ``hy.geometry.fill_holes`` is the same
    decision made per occupant rather than per ring - a ring with something in it closes *around*
    what stands in it, so what is added back is the hole minus that basin - and step 6 already
    dissolves the group outlines with it. Re-measured on the same region's level-4 band, it adds
    1.4 km2 rather than 11,425 km2 and still closes 34 of the 46 rings.

    The tree is this band only, so a hole occupied by a basin in the *neighbouring region* is still
    closed. Fixing that needs every region's band at once, which is the one thing that would make
    these builds depend on each other; measured globally at level 4 it is 28,575 km2 against the
    651,711 km2 this removes.
    """
    tree = shapely.STRtree(geometries)
    filled = np.empty(len(geometries), dtype=object)
    closed = trimmed = held = 0
    for index, geometry in enumerate(geometries):
        try:
            filled[index], shut, around = hy.geometry.fill_holes(geometry, tree, skip=index)
        except shapely.errors.GEOSException:
            # one basin GEOS will not rebuild keeps its holes. That is a ragged outline for one
            # basin, against losing the whole region - and an hour of the build - to an exception
            # raised on the last of its seven levels
            filled[index], held = geometry, held + 1
            continue
        closed += shut
        trimmed += around
    return filled, closed, trimmed, held


def promote_to_multi(geometries: np.ndarray) -> np.ndarray:
    parts, index = shapely.get_parts(geometries, return_index=True)
    promoted = shapely.multipolygons(parts, indices=index)
    if len(promoted) != len(geometries):
        raise RuntimeError(f'{len(geometries) - len(promoted)} basin(s) have no polygon to promote')
    return promoted


def basin_attributes(raw: pd.DataFrame, codes: pd.Series, outlets: pd.DataFrame,
                     level: int, k: int) -> pd.DataFrame:
    code = codes.to_numpy().astype(np.int64)
    prefix = code // 10 ** (len(LEVELS) - k)
    frame = pd.DataFrame({
        'prefix': prefix,
        hy.schema.area: raw[hy.schema.area].to_numpy(),
        hy.schema.strahler_order: raw[hy.schema.strahler_order].to_numpy(),
    })
    stats = frame.groupby('prefix', sort=True).agg(
        riverCount=(hy.schema.area, 'size'),
        areaM2=(hy.schema.area, 'sum'),
        strahlerOrder=(hy.schema.strahler_order, 'max'),
    )
    pour = raw.merge(outlets[[hy.schema.tdx_link_no_field, f'level{level}']],
                     left_on=hy.schema.river_id, right_on=hy.schema.tdx_link_no_field)
    pour = pour[pour[f'level{level}']]
    pour = pour.assign(prefix=pour[hy.schema.river_id].map(
        pd.Series(prefix, index=raw[hy.schema.river_id].to_numpy())))
    pour = pour.sort_values(['prefix', hy.schema.tdx_ds_area_field, hy.schema.river_id],
                            ascending=[True, False, True]).groupby('prefix', sort=True).first()
    stats[hy.schema.tdx_link_no_field] = pour[hy.schema.river_id]
    if stats[hy.schema.tdx_link_no_field].isna().any():
        raise RuntimeError(f'level {level}: a basin has no pour point')
    stats['pfafCode'] = [str(p).zfill(k) for p in stats.index]
    return stats.reset_index(names='prefix')


def write_part(frame: pd.DataFrame, geometries: np.ndarray, level: int, region: str) -> None:
    solid, closed, trimmed, held = close_holes(geometries)
    if closed or trimmed:
        logging.info(f'{region} level {level}: {closed:,} hole(s) closed outright, {trimmed:,} '
                     f'closed around a basin standing in them')
    if held:
        logging.warning(f'{region} level {level}: {held:,} basin(s) kept their holes, GEOS would '
                        f'not rebuild them')
    # A basin whose fill will not repair keeps its holes rather than its fill. That is the safe
    # direction: an open ring is ground this basin does not draw, which costs nothing but a
    # ragged outline, where a bad fill is ground two basins draw. The blind 1 m refill that used
    # to sit here undid exactly what close_holes leaves open, so it is gone.
    solid, lost = hy.geometry.repair(solid, fallback=geometries)
    if lost:
        logging.info(f'{region} level {level}: {lost:,} basin(s) kept their unfilled geometry')
    basins = gpd.GeoDataFrame(frame.reset_index(drop=True),
                              geometry=promote_to_multi(solid), crs=MERCATOR)
    basins.insert(0, 'basinId', np.arange(len(basins), dtype=np.int32))
    basins.insert(1, 'level', np.int32(level))
    basins[hy.schema.tdx_region_field] = region
    basins['riverCount'] = basins['riverCount'].astype(np.int64)
    basins['strahlerOrder'] = basins['strahlerOrder'].astype(np.int32)
    basins[hy.schema.tdx_link_no_field] = basins[hy.schema.tdx_link_no_field].astype(np.int64)
    basins = basins[['basinId', 'level', 'pfafCode', hy.schema.tdx_region_field,
                     hy.schema.tdx_link_no_field, 'riverCount', hy.schema.area,
                     'strahlerOrder', hy.schema.geometry]]
    path = level_output(level, region)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f'{path.name}.partial')
    hy.parquet.write_geoparquet(basins, partial)
    partial.replace(path)


def build_region_polygons(region: str, raw: pd.DataFrame, codes: pd.Series,
                          outlets: pd.DataFrame, basins_path: Path) -> None:
    started = time.time()
    catchments = gpd.read_parquet(basins_path)
    cat_ids = global_ids(catchments)
    geometry = catchments.geometry
    source_crs = catchments.crs or WGS84

    code_of = pd.Series(codes.to_numpy().astype(np.int64),
                        index=raw[hy.schema.river_id].to_numpy())
    full = code_of.reindex(cat_ids)
    missing = int(full.isna().sum())
    if missing:
        logging.info(f'{region}: {missing:,} catchment polygon(s) have no coded reach, dropped')
    keep = full.notna().to_numpy()
    parts = geometry.to_numpy()[keep]
    full = full.to_numpy()[keep].astype(np.int64)
    del catchments, geometry

    keys, merged, fallbacks = dissolve_by(parts, full, coverage=True)
    logging.info(f'{region}: leaf -> level 8, {len(keys):,} basins, {fallbacks:,} not a coverage, '
                 f'{time.time() - started:.0f}s')
    if fallbacks:
        print(f'{region}: {fallbacks:,} of {len(keys):,} level-8 basins did not dissolve as a '
              f'coverage - run 1_translate_tdxhydro.py to node this region\'s source')
    del parts
    merged = gpd.GeoSeries(merged, crs=source_crs).to_crs(MERCATOR).to_numpy()

    for level in reversed(LEVELS):
        k = level - 2
        if level != LEVELS[-1]:
            keys, merged, fallbacks = dissolve_by(merged, keys // 10, coverage=True)
            if fallbacks:
                logging.info(f'{region} level {level}: {fallbacks:,} of {len(keys):,} took the '
                             f'general union')
        tolerance = max(zoom_tolerance(LEVEL_ZOOMS[level][1]), 1.0)
        simplified = simplify_coverage(merged, tolerance)
        simplified, _ = hy.geometry.repair(simplified, fallback=merged)
        snapped, _ = snap(simplified, tolerance)
        merged, _ = hy.geometry.repair(snapped, fallback=simplified)

        attributes = basin_attributes(raw, codes, outlets, level, k)
        attributes = attributes[attributes['prefix'].isin(set(keys.tolist()))]
        if not np.array_equal(attributes['prefix'].to_numpy(), keys):
            raise RuntimeError(f'{region} level {level}: polygon keys and attribute rows disagree')
        write_part(attributes.drop(columns=['prefix']), merged, level, region)
        logging.info(f'{region} level {level}: {len(keys):,} basins at {tolerance:,.0f} m, '
                     f'{time.time() - started:.0f}s')

    footprint = hy.geometry.union_coverage(merged)
    terminal = raw[raw[hy.schema.next_river_id] == -1]
    biggest = terminal.sort_values([hy.schema.tdx_ds_area_field, hy.schema.river_id],
                                   ascending=[False, True]).iloc[0]
    level2 = pd.DataFrame({
        'riverCount': [np.int64(len(raw))],
        hy.schema.area: [float(raw[hy.schema.area].sum())],
        'strahlerOrder': [int(raw[hy.schema.strahler_order].max())],
        hy.schema.tdx_link_no_field: [int(biggest[hy.schema.river_id])],
        'pfafCode': [''],
    })
    write_part(level2, np.array([footprint], dtype=object), 2, region)
    logging.info(f'{region}: level 2 footprint written, {time.time() - started:.0f}s total')


def publish_levels(regions: list) -> None:
    for level in [2, *LEVELS]:
        out = level_output(level)
        if out.exists():
            continue
        parts = [level_output(level, r) for r in regions]
        if any(not p.exists() for p in parts):
            print(f'not writing {out.name}: a region is missing this level')
            continue
        partial = out.with_name(f'{out.name}.partial')
        rows = hy.parquet.concat_geoparquet(parts, partial)
        partial.replace(out)
        print(f'{rows:,} level-{level} basins -> {out.name}')


if __name__ == '__main__':
    if '--bands' in sys.argv:
        for level, (lo, hi) in sorted(LEVEL_ZOOMS.items()):
            print(f'{level}:{lo}:{hi}')
        print(f'leaf:{hy.basins.LEAF_ZOOMS[0]}:{hy.basins.LEAF_ZOOMS[1]}')
        sys.exit(0)

    hy.console.banner('Generate global basins (one time, not per release)')

    streamnets = sorted(hy.paths.tdx_root.glob('TDX_streamnet_*_01.parquet'))
    if not streamnets:
        sys.exit(f'no TDX_streamnet parquet found under {hy.paths.tdx_root}')
    regions = [p.name.split('_')[2] for p in streamnets]

    outputs = [codes_output, outlets_output] + [level_output(lv) for lv in [2, *LEVELS]]
    if all(path.exists() for path in outputs):
        # exit 0, not 1: pipeline.sh runs under `set -e`, so a step with nothing to do must report
        # success or it aborts the whole run. sys.exit(<string>) prints to stderr and exits 1.
        print('all outputs exist, nothing to do')
        sys.exit(0)
    need_codes = not (codes_output.exists() and outlets_output.exists())

    hy.paths.logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=hy.paths.logs_root / 'global_basins.log', filemode='w',
                        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    started = time.time()
    for path, region in zip(streamnets, regions):
        t0 = time.time()
        have_polygons = all(level_output(lv, region).exists() for lv in [2, *LEVELS])
        have_codes = not need_codes or (codes_part(region).exists()
                                        and outlets_part(region).exists())
        if have_polygons and have_codes:
            print(f'{region}: complete, skipped')
            continue

        raw = load_raw_region(path)
        codes = hy.basins.assign_basin_codes(raw, LEVELS, log=logging.info, growth=LEVEL_GROWTH)
        outlets = outlet_flags(raw, codes)
        write_codes_parts(region, raw, codes, outlets)

        if have_polygons:
            print(f'{region}: polygons already built, {time.time() - t0:.0f}s (codes only)')
            continue
        basins_path = path.with_name(path.name.replace('streamnet', 'streamreach_basins'))
        if not basins_path.exists():
            raise FileNotFoundError(f'{basins_path} not found; the polygon build needs it')
        build_region_polygons(region, raw, codes, outlets, basins_path)
        print(f'{region}: {len(raw):,} reaches, polygons built, {time.time() - t0:.0f}s')

    if need_codes:
        missing = [r for r in regions if not (codes_part(r).exists() and outlets_part(r).exists())]
        if missing:
            raise RuntimeError(f'{len(missing)} region(s) have no code part: {missing[:5]}')
        codes = pd.concat([pd.read_parquet(codes_part(r)) for r in regions], ignore_index=True)
        outlets = pd.concat([pd.read_parquet(outlets_part(r)) for r in regions], ignore_index=True)
        for frame in (codes, outlets):
            ids = frame[hy.schema.tdx_link_no_field]
            if ids.duplicated().any():
                raise RuntimeError('raw ids are not globally unique across regions')
            if int(ids.max()) > np.iinfo(np.int32).max:
                raise RuntimeError('an id is outside int32; widen the dtype, do not let it wrap')
            frame[hy.schema.tdx_link_no_field] = ids.astype(np.int32)
        out_dir.mkdir(parents=True, exist_ok=True)
        for frame, output in ((codes, codes_output), (outlets, outlets_output)):
            partial = output.with_name(f'{output.name}.partial')
            hy.parquet.write_parquet(frame, partial, index=False)
            partial.replace(output)
        print(f'{len(codes):,} reaches coded, {len(outlets):,} outlet reaches '
              f'-> {codes_output.name}, {outlets_output.name}')

    publish_levels(regions)
    print(f'done, {time.time() - started:.0f}s')
