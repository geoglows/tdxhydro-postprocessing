#!/usr/bin/env python
"""
Generate the global basin product once, deterministically, from the raw TDX-Hydro inputs.

This is the step that fixes the basins for good - codes, outlet registry and polygons. Everything
after it revises the stream network, and every revision changes what a recomputed Pfafstetter run
would produce, mostly by reshuffling digits rather than moving boundaries (measured globally:
96-99% of area keeps its basin at every level while literal codes fall to 64% agreement by level
8). Generating the basins here, from the raw network and raw catchments that every future revision
descends from, makes them a permanent fact: releases do not rebuild basins, they re-stamp them -
step 6 duplicates these files and adds the release's ids and indices so lookups stay valid.

The hierarchy starts from the TDX regions: **a region is a level-2 basin**, and level 3 is the
first split inside one, so codes run levels 3-8, one digit per level. There is no level 9. Codes
are assigned by hydrography/basins.py with the budget ramp fixed at ``LEVEL_GROWTH`` per level, so
the hierarchy is a property of the raw hydrography and the pyramid rate, not of any network's
reach count. Every region parquet in $TDXHYDRO_ROOT is processed, including regions the revision
steps currently exclude: excluded is a revision decision, and this step is upstream of all of
those, so a region added to a build later already has its basins.

Ids are read as step 1 stamped them and never derived here: the global id is ``TDXHydroLinkNo``
where that column exists (the converted tree on disk) and the already-stamped ``LINKNO``
otherwise (what 1_translate_tdxhydro.py writes); downstream ids come from mapping ``DSLINKNO``
through the file's own local-to-global pairing. The region header arithmetic lives in step 1 only.

Everything goes to $TDXHYDRO_ROOT/global_basins - inside the raw tree, not the data root, because
this product shares the raw data's lifecycle: derived from nothing else, consumed by every
release, regenerated only if the raw data changes. See hydrography/paths.py.

    writes  global_basins/pfaf_codes.parquet     TDXHydroLinkNo, TDXHydroRegion, pfafCode -
                                                 one row per raw reach
            global_basins/basin_outlets.parquet  TDXHydroLinkNo, TDXHydroRegion, level3..level8 -
                                                 one row per reach that is a basin pour point
            global_basins/parts/<region>/basins_level{2..8}_<region>.geo.parquet
                                                 per-region polygon intermediates, resumable
            global_basins/basins_level{2..8}.geo.parquet
                                                 the basin polygons, one file per level

The outlet table marks every pour point of every basin (a level-3 outlet is an outlet at every
deeper level, so the columns are monotone and the rows are the level-8 outlet set). It exists to
be the constraint the revision steps respect: feed the ids to the ``protected`` set of the
stream-revision step at whatever depth is worth preserving.

The polygons are the raw per-reach catchments dissolved by code prefix, cut per level at the
tolerance of the finest zoom its band is drawn at (level 8 spans z8-9, so z9 draws the same
features), and published solid: interior rings are enclaves of other coastal groups and are
filled, with the enclave basins still present as features of their own. The leaf catchments are
not published here - they are per-release data and stay with the release pipeline.

The leaf-to-level-8 dissolve runs in the source CRS, where the TauDEM catchments share exact
pixel-edge vertices and the union cancels shared edges cleanly; the result is projected to web
mercator and the coarser levels telescope out of it exactly as the retired per-build step did
(dissolve, simplify at the band tolerance, snap onto the matching power-of-two lattice, repair).
This is hours of GEOS for the planet, once; the per-region parts make a stopped run resume
instead of restart.

    TDXHYDRO_ROOT=... RFS_DATA_ROOT=... python 2_global_basins.py
"""
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

# one digit per level, level 3 (the first split of a region) through level 8, straight from the
# shared zoom banding in hydrography/basins.py. The region itself is level 2 and needs no digit;
# there is deliberately no level 9.
LEVEL_ZOOMS = hy.basins.LEVEL_ZOOMS
zoom_tolerance = hy.basins.zoom_tolerance
LEVELS = sorted(LEVEL_ZOOMS)

# Each level's basin budget is this multiple of the level above it, anchored at whatever the first
# split realised. 4x is the tile-pyramid rate: a zoom step quadruples the tile count, so each level
# wants ~4x the features of the one above. Fixed here rather than taking ``level_targets``' natural
# ramp so the hierarchy depends on nothing but the split radix and the pyramid rate - the natural
# ramp ends at one basin per raw reach, measured at a 5.8x step and 2.8M level-8 basins globally,
# an order finer than any band would draw.
LEVEL_GROWTH = 4.0

# threads for the per-group unions; GEOS releases the GIL, so threads scale
UNION_THREADS = max(1, os.cpu_count() or 8)

out_dir = hy.paths.global_basins_root
parts_dir = out_dir / 'parts'
codes_output = out_dir / 'pfaf_codes.parquet'
outlets_output = out_dir / 'basin_outlets.parquet'

# raw streamnet columns needed for the code assignment; geometry is deliberately absent
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
    """This region's rows of the two registries, so a stopped run resumes them like the polygons.

    Without these the codes were the one product with no part: every rerun re-ran
    ``assign_basin_codes`` for every region just to rebuild the two global tables at the end, ~3.5 s
    a region on the raw network, thrown away whenever the tables were already current. The polygons
    have resumed from parts since this step was written; now the registries do too.
    """
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
    """The globally unique reach ids step 1 stamped, whichever column vintage the file carries."""
    column = hy.schema.tdx_link_no_field if hy.schema.tdx_link_no_field in frame.columns \
        else hy.schema.tdx_link_field
    return frame[column].to_numpy().astype(np.int64)


def load_raw_region(path: Path) -> pd.DataFrame:
    """One raw streamnet region as the attribute frame ``assign_basin_codes`` needs, in
    topological order, with ids exactly as step 1 stamped them.

    Downstream ids come from mapping ``DSLINKNO`` through the file's own LINKNO-to-global pairing
    (the identity on files whose LINKNO is already global), so no header arithmetic happens here.
    The (strahler, DSContArea, id) sort is a valid upstream-before-downstream order on the raw
    network - ``assign_basin_codes`` verifies rather than trusts that - and the outlet sweep runs
    back-to-front so every reach copies from a downstream row already resolved.
    """
    try:
        df = pd.read_parquet(path, columns=RAW_COLUMNS + [hy.schema.tdx_link_no_field])
    except Exception:                                   # a new-vintage file: LINKNO is global
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
    """Every reach whose downstream crosses a basin boundary, with one boolean per level saying at
    which depths it is a pour point."""
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


# ---------------------------------------------------------------------------
# The dissolves. Same machinery the retired per-build basins step proved out.
# ---------------------------------------------------------------------------
def dissolve_by(geometries: np.ndarray, group: np.ndarray, workers: int = None,
                coverage: bool = True) -> tuple:
    """One polygon per distinct ``group`` value, group ids ascending.

    ``coverage=True`` tries the edge-cancelling fast path first (right once the carry is snapped),
    falling back per group to the general union; ``coverage=False`` goes straight to the general
    union, which the raw leaf catchments need - they are exact but not an edge-matched coverage in
    GEOS's eyes. The unions run across threads because GEOS releases the GIL.
    """
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
        try:
            if coverage:
                merged[index] = shapely.coverage_union_all(geometries[rows])
            else:
                merged[index] = shapely.union_all(geometries[rows])
        except shapely.errors.GEOSException:
            retry.append(index)      # list.append is atomic; no lock needed

    with ThreadPoolExecutor(max_workers=workers or UNION_THREADS) as pool:
        list(pool.map(union, range(len(blocks))))
    for index in retry:
        members, _ = hy.geometry.repair(geometries[blocks[index]])
        try:
            merged[index] = hy.geometry.hierarchical_union(list(members), workers=workers)
        except shapely.errors.GEOSException:
            merged[index] = shapely.union_all(members, grid_size=1.0)
    return keys, merged, len(retry)


# the band cut and the lattice snap are shared with the leaf-band cut in 5_concatenate_global.py -
# same operations, same reasons - so they live in hydrography/geometry.py
simplify_coverage = hy.geometry.simplify_coverage
snap = hy.geometry.snap


def fill_holes(geometries: np.ndarray) -> tuple:
    """Every basin polygon with its interior rings removed, and how many rings were removed.

    Holes are enclaves of other coastal groups; the basins are published solid, with the enclave
    basins still present as features of their own. Runs to a fixed point because two ringless
    parts touching at points can jointly enclose a void that only becomes a ring once the union
    merges them; converges in a pass or two, the bound is a backstop.
    """
    total = 0
    for _ in range(8):
        parts, index = shapely.get_parts(geometries, return_index=True)
        ring_count = np.bincount(index, weights=shapely.get_num_interior_rings(parts),
                                 minlength=len(geometries)).astype(np.int64)
        if not ring_count.any():
            break
        shells = shapely.polygons(shapely.get_exterior_ring(parts))
        geometries = geometries.copy()
        edges = np.flatnonzero(np.r_[True, index[1:] != index[:-1], True])
        for start, end in zip(edges[:-1], edges[1:]):
            i = index[start]
            if not ring_count[i]:
                continue
            group = shells[start:end]
            try:
                geometries[i] = group[0] if end - start == 1 else shapely.union_all(group)
            except shapely.errors.GEOSException:
                geometries[i] = shapely.union_all(shapely.make_valid(group))
        total += int(ring_count.sum())
    return geometries, total


def promote_to_multi(geometries: np.ndarray) -> np.ndarray:
    """Every polygon as a MultiPolygon so all files share one parquet schema."""
    parts, index = shapely.get_parts(geometries, return_index=True)
    promoted = shapely.multipolygons(parts, indices=index)
    if len(promoted) != len(geometries):
        raise RuntimeError(f'{len(geometries) - len(promoted)} basin(s) have no polygon to promote')
    return promoted


# ---------------------------------------------------------------------------
# Per-region attribute and polygon builds
# ---------------------------------------------------------------------------
def basin_attributes(raw: pd.DataFrame, codes: pd.Series, outlets: pd.DataFrame,
                     level: int, k: int) -> pd.DataFrame:
    """One row per level-``k``-prefix basin: the raw-network facts that never change - member
    count, summed local area, max order, and the pour point (largest-drainage boundary-crossing
    reach, ties on the lower id, the convention that names a coastal group after its dominant
    river)."""
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
    """One region's one level, written aside and renamed so a part is whole or absent."""
    solid, plugged = fill_holes(geometries)
    if plugged:
        logging.info(f'{region} level {level}: {plugged:,} enclave hole(s) filled')
    # the shell unions in fill_holes can hand back a self-intersecting ring. Repair fixes most;
    # where make_valid refuses, repair falls back to the holed geometry, so any basin still
    # carrying rings afterwards gets the last resort every dissolve here shares: its shells
    # unioned on the 1 m grid, which snap-rounding makes valid by construction
    solid, _ = hy.geometry.repair(solid, fallback=geometries)
    parts_of, part_index = shapely.get_parts(solid, return_index=True)
    ringed = np.flatnonzero(np.bincount(
        part_index, weights=shapely.get_num_interior_rings(parts_of), minlength=len(solid)) > 0)
    for i in ringed:
        shells = shapely.polygons(shapely.get_exterior_ring(shapely.get_parts(solid[i])))
        solid[i] = shapely.union_all(shells, grid_size=1.0)
    if plugged or len(ringed):
        logging.info(f'{region} level {level}: {plugged:,} enclave hole(s) filled' + (
            f', {len(ringed)} refilled on the 1 m grid after a failed repair' if len(ringed) else ''))
    # a filtered frame carries a gappy index, and a gappy index gets written as a column that a
    # region without the filter does not have - the parts must share one schema to stack
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
    """The telescope for one region: raw catchments -> level 8 in the source CRS, then each
    coarser level dissolved from the one below in web mercator, each level written solid at its
    band's tolerance, ending with the region itself as level 2."""
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
        # a handful of raw reaches (zero-length connectors) have no catchment and vice versa
        logging.info(f'{region}: {missing:,} catchment polygon(s) have no coded reach, dropped')
    keep = full.notna().to_numpy()
    parts = geometry.to_numpy()[keep]
    full = full.to_numpy()[keep].astype(np.int64)
    del catchments, geometry

    # leaf -> level 8 in the source CRS, where the pixel-edge vertices are exact
    keys, merged, fallbacks = dissolve_by(parts, full, coverage=False)
    logging.info(f'{region}: leaf -> level 8, {len(keys):,} basins, {fallbacks:,} repaired, '
                 f'{time.time() - started:.0f}s')
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

        # attributes cover every coded reach; a basin made only of catchmentless reaches has no
        # polygon, so align on the keys the dissolve actually produced
        attributes = basin_attributes(raw, codes, outlets, level, k)
        attributes = attributes[attributes['prefix'].isin(set(keys.tolist()))]
        if not np.array_equal(attributes['prefix'].to_numpy(), keys):
            raise RuntimeError(f'{region} level {level}: polygon keys and attribute rows disagree')
        write_part(attributes.drop(columns=['prefix']), merged, level, region)
        logging.info(f'{region} level {level}: {len(keys):,} basins at {tolerance:,.0f} m, '
                     f'{time.time() - started:.0f}s')

    # the region itself: level 2, one solid polygon, named after its largest terminal drainage
    try:
        footprint = shapely.coverage_union_all(merged)
    except shapely.errors.GEOSException:
        footprint = hy.geometry.hierarchical_union(list(merged))
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
    """Concat every region's parts into the per-level finals, mtime-skipped per level."""
    for level in [2, *LEVELS]:
        out = level_output(level)
        parts = [level_output(level, r) for r in regions]
        if any(not p.exists() for p in parts):
            print(f'not writing {out.name}: a region is missing this level')
            continue
        if out.exists() and all(p.stat().st_mtime <= out.stat().st_mtime for p in parts):
            continue
        partial = out.with_name(f'{out.name}.partial')
        rows = hy.parquet.concat_geoparquet(parts, partial)
        partial.replace(out)
        print(f'{rows:,} level-{level} basins -> {out.name}')


if __name__ == '__main__':
    if '--bands' in sys.argv:
        # every band as "level:minzoom:maxzoom", coarsest first, leaf last, for tile_catchments.sh
        # - the tiles and the polygons must agree on what a band is, and this is the one authority
        for level, (lo, hi) in sorted(LEVEL_ZOOMS.items()):
            print(f'{level}:{lo}:{hi}')
        print(f'leaf:{hy.basins.LEAF_ZOOMS[0]}:{hy.basins.LEAF_ZOOMS[1]}')
        sys.exit(0)

    # after --bands, never before: that mode's stdout is parsed by tile_catchments.sh
    hy.console.banner('Generate global basins (one time, not per release)')

    streamnets = sorted(hy.paths.tdx_root.glob('TDX_streamnet_*_01.parquet'))
    if not streamnets:
        sys.exit(f'no TDX_streamnet parquet found under {hy.paths.tdx_root}')
    regions = [p.name.split('_')[2] for p in streamnets]

    finals = [codes_output, outlets_output] + [level_output(lv) for lv in [2, *LEVELS]]
    parts_done = all(level_output(lv, r).exists() for r in regions for lv in [2, *LEVELS])
    # existence is not enough for the codes: a region added to the raw tree must grow them too -
    # caught by a sandbox run where an added region's reaches had no frozen code. The region
    # column is dictionary-encoded, so probing it alone is cheap
    codes_current = False
    if codes_output.exists() and outlets_output.exists():
        have = set(pd.read_parquet(codes_output, columns=[hy.schema.tdx_region_field])
                   [hy.schema.tdx_region_field].unique().tolist())
        codes_current = have == set(regions)
        if not codes_current:
            print(f'{codes_output.name} covers {len(have)} region(s) but the raw tree has '
                  f'{len(regions)}; the codes and outlet registry will be rewritten')
    if all(p.exists() for p in finals) and parts_done and codes_current:
        print('global basins already generated, skipping (delete the files to regenerate)')
        sys.exit(0)

    hy.paths.logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=hy.paths.logs_root / 'global_basins.log', filemode='w',
                        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    started = time.time()
    for path, region in zip(streamnets, regions):
        t0 = time.time()
        have_polygons = all(level_output(lv, region).exists() for lv in [2, *LEVELS])
        # ``codes_current`` covers a tree whose global tables predate the per-region code parts:
        # the rows are already frozen in them, so there is nothing to recover for this region.
        have_codes = (codes_part(region).exists() and outlets_part(region).exists()) or codes_current
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

    if not codes_current:    # probed against the region set at the top, before any skip
        # from the parts rather than a list built during the loop, so a region skipped as complete
        # still contributes its rows and the planet is never all held in memory at once
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
