#!/usr/bin/env python
"""
Aggregate one region's catchments into Pfafstetter basins, one polygon set per zoom band.

Everything about the basin hierarchy lives in this one file: the zoom banding, the code assignment,
the dissolve, and the attributes the tiles carry. It reads only region-level outputs that steps 2
and 4 already wrote, and it writes only files that nothing else in the pipeline reads, so it can be
re-run, re-tuned or deleted without touching anything upstream of it.

    reads   regions/<region>/metadata_<region>.parquet
            regions/<region>/catchments_<region>.geo.parquet
    writes  regions/<region>/basin_level<k>_<region>.geo.parquet, one per band

The levels are built finest first and each one is dissolved out of the one below it rather than out
of the leaf catchments, so only the first dissolve ever sees the whole region. The levels are
nested, so this is the same union either way; what changes is that the six coarse levels union a
few thousand polygons instead of a few hundred thousand. It also means the levels are built as one
chain: a run rebuilds all of them or none, which is why there is no per-level skip.

## Why the map needs this at all

A catchment layer is a coverage: it tiles the ground. So the low zooms cannot be a thinned sample of
the leaf polygons the way the streams are - thinning a coverage leaves holes. A z4 tile can carry
~10^3 polygons and a region has 10^5 catchments, so the low zooms need *fewer, bigger* basins, and
the only way to get them is to merge. That is what the codes decide.

## The codes

A Pfafstetter code is one digit per level, and **a prefix of length k names the level-k basin a
reach belongs to** - so level k's polygons are simply the catchments grouped by that prefix, and the
sets are strictly nested: a level-4 basin is a whole number of level-5 basins, so a selection
survives zooming.

The network is a *forest* of terminal watersheds, not one basin, so a plain Pfafstetter recursion
cannot start: it has no root. The recursion here is uniform and only the split rule switches:

    below the level's budget       -> leaf; pad the rest of the code with 0
    spans >1 terminal watershed    -> COASTAL rule
    otherwise                      -> PFAFSTETTER rule

Both share the digit convention, which is what keeps the code readable as Pfafstetter: the four
largest members take even digits 2, 4, 6, 8 in downstream/along-coast order, and the runs between
them take odd digits 1, 3, 5, 7, 9 with 1 the most downstream. Only what counts as a member differs
- a terminal watershed ranked by total area for the coastal rule, a tributary subtree ranked by
DSContArea for the Pfafstetter rule.

**Which basins may split is what makes this a usable zoom pyramid.** Recursed blindly, Pfafstetter
is badly unbalanced: an interbasin that is already tiny still spawns nine children. An area
threshold does not fix it either - a zoom step quadruples the tile count so each level wants ~4x the
features of the one above, but a split makes up to nine children at once, and measured on
7020000010 an area rule stepped 8.8x, 7.9x, 7.5x, 4.4x and then stalled at 2.3x, 1.4x. So the rule
here is a **budget**: each level gets a target count and basins are refined largest-first until it
is met. That is the right refinement order for a map - the biggest thing on screen is the one most
worth subdividing - and it makes the counts follow the zooms instead of the split radix.

Two limits, both measured and both fine. Pfafstetter only cuts at tributary junctions, so a basin
that is a pure chain never subdivides; that is harmless because the deepest zoom draws individual
catchments keyed by riverId and needs no code. And a handful of reaches have enormous catchments
because step 2 collapses a lake's interior into its outlet (Superior, Baikal, Victoria), so a few
basins stay large at every level.

    RFS_DATA_ROOT=... TDXHYDRO_ROOT=... python 8_pfafstetter_basins.py <region> [--force]
    python 8_pfafstetter_basins.py --bands       # "level:minzoom:maxzoom" per line, for the shell
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

region_root = hy.paths.region_root
logs_root = hy.paths.logs_root

# ---------------------------------------------------------------------------
# The zoom banding. This is the one dial worth turning.
# ---------------------------------------------------------------------------
# basin level -> (min zoom, max zoom). One level per zoom from the first refinement to the leaf, so
# detail arrives in even steps instead of lurching. The coarsest level holds z0-2 because there is
# nothing to refine into above it: it is the first split of the region, and the split radix rather
# than the budget decides how many basins it has.
#
# The banding drives the hierarchy, not the other way round - the code assignment below builds
# exactly these levels and ramps the budgets to the leaf count, so the counts follow the zooms. Add
# a level and the ramp gets gentler; take one away and it gets steeper.
LEVEL_ZOOMS = {
    3: (0, 2),
    4: (3, 3),
    5: (4, 4),
    6: (5, 5),
    7: (6, 6),
    8: (7, 7),
    9: (8, 8),
}

# Full-resolution catchments, one polygon per reach, from here down. Where this starts is the
# expensive decision: moving it one zoom earlier quadruples the features in every tile of that zoom,
# because the same polygons are spread over a quarter as many tiles. See catchment_tiling_design.md.
#
# The band ends at z11, not z12. The leaf geometry is coverage-simplified in step 4 at a tolerance
# that is sub-pixel at z11, so a z12 tile carries no vertex a z11 tile does not already have -- it
# is a second full-resolution copy of the largest tileset in the pipeline for nothing. Clients
# overzoom past the maximum, which draws the same geometry at the same fidelity.
LEAF_ZOOMS = (9, 11)

# web mercator resolution at zoom 0, metres per pixel at the equator, and the vertex spacing worth
# keeping: a vertex closer than this many pixels to its neighbour cannot be seen at the band's
# finest zoom.
MERCATOR_M_PER_PX_Z0 = 156543.03392
PIXELS_PER_VERTEX = 1.0


def zoom_tolerance(zoom: int, pixels: float = PIXELS_PER_VERTEX) -> float:
    """Web mercator metres per pixel at ``zoom``, times the vertex spacing worth keeping."""
    return pixels * MERCATOR_M_PER_PX_Z0 / 2 ** zoom


def bands() -> list:
    """Every band as ``level:minzoom:maxzoom``, coarsest first, ``leaf`` last. Printed for the
    shell so the tiles and the polygon sets they are built from come from this one definition."""
    out = [f'{level}:{lo}:{hi}' for level, (lo, hi) in sorted(LEVEL_ZOOMS.items())]
    return out + [f'leaf:{LEAF_ZOOMS[0]}:{LEAF_ZOOMS[1]}']


# The code assignment itself lives in hydrography/basins.py: it is the one part of this step that is
# a self-contained algorithm rather than a driver, and the write-up of how the digits are chosen is
# its module docstring.
assign_basin_codes = hy.basins.assign_basin_codes


# ---------------------------------------------------------------------------
# The dissolve
# ---------------------------------------------------------------------------
def dissolve_by(geometries: np.ndarray, group: np.ndarray, workers: int = None) -> tuple:
    """One polygon per distinct value of ``group``, with the group ids ascending.

    ``coverage_union_all`` cancels shared edges instead of running a general overlay, which is the
    fast path and is available only because step 4 rebuilt every shared edge to match on both sides
    - the raw catchments are not edge-matched and it fails on them outright. Some groups still fail
    ``coverage_is_valid`` because snapping to the 1 m grid leaves a pair of neighbours fractionally
    overlapping, and GEOS refuses a whole group for one bad edge; those fall back to the general
    union, which gives the same answer more slowly. The fallback count comes back so the log can say
    how much of the region took the slow path.

    The unions run across threads because GEOS releases the GIL, and a group of one is passed
    through untouched rather than sent into GEOS to union with nothing. The fallbacks are collected
    and run afterwards on the main thread: ``hierarchical_union`` has a thread pool of its own, and
    nesting it inside this one would oversubscribe the machine for the few groups that need it.
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
            merged[index] = shapely.coverage_union_all(geometries[rows])
        except shapely.errors.GEOSException:
            retry.append(index)      # list.append is atomic, so no lock is needed here

    with ThreadPoolExecutor(max_workers=workers or os.cpu_count() or 8) as pool:
        list(pool.map(union, range(len(blocks))))
    for index in retry:
        # the general union is no more tolerant of a self-intersecting ring than the coverage one,
        # so the group is repaired on the way in rather than raising a second time
        members, _ = repair(geometries[blocks[index]])
        try:
            merged[index] = hy.geometry.hierarchical_union(members)
        except shapely.errors.GEOSException:
            # a repaired ring can still put GEOS in a state it asserts its way out of. Unioning on
            # a fixed grid is the standard answer: it collapses the near-coincident vertices the
            # overlay could not order, at a metre, which is the precision the geometry is on anyway
            merged[index] = shapely.union_all(members, grid_size=1.0)
    return keys, merged, len(retry)


def simplify_coverage(geometries: np.ndarray, tolerance: float) -> np.ndarray:
    """``coverage_simplify``, bisecting around whatever GEOS refuses to simplify.

    At the coarse levels the tolerance is tens of kilometres and a basin can be smaller than that,
    so GEOS simplifies its ring away and then raises rebuilding it - and it raises for the whole
    array, so one basin costs the level its simplification.

    Bisecting is a safe response rather than a different answer, and only because
    ``simplify_boundary=False`` is on. It pins the outline of whatever it is handed, so a half is
    simplified less than the whole would be - more of its edges are outline - but an edge shared by
    the two halves lies on both their outlines and neither half moves it. The coverage cannot crack
    along the split. What is lost is detail removed, not detail kept, and only near the cut.
    """
    try:
        return shapely.coverage_simplify(geometries, tolerance, simplify_boundary=False)
    except shapely.errors.GEOSException:
        if len(geometries) == 1:
            return geometries       # nowhere left to bisect: this one stays at full detail
    half = len(geometries) // 2
    return np.concatenate([simplify_coverage(geometries[:half], tolerance),
                           simplify_coverage(geometries[half:], tolerance)])


def snap(geometries: np.ndarray, grid: float) -> tuple:
    """Round every vertex onto a ``grid``-metre lattice, dropping the ones that merge.

    This exists to get the region's outer boundary out of the way. ``coverage_simplify`` is run with
    ``simplify_boundary=False``, which pins the outline of whatever it is handed, because a region's
    basins have to abut the next region's exactly and two regions simplifying their shared divide
    independently would crack it open. The cost is that the outline never gets cheaper: measured on
    5020049720, the level-3 polygons carry 7,536,693 vertices of which 7,500,525 are the pinned
    region footprint and 36,168 are the interior divides the level is actually about, and every one
    of the seven levels re-unions, re-validates and re-simplifies that same footprint.

    Snapping is the one reduction that does not need the two regions to agree on anything, because
    it does not look at the linework at all - a vertex goes to the nearest lattice point and that is
    a function of its own coordinates. Both sides of a shared divide start from the same vertices,
    so both land on the same points, and the seam closes the same way it did before. The lattice is
    the band's own tolerance, one pixel at the finest zoom the band is drawn at, so anything it
    moves was already below what that zoom can resolve.

    It is what makes the telescope actually telescope. On the same region the levels went 4.63M,
    2.69M, 2.15M, 2.01M, 1.92M, 1.78M, 1.21M vertices - barely falling, because the footprint was
    almost all of it - and now go 2.87M, 693k, 194k, 58k, 19k, 6.3k, 1.9k. Level 3 costs under a
    second where it used to cost longer than level 9. Total area moves by 0.4% over all seven.

    A basin small enough to vanish into a single lattice cell keeps its unsnapped geometry instead:
    the levels are a coverage and a dropped basin is a hole, which is a worse artifact at any zoom
    than an unsnapped edge on something under a pixel wide. So does one GEOS refuses to snap at all
    - it is an overlay underneath and it raises on rings that ``is_valid`` accepts - which is why the
    whole-array call, the fast one, is retried element by element rather than given up on. Both
    kinds are counted together for the log; they are the same thing from the map's point of view,
    a basin that kept more detail than the band needs.
    """
    try:
        snapped = shapely.set_precision(geometries, grid)
    except shapely.errors.GEOSException:
        snapped = np.empty(len(geometries), dtype=object)
        for i, geometry in enumerate(geometries):
            try:
                snapped[i] = shapely.set_precision(geometry, grid)
            except shapely.errors.GEOSException:
                snapped[i] = None
    kept = shapely.is_empty(snapped) | shapely.is_missing(snapped)
    if kept.any():
        snapped[kept] = geometries[kept]
    return snapped, int(kept.sum())


def polygonal(geometries: np.ndarray) -> np.ndarray:
    """Reduce anything that is not a polygon to the polygonal parts of itself.

    A repair can hand back a GeometryCollection - the noded linework, or the areas alongside the
    cut-lines that produced them - and a basin level is a polygon layer: the parquet writer has no
    geoarrow layout for a collection and refuses the file outright.
    """
    odd = np.flatnonzero(~np.isin(shapely.get_type_id(geometries), (3, 6)))
    if len(odd):
        geometries = geometries.copy()
        for index in odd:
            parts = [p for p in shapely.get_parts(geometries[index])
                     if shapely.get_type_id(p) in (3, 6)]
            geometries[index] = shapely.union_all(parts) if parts else None
    return geometries


def _make_valid(geometries: np.ndarray) -> np.ndarray:
    """``make_valid`` over an array, element by element only if the array call raises."""
    try:
        return shapely.make_valid(geometries, method='structure', keep_collapsed=False)
    except shapely.errors.GEOSException:
        pass
    out = np.empty(len(geometries), dtype=object)
    for i, geometry in enumerate(geometries):
        for kwargs in ({'method': 'structure', 'keep_collapsed': False}, {}):
            try:
                out[i] = shapely.make_valid(geometry, **kwargs)
                break
            except shapely.errors.GEOSException:
                continue
    return out


def repair(geometries: np.ndarray, fallback: np.ndarray = None) -> tuple:
    """``make_valid`` for the ones that need it, as polygons and never as nothing.

    ``coverage_simplify`` routinely leaves self-intersecting rings - measured on one region, 14,000
    of 21,000 level-9 basins - because it moves each shared edge without re-noding the rings the
    edge belongs to. Nothing downstream of a written file minded, tippecanoe included, but the level
    above does: it unions these polygons, and GEOS refuses an invalid ring in either the coverage
    union or the general one.

    ``method='structure'`` rebuilds the polygon's areas rather than returning the noded linework as
    a collection, so what comes back is still polygonal and can be unioned again. Measured on one
    region it costs 3.7 s, removes 8 % of the vertices, and changes total area by zero.

    This is also where an emptied basin is caught, which is why the test is not simply
    ``is_valid``. **An empty polygon is a valid polygon**, so nothing here or in GEOS objects to
    one, and it travels all the way to the parquet writer and to the next level's union before
    anything does. It arrives two ways: ``make_valid`` collapses a basin whose repair leaves no
    area, and ``coverage_simplify`` annihilates a basin smaller than the tolerance outright -
    measured, a 1 km² single-reach basin against level 8's 1,223 m.

    Either way, and however ``make_valid`` itself raised - it does, on the degenerate zero-length
    segments snapping to a lattice leaves behind - the basin falls back to ``fallback``: the
    geometry as it was before whichever step broke it, which is the last version known to be both
    valid and non-empty. So a level always writes one non-empty polygon per basin, and the level
    above it always gets a coverage it can union. The count comes back for the log.
    """
    fallback = geometries if fallback is None else fallback
    broken = (~shapely.is_valid(geometries)) | shapely.is_empty(geometries) \
        | shapely.is_missing(geometries)
    if not broken.any():
        return geometries, 0

    geometries = geometries.copy()
    fixed = polygonal(_make_valid(geometries[broken]))
    lost = shapely.is_missing(fixed) | shapely.is_empty(fixed)
    if lost.any():
        fixed[lost] = fallback[broken][lost]
    geometries[broken] = fixed
    return geometries, int(lost.sum())


def outlet_candidates(frame: pd.DataFrame, label: np.ndarray) -> pd.DataFrame:
    """The rows that could be a basin outlet at ``label``'s level, or at any coarser one.

    A reach is an outlet of its basin when its downstream reach is in a different basin or the
    network ends there. The levels are strictly nested - a coarse basin is a whole number of fine
    ones - so a reach that leaves a coarse basin also leaves the fine basin it sits in, and the
    coarse level's outlets are a subset of the fine level's. Computing this once at the finest level
    therefore serves every level above it, which is what keeps the six coarse levels off the full
    5-million-row frame: they only ever re-test the few thousand rows kept here.

    Each candidate carries the frame row it occupies and the frame row it drains into, so a coarser
    level can decide whether it is still an outlet by indexing that level's labels with two integers
    rather than walking the network again.
    """
    river = frame[hy.schema.river_id].to_numpy()
    row_of = pd.Series(np.arange(len(frame)), index=river)
    downstream = row_of.reindex(frame[hy.schema.next_river_id].to_numpy()).to_numpy()
    leaves_network = np.isnan(downstream)
    downstream_row = np.where(leaves_network, 0, np.nan_to_num(downstream, nan=0)).astype(np.int64)
    is_outlet = leaves_network | (label[downstream_row] != label)

    rows = np.flatnonzero(is_outlet)
    return frame.iloc[rows][[hy.schema.river_id, hy.schema.river_index,
                             hy.schema.tdx_ds_area_field]].assign(
        _row=rows, _down=downstream_row[rows], _end=leaves_network[rows])


def basin_outlets(candidates: pd.DataFrame, label: np.ndarray, n_basins: int) -> pd.DataFrame:
    """The reach each basin drains through, as riverId/riverIndex indexed by basin id.

    A basin's outlet is the member whose downstream reach is outside the basin (or is the end of the
    network). For anything the Pfafstetter rule produced that member is unique - a tributary basin is
    a subtree, and an interbasin is a run of main stem plus the small tributaries hanging off it, so
    either way everything drains through one reach.

    A basin the coastal rule produced is the exception: it is several whole terminal watersheds
    grouped along the coast, so it has one outlet per watershed and no single reach all of it drains
    through. There the largest contributing area wins, which names the basin after its dominant
    river - the convention HydroBASINS uses for a coastal group. Ties break on the lower riverId so
    the choice is reproducible.

    ``candidates`` comes from ``outlet_candidates`` and ``label`` is this level's basin id for every
    row of the full frame.
    """
    own = label[candidates['_row'].to_numpy()]
    keep = candidates['_end'].to_numpy() | (label[candidates['_down'].to_numpy()] != own)

    picked = candidates.loc[keep].assign(_basin=own[keep])
    picked = picked.sort_values(['_basin', hy.schema.tdx_ds_area_field, hy.schema.river_id],
                                ascending=[True, False, True])
    picked = picked.groupby('_basin', sort=True).first()
    if len(picked) != n_basins:
        missing = sorted(set(range(n_basins)) - set(picked.index))
        raise RuntimeError(f'{len(missing)} basin(s) have no outlet reach, e.g. {missing[:5]}')
    return picked[[hy.schema.river_id, hy.schema.river_index]]


if __name__ == '__main__':
    if '--bands' in sys.argv:
        print('\n'.join(bands()))
        sys.exit(0)

    force = '--force' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) != 1:
        sys.exit('usage: 8_pfafstetter_basins.py <region> [--force]  |  --bands')
    region = int(args[0])

    outputs_dir = region_root / f'{region}'
    levels = sorted(LEVEL_ZOOMS)
    level_outputs = {lv: outputs_dir / f'basin_level{lv}_{region}.geo.parquet' for lv in levels}
    if not force and all(p.exists() for p in level_outputs.values()):
        print(f'All basin levels for region {region} already exist, skipping')
        sys.exit(0)

    # catchments are built one region at a time and the set on disk is often partial, so a region
    # that has not been through step 4 yet is skipped rather than failing the whole pipeline
    catchments_path = outputs_dir / f'catchments_{region}.geo.parquet'
    if not catchments_path.exists():
        print(f'region {region}: no catchments, run step 4 first, skipping')
        sys.exit(0)

    logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=logs_root / f'pfafstetter_basins_{region}.log', filemode='w',
                        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    metadata = pd.read_parquet(outputs_dir / f'metadata_{region}.parquet')
    # only the geometry is needed off the catchment file - every attribute the basins carry comes
    # from the metadata, which is asserted below to be in the same row order. Reading two columns
    # instead of the whole frame is most of a large region's read time.
    catchments = gpd.read_parquet(catchments_path, columns=[hy.schema.river_id, hy.schema.geometry])
    if not np.array_equal(catchments[hy.schema.river_id].to_numpy(),
                          metadata[hy.schema.river_id].to_numpy()):
        raise RuntimeError('the catchments and the metadata are not in the same row order; '
                           'rerun step 4, or step 3 if the ordering itself changed')
    logging.info(f'{len(catchments):,} catchments joined to their metadata')

    codes = assign_basin_codes(metadata, levels, log=logging.info)
    # basin ids are dense, 0..n-1, and ordered by code, so the id of a reach's basin at any level is
    # a plain array lookup and the level-to-level map below is one too
    basin_ids = {level: pd.factorize(codes.str[:i + 1], sort=True)[0].astype(np.int32)
                 for i, level in enumerate(levels)}
    metadata['pfafCode'] = codes.to_numpy()

    # the finest level's outlet set, computed once against the full frame and re-tested per level
    candidates = outlet_candidates(metadata, basin_ids[levels[-1]])
    logging.info(f'{len(candidates):,} of {len(metadata):,} reaches can be a basin outlet')

    # The telescope. The finest level is dissolved out of the leaf catchments; every level above it
    # is dissolved out of the level below, which is already both aggregated and simplified. That is
    # the speed argument: running every level against the leaves means seven passes over the whole
    # region, where this is one pass over the leaves and six over a few thousand polygons each. It
    # costs nothing in fidelity because the levels are nested - the same polygons are being unioned
    # either way - and because the tolerance only ever coarsens going up, so a level never needs a
    # vertex the level below it already dropped.
    crs = catchments.crs
    parts = catchments[hy.schema.geometry].to_numpy()
    part_basin = basin_ids[levels[-1]]
    del catchments

    for position, level in reversed(list(enumerate(levels))):
        started = time.time()
        min_zoom, max_zoom = LEVEL_ZOOMS[level]
        keys, geometries, fallbacks = dissolve_by(parts, part_basin)
        if fallbacks:
            logging.info(f'level {level}: {fallbacks:,} of {len(keys):,} basin(s) took the general '
                         f'union because GEOS rejected their coverage')

        raw = int(shapely.get_num_coordinates(geometries).sum())
        # simplify at a tolerance matched to the FINEST zoom in the band - the one that still has to
        # look right - so a level-3 basin drawn z0-z2 keeps what z2 can resolve and drops the rest.
        # The outer boundary stays pinned, as it is for the leaves, because a region's aggregate
        # basins abut the next region's and the two have to agree.
        tolerance = max(zoom_tolerance(max_zoom), 1.0)
        simplified = simplify_coverage(geometries, tolerance)
        broken = int(((~shapely.is_valid(simplified)) | shapely.is_empty(simplified)).sum())
        # each repair falls back to the array it was handed, which is the last one known good
        simplified, lost = repair(simplified, fallback=geometries)
        # and then the pinned outline, which is the only thing simplification can never touch.
        # after the repair, not before it: snapping is a GEOS overlay and it refuses invalid rings
        # for the same reason the unions do. It leaves a handful invalid again - the ones it
        # declined to snap, and the ones it collapsed a ring inside - so the repair runs either side
        snapped, collapsed = snap(simplified, tolerance)
        geometries, lost_again = repair(snapped, fallback=simplified)
        kept = int(shapely.get_num_coordinates(geometries).sum())
        if collapsed or broken:
            logging.info(f'level {level}: {broken:,} basin(s) of {len(keys):,} repaired after '
                         f'simplification, {collapsed:,} too small to snap onto the '
                         f'{tolerance:,.0f} m lattice, {lost + lost_again:,} kept an earlier '
                         f'geometry because the repair collapsed them')

        # the code is constant within a basin by construction, so the first row's is the basin's
        label = basin_ids[level]
        attributes = metadata.groupby(label, sort=True).agg(
            pfafCode=('pfafCode', 'first'),
            riverCount=(hy.schema.river_id, 'size'),
            areaM2=(hy.schema.area, 'sum'),
            strahlerOrder=(hy.schema.strahler_order, 'max'),
        )
        merged = gpd.GeoDataFrame({'basinId': keys}, geometry=geometries, crs=crs)
        merged = merged.merge(attributes, left_on='basinId', right_index=True, how='left')
        merged = merged.merge(basin_outlets(candidates, label, len(keys)),
                              left_on='basinId', right_index=True, how='left', validate='one_to_one')
        merged['level'] = np.int32(level)
        merged = hy.schema.enforce_int32(merged)

        merged = merged[[hy.schema.river_id, hy.schema.river_index, 'basinId', 'level', 'pfafCode',
                         'riverCount', 'areaM2', 'strahlerOrder', hy.schema.geometry]]
        hy.parquet.write_geoparquet(merged, level_outputs[level])
        logging.info(f'level {level} (z{min_zoom}-{max_zoom}): {len(merged):,} basins, '
                     f'{raw:,} -> {kept:,} vertices at {tolerance:,.0f} m, '
                     f'{time.time() - started:.0f}s -> {level_outputs[level].name}')
        print(f'level {level} (z{min_zoom}-{max_zoom}): {len(merged):,} basins, {kept:,} vertices')

        # this level's polygons are the next one's input, mapped onto the next one's basin ids. the
        # levels are nested, so every row of a level-k basin shares one level-(k-1) id and a scatter
        # by id is enough to build the map
        if position:
            coarser = np.zeros(len(keys), dtype=np.int32)
            coarser[label] = basin_ids[levels[position - 1]]
            parts, part_basin = geometries, coarser[keys]
