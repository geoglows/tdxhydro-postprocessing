#!/usr/bin/env python
"""
Aggregate one region's catchments into Pfafstetter basins, one polygon set per zoom band.

Everything about the basin hierarchy lives in this one file: the zoom banding, the code assignment,
the dissolve, and the attributes the tiles carry. It reads only region-level outputs that steps 2
and 4 already wrote, and it writes only files that nothing else in the pipeline reads, so it can be
re-run, re-tuned or deleted without touching anything upstream of it.

    reads   regions/<region>/metadata_<region>.parquet
            regions/<region>/catchments_<region>.geo.parquet
    writes  regions/<region>/catchments_tile_<region>.geo.parquet, the leaf band
            regions/<region>/basin_level<k>_<region>.geo.parquet, one per aggregate band

**Every band's geometry is cut here, including the leaf's.** Step 4 publishes the catchments at the
resolution the source DEM has, which is the right thing for a data product and far more than any
tile needs; the tolerance a band is drawn at is a function of the zooms it covers and so belongs
with the banding, which is here. So the leaf is a band like any other: it is simplified at its own
zooms' tolerance into catchments_tile_<region>.geo.parquet, and that -- not the published
catchments -- is what tile_catchments.sh tiles and what the aggregate levels are dissolved out of.

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

    RFS_DATA_ROOT=... TDXHYDRO_ROOT=... python 7_pfafstetter_basins.py <region>
    python 7_pfafstetter_basins.py --bands       # "level:minzoom:maxzoom" per line, for the shell
"""
import logging
import math
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
# detail arrives in even steps instead of lurching. The coarsest level holds z0-3 because there is
# nothing to refine into above it: it is the first split of the region, and the split radix rather
# than the budget decides how many basins it has.
#
# The banding drives the hierarchy, not the other way round - the code assignment below builds
# exactly these levels and ramps the budgets to the leaf count, so the counts follow the zooms. Add
# a level and the ramp gets gentler; take one away and it gets steeper.
LEVEL_ZOOMS = {
    3: (0, 3),
    4: (4, 4),
    5: (5, 5),
    6: (6, 6),
    7: (7, 7),
    8: (8, 8),
    9: (9, 9),
}

# One polygon per reach from here down - no more aggregation, only the leaf catchments themselves.
# Where this starts is the expensive decision: moving it one zoom earlier quadruples the features in
# every tile of that zoom, because the same polygons are spread over a quarter as many tiles. See
# catchment_tiling_design.md.
#
# The band ends at z10 and clients overzoom past it, which draws the same geometry at the same
# fidelity. z10 is therefore the zoom the leaf tolerance below is derived from, and the deepest
# detail anything downstream of this file can show.
LEAF_ZOOMS = (10, 10)

# web mercator resolution at zoom 0, metres per pixel at the equator, and the vertex spacing worth
# keeping: a vertex closer than this many pixels to its neighbour cannot be seen at the band's
# finest zoom.
#
# A quarter pixel rather than a whole one. At one pixel the tolerance is the largest error that is
# invisible *in the limit*, which is the wrong target for two reasons: an error that size lands on
# the pixel grid as a visibly moved edge about as often as not, and it leaves nothing for
# tippecanoe's own per-zoom simplification to work with, so the two compose into something coarser
# than either. Quartering it is the whole fix for faceted outlines at the low zooms and costs
# almost nothing there - the aggregate bands are a rounding error next to the leaf - while at the
# leaf it is the difference between a boundary that follows the terrain and one that does not.
MERCATOR_M_PER_PX_Z0 = 156543.03392
PIXELS_PER_VERTEX = 0.25

# The leaf band is simplified in chunks, for the reason step 4 chunks its coverage pass: a single
# coverage_simplify call over a whole region's catchments does not fit in memory (31.7 GB without
# completing, measured). Chunking is safe only because simplify_boundary=False pins each chunk's
# outline, so chunk seams come out exact, and it is only cheap because the rows are in the published
# riverIndex order, which makes a contiguous run a compact clump rather than a scattered set - a
# pinned outline costs whatever is on it. Step 4 establishes the order; the row-order check below
# is what confirms this file is reading it.
CHUNK_SIZE = 20_000


def zoom_tolerance(zoom: int, pixels: float = PIXELS_PER_VERTEX) -> float:
    """Web mercator metres per pixel at ``zoom``, times the vertex spacing worth keeping, rounded
    to the nearest power of two.

    **The rounding is not cosmetic.** This value is the lattice ``snap`` rounds every vertex onto,
    and a lattice that is not a power of two puts the coordinates somewhere float64 cannot say
    exactly: ``set_precision`` computes ``round(x / g) * g``, which is exact when ``g`` is a power
    of two and one ulp of noise in the low mantissa bytes when it is not. Those bytes are exactly
    what ``BYTE_STREAM_SPLIT`` and zstd need to be zero -- parquet.py spells out that the encoding
    makes files *larger* on full-precision floats, because the low byte planes become noise it has
    to store instead of a run of zeros it can collapse.

    Measured on 7020000010: the leaf band snapped onto the raw 19.109 m lattice came out at 565 MB
    for 65.8 M vertices, against 229 MB for the 205.6 M vertices of the file it was cut from -- 2.5x
    the bytes for a third of the geometry, and 0 % of its coordinates integer-valued where the
    source's are 100 %.

    Rounding rather than flooring keeps the value within 2^(1/2) of the pixel budget asked for, and
    since the input halves per zoom the bands stay exactly one doubling apart: 16 m at z11, 32 at
    z10, 64 at z9, and so on.
    """
    raw = pixels * MERCATOR_M_PER_PX_Z0 / 2 ** zoom
    return float(max(1, 2 ** round(math.log2(raw))))


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
    the band's own tolerance - a quarter pixel at the finest zoom the band is drawn at, rounded to a
    power of two - so anything it moves was already below what that zoom can resolve. See
    ``zoom_tolerance`` for why the power of two is not optional.

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


# Both moved to hydrography/geometry.py: step 4 now needs the same repair for the same reason -
# it rounds onto a lattice without re-noding what the rounding moved - and two copies of a
# fallback this subtle is one too many. Imported under the old names; the call sites are unchanged.
polygonal = hy.geometry.polygonal
repair = hy.geometry.repair


def simplify_leaf(geometries: np.ndarray, tolerance: float, chunk_size: int = CHUNK_SIZE) -> tuple:
    """Cut the leaf catchments to their band's tolerance, in place, chunk by chunk.

    The same simplify -> repair -> snap -> repair sequence the aggregate levels run, for the same
    reasons, with one difference: it is chunked. There are 10^5 polygons here against 10^4 at the
    finest aggregate level and 10^1 at the coarsest, and a whole-region ``coverage_simplify`` does
    not fit in memory - see ``CHUNK_SIZE``.

    In place so the source geometry is released as it goes. A region's raw catchments are ~78 bytes
    per vertex once GEOS holds them, which is already most of this script's peak, and holding the
    cut copy alongside all of it would double that for no reason: the caller wants only the cut
    version, both to write and to dissolve the levels out of.

    Returns the vertex count before and after, and how many polygons kept a less simplified geometry
    because a repair or the snap would have collapsed them.
    """
    before = after = held = 0
    for start in range(0, len(geometries), chunk_size):
        block = slice(start, start + chunk_size)
        raw = geometries[block]
        before += int(shapely.get_num_coordinates(raw).sum())
        clean, lost = repair(simplify_coverage(raw, tolerance), fallback=raw)
        snapped, collapsed = snap(clean, tolerance)
        clean, lost_again = repair(snapped, fallback=clean)
        after += int(shapely.get_num_coordinates(clean).sum())
        held += lost + collapsed + lost_again
        geometries[block] = clean
    return before, after, held


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
        sys.stdout.flush()
        os._exit(0)

    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) != 1:
        sys.exit('usage: 7_pfafstetter_basins.py <region>  |  --bands')
    region = int(args[0])

    outputs_dir = region_root / f'{region}'
    levels = sorted(LEVEL_ZOOMS)
    level_outputs = {lv: outputs_dir / f'basin_level{lv}_{region}.geo.parquet' for lv in levels}
    # the leaf band is written here too, and the aggregate levels are dissolved out of it, so it is
    # part of the same all-or-nothing chain as the levels rather than a separate skip
    leaf_output = outputs_dir / f'catchments_tile_{region}.geo.parquet'
    if leaf_output.exists() and all(p.exists() for p in level_outputs.values()):
        print(f'All bands for region {region} already exist, skipping')
        # os._exit, not sys.exit: pyarrow's thread pool destructor can hang at interpreter exit, and
        # this step runs under xargs, where one wedged process holds its slot and stalls the run.
        # See 5_generate_groups.py, where it happened. The flush is because print buffers to a pipe.
        sys.stdout.flush()
        os._exit(0)

    # catchments are built one region at a time and the set on disk is often partial, so a region
    # that has not been through step 4 yet is skipped rather than failing the whole pipeline
    catchments_path = outputs_dir / f'catchments_{region}.geo.parquet'
    if not catchments_path.exists():
        print(f'region {region}: no catchments, run step 4 first, skipping')
        sys.stdout.flush()
        os._exit(0)

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

    crs = catchments.crs
    parts = catchments[hy.schema.geometry].to_numpy()
    del catchments

    # The leaf band, cut at the tolerance z11 can resolve. This is the first thing done with the
    # geometry, not the last, because it serves both purposes at once: it is what gets tiled, and
    # it is what the levels telescope out of. Dissolving the levels out of the published catchments
    # instead would union geometry at DEM resolution to produce basins drawn at z8 and coarser.
    leaf_tolerance = max(zoom_tolerance(LEAF_ZOOMS[1]), 1.0)
    started = time.time()
    raw, kept, held = simplify_leaf(parts, leaf_tolerance)
    # the row order is the catchments', asserted against the metadata's above, so the ids can be
    # taken off either. riverIndex rides along for the same reason it does on the published
    # catchments: a leaf tile carries the same id *and* index the stream network does.
    leaf = gpd.GeoDataFrame(
        metadata[[hy.schema.river_id, hy.schema.river_index]].copy(), geometry=parts, crs=crs)
    hy.parquet.write_geoparquet(hy.schema.enforce_int32(leaf), leaf_output)
    del leaf
    logging.info(f'leaf (z{LEAF_ZOOMS[0]}-{LEAF_ZOOMS[1]}): {len(parts):,} catchments, '
                 f'{raw:,} -> {kept:,} vertices at {leaf_tolerance:,.0f} m '
                 f'({100 * kept / raw:.2f}%), {held:,} kept an earlier geometry, '
                 f'{time.time() - started:.0f}s -> {leaf_output.name}')
    print(f'leaf (z{LEAF_ZOOMS[0]}-{LEAF_ZOOMS[1]}): {len(parts):,} catchments, {kept:,} vertices')

    # The telescope. The finest level is dissolved out of the leaf band above; every level after it
    # is dissolved out of the level below, which is already both aggregated and simplified. That is
    # the speed argument: running every level against the leaves means seven passes over the whole
    # region, where this is one pass over the leaves and six over a few thousand polygons each. It
    # costs nothing in fidelity because the levels are nested - the same polygons are being unioned
    # either way - and because the tolerance only ever coarsens going up, so a level never needs a
    # vertex the level below it already dropped.
    part_basin = basin_ids[levels[-1]]

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
