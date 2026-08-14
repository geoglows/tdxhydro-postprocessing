"""
Geometry reductions that are too slow done the obvious way, and the repair every one of them needs.

The reduction is unioning a very large set of adjacent polygons into their outline, which is what
turns a group's catchments into the group's boundary.

The repair is ``repair``: the polygons these steps produce are routinely self-intersecting, because
both ways of cheapening a coverage - simplifying its shared edges, rounding its vertices onto a
lattice - move linework without re-noding the rings it belongs to. Steps 4 and 8 both need it, so
it lives here rather than in either.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shapely

__all__ = [
    'hierarchical_union',
    'union_chunk_size',
    'polygonal',
    'repair',
    'simplify_coverage',
    'snap',
]

# How many polygons one parallel union task takes. Measured on a 6,771-catchment group: 64/256/1024
# ran in 13.5/10.9/11.8 s, so the curve is flat and the only thing that matters is that it is
# neither 1 nor everything.
union_chunk_size = 256

# Threads for the per-geometry calls below. These are single GEOS calls over the whole array, so
# they run on one core however many threads exist - and on a big band they are most of the wall
# clock: measured on 2_global_basins.py's level-8 band, the threaded dissolve took 80 s and the
# single-threaded chain after it took 106 s of a 196 s region.
#
# They are independent per polygon, so splitting them across threads is a scheduling change and
# nothing else. Measured on a 9,220-basin level-8 part (6.2M vertices), 32 threads:
# set_precision 4.74 s -> 0.27 s (17.5x) and make_valid 0.79 s -> 0.04 s (18.1x), both
# bit-identical to the serial result under shapely.equals_exact.
elementwise_threads = max(1, os.cpu_count() or 8)

# Under this many polygons the split costs more than it saves, so the array goes in one call.
elementwise_min = 512


def _elementwise(function, geometries: np.ndarray, workers: int = None) -> np.ndarray:
    """Run a per-geometry array call across threads. Identical to calling it on the whole array.

    Only for operations that treat each polygon on its own. ``coverage_simplify`` is deliberately
    not one of them - see ``simplify_coverage``.
    """
    workers = workers or elementwise_threads
    if workers == 1 or len(geometries) < elementwise_min:
        return function(geometries)
    size = -(-len(geometries) // workers)
    chunks = [geometries[i:i + size] for i in range(0, len(geometries), size)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return np.concatenate(list(pool.map(function, chunks)))


def hierarchical_union(geometries, workers: int = None, chunk: int = union_chunk_size):
    """
    Union many polygons by unioning small chunks in parallel, then unioning the results, repeatedly
    until one geometry is left.

    Two reasons this beats a single ``shapely.union_all`` over the whole set. Each chunk cancels the
    interior edges of its own neighbourhood, so every round hands the next one far fewer edges than
    it received - on one group, 12.3M input vertices collapse to 931k. And ``union_all`` releases the
    GIL inside GEOS, so the chunks genuinely run in parallel rather than taking turns.

    Measured on group 719 (718 when this was measured - see network_data/group_renumbering.csv;
    6,771 catchments, 12.3M vertices): 36.5 s for the single call against
    10.9 s for chunks of 256 across 16 threads, and the results are identical - same vertex count,
    same area to the last bit. It is a scheduling change, not an approximation.

    ``shapely.coverage_union_all`` is faster still, and whether it is usable depends entirely on
    which catchments these are. On the *raw* ones it is not: measured on the same group, 6,769 of
    the 6,771 have invalid coverage edges, and GEOS raises a side-location conflict at every
    precision tried, including the 1 m grid the geometry is already snapped to.

    **Simplification does not fix that, contrary to what this docstring used to say.** Re-measured
    on 7020000010 straight out of step 4: an 8,000-polygon run leaves 7,999 with invalid coverage
    edges and the same side-location conflict, at 10, 20, 30, 50, 100 and 300 m alike. Coarsening
    never helped. What does is *snapping* onto a lattice coarse enough to merge the mismatched
    vertex pair - which is why the retired basins step, whose bands were snapped, unioned 29,086 of 30,445 basins on
    the fast path, and why step 5, whose catchments are not, mostly lands here.

    So a caller should still try the fast path first and keep this as the fallback, but should
    expect to use the fallback whenever it is holding step 4's output rather than a snapped band.
    """
    workers = workers or os.cpu_count() or 8
    merged = list(geometries)
    if not merged:
        return None
    while len(merged) > 1:
        parts = [merged[i:i + chunk] for i in range(0, len(merged), chunk)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            merged = list(pool.map(shapely.union_all, parts))
    geometry = shapely.make_valid(merged[0])
    if geometry.geom_type not in ('Polygon', 'MultiPolygon'):
        # a union that produced stray lines or points from degenerate input keeps only its areas
        geometry = shapely.union_all([g for g in shapely.get_parts(geometry)
                                      if g.geom_type in ('Polygon', 'MultiPolygon')])
    return None if geometry.is_empty else geometry


def polygonal(geometries: np.ndarray) -> np.ndarray:
    """Reduce anything that is not a polygon to the polygonal parts of itself.

    A repair can hand back a GeometryCollection - the noded linework, or the areas alongside the
    cut-lines that produced them - and a catchment or basin layer is a polygon layer: the parquet
    writer has no geoarrow layout for a collection and refuses the file outright.
    """
    odd = np.flatnonzero(~np.isin(shapely.get_type_id(geometries), (3, 6)))
    if len(odd):
        geometries = geometries.copy()
        for index in odd:
            parts = [p for p in shapely.get_parts(geometries[index])
                     if shapely.get_type_id(p) in (3, 6)]
            geometries[index] = shapely.union_all(parts) if parts else None
    return geometries


def _make_valid_chunk(geometries: np.ndarray) -> np.ndarray:
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


def _make_valid(geometries: np.ndarray) -> np.ndarray:
    """``_make_valid_chunk`` across threads. The element-by-element escape is now per chunk rather
    than for the whole array, which is the same answer - an element the array call already handled
    is retried with the kwargs that handled it - reached without dragging every other element down
    with it."""
    return _elementwise(_make_valid_chunk, geometries)


def repair(geometries: np.ndarray, fallback: np.ndarray = None) -> tuple:
    """``make_valid`` for the ones that need it, as polygons and never as nothing.

    Two things upstream produce invalid rings, and both do it the same way - by moving linework
    without re-noding the ring it belongs to. ``coverage_simplify`` moves each shared edge:
    measured on one region, 14,000 of 21,000 level-9 basins come out self-intersecting. Rounding
    onto a lattice moves every vertex a little: measured on a 4,000-catchment chunk of step 5's
    output, one does. Nothing downstream of a written file minded, tippecanoe included, but the
    next dissolve does - GEOS refuses an invalid ring in either the coverage union or the general
    one - so it is fixed where it is made.

    ``method='structure'`` rebuilds the polygon's areas rather than returning the noded linework as
    a collection, so what comes back is still polygonal and can be unioned again. Measured on one
    region it costs 3.7 s, removes 8 % of the vertices, and changes total area by zero.

    This is also where an emptied polygon is caught, which is why the test is not simply
    ``is_valid``. **An empty polygon is a valid polygon**, so nothing here or in GEOS objects to
    one, and it travels all the way to the parquet writer and to the next dissolve before anything
    does. It arrives two ways: ``make_valid`` collapses one whose repair leaves no area, and
    ``coverage_simplify`` annihilates one smaller than the tolerance outright - measured, a 1 km2
    single-reach basin against level 8's 1,223 m.

    Either way, and however ``make_valid`` itself raised - it does, on the degenerate zero-length
    segments snapping to a lattice leaves behind - the polygon falls back to ``fallback``: the
    geometry as it was before whichever step broke it, which is the last version known to be both
    valid and non-empty. So a file always carries one non-empty polygon per row, and whatever
    dissolves it next always gets a coverage it can union. The count comes back for the log.
    """
    fallback = geometries if fallback is None else fallback
    # is_valid walks every ring, so it is threaded like the repair itself; is_empty and is_missing
    # only read a header and are not worth splitting.
    broken = (~_elementwise(shapely.is_valid, geometries)) | shapely.is_empty(geometries) \
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


def simplify_coverage(geometries: np.ndarray, tolerance: float) -> np.ndarray:
    """``coverage_simplify`` with the boundary pinned, bisecting around whatever GEOS refuses.

    At coarse tolerances a polygon can be smaller than the tolerance, and GEOS simplifies its ring
    away and then raises rebuilding it - for the whole array. Bisecting is a safe response rather
    than a different answer, and only because ``simplify_boundary=False`` is on: it pins the
    outline of whatever it is handed, so an edge shared by the two halves lies on both their
    outlines and neither half moves it - the coverage cannot crack along the split. What is lost
    is detail removed, not detail kept, and only near the cut.

    **This one is not threaded, unlike ``repair`` and ``snap``.** It is a coverage operation, not an
    elementwise one - the whole point is that one call sees every shared edge - so splitting it is
    not a scheduling change, it is a different answer: each chunk's outline gets pinned, and those
    outlines are interior edges of the real coverage, so the cuts keep detail the single call
    removes. Splitting by connected component would be exact, but there is nothing to split -
    measured on a level-8 band, all 9,220 basins are one component. So the bisection above stays
    what it is: an escape from an exception, never a speedup."""
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

    Both sides of a shared divide start from the same vertices, so both land on the same lattice
    points and seams close the same way they did before - which is what lets neighbours snap
    independently. A polygon small enough to vanish into one cell keeps its unsnapped geometry (a
    dropped polygon is a hole in a coverage, worse at any zoom than an unsnapped edge), and so
    does one GEOS refuses outright - the whole-array call is retried element by element rather
    than given up on. Returns the array and how many kept their unsnapped geometry.

    Snapping is per polygon by construction - which is the same independence this docstring already
    relies on for neighbours - so the array is rounded across threads. Same answer, same bits."""
    def chunk(part: np.ndarray) -> np.ndarray:
        try:
            return shapely.set_precision(part, grid)
        except shapely.errors.GEOSException:
            out = np.empty(len(part), dtype=object)
            for i, geometry in enumerate(part):
                try:
                    out[i] = shapely.set_precision(geometry, grid)
                except shapely.errors.GEOSException:
                    out[i] = None
            return out

    snapped = _elementwise(chunk, geometries)
    kept = shapely.is_empty(snapped) | shapely.is_missing(snapped)
    if kept.any():
        snapped[kept] = geometries[kept]
    return snapped, int(kept.sum())
