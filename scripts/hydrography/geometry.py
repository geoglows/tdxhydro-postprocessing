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
    'fill_holes',
    'hierarchical_union',
    'union_coverage',
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
    vertex pair - which is why the retired basins step, whose bands were snapped, unioned 29,086 of
    30,445 basins on the fast path, and why step 5, whose catchments are not, used to land here.

    What fixes it at the root is noding the source, which 1_translate_tdxhydro.py now does once per
    region with GEOS 3.14's coverage cleaner (see coverage.py). On a tree that has had that, the
    catchments are a coverage and ``union_coverage`` takes the fast path; on one that has not, it
    lands here. So this stays the fallback and callers should reach it through ``union_coverage``
    rather than directly.
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


# How far a coverage union's area may sit from the summed area of its parts before the answer is
# thrown away. A coverage is a partition, so the two are the same number, and every way the fast
# path goes wrong - a sliver counted twice, a sliver lost - moves it. Measured, as a relative gap:
#
#     cleaned group 122 catchments   0          (13,535 polygons)
#     cleaned group 108 catchments   0          (153,602 polygons)
#     cleaned source basins, deg2    5.2e-14    (float cancellation on numbers this small)
#     un-noded group 122, fast path  1.0e-09    <- the answer this exists to reject
#     un-noded group 122, exact      3.4e-11    (the parts really do overlap: it is not a coverage)
#
# On a coverage the two agree bit for bit, so this is set four orders above the worst clean
# measurement and three below the defect, rather than anywhere near the middle of them.
coverage_area_tolerance = 1e-12


def union_coverage(geometries, workers: int = None, chunk: int = union_chunk_size):
    """Dissolve a coverage, by edge cancellation if the input really is one and the long way if not.

    ``coverage_union_all`` is the right operation for these inputs and cannot be trusted blind.
    Handed an un-noded coverage it raises a side-location conflict on some inputs - group 108's
    153,602 catchments - and on others it *returns*, with slivers in it: on group 122 its answer is
    390 m2 from the exact union and ``symmetric_difference`` against that union throws
    ``unable to assign free hole to a shell``. A silently wrong dissolve is the one outcome worth
    spending something to avoid.

    What it is not worth spending is ``coverage_is_valid``, which is the obvious gate and costs
    more than it saves: 37.7 s on group 108 against the 40.5 s ``hierarchical_union`` it would be
    avoiding. So the check is on the *answer* rather than the input - a coverage is a partition, so
    its union's area is the sum of its parts' areas, and ``shapely.area`` over the array is an
    elementwise call that threads (see ``_elementwise``). Anything the fast path gets wrong shows
    up there.

    Measured on the cleaned files, against ``hierarchical_union`` on the same input:
    group 122 0.47 s against 4.3 s, group 108 8.0 s against 40.5 s, both to the same area.
    """
    geometries = np.asarray(geometries, dtype=object)
    if not len(geometries):
        return None
    expected = float(_elementwise(shapely.area, geometries).sum())
    try:
        geometry = shapely.coverage_union_all(geometries)
    except shapely.errors.GEOSException:
        geometry = None
    if geometry is not None and not geometry.is_empty:
        if abs(geometry.area - expected) <= coverage_area_tolerance * max(expected, 1.0):
            return geometry
    return hierarchical_union(geometries, workers=workers, chunk=chunk)


def _occupant(hole, tree, skip: int):
    """Whatever in ``tree``, other than ``skip``, has area inside ``hole``. None if nothing does.

    The tree predicate does the work and is prepared, so the usual answer - nothing but the ring's
    own owner - costs a bounding-box test. An intersection is only computed for a candidate that
    survives it, and it is an *area* test rather than a hit test because everything sharing a ring
    intersects it: a hole's own polygon runs along its whole edge and encloses none of it.
    """
    if tree is None:
        return None
    pieces = []
    for candidate in tree.query(hole, predicate='intersects'):
        if candidate == skip:
            continue
        piece = shapely.intersection(hole, tree.geometries[candidate])
        if shapely.area(piece) > 0:
            pieces.append(piece)
    if not pieces:
        return None
    return pieces[0] if len(pieces) == 1 else shapely.union_all(pieces)


def fill_holes(geometry, occupied=None, skip: int = None):
    """Close a polygon's interior rings, keeping open only the ground something else is standing on.

    A dissolve of a coverage keeps a hole wherever the coverage has one, and the holes are of two
    kinds. Most are ground the coverage never claimed - a watershed this release dropped, a
    no-runoff basin, an endorheic sink - and an outline drawn with those punched out of it reads as
    shrapnel rather than as the region it is meant to bound. The rest are real: another group, or
    another part of this one, that this outline happens to enclose. Filling that kind is how an
    outline comes to claim ground another already claims, which is the overlap the caller reports.

    **The unit is the occupant, not the ring.** Deciding per ring - keep the whole hole if anything
    at all is inside it - was measured on the 30 published outlines and is far too blunt: 196 holes
    totalling 86,210 km2 would stay open on account of an occupant covering less than a thousandth
    of them, which is a 1,000 km2 hole held open by a sliver. So an occupied ring is closed *around*
    its occupant: what gets added back is the hole minus whatever stands in it, and what stays open
    is exactly that occupant's ground. Nothing this returns can overlap anything in ``occupied``
    that it did not already overlap, and the ring being 99.9% empty no longer decides anything.

    Measured over those 30 outlines, 91 s for all of them: of 4,554 rings, 3,279 close outright and
    1,275 close around an occupant, leaving 1,060 rings that are somebody else's ground and adding
    128,809 km2 - against 42,598 km2 for the per-ring rule, which is the 86,210 above.

    ``occupied`` is an ``STRtree`` over every outline including this one, and ``skip`` is this
    one's index in it. The geometry's own parts are checked separately against each other, and are
    the one occupant that is not left standing in its hole: ground this outline already owns is
    ground it can close over, so the ring and the island in it are merged into one solid part
    rather than emitted as two that share a line - which is not a valid MultiPolygon.

    Returns the geometry, how many rings were closed outright, and how many were closed around an
    occupant. The geometry itself comes back unrebuilt when both counts are zero.
    """
    if geometry is None or shapely.is_empty(geometry):
        return geometry, 0, 0
    parts = shapely.get_parts(geometry)
    mine = shapely.STRtree(parts) if len(parts) > 1 else None
    shells, pieces, neighbours, closed, trimmed = [], {}, {}, 0, 0
    for index, part in enumerate(parts):
        rings = list(part.interiors)
        if not rings:
            shells.append(part)
            continue
        kept = []
        for ring in rings:
            hole = shapely.Polygon(ring)
            theirs = _occupant(hole, occupied, skip)
            ours = _occupant(hole, mine, index)
            if theirs is None and ours is None:
                closed += 1
                continue
            # the ring stays, and the ground inside it that nobody else claims comes back as a
            # piece of its own, for the union below to put back into the part it came out of
            kept.append(ring)
            standing = [g for g in (theirs, ours) if g is not None]
            free = shapely.difference(hole, standing[0] if len(standing) == 1
                                      else shapely.union_all(standing))
            if not shapely.is_empty(free) and shapely.area(free) > 0:
                pieces.setdefault(index, []).append(free)
            if ours is not None:
                # a part of this same geometry standing in this one's hole. The two have to be
                # merged rather than emitted side by side: the piece added back runs up to that
                # part's edge, and a MultiPolygon whose members share a line rather than a point
                # is not a valid one
                neighbours.setdefault(index, []).extend(
                    int(j) for j in mine.query(hole, predicate='intersects') if j != index)
            trimmed += 1
        shells.append(shapely.Polygon(part.exterior, kept))
    if not closed and not trimmed:
        return geometry, 0, 0

    # Only what interlocks gets unioned. A piece lies inside a ring of the part it came from, so
    # everything else is already final - which is what keeps this off the whole continent: unioning
    # each outline as a whole instead had not finished the 30 of them after 8 minutes, against 91 s
    # for all of them this way.
    owner = list(range(len(shells)))

    def root(node: int) -> int:
        while owner[node] != node:
            owner[node] = owner[owner[node]]
            node = owner[node]
        return node

    for index, standing in neighbours.items():
        for other in standing:
            owner[root(index)] = root(other)
    components = {}
    for index in range(len(shells)):
        components.setdefault(root(index), []).append(index)

    rebuilt = []
    for members in components.values():
        merged = [shells[i] for i in members] + [p for i in members for p in pieces.get(i, ())]
        if len(merged) == 1:
            rebuilt.append(merged[0])
        else:
            rebuilt.extend(shapely.get_parts(_union_parts(merged)))
    filled = rebuilt[0] if len(rebuilt) == 1 else shapely.multipolygons(rebuilt)
    return filled, closed, trimmed


def _union_parts(merged: list):
    """``union_all`` over a rebuilt component, escalating rather than raising.

    A shell and the piece put back inside it were cut from the same ring, so they share linework
    that is coincident and not necessarily noded, and GEOS raises a side-location conflict on it.
    Step 6 never saw this because a group outline is a small, already-snapped thing; a region's
    level-2 footprint is not - measured on 2020024230, whose footprint carries over a thousand
    rings, the plain call raises at 4115114.667 8880128.

    The escalation is the same one ``dissolve_by`` uses and in the same order: repair the members,
    then round onto the 1 m grid, which merges the mismatched vertex pair that the noding is
    missing. Only the last step moves anything, and only by less than the metre these coordinates
    are already snapped to.
    """
    attempts = (lambda: shapely.union_all(merged),
                lambda: shapely.union_all(_make_valid(np.asarray(merged, dtype=object))),
                lambda: shapely.union_all(_make_valid(np.asarray(merged, dtype=object)),
                                          grid_size=1.0))
    failed = None
    for attempt in attempts:
        try:
            return attempt()
        except shapely.errors.GEOSException as error:
            failed = error
    raise failed


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
