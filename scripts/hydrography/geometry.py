"""
Geometry reductions that are too slow done the obvious way.

Right now that is one thing: unioning a very large set of adjacent polygons into their outline,
which is what turns a group's catchments into the group's boundary.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import shapely

__all__ = [
    'hierarchical_union',
    'union_chunk_size',
]

# How many polygons one parallel union task takes. Measured on a 6,771-catchment group: 64/256/1024
# ran in 13.5/10.9/11.8 s, so the curve is flat and the only thing that matters is that it is
# neither 1 nor everything.
union_chunk_size = 256


def hierarchical_union(geometries, workers: int = None, chunk: int = union_chunk_size):
    """
    Union many polygons by unioning small chunks in parallel, then unioning the results, repeatedly
    until one geometry is left.

    Two reasons this beats a single ``shapely.union_all`` over the whole set. Each chunk cancels the
    interior edges of its own neighbourhood, so every round hands the next one far fewer edges than
    it received - on one group, 12.3M input vertices collapse to 931k. And ``union_all`` releases the
    GIL inside GEOS, so the chunks genuinely run in parallel rather than taking turns.

    Measured on group 718 (6,771 catchments, 12.3M vertices): 36.5 s for the single call against
    10.9 s for chunks of 256 across 16 threads, and the results are identical - same vertex count,
    same area to the last bit. It is a scheduling change, not an approximation.

    ``shapely.coverage_union_all`` is faster still, and whether it is usable depends entirely on
    which catchments these are. On the *raw* ones it is not: measured on the same group, 6,769 of
    the 6,771 have invalid coverage edges, and GEOS raises a side-location conflict at every
    precision tried, including the 1 m grid the geometry is already snapped to. Step 4's
    ``coverage_simplify`` rebuilds every shared edge to match on both sides, and after it the fast
    path works for most groups - so a caller holding simplified catchments should try it first and
    keep this as the fallback for the few groups GEOS still refuses.
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
