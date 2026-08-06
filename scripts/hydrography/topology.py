import geopandas as gpd
import networkx
import numpy as np
import pandas as pd

from . import schema

__all__ = [
    'compute_topology',
    'recompute_outlets',
    'find_topology_violations',
    'topology_is_valid',
    'assert_topology_is_valid',
    'get_all_upstream',
    'make_upstream_id_map',
    'make_downstream_id_map',
    'hilbert_index',
    'topological_order_hilbert_tiebreak',
]


def hilbert_index(lon: np.ndarray, lat: np.ndarray, bits: int = 16) -> np.ndarray:
    """
    Position of each lon/lat along a Hilbert curve over a 2**bits grid of the globe.

    Vectorized form of the standard xy2d. Note that the rotation reflects about the full grid
    (``n - 1 - x``), not the current level's ``s``; reflecting about ``s`` still yields a bijection,
    so the mistake produces a plausible-looking index that is not the Hilbert curve and lacks its
    adjacency property.
    """
    n = 1 << bits
    x = np.clip(((lon + 180.0) / 360.0 * n).astype(np.int64), 0, n - 1)
    y = np.clip(((lat + 90.0) / 180.0 * n).astype(np.int64), 0, n - 1)
    d = np.zeros(x.shape, dtype=np.int64)
    s = n >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        swap = ry == 0
        flip = swap & (rx == 1)
        x[flip] = n - 1 - x[flip]
        y[flip] = n - 1 - y[flip]
        tmp = x[swap].copy()
        x[swap] = y[swap]
        y[swap] = tmp
        s >>= 1
    return d


def topological_order_hilbert_tiebreak(gdf: gpd.GeoDataFrame, bits: int = 16) -> gpd.GeoDataFrame:
    """
    Sort reaches upstream-to-downstream, breaking ties spatially, and renumber topologySortedOrder.

    **The ordering is topological first.** Every reach lands after every reach that drains into it,
    which is asserted before this returns. The Hilbert curve is only a tie-breaker, consulted at
    the two points where topology constrains nothing: which order to emit whole watersheds (they
    are disjoint components, so no edge crosses between them) and which sibling branch to descend
    first at a junction. Neither can place a reach ahead of its own headwaters. Sorting rows by a
    Hilbert index directly would be a different thing entirely and would violate about half the
    edges; that is not what this does.

    A topological sort is not unique, and which valid linearization is chosen decides how the whole
    published dataset behaves: a linearization that scatters a watershed across the file makes every
    subset a scatter-read rather than a range, and shows the compressor coordinates that jump
    between continents row to row. This one is built to do neither. It orders by:

    1. terminal watersheds along a Hilbert curve through their outlet points, so neighbouring
       basins are neighbours in the file, and
    2. within each watershed, depth-first **post-order** from the outlet, visiting the upstream
       reaches at each junction in Hilbert order.

    Both properties fall out of it:

    - **Still a valid topological sort.** Post-order emits a node after all of its upstream
       children, which is exactly upstream-before-downstream, and watersheds are disjoint
       components so ordering whole watersheds among themselves cannot create a violation.
    - **Every upstream subset is one contiguous range.** The network is a forest, so the reaches
       upstream of X are exactly the subtree rooted at X, and in DFS post-order a subtree occupies a
       contiguous interval. Because post-order puts the root last, that interval is precisely
       ``[topologySortedOrder - upstreamCount + 1, topologySortedOrder]``.

    Caller must have lon/lat populated; the geometry is not consulted.
    """
    n = len(gdf)
    if n == 0:
        return gdf

    river = gdf[schema.river_id].to_numpy()
    nxt = gdf[schema.next_river_id].to_numpy()
    hilbert = hilbert_index(gdf[schema.lon_field].to_numpy(), gdf[schema.lat_field].to_numpy(), bits)

    row_of = pd.Series(np.arange(n, dtype=np.int64), index=river)
    parent = row_of.reindex(nxt).to_numpy()  # NaN where nextRiverId is -1 (an outlet)
    is_root = np.isnan(parent)
    if (~is_root & np.isnan(parent)).any():  # pragma: no cover - defensive
        raise ValueError('a reach flows into an id that is not in the network')
    parent_row = np.where(is_root, -1, np.nan_to_num(parent, nan=-1)).astype(np.int64)

    # CSR-style child lists, each already sorted by Hilbert position so a junction is descended
    # in a spatially coherent order rather than in whatever order the rows happened to be in.
    kids = np.flatnonzero(parent_row >= 0)
    kids = kids[np.lexsort((hilbert[kids], parent_row[kids]))]
    counts = np.bincount(parent_row[parent_row >= 0], minlength=n)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

    roots = np.flatnonzero(parent_row < 0)
    roots = roots[np.argsort(hilbert[roots], kind='stable')]

    # Iterative DFS. A node is pushed as itself to descend into and as its bitwise complement
    # (always negative) to emit, which keeps the stack a flat list of ints.
    out = np.empty(n, dtype=np.int64)
    k = 0
    stack = []
    for root in roots:
        stack.append(int(root))
        while stack:
            v = stack.pop()
            if v < 0:
                out[k] = ~v
                k += 1
                continue
            stack.append(~v)
            for j in range(offsets[v + 1] - 1, offsets[v] - 1, -1):
                stack.append(int(kids[j]))
    if k != n:
        raise ValueError(
            f'Hilbert ordering reached {k:,} of {n:,} reaches. Reaches unreachable from any '
            f'terminal outlet imply a cycle or a dangling nextRiverId.'
        )

    gdf = gdf.iloc[out].reset_index(drop=True)
    gdf[schema.topo_sort] = np.arange(n, dtype=np.int32)

    # The guarantee this function exists to provide, checked rather than assumed: every reach sits
    # after everything draining into it. O(V) and a few tens of ms even on the largest region, so
    # there is no reason for a caller to have to remember to do it.
    row = pd.Series(np.arange(n), index=gdf[schema.river_id].to_numpy())
    flows = gdf[schema.next_river_id].to_numpy() != -1
    upstream_row = np.arange(n)[flows]
    downstream_row = row.reindex(gdf.loc[flows, schema.next_river_id].to_numpy()).to_numpy()
    violations = int((upstream_row >= downstream_row).sum())
    if violations:
        raise ValueError(
            f'{violations:,} reach(es) were ordered at or after the reach they drain into. '
            f'The result is not a topological order.'
        )
    return gdf


def make_upstream_id_map(gdf: gpd.GeoDataFrame) -> dict:
    """Return a dict mapping each river id to a list of its direct upstream river ids."""
    return (
        gdf[[schema.river_id, schema.next_river_id]]
        .groupby(schema.next_river_id)[schema.river_id]
        .apply(list)
        .to_dict()
    )


def make_downstream_id_map(gdf: gpd.GeoDataFrame) -> dict:
    """Return a dict mapping each river id to its direct downstream river id (or -1 if outlet)."""
    return gdf.set_index(schema.river_id)[schema.next_river_id].to_dict()


def get_all_upstream(root_id: int, id_map: dict) -> list:
    """Return all transitive upstream river ids of root_id (root itself excluded)."""
    all_upstream = []
    stack = list(id_map.get(root_id, []))
    while stack:
        current = stack.pop()
        all_upstream.append(current)
        stack.extend(id_map.get(current, []))
    return all_upstream


def recompute_outlets(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    (Re)assign the OutletRiverID attribute for every reach from the current
    riverId/nextRiverId topology, without re-sorting or renumbering
    topologySortedOrder. Use this after edits that create new outlets (e.g. a
    zero-length outlet being removed repoints its upstreams to -1) so the stale
    outlet ids left behind are corrected. Only outletRiverId is written; groupId and
    every other column are untouched.
    """
    G = networkx.DiGraph()
    G.add_edges_from(gdf[[schema.river_id, schema.next_river_id]].values)

    gdf[schema.last_river_id] = -1
    outlets = gdf[gdf[schema.next_river_id] == -1][schema.river_id].tolist()
    for outlet in outlets:
        ancestors = networkx.ancestors(G, outlet)
        ancestors.add(outlet)  # include the outlet itself
        gdf.loc[gdf[schema.river_id].isin(ancestors), schema.last_river_id] = outlet
    # if any -1 are left, then an unanticipated error has occurred so raise an error
    if (gdf[schema.last_river_id] == -1).any():
        raise ValueError('Some rivers still have outletRiverId of -1 after processing. Please investigate.')

    return gdf


def compute_topology(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # assign OutletRiverID attribute to all rivers in the watershed
    gdf = recompute_outlets(gdf)

    # compute topological sort - use strahler order and drainage area to avoid expensive graph algorithms
    gdf.sort_values(
        [schema.strahler_order, schema.tdx_ds_area_field],
        ascending=True, kind='stable', inplace=True,
    )
    gdf.reset_index(drop=True, inplace=True)
    gdf[schema.topo_sort] = np.arange(len(gdf), dtype=np.int32)

    assert_topology_is_valid(gdf)
    return gdf


def find_topology_violations(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Return the reaches that violate topologically-sorted validity. A row is valid
    when, unless it is an outlet (next id == -1), its downstream reach both exists
    in the network and appears strictly later in topological order. The returned
    frame is empty iff the network is a valid topological sort. This is the
    vectorized equivalent of the original per-row iterrows check (O(V) instead of
    O(V**2)) and additionally reports which reaches failed and why.
    """
    duplicate_ids = gdf[schema.river_id][gdf[schema.river_id].duplicated()].tolist()
    if duplicate_ids:
        raise ValueError(
            f'Cannot validate topology: river ids are not unique. '
            f'{len(duplicate_ids)} duplicated id(s), e.g. {duplicate_ids[:10]}'
        )

    topo_by_id = gdf.set_index(schema.river_id)[schema.topo_sort]
    has_downstream = gdf[schema.next_river_id] != -1
    checked = gdf.loc[
        has_downstream,
        [schema.river_id, schema.next_river_id, schema.topo_sort],
    ].copy()
    downstream_topo = checked[schema.next_river_id].map(topo_by_id)

    downstream_missing = downstream_topo.isna()
    # NaN comparisons are False, so missing rows are caught only by the mask above
    downstream_not_later = downstream_topo <= checked[schema.topo_sort]

    checked['downstreamTopoSort'] = downstream_topo
    checked['violation'] = ''
    checked.loc[downstream_missing, 'violation'] = 'downstream reach not in network'
    checked.loc[downstream_not_later, 'violation'] = 'downstream reach not later in topological order'
    return checked[downstream_missing | downstream_not_later]


def topology_is_valid(gdf: gpd.GeoDataFrame) -> bool:
    return find_topology_violations(gdf).empty


def assert_topology_is_valid(gdf: gpd.GeoDataFrame) -> None:
    violations = find_topology_violations(gdf)
    if not violations.empty:
        raise ValueError(
            f'The computed topology is not valid: {len(violations)} reach(es) '
            f'violate topological ordering. First offenders:\n'
            f'{violations.head(10).to_string(index=False)}'
        )
