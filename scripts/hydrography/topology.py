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
    'find_nested_set_violations',
    'assert_nested_set_is_valid',
    'get_all_upstream',
    'make_upstream_id_map',
    'make_downstream_id_map',
    'hilbert_index',
    'nested_set_order',
]

TERMINAL = -1


def hilbert_index(lon: np.ndarray, lat: np.ndarray, bits: int = 16) -> np.ndarray:
    """
    Position of each lon/lat along a Hilbert curve over a 2**bits grid of the globe.

    Vectorized form of the standard xy2d, which reflects about the full grid (``n - 1 - x``) rather
    than the current level's ``s``. That is the correct formulation and not, as an earlier version of
    this docstring claimed, a bug: verified at bits 3, 4 and 5 that the result is a bijection onto
    ``0 .. 4**bits - 1`` and that every pair of consecutive indices is grid-adjacent, which is the
    adjacency property the curve is used for.
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


def parent_rows(river: np.ndarray, next_river: np.ndarray) -> np.ndarray:
    """
    Row position of each reach's downstream reach, or -1 where it leaves the network.

    This is the only place the id-space network is turned into a row-space one, so it is also where
    the two ways that translation can fail are caught: a duplicated riverId (which would make the
    lookup ambiguous) and a nextRiverId that is not TERMINAL and not present in the frame (a
    dangling edge, which would silently turn its reach into a spurious extra outlet).
    """
    n = len(river)
    row_of = pd.Series(np.arange(n, dtype=np.int64), index=river)
    if not row_of.index.is_unique:
        duplicated = pd.Index(river)[pd.Index(river).duplicated()].unique()
        raise ValueError(f'{len(duplicated):,} duplicated riverId(s), e.g. {duplicated[:10].tolist()}')
    parent = row_of.reindex(next_river).to_numpy()  # NaN at TERMINAL and at any dangling id
    missing = np.isnan(parent)
    dangling = missing & (next_river != TERMINAL)
    if dangling.any():
        raise ValueError(
            f'{int(dangling.sum()):,} reach(es) flow into an id that is not in the network, e.g. '
            f'{river[dangling][:10].tolist()} -> {next_river[dangling][:10].tolist()}'
        )
    return np.where(missing, -1, np.nan_to_num(parent, nan=-1)).astype(np.int64)


def _child_lists(parent_row: np.ndarray, keys: list) -> tuple:
    """
    CSR-style child lists: ``kids[offsets[v]:offsets[v + 1]]`` are the rows draining into row ``v``,
    ordered by ``keys`` (most significant first) within each parent.
    """
    n = len(parent_row)
    kids = np.flatnonzero(parent_row >= 0)
    # lexsort takes its primary key last, so the parent has to come last and the caller's keys are
    # reversed into the slots ahead of it
    kids = kids[np.lexsort(tuple(k[kids] for k in reversed(keys)) + (parent_row[kids],))]
    counts = np.bincount(parent_row[parent_row >= 0], minlength=n)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    return kids, offsets


def _depth_first_postorder(kids: np.ndarray, offsets: np.ndarray, roots: np.ndarray, n: int) -> np.ndarray:
    """
    Rows in depth-first post-order from each root in turn, descending children in ``kids`` order.

    A node is pushed as itself to descend into and as its bitwise complement (always negative) to
    emit, which keeps the stack a flat list of ints. Post-order is what makes a subtree contiguous:
    every descendant is emitted before the node itself, so the node lands at the end of its own
    block.
    """
    # python lists rather than numpy scalars: this is 5.5M iterations of scalar work on the global
    # network and indexing a list is several times cheaper than indexing an ndarray
    kid_list = kids.tolist()
    offset_list = offsets.tolist()
    out = [0] * n
    k = 0
    for root in roots.tolist():
        stack = [root]
        while stack:
            v = stack.pop()
            if v < 0:
                out[k] = ~v
                k += 1
                continue
            stack.append(~v)
            for j in range(offset_list[v + 1] - 1, offset_list[v] - 1, -1):
                stack.append(kid_list[j])
    if k != n:
        raise ValueError(
            f'Depth-first traversal reached {k:,} of {n:,} reaches. Reaches unreachable from any '
            f'terminal outlet imply a cycle or a dangling nextRiverId.'
        )
    return np.array(out, dtype=np.int64)


def _accumulate_upstream(parent_row: np.ndarray, post_order: np.ndarray) -> tuple:
    """
    Subtree size and Shreve magnitude for every row, from one pass in post-order.

    Post-order visits every child before its parent, so a single forward sweep accumulating into the
    parent is enough - no repeated traversal, no recursion depth limit. Subtree size counts the reach
    itself, so ``upstreamCount = subtree_size - 1``. Shreve magnitude counts upstream headwaters,
    which is a different quantity entirely and is why it cannot stand in for the upstream count.
    """
    n = len(parent_row)
    pr = parent_row.tolist()
    child_counts = np.bincount(parent_row[parent_row >= 0], minlength=n)
    subtree = [1] * n
    shreve = (child_counts == 0).astype(np.int64).tolist()  # a headwater is its own magnitude 1
    for i in post_order.tolist():
        p = pr[i]
        if p >= 0:
            subtree[p] += subtree[i]
            shreve[p] += shreve[i]
    return np.array(subtree, dtype=np.int64), np.array(shreve, dtype=np.int64)


def nested_set_order(gdf: gpd.GeoDataFrame, bits: int = 16) -> gpd.GeoDataFrame:
    """
    Reorder rows into the published nested-set ordering and stamp topologySortedOrder,
    upstreamCount, and a recomputed shreveOrder. See docs/river-index.md for the full rationale.

    **The ordering is topological first.** Every reach lands after every reach that drains into it,
    which is asserted before this returns. Everything else is a tie-break, consulted only where
    topology constrains nothing. Sorting rows by a Hilbert index directly would be a different thing
    entirely and would violate about half the edges; that is not what this does.

    A topological sort is not unique, and which valid linearization is chosen decides how the whole
    published dataset behaves. Three nested levels of blocking are imposed, each on a different axis:

    1. **groups, by groupId.** No reach drains across a group boundary, so whole groups can be
       ordered freely. Doing it this way gives every group one contiguous riverIndex range, which is
       what lets a routing engine treat a group file as a dense array whose local index is
       ``riverIndex - riverIndexStart``. Ordering by anything else fragments them: measured on the
       published network before this change, the groups (125 of them at the time, 127 now) occupied
       625 separate runs.
    2. **terminal watersheds within a group, by the Hilbert index of the outlet.** Watersheds are
       disjoint components, so this too is unconstrained, and putting neighbouring basins next to
       each other in the file is what keeps the geometry compressible and the tiles coherent.
    3. **reaches within a watershed, depth-first post-order, descending the largest subtree first.**

    Level 3 is the one that decides routing performance, and it is not a heuristic. In post-order a
    parent is emitted immediately after its last child, so a child sits ``1 + (total size of every
    sibling subtree emitted after it)`` slots ahead of the parent it feeds. Summing that over one
    junction gives ``k + sum_j size_j * (j - 1)``, which is minimised by giving the largest subtree
    the earliest slot - independently at every junction, so descending subtree size is the exact
    minimiser of total gather distance over all post-order linearizations. Measured over all 3.89M
    edges of the global network it takes the mean distance from a reach to the reach it drains into
    from 135.3 to 3.5, the 99th percentile from 1,451 to 28, and the number of edges reaching more
    than 4,096 slots away from 25,814 to 201. Descending the *smallest* tributary first - which is
    what "follow the low-order, low-drainage headwater" amounts to - measures at 169.1, worse than
    doing nothing, because it buys a perfectly contiguous trunk by pushing every one of the ~1.53M
    tributary mouths an entire upstream basin away from the junction it feeds.

    Two properties fall out of post-order and are both asserted here:

    - **Still a valid topological sort.** Post-order emits a node after all of its upstream children,
      which is exactly upstream-before-downstream, and watersheds are disjoint components so
      ordering whole watersheds among themselves cannot create a violation.
    - **Every upstream subset is one contiguous range.** The network is a forest, so the reaches
      upstream of X are exactly the subtree rooted at X, and in DFS post-order a subtree occupies a
      contiguous interval. Because post-order puts the root last, that interval is precisely
      ``[position - upstreamCount, position]``.

    Caller must have lon/lat and groupId populated; the geometry is not consulted.
    """
    n = len(gdf)
    if n == 0:
        return gdf

    parent_row = parent_rows(gdf[schema.river_id].to_numpy(), gdf[schema.next_river_id].to_numpy())

    # A first traversal in arbitrary child order, purely to get subtree sizes - they are what the
    # real traversal sorts siblings by, and they do not depend on the order children are visited in.
    kids, offsets = _child_lists(parent_row, [np.arange(n, dtype=np.int64)])
    roots = np.flatnonzero(parent_row < 0)
    subtree, shreve = _accumulate_upstream(parent_row, _depth_first_postorder(kids, offsets, roots, n))

    river = gdf[schema.river_id].to_numpy().astype(np.int64)
    hilbert = hilbert_index(gdf[schema.lon_field].to_numpy(), gdf[schema.lat_field].to_numpy(), bits)
    # riverId last in both lists so the ordering is fully determined and a rebuild is reproducible
    kids, offsets = _child_lists(parent_row, [-subtree, -gdf[schema.tdx_ds_area_field].to_numpy(), river])
    roots = roots[np.lexsort((river[roots], hilbert[roots], gdf[schema.group_id].to_numpy()[roots]))]
    out = _depth_first_postorder(kids, offsets, roots, n)

    gdf = gdf.iloc[out].reset_index(drop=True)
    gdf[schema.topo_sort] = np.arange(n, dtype=np.int32)
    gdf[schema.upstream_count] = (subtree[out] - 1).astype(np.int32)
    gdf[schema.shreve_order] = shreve[out].astype(np.int32)

    # The guarantees this function exists to provide, checked rather than assumed. Both are O(V)
    # numpy and a few tens of ms even on the global network, so there is no reason for a caller to
    # have to remember to do it.
    assert_nested_set_is_valid(gdf)
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
    #
    # riverId is the third key, and it is not decoration. The first two tie for 41.7% of raw reaches
    # (measured on 7020000010 and 1020000010), and a stable sort resolves a tie by source row order -
    # so without it, topologySortedOrder is a fact about how the input file happened to be written.
    # That rank is what dissolve_groups and find_short_streams pick a merge keeper from
    # (streams.py:242, streams.py:521), so reordering or repartitioning the source silently changes
    # which reach survives a merge. riverId is unique, so the sort is now total and the whole step is
    # reproducible from the ids alone.
    gdf.sort_values(
        [schema.strahler_order, schema.tdx_ds_area_field, schema.river_id],
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


def find_nested_set_violations(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Return the reaches that break the nested-set property, i.e. the rows for which the reaches
    upstream are *not* exactly the block of rows ``[position - upstreamCount, position]``.

    Every consumer of upstreamCount depends on that block being right, and the failure mode if it is
    not is silent: an upstream query returns a plausible set of reaches that is simply the wrong one.
    So it is checked rather than argued for, in four vectorized O(V) tests which together prove it by
    induction on the forest:

    1. every reach is ordered before the reach it drains into (a valid topological sort, so a
       subtree's maximum row is its own root);
    2. a reach's upstreamCount is exactly the sum of its children's subtree sizes (the block is the
       right *length*);
    3. every child's block starts at or after its parent's block start (the block is in the right
       *place*);
    4. the roots' blocks tile ``[0, n)`` exactly, with no gap and no overlap.

    Given (1), a subtree ends at its own root, so (2) and (3) force each child's block to be a
    sub-interval of its parent's, and the lengths leave no room for a gap - which is contiguity.
    """
    columns = ['riverId', 'rowPosition', 'violation']
    n = len(gdf)
    if n == 0:
        return pd.DataFrame(columns=columns)

    river = gdf[schema.river_id].to_numpy()
    parent_row = parent_rows(river, gdf[schema.next_river_id].to_numpy())
    upstream = gdf[schema.upstream_count].to_numpy().astype(np.int64)
    rows = np.arange(n, dtype=np.int64)
    has_downstream = parent_row >= 0
    child = rows[has_downstream]
    parent = parent_row[has_downstream]

    found = []

    def record(bad_rows, reason):
        if len(bad_rows):
            found.append(pd.DataFrame({
                columns[0]: river[bad_rows], columns[1]: bad_rows, columns[2]: reason,
            }))

    record(child[child >= parent], 'ordered at or after the reach it drains into')
    # bincount's weights are float64, exact well past any reach count this network will ever have
    children_total = np.bincount(parent, weights=(upstream[child] + 1).astype(float), minlength=n)
    record(np.flatnonzero(children_total.astype(np.int64) != upstream),
           'upstreamCount does not equal the total size of the subtrees draining into it')
    record(child[(child - upstream[child]) < (parent - upstream[parent])],
           'upstream block starts before the block of the reach it drains into')

    roots = np.flatnonzero(~has_downstream)  # already ascending
    if len(roots):
        expected_starts = np.concatenate(([0], roots[:-1] + 1))
        record(roots[(roots - upstream[roots]) != expected_starts],
               'terminal watershed does not begin where the previous one ended')
        if roots[-1] != n - 1:
            record(roots[-1:], 'last terminal watershed does not end at the last row')

    return pd.concat(found, ignore_index=True) if found else pd.DataFrame(columns=columns)


def assert_nested_set_is_valid(gdf: gpd.GeoDataFrame) -> None:
    violations = find_nested_set_violations(gdf)
    if not violations.empty:
        raise ValueError(
            f'The ordering is not a valid nested set: {len(violations):,} reach(es) whose upstream '
            f'reaches are not the contiguous block [position - upstreamCount, position]. '
            f'First offenders:\n{violations.head(10).to_string(index=False)}'
        )
