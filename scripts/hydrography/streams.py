import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from . import schema
from .topology import make_upstream_id_map

log = logging.getLogger(__name__)

# Explicit per-attribute reduction rules for dissolving a group of reaches into
# their downstream-most keeper. dissolve_groups topo-sorts ascending first, so
# 'last' == the downstream-most member and 'first' == the most-upstream member.
# Any column not listed here falls back to 'last' (the keeper's value), and the
# length-like columns are overridden per step (summed along a chain, maxed when a
# headwater tree collapses into one reach) - see _build_aggfunc.
_agg_rules = {
    schema.river_id: 'last',
    schema.next_river_id: 'last',
    schema.last_river_id: 'last',  # constant within a connected group
    schema.group_id: 'last',  # constant within a connected group
    schema.topo_sort: 'last',
    schema.strahler_order: 'max',
    schema.tdx_magnitude_field: 'max',  # Shreve magnitude; the downstream-most reach already counts its upstreams
    schema.tdx_us_area_field: 'min',  # upstream-contributing area at the merged reach's upstream end
    schema.tdx_ds_area_field: 'max',  # DSContArea: contributing area at the merged reach's downstream end
    schema.area: 'sum',  # local catchment areas are additive and must be conserved
    schema.tdx_region_field: 'first',  # constant within a region
    schema.tdx_us_link_1_field: 'first',
    schema.tdx_us_link_2_field: 'first',
    schema.tdx_ds_node_id_field: 'last',
    schema.tdx_ws_no_field: 'last',
    schema.tdx_dout_end_field: 'last',
    schema.tdx_dout_start_field: 'first',
    schema.tdx_dout_mid_field: 'last',
    schema.tdx_straight_length_field: 'last',
    schema.tdx_slope_field: 'last',
    schema.lon_field: 'last',
    schema.lat_field: 'last',
    schema.z_field: 'last',
}

# length-like attributes accumulate along a linear chain but must not be summed
# across the parallel tributaries of a collapsed headwater tree
_length_like_fields = (schema.length, schema.tdx_length_field, schema.tdx_strm_drop_field)

# transient grouping key used during a dissolve; never written to the output
_group_field = 'group'


def _build_aggfunc(gdf: gpd.GeoDataFrame, length_rule: str) -> dict:
    """
    Build a dissolve aggfunc covering every column actually present so no
    attribute is dropped (and NaN-corrupted) on a merged row. Known columns use
    _agg_rules; length-like columns use the step-specific length_rule; anything
    else defaults to 'last' (the downstream-most keeper's value).
    """
    agg = {}
    for col in gdf.columns:
        if col in (schema.geometry, _group_field):
            continue
        if col in _length_like_fields:
            agg[col] = length_rule
        else:
            agg[col] = _agg_rules.get(col, 'last')
    return agg


tdx_geoparquet_dir = '../data/TDXHydroGeoParquet'
mods_dir = '../data/modifications'

__all__ = [
    'find_zero_length',
    'remove_zero_length',
]


def find_zero_length(gdf: gpd.GeoDataFrame) -> dict:
    """
    Fix streams that have 0 length.
    General error cases and their unwanted behavior to be resolved:
    1) Feature is coastal w/ no upstream or downstream
        -> Delete the stream and its basin
    2) Feature is bridging a 3-river confluence (Has downstream and upstreams)
        -> Remove the stream, modify upstreams DSLINKNO to point to downstream, delete the temporary basin
    3) Feature is generally costal w/ upstreams but no downstream
        -> Delete the basin, modify upstreams DSLINKNO to -1, revise OutletLinkNo
    4) Feature doesn't match any previous case
        -> Raise an error for now

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Stream network
    """
    case1_ids = []
    case2_ids = []
    case3_ids = []
    case4_ids = []

    c1_type = 'No upstream or downstream segments'
    c1_action = 'Delete the stream and its basin'
    c1_desc = 'Normally on coasts'

    c2_type = 'Upstream and downstream segments exist'
    c2_action = 'Remove the stream, modify upstreams DSLINKNO to point to downstream, delete the basin'
    c2_desc = 'Normally found bridging a 3-river confluence where delineation allowed max 2 connectors'

    c3_type = 'Upstream segments exist but no downstream segments'
    c3_action = 'Delete the basin, modify upstreams DSLINKNO to -1, revise OutletLinkNo'
    c3_desc = 'Normally on coasts where the stream should drain into the sea delineation did not allow it'

    c4_type = 'Does not fit into any of the previous cases'
    c4_action = 'Needs manual review'
    c4_desc = 'Needs manual review'

    for rivid in gdf[gdf[schema.length] <= 0.1][schema.river_id].values:
        feat = gdf[gdf[schema.river_id] == rivid]

        upstreams = gdf[gdf[schema.next_river_id] == rivid][schema.river_id].values
        ds_id = feat[schema.next_river_id].values[0]

        # Case 1
        if ds_id == -1 and all([x == -1 for x in upstreams]):
            case1_ids.append(rivid)

        # Case 2
        elif ds_id != -1 and all([x != -1 for x in upstreams]):
            case2_ids.append(rivid)

        # Case 3
        elif ds_id == -1 and all([x != -1 for x in upstreams]):
            case3_ids.append(rivid)

        # Case 4
        else:
            logging.warning(f"The stream segment {feat[schema.river_id]} has conditions we've not yet considered")
            case4_ids.append(rivid)

    # write a summary json
    case1_ids = np.array(case1_ids).astype(int).tolist()
    case2_ids = np.array(case2_ids).astype(int).tolist()
    case3_ids = np.array(case3_ids).astype(int).tolist()
    case4_ids = np.array(case4_ids).astype(int).tolist()

    return {
        'case1': {
            'type': c1_type,
            'action': c1_action,
            'description': c1_desc,
            'ids': case1_ids,
            'count': len(case1_ids)
        },
        'case2': {
            'type': c2_type,
            'action': c2_action,
            'description': c2_desc,
            'ids': case2_ids,
            'count': len(case2_ids)
        },
        'case3': {
            'type': c3_type,
            'action': c3_action,
            'description': c3_desc,
            'ids': case3_ids,
            'count': len(case3_ids)
        },
        'case4': {
            'type': c4_type,
            'action': c4_action,
            'description': c4_desc,
            'ids': case4_ids,
            'count': len(case4_ids)
        }
    }


def remove_zero_length(gdf: gpd.GeoDataFrame, zero_length_json: dict, ) -> gpd.GeoDataFrame:
    """Apply fixes to streams that have 0 length"""
    case1 = zero_length_json['case1']['ids']
    case2 = zero_length_json['case2']['ids']
    case3 = zero_length_json['case3']['ids']

    # Case 1 - Coastal w/ no upstream or downstream - Delete the stream and its basin
    gdf = gdf[~gdf[schema.river_id].isin(case1)]

    # Case 2 - Allow 3-river confluence - Delete river and basin, modify upstreams to point downstream
    # Apply before case 3 to handle some edges cases where zero length basins drain into other zero length basins
    # Sort by next_river_id to handle some edges cases where zero length basins drain into other zero length basins
    sorted_c2_order = gdf[gdf[schema.river_id].isin(case2)].sort_values(schema.next_river_id, ascending=True)[
        schema.river_id].values
    for river_id in sorted_c2_order:
        downstream_id = gdf[gdf[schema.river_id] == river_id][schema.next_river_id].values[0]
        gdf.loc[gdf[schema.next_river_id] == river_id, schema.next_river_id] = downstream_id
    gdf = gdf[~gdf[schema.river_id].isin(case2)]

    # Case 3 - Coastal w/ upstreams but no downstream - delete stream and basin, modify upstreams to have no downstream
    for river_id in case3:
        gdf.loc[gdf[schema.next_river_id] == river_id, schema.next_river_id] = -1
    gdf = gdf[~gdf[schema.river_id].isin(case3)]

    return gdf


def dissolve_groups(gdf: gpd.GeoDataFrame, groups: dict, aggfunc: dict) -> gpd.GeoDataFrame:
    # the aggfuncs use 'last' to keep the downstream-most reach's values, which only
    # holds if rows are in topological order. A prior dissolve appends its merged rows
    # at the end, so re-sort here to restore the invariant before aggregating.
    gdf = gdf.sort_values(schema.topo_sort)

    # label the rows that need to be dissolved together based on the key value dictionary
    group_label = {}
    member_to_keeper = {}
    for index, (keeper, merge_list) in enumerate(groups.items()):
        group_label[keeper] = index
        for merge in merge_list:
            group_label[merge] = index
            # every non-keeper member is collapsed into the keeper id
            member_to_keeper[merge] = keeper
    gdf[_group_field] = gdf[schema.river_id].map(group_label).fillna(-1).astype(int)

    # dissolve the rows together and update the geometry to be the merged geometry of all the rows
    dissolved = (
        gdf
        [gdf[_group_field] != -1]
        .dissolve(
            by=_group_field,
            as_index=False,
            aggfunc=aggfunc
        )
        # as_index=False re-emits the 'group' key as a column; drop it so the merged
        # rows have the same columns as the untouched rows and it does not leak out
        .drop(columns=[_group_field, ])
    )
    result = pd.concat([gdf[gdf[_group_field] == -1].drop(columns=[_group_field, ]), dissolved], ignore_index=True)

    # a reach outside a group may flow into a member that was just collapsed into its
    # keeper (common when consolidating chains of short streams); repoint those so they
    # reference the surviving keeper instead of a now-deleted id
    if member_to_keeper:
        result[schema.next_river_id] = result[schema.next_river_id].map(lambda x: member_to_keeper.get(x, x))

    return result


def find_headwater_mergers(gdf: gpd.GeoDataFrame, min_order: int) -> dict:
    if min_order == 1:
        return dict()

    upstream_id_map = make_upstream_id_map(gdf)
    order_of = gdf.set_index(schema.river_id)[schema.strahler_order].to_dict()

    # candidates: order min_order reaches with 2+ upstreams that are ALL order min_order - 1.
    # group every reach by its downstream id, so each group is the upstreams of one reach;
    # per group, 'size' counts the upstreams and 'sum' counts those of order min_order - 1.
    # a downstream qualifies when it has 2+ upstreams that are all order min_order - 1.
    upstreams = gdf[[schema.next_river_id, schema.strahler_order]].assign(
        is_order_minus_1=lambda df: df[schema.strahler_order] == (min_order - 1)
    )
    counts = upstreams.groupby(schema.next_river_id)['is_order_minus_1'].agg(['size', 'sum'])
    qualifying_ds = counts.index[(counts['size'] >= 2) & (counts['size'] == counts['sum'])]

    candidate_ids = gdf.loc[
        (gdf[schema.strahler_order] == min_order) & (gdf[schema.river_id].isin(qualifying_ds)),
        schema.river_id,
    ].astype(int).tolist()
    if not candidate_ids:
        return dict()

    def upstream_below_order(root: int) -> list:
        """
        Reaches upstream of ``root`` whose order is < min_order, collected with a
        bounded walk that stops at (never collects or traverses past) any reach of
        order >= min_order. ``root`` is itself the first downstream order-min_order
        reach, so it absorbs exactly its directly-feeding lower-order tributaries
        (and any lower-order chains) and never another same-or-higher-order
        segment. Without this bound, a chain of order-min_order segments would all
        collapse into the most-downstream candidate.
        """
        collected = []
        stack = list(upstream_id_map.get(root, []))
        while stack:
            node = stack.pop()
            if order_of.get(node, min_order) >= min_order:
                continue
            collected.append(node)
            stack.extend(upstream_id_map.get(node, []))
        return collected

    headwaters_dict = {
        candidate: [int(u) for u in upstream_below_order(candidate)]
        for candidate in candidate_ids
    }
    return headwaters_dict


def merge_headwaters(gdf: gpd.GeoDataFrame, header_mergers: dict) -> gpd.GeoDataFrame:
    return dissolve_groups(gdf, header_mergers, _build_aggfunc(gdf, length_rule='max'))


def merge_headwaters_order2_geom(gdf: gpd.GeoDataFrame, header_mergers: dict) -> gpd.GeoDataFrame:
    """
    Same as merge_headwaters (attributes are aggregated identically: max length,
    summed area, etc.), but the merged reach keeps ONLY the order-2 keeper geometry
    instead of the union of the order-2 line and its order-1 upstream tributaries.
    The order-1s are never mapped, so their geometry is dropped rather than merged in.
    """
    keeper_geom = gdf.set_index(schema.river_id)[schema.geometry]
    merged = dissolve_groups(gdf, header_mergers, _build_aggfunc(gdf, length_rule='max'))
    keeper_ids = set(header_mergers.keys())
    mask = merged[schema.river_id].isin(keeper_ids)
    merged.loc[mask, schema.geometry] = merged.loc[mask, schema.river_id].map(keeper_geom)
    return merged


def prune_branches(gdf: gpd.GeoDataFrame, branches_to_prune: dict) -> gpd.GeoDataFrame:
    """
    Drop the rows whose ids appear in the value-lists of branches_to_prune and
    fold each dropped branch's local catchment area into its keeper (the dict key)
    so total drained area is conserved. Pruning removes parallel headwater siblings
    rather than an upstream chain, so only areaM2 changes; the keeper's own
    USContArea/DSContArea (its upstream and downstream contributing areas) are
    unaffected.
    """
    to_drop = {rivid for drops in branches_to_prune.values() for rivid in drops}
    if not to_drop:
        return gdf

    if schema.area in gdf.columns:
        area_by_id = gdf.set_index(schema.river_id)[schema.area]
        added_area = {
            keeper: float(area_by_id.reindex(drops).sum())
            for keeper, drops in branches_to_prune.items()
        }
        keeper_mask = gdf[schema.river_id].isin(added_area)
        gdf.loc[keeper_mask, schema.area] = (
                gdf.loc[keeper_mask, schema.area] + gdf.loc[keeper_mask, schema.river_id].map(added_area)
        )

    return gdf[~gdf[schema.river_id].isin(to_drop)].reset_index(drop=True)


def find_branches_to_prune(gdf: gpd.GeoDataFrame) -> dict:
    """
    Return {keeper_river_id: [river_ids_to_merge_into_keeper, ...]}.

    For each order-1 stream that flows into an order-2+ stream, pick a sibling at
    the same confluence to merge it into. Multiple order-1s can map to the same
    keeper, so values are lists.
    """
    # select order 1 streams whose downstream is order 2+
    # because headwaters are merged first, a 1 that flows into a 2+ is not a headwater confluence anymore
    order2plus_ids = gdf.loc[gdf[schema.strahler_order] >= 2, schema.river_id].values
    order1s = gdf.loc[
        (gdf[schema.strahler_order] == 1) & (gdf[schema.next_river_id].isin(order2plus_ids)),
        [schema.river_id, schema.next_river_id, schema.strahler_order, schema.geometry]
    ]

    # reverse adjacency replaces nx.DiGraph.predecessors; lookups replace per-row gdf filters
    upstream_id_map = make_upstream_id_map(gdf)
    order_by_id = gdf.set_index(schema.river_id)[schema.strahler_order].to_dict()
    geom_by_id = gdf.set_index(schema.river_id)[schema.geometry].to_dict()

    merges: dict = {}
    do_not_delete = set()
    for _, rivid, ds_rivid, strm_order, geometry in order1s.itertuples():
        if rivid in do_not_delete:
            continue

        siblings = [s for s in upstream_id_map.get(ds_rivid, []) if s != rivid]

        if len(siblings) > 2:
            # This is an inlet to a lake. It should merge with the downstream stream
            siblings = [ds_rivid]
        elif len(siblings) > 1:
            sibling_orders = [order_by_id[s] for s in siblings]
            # Case: 1 sibling shares this order and another has higher order -> pick the higher
            if strm_order in sibling_orders and strm_order < max(sibling_orders):
                siblings = [s for s, o in zip(siblings, sibling_orders) if o > strm_order]
            # otherwise 2 higher-order siblings -> pick the nearest by centroid distance
            else:
                centroid = geometry.centroid
                siblings = [min(siblings, key=lambda s: geom_by_id[s].distance(centroid))]

        if not siblings:
            siblings = [ds_rivid]

        keeper = siblings[0]
        if keeper in do_not_delete:
            continue

        merges.setdefault(int(keeper), []).append(int(rivid))
        do_not_delete.add(rivid)

    return merges


def find_orphaned_coastal_outlets(gdf: gpd.GeoDataFrame) -> dict:
    """
    Find order-1 reaches left as tiny standalone outlets by zero-length outlet
    removal, and map each to the neighbor it should be folded into.

    When remove_zero_length drops a zero-length reach that was itself an outlet
    (case 3), every direct upstream is repointed to -1 and becomes its own outlet.
    A small order-1 among those upstreams then drains straight to the network edge
    instead of joining the larger river it shared the (now deleted) confluence
    with. Those siblings can no longer be related through the topology (they all
    point to -1), but they still carry the STALE outletRiverId of the deleted
    zero-length reach, which recovers the sibling set. This must therefore run
    before topology.recompute_outlets overwrites those stale ids.

    Returns {keeper_river_id: [order1_ids_to_fold, ...]} in the shape that
    prune_branches consumes: each order-1 sibling is folded (area-conserving, no
    geometry merge) into the largest-drainage-area member of its group. Groups
    with a single orphan (the sole upstream of the removed reach, a legitimate new
    outlet) are skipped, and non-order-1 members are left as independent outlets.
    """
    surviving_ids = set(gdf[schema.river_id].values)
    orphans = gdf[
        (gdf[schema.next_river_id] == -1) & (~gdf[schema.last_river_id].isin(surviving_ids))
        ]
    if orphans.empty:
        return {}

    merges: dict = {}
    for _, group in orphans.groupby(schema.last_river_id):
        if len(group) < 2:
            continue
        ids = group[schema.river_id].to_numpy()
        areas = group[schema.tdx_ds_area_field].to_numpy()
        orders = group[schema.strahler_order].to_numpy()
        keeper = int(ids[areas.argmax()])
        folds = [int(i) for i, o in zip(ids, orders) if o == 1 and int(i) != keeper]
        if folds:
            merges[keeper] = folds
    return merges


def find_short_streams(gdf: gpd.GeoDataFrame, min_length: float) -> dict:
    """
    make a dictionary of {downstream_most_id: [stream_ids_to_merge_in, ...]}

    look for short streams that are connected to a stream upstream or downstream of them no split by a confluence
    we would like to merge together those streams to eliminate the short streams so that routing is numerically
    stabler and more accurate and reduce the number of stored results.

    Each short stream chooses one merge partner: the shorter of its sole-feed upstream
    or downstream neighbor (skipped if there's a confluence on both sides). Union-find
    over those choices collapses chains of consecutive shorts, cascades where multiple
    shorts neighbor the same stream, and isolated shorts that get absorbed into a long
    neighbor (matching the old behavior). The keeper of each group is the member with
    the largest topologySortedOrder, i.e. the most-downstream one. The keeper may be a
    non-short stream.
    """
    short_ids = set(gdf.loc[gdf[schema.length] < min_length, schema.river_id].values)
    if not short_ids:
        return {}

    next_of = gdf.set_index(schema.river_id)[schema.next_river_id].to_dict()
    length_of = gdf.set_index(schema.river_id)[schema.length].to_dict()
    topo_of = gdf.set_index(schema.river_id)[schema.topo_sort].to_dict()
    upstream_id_map = make_upstream_id_map(gdf)

    parent: dict = {}

    def find(x):
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while x != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # keeper of the merged component is the more-downstream node
        if topo_of.get(ra, -1) > topo_of.get(rb, -1):
            parent[rb] = ra
        else:
            parent[ra] = rb

    for s in short_ids:
        up = upstream_id_map.get(s, [])
        ds = next_of.get(s, -1)
        ds_up = upstream_id_map.get(ds, []) if ds != -1 else []

        # confluence on both sides → no safe neighbor to merge with
        if len(up) != 1 and len(ds_up) != 1:
            continue

        partner_up = up[0] if len(up) == 1 else None
        partner_ds = ds if (ds != -1 and len(ds_up) == 1) else None
        if partner_up is None and partner_ds is None:
            continue

        # pick the shorter neighbor (old code did this via musk_k)
        if partner_up is not None and partner_ds is not None:
            partner = (
                partner_up
                if length_of.get(partner_up, float('inf')) <= length_of.get(partner_ds, float('inf'))
                else partner_ds
            )
        else:
            partner = partner_up if partner_up is not None else partner_ds

        parent.setdefault(s, s)
        parent.setdefault(partner, partner)
        union(s, partner)

    groups: dict = {}
    for node in list(parent):
        root = find(node)
        if node != root:
            groups.setdefault(int(root), []).append(int(node))
    return groups


def consolidate_short_streams(gdf: gpd.GeoDataFrame, consolidations: dict) -> gpd.GeoDataFrame:
    return dissolve_groups(gdf, consolidations, _build_aggfunc(gdf, length_rule='sum'))
