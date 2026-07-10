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
]


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
    outlet ids left behind are corrected. Only outletRiverId is written; vpuId and
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
