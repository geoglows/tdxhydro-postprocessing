"""
Lake simplification.

Consolidates the lake-collapse logic from the old ``run_1.1_lakes.py`` /
``tdxhydrorapid.network`` into the same two-step shape as the other revision
steps: :func:`find_lake_edits` analyzes the network and returns a JSON-style
description of the edits, and :func:`apply_lake_edits` mutates the gdf to
realize them.

The interior of each lake is found with one bounded upstream walk
(``_strict_interior``) instead of repeated whole-graph ``nx.ancestors``
traversals.
"""
import os

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.ops import linemerge

from . import schema

__all__ = [
    'find_lake_edits',
    'apply_lake_edits',
]

# lake_table_path = '../network_data/lake_table.csv'
lake_table_path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'network_data', 'lake_table.csv')


def _build_digraph(df: pd.DataFrame) -> nx.DiGraph:
    g = nx.from_pandas_edgelist(
        df[df[schema.next_river_id] != -1], source=schema.river_id, target=schema.next_river_id, create_using=nx.DiGraph
    )
    g.add_nodes_from(df[schema.river_id].values)
    return g


def _strict_interior(graph: nx.DiGraph, outlet, barriers: set) -> set:
    """
    Nodes strictly upstream of ``outlet`` whose downstream path to ``outlet``
    does not pass through (or start at) any node in ``barriers``. ``outlet``
    and the barrier nodes themselves are excluded.

    On a flow forest this equals the original
    ``nx.ancestors(outlet) - U(nx.ancestors(i) | {i} for i in barriers)`` but
    is one bounded walk over the lake interior instead of N traversals of the
    entire region graph.
    """
    interior = set()
    stack = [p for p in graph.predecessors(outlet) if p not in barriers]
    while stack:
        node = stack.pop()
        if node in interior:
            continue
        interior.add(node)
        stack.extend(
            p for p in graph.predecessors(node) if p not in barriers and p not in interior
        )
    return interior


def find_lake_edits(gdf: gpd.GeoDataFrame) -> dict:
    """
    Analyze the network against the lake table and describe the edits to make,
    without mutating the gdf.

    Returns ``{outlet_id: {'inlets': [...], 'delete': [...], 'geometry_path': [...]}}``:
      - inlets:        reaches whose nextRiverId should be set to the outlet
      - delete:        interior reaches to remove from the network
      - geometry_path: reaches from the largest inlet's downstream through the
                       outlet whose merged line becomes the outlet's geometry

    Raises if any lake outlet would itself be deleted, which means the lake is
    nested inside another (or the lake table fragments one lake across several
    outlets). That must be resolved in lake_table.csv, not here.
    """
    lake_table = pd.read_csv(lake_table_path)
    lake_table = lake_table[lake_table[schema.inlet_field].isin(gdf[schema.river_id])]
    if lake_table.empty:
        return {}

    graph = _build_digraph(gdf)

    # id lookup tables for faster network traversal compared to repeated gdf.loc[condition, field]
    next_id_for = gdf.set_index(schema.river_id)[schema.next_river_id].to_dict()
    drainage_area_for = gdf.set_index(schema.river_id)[schema.tdx_ds_area_field].to_dict()
    ids_present = set(gdf[schema.river_id].values)

    edits = {}
    all_to_delete = set()
    unique_outlets = lake_table[schema.outlet_field].unique()
    for outlet in unique_outlets:
        inlets = lake_table[lake_table[schema.outlet_field] == outlet][schema.inlet_field].tolist()
        if outlet not in ids_present:
            raise ValueError(f'Lake outlet {outlet} not found in gdf, which should not be possible')
        largest_inlet = max(inlets, key=lambda i: drainage_area_for.get(i, -1))

        direct_path = []
        start_id = next_id_for[largest_inlet]
        while start_id != outlet:
            if start_id == -1 or start_id not in ids_present:
                break
            direct_path.append(start_id)
            start_id = next_id_for[start_id]
        if start_id != outlet:
            raise RuntimeError(
                f'Lake outlet {outlet} not reachable from inlet {largest_inlet}, which should be impossible'
            )
        direct_path.append(outlet)

        interior = _strict_interior(graph, outlet, set(inlets))
        interior.discard(outlet)
        all_to_delete |= interior

        edits[int(outlet)] = {
            'inlets': [int(i) for i in inlets],
            'delete': sorted(int(s) for s in interior),
            'geometry_path': [int(s) for s in direct_path],
        }

    # A lake outlet falling in another lake's interior (all_to_delete) is only a
    # problem when that lake still has a *surviving* inlet pointing at it: the
    # inlet would be left referencing a vanished reach. If the inner lake's inlets
    # are themselves deleted, the inner lake is simply absorbed into the outer one,
    # which is fine. Only the former is a lake_table.csv defect we refuse to guess
    # at; refine the table so each lake has a single downstream-most outlet.
    dangling_outlets = sorted(
        outlet
        for outlet, edit in edits.items()
        if outlet in all_to_delete
        and any(inlet not in all_to_delete for inlet in edit['inlets'])
    )
    if dangling_outlets:
        raise ValueError(
            f'{len(dangling_outlets)} lake outlet(s) would be deleted as the interior of '
            f'another lake while still having surviving inlets (nested or multi-outlet '
            f'lakes in lake_table.csv): {dangling_outlets}. Refine the lake table so each '
            f'lake has a single downstream-most outlet.'
        )

    return edits


def apply_lake_edits(gdf: gpd.GeoDataFrame, lake_edits: dict) -> gpd.GeoDataFrame:
    """
    Apply the edits from :func:`find_lake_edits`: merge each lake's direct-path
    geometry into its outlet, repoint that lake's inlets at the outlet, and drop
    the interior reaches.
    """
    if not lake_edits:
        return gdf

    next_dtype = gdf[schema.next_river_id].dtype
    geom_for = gdf.set_index(schema.river_id)['geometry'].to_dict()

    inlet_to_outlet = {}
    all_to_delete = set()
    for outlet, edit in lake_edits.items():
        outlet = int(outlet)
        path = [int(p) for p in edit['geometry_path']]
        # todo handle correcting the rest of the attributes instead of only handling geometry
        if len(path) > 1:
            merged = linemerge([geom_for[p] for p in path if p in geom_for])
            gdf.loc[gdf[schema.river_id] == outlet, 'geometry'] = merged
        for inlet in edit['inlets']:
            inlet_to_outlet[int(inlet)] = outlet
        all_to_delete.update(int(d) for d in edit['delete'])

    # update the inlet rows to point to the outlet (next_id is overridden by the inlet to outlet map)
    gdf[schema.next_river_id] = (
        gdf[schema.river_id]
        .map(inlet_to_outlet)
        .fillna(gdf[schema.next_river_id])
        .astype(next_dtype)
    )

    return gdf[~gdf[schema.river_id].isin(all_to_delete)]
