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
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from shapely.ops import linemerge

from . import paths
from . import projection
from . import schema

__all__ = [
    'lake_outlets',
    'find_lake_edits',
    'apply_lake_edits',
]

lake_table_path = paths.network_data_root / 'lake_table.csv'

# An inlet whose drainage area (DSContArea, the contributing area at its downstream
# end where it meets the lake) is below this is too small to keep as its own routed
# reach; its whole upstream branch is absorbed into the lake instead. 100 km^2.
min_lake_inlet_area = 100_000_000

# Douglas-Peucker tolerance for the traced lake lines, and only for those. Every other
# reach is published at source resolution because generalizing for a zoom is tippecanoe's
# job (see 2_simplify_streams.py) - a tolerance baked into the file applies at every zoom
# and can never be undone. A lake trace is different in kind: it is not a channel that was
# surveyed, it is a synthetic line drawn across open water to show which inlet connects to
# which outlet and which way the water goes. The meander detail it inherits from the
# reaches it was merged from is describing a river bed that the lake drowned, so keeping it
# costs vertices to draw something the map should not be asserting. 100 m is coarse enough
# to flatten that inherited wiggle and fine enough to keep the line inside its own lake.
lake_simplify_meters = 100.0


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


def _generalize(geometries: gpd.GeoSeries, tolerance_meters: float) -> gpd.GeoSeries:
    """Douglas-Peucker the given lines at a ground tolerance in metres, returning them in
    the CRS they arrived in.

    The work is done in web mercator, where the projection is conformal so one tolerance
    applies in both directions, unlike the lat/lon the network is carried in. Mercator
    metres are inflated by 1/cos(lat), so a fixed *ground* tolerance is tolerance/cos(lat)
    mercator metres - taken per geometry, since a lake is small enough that one latitude
    describes all of it but the set spans the equator to 80 N.
    """
    mercator = geometries.to_crs(epsg=projection.web_mercator_epsg)
    latitude = np.radians(geometries.representative_point().y.to_numpy())
    simplified = shapely.simplify(mercator.values, tolerance_meters / np.cos(latitude))
    return gpd.GeoSeries(simplified, index=geometries.index, crs=mercator.crs).to_crs(geometries.crs)


def lake_outlets(gdf: gpd.GeoDataFrame) -> set:
    """Every lake outlet in the table that is a reach of ``gdf``.

    This is the protected set the simplification steps are given, and it is taken from the table
    rather than from :func:`find_lake_edits` because a lake whose inlets are all below
    ``min_inlet_area`` produces no edit at all and would otherwise go unprotected.
    """
    lake_table = pd.read_csv(lake_table_path)
    return set(lake_table[schema.outlet_field]) & set(gdf[schema.river_id])


def find_lake_edits(gdf: gpd.GeoDataFrame, min_inlet_area: float = min_lake_inlet_area) -> dict:
    """
    Analyze the network against the lake table and describe the edits to make,
    without mutating the gdf.

    Returns ``{outlet_id: {'inlets': [...], 'delete': [...], 'geometry_path': [...]}}``:
      - inlets:        reaches whose nextRiverId should be set to the outlet
      - delete:        interior reaches to remove from the network
      - geometry_path: reaches from each traced inlet's downstream through the
                       outlet whose merged line becomes the outlet's geometry. The
                       traced inlets are the kept inlets flagged 1 in the
                       ``trace_inlet`` column of lake_table.csv, which is written by
                       scripts/flag_lake_trace_inlets.py and flags every inlet of
                       Strahler order 4 or more plus each lake's largest one; a lake
                       that flags none defaults to its single largest-drainage inlet.
                       The merged line branches into a MultiLineString for two or more.

    Only inlets whose drainage area (DSContArea) is at least ``min_inlet_area`` are
    kept as true inlets. A smaller inlet is too minor to route into the lake on its
    own, so it is dropped from the barrier set and its whole upstream branch falls
    into the interior to be absorbed into the lake (its catchment folds into the
    outlet downstream, in 3_create_catchments). A lake whose inlets are all below
    the threshold collapses its entire contributing network into the outlet.

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

    def _path_to_outlet(inlet, outlet):
        # interior reaches from an inlet's downstream neighbour to the outlet (inclusive)
        path = []
        start_id = next_id_for[inlet]
        while start_id != outlet:
            if start_id == -1 or start_id not in ids_present:
                break
            path.append(start_id)
            start_id = next_id_for[start_id]
        if start_id != outlet:
            raise RuntimeError(
                f'Lake outlet {outlet} not reachable from inlet {inlet}, which should be impossible'
            )
        path.append(outlet)
        return path

    edits = {}
    all_to_delete = set()
    unique_outlets = lake_table[schema.outlet_field].unique()
    for outlet in unique_outlets:
        group = lake_table[lake_table[schema.outlet_field] == outlet]
        inlets = group[schema.inlet_field].tolist()
        if outlet not in ids_present:
            raise ValueError(f'Lake outlet {outlet} not found in gdf, which should not be possible')

        # only inlets draining at least min_inlet_area stay as true (routed) inlets; the
        # rest are not barriers, so _strict_interior walks past them and absorbs their
        # whole upstream branch into the lake interior
        kept_inlets = [i for i in inlets if drainage_area_for.get(i, 0) >= min_inlet_area]

        # which kept inlets get a line traced through the lake to the outlet, forming the
        # outlet's dissolved geometry. per-inlet 1/0 in the trace_inlet column (blank
        # counts as 0). a lake that flags no inlet defaults to its single largest-drainage
        # kept inlet, so an unset lake reproduces the historic single-trace output.
        traced_inlets = []
        if kept_inlets and schema.trace_inlet_field in group.columns:
            flags = pd.to_numeric(group[schema.trace_inlet_field], errors='coerce').fillna(0)
            flagged = set(group.loc[flags > 0, schema.inlet_field])
            traced_inlets = [i for i in kept_inlets if i in flagged]
        if not traced_inlets and kept_inlets:
            traced_inlets = [max(kept_inlets, key=lambda i: drainage_area_for.get(i, -1))]

        # the outlet geometry is the merged line from each traced inlet down to the
        # outlet; a set dedups the shared downstream trunk so no reach is drawn twice and
        # linemerge is order-independent (multiple inlets -> branching MultiLineString).
        # with no kept inlet the whole network collapses into the outlet, which then just
        # keeps its own geometry.
        if traced_inlets:
            geometry_reaches = set()
            for inlet in traced_inlets:
                geometry_reaches.update(_path_to_outlet(inlet, outlet))
            direct_path = sorted(geometry_reaches)
        else:
            direct_path = [outlet]

        interior = _strict_interior(graph, outlet, set(kept_inlets))
        interior.discard(outlet)
        all_to_delete |= interior

        edits[int(outlet)] = {
            'inlets': [int(i) for i in kept_inlets],
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


def apply_lake_edits(
        gdf: gpd.GeoDataFrame, lake_edits: dict, simplify_meters: float = lake_simplify_meters
) -> gpd.GeoDataFrame:
    """
    Apply the edits from :func:`find_lake_edits`: merge each lake's direct-path
    geometry into its outlet, generalize that merged line at ``simplify_meters``
    (see :data:`lake_simplify_meters` - nothing outside a lake is touched), fold
    each interior reach's local catchment area (areaM2) into the outlet so total
    drained area is conserved (matching the catchment dissolve in
    3_create_catchments, which redirects every deleted reach's basin to the
    outlet), repoint that lake's inlets at the outlet, and drop the interior
    reaches. DSContArea/USContArea are unchanged - the outlet's contributing areas
    already account for everything upstream.
    """
    if not lake_edits:
        return gdf

    next_dtype = gdf[schema.next_river_id].dtype
    geom_for = gdf.set_index(schema.river_id)['geometry'].to_dict()
    area_for = gdf.set_index(schema.river_id)[schema.area] if schema.area in gdf.columns else None

    inlet_to_outlet = {}
    all_to_delete = set()
    added_area: dict = {}
    merged_geometries: dict = {}
    for outlet, edit in lake_edits.items():
        outlet = int(outlet)
        path = [int(p) for p in edit['geometry_path']]
        if len(path) > 1:
            merged_geometries[outlet] = linemerge([geom_for[p] for p in path if p in geom_for])
        for inlet in edit['inlets']:
            inlet_to_outlet[int(inlet)] = outlet
        deletes = [int(d) for d in edit['delete']]
        all_to_delete.update(deletes)
        if area_for is not None:
            added_area[outlet] = float(area_for.reindex(deletes).sum())

    # generalize every lake trace in one pass, then write them all back in one pass. Douglas-Peucker
    # keeps the endpoints, so the outlet's downstream end and the point each inlet meets the lake at
    # are exactly where they were and the drawn network stays joined at its junctions.
    if merged_geometries:
        traces = gpd.GeoSeries(list(merged_geometries.values()), index=list(merged_geometries), crs=gdf.crs)
        if simplify_meters:
            traces = _generalize(traces, simplify_meters)
        outlet_rows = gdf[schema.river_id].isin(merged_geometries)
        gdf.loc[outlet_rows, schema.geometry] = gdf.loc[outlet_rows, schema.river_id].map(traces)

    # grow each outlet's areaM2 by the sum of the interior areas it absorbs, before
    # those rows are dropped, so the dissolved area equals the sum of its parts
    if added_area:
        keeper_mask = gdf[schema.river_id].isin(added_area)
        gdf.loc[keeper_mask, schema.area] = (
                gdf.loc[keeper_mask, schema.area] + gdf.loc[keeper_mask, schema.river_id].map(added_area)
        )

    # update the inlet rows to point to the outlet (next_id is overridden by the inlet to outlet map)
    gdf[schema.next_river_id] = (
        gdf[schema.river_id]
        .map(inlet_to_outlet)
        .fillna(gdf[schema.next_river_id])
        .astype(next_dtype)
    )

    return gdf[~gdf[schema.river_id].isin(all_to_delete)]
