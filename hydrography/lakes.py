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
    'outlet_links',
    'apply_outlet_links',
    'find_lake_edits',
    'apply_lake_edits',
]

# Which reaches enter each lake and where it drains, as raw TDX-Hydro routes them: every inlet drains to
# its row's outlet in the source network, unmodified. A lake TDX ends on several terminal reaches therefore
# has several outlets here, one per terminal, until lake_outlet_links.csv joins them.
lake_table_path = paths.network_data_root / 'lake_table.csv'

# The connectivity overrides, and only those: each repoints a terminal reach (DSLINKNO -1) of a lake at
# the lake's outlet, so the lake drains and dissolves as one. See :func:`apply_outlet_links`.
outlet_links_path = paths.network_data_root / 'lake_outlet_links.csv'

# An inlet whose drainage area (DSContArea, the contributing area at its downstream
# end where it meets the lake) is below this is too small to keep as its own routed
# reach; its whole upstream branch is absorbed into the lake instead. 100 km^2.
min_lake_inlet_area = 100_000_000

# Douglas-Peucker tolerance for the traced lake lines, and only for those. Every other
# reach is published at source resolution because generalizing for a zoom is tippecanoe's
# job (see simplify_streams.py) - a tolerance baked into the file applies at every zoom
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


def _read_lake_table() -> pd.DataFrame:
    """lake_table.csv as written: every inlet drains to its row's outlet in raw TDX-Hydro."""
    return pd.read_csv(lake_table_path)


def _hand_links() -> dict:
    """``lake_outlet_links.csv`` as ``{riverId: nextRiverId}``."""
    if not outlet_links_path.exists():
        return {}
    links = pd.read_csv(outlet_links_path)
    if links[schema.river_id].duplicated().any():
        repeated = links.loc[links[schema.river_id].duplicated(), schema.river_id].unique()
        raise RuntimeError(f'{outlet_links_path.name} repoints {repeated[:5].tolist()} more than once')
    return dict(zip(links[schema.river_id], links[schema.next_river_id]))


def _resolve(links: dict) -> dict:
    """Each linked reach's final target, following targets that are themselves linked onward, so every
    arm points straight at the reach its lake drains through. Refuses a cycle."""
    resolved = {}
    for source in links:
        node, seen = source, {source}
        while node in links:
            node = links[node]
            if node in seen:
                raise RuntimeError(f'lake outlet links form a cycle through {sorted(seen)[:5]}')
            seen.add(node)
        resolved[source] = node
    return resolved


def outlet_links(gdf: gpd.GeoDataFrame) -> dict:
    """The rows of ``lake_outlet_links.csv`` for this gdf, as ``{riverId: lake outlet}``.

    Read from the raw network, before :func:`apply_outlet_links` changes it, and handed to every other
    function here. Every override is explicit in the file; none is inferred. Chains are resolved so
    every arm points at its lake's final outlet.

    Raises unless every ``lake_id`` of lake_table.csv with an outlet in ``gdf`` drains through exactly
    one outlet once the links are applied. A lake the links leave with two or more outlets is either
    missing an override (a terminal arm that should be linked to the lake's outlet) or is not one lake
    (separate water bodies under one id); either way it is fixed in the tables, not guessed at here.
    """
    here = set(gdf[schema.river_id])
    links = _resolve({s: t for s, t in _hand_links().items() if s in here and t in here})

    lake_table = _read_lake_table()
    lake_table = lake_table[lake_table[schema.outlet_field].isin(here)]
    effective = _effective_outlets(lake_table, links)
    outlets = effective.groupby(lake_table[schema.lake_id_field]).unique()
    several = outlets[outlets.map(len) > 1]
    if not several.empty:
        listing = '; '.join(f'lake {lake}: {sorted(int(o) for o in found)}' for lake, found in several.items())
        raise RuntimeError(
            f'{len(several)} lake(s) in {lake_table_path.name} still have more than one outlet after '
            f'{outlet_links_path.name} is applied. Link each extra terminal outlet to the lake\'s outlet '
            f'in {outlet_links_path.name}, or give separate water bodies their own lake_id. {listing}')
    return links


def _effective_outlets(lake_table: pd.DataFrame, links: dict) -> pd.Series:
    """Each row's outlet once the links are applied: a linked arm's inlets drain on to its lake's outlet."""
    outlet = lake_table[schema.outlet_field]
    return outlet.map(links).fillna(outlet).astype(outlet.dtype)


def lake_outlets(gdf: gpd.GeoDataFrame, links: dict) -> set:
    """Every lake outlet in the table that is a reach of ``gdf``, after the links (:func:`outlet_links`).

    This is the protected set the simplification steps are given, and it is taken from the table
    rather than from :func:`find_lake_edits` because a lake whose inlets are all below
    ``min_inlet_area`` produces no edit at all and would otherwise go unprotected.
    """
    return set(_effective_outlets(_read_lake_table(), links)) & set(gdf[schema.river_id])


def apply_outlet_links(gdf: gpd.GeoDataFrame, links: dict) -> gpd.GeoDataFrame:
    """Join the separate terminal arms of one lake into a single drainage.

    A lake sitting in a closed basin often reaches the network as several terminal reaches
    rather than one. Lake Van arrives as four, within sixty metres of each other; Uvs Nuur as
    three. Each is a river ending at the shore, and the DEM cannot route between them because
    the surface they would route across is flat water, so each is marked DSLINKNO -1 and the
    collapse builds a separate lake on each one. One lake becomes four.

    Nothing in the source data is wrong, and nothing here corrects it: the arms really do end
    where TDX says they end, and lake_table.csv records them that way, each arm the outlet of the
    inlets that reach it. What the source cannot know is that they end in the *same water*. That
    is recorded by the lake table's ``lake_id`` and, for arms the table has no rows for, by hand in
    ``lake_outlet_links.csv``; :func:`outlet_links` combines the two. The links are applied to the
    working copy only. The parquet under ``$TDXHYDRO_GEOPARQUET_FINISHED`` is a faithful translation
    of the source and stays that way.

    Each link repoints one terminal reach at the outlet of its lake. Applied before the topology is
    computed, so accumulation, the sub-250 km² terminal drop and the lake collapse all see one lake:
    :func:`find_lake_edits` reads each arm's inlets as the outlet's, and the arm, no longer an outlet
    or an inlet, dissolves into it.

    Refuses to touch a reach that already flows somewhere. A link that redirects live drainage
    is far more likely to be a mistake than an intention; the check is what makes the override
    table safe to hand-edit.
    """
    if not links:
        return gdf

    next_river = gdf.set_index(schema.river_id)[schema.next_river_id]
    live = [source for source in links if next_river[source] != -1]
    if live:
        raise RuntimeError(
            f'lake outlet links repoint {len(live)} reach(es) that already flow somewhere: '
            f'{live[:5]}. Only a terminal reach (DSLINKNO -1) may be linked across a lake.')
    targets = links

    gdf = gdf.copy()
    column = gdf[schema.next_river_id]
    gdf[schema.next_river_id] = (
        gdf[schema.river_id].map(targets).fillna(column).astype(column.dtype))

    # The arm's water now reaches the target, so the target and everything below it drain the
    # arm's area too. TDX's contributing areas and Strahler orders were computed with the arm
    # ending at the shore, and they are not recomputed downstream, so carry the arm into them
    # here: its DSContArea is added to the US and DS areas along the target's path (each reach's
    # own local area is unchanged) and the Strahler order there is raised to at least the arm's.
    # Without this a 2,821 km2 arm linked into a 1,488 km2 reach leaves the reach that now
    # drains both reporting the smaller area, and compute_topology, which orders reaches by
    # (Strahler, DSContArea), puts the arm after the reach it flows into.
    row = pd.Series(np.arange(len(gdf)), index=gdf[schema.river_id].to_numpy())
    down = row.reindex(gdf[schema.next_river_id].to_numpy()).to_numpy()
    us_area = gdf[schema.tdx_us_area_field].to_numpy(dtype=float, copy=True)
    ds_area = gdf[schema.tdx_ds_area_field].to_numpy(dtype=float, copy=True)
    strahler = gdf[schema.strahler_order].to_numpy(copy=True)
    # each arm adds its own area as TDX gives it, read from the unmodified columns, so the result does
    # not depend on the order the arms are applied in
    raw_area = gdf[schema.tdx_ds_area_field].to_numpy(dtype=float)
    raw_order = gdf[schema.strahler_order].to_numpy()
    for source, target in targets.items():
        arm = row[source]
        added, order = raw_area[arm], raw_order[arm]
        node = row[target]
        while True:
            us_area[node] += added
            ds_area[node] += added
            strahler[node] = max(strahler[node], order)
            if np.isnan(down[node]):
                break
            node = int(down[node])
    gdf[schema.tdx_us_area_field] = us_area.astype(gdf[schema.tdx_us_area_field].dtype)
    gdf[schema.tdx_ds_area_field] = ds_area.astype(gdf[schema.tdx_ds_area_field].dtype)
    gdf[schema.strahler_order] = strahler
    return gdf


def find_lake_edits(gdf: gpd.GeoDataFrame, links: dict, min_inlet_area: float = min_lake_inlet_area) -> dict:
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
    outlet downstream, in create_catchments.py). A lake whose inlets are all below
    the threshold collapses its entire contributing network into the outlet.

    Raises if any lake outlet would itself be deleted, which means the lake is
    nested inside another (or the lake table fragments one lake across several
    outlets). That must be resolved in lake_table.csv, not here.
    """
    lake_table = _read_lake_table()
    lake_table[schema.outlet_field] = _effective_outlets(lake_table, links)
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
    # one lake per outlet the links leave. outlet_links has checked that each lake_id drains through
    # exactly one, so this is one group per lake_id, except that lake_ids sharing an outlet (77 outlets
    # are shared, 8 lake_ids on the Lake Huron one) are kept together: they are one water surface
    # draining through one reach, and built apart each would claim the outlet with a partial inlet list
    for outlet, group in lake_table.groupby(schema.outlet_field, sort=False):
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
    create_catchments.py, which redirects every deleted reach's basin to the
    outlet), repoint that lake's inlets at the outlet, and drop the interior
    reaches. DSContArea/USContArea are unchanged - the outlet's contributing areas
    already account for everything upstream.
    """
    if not lake_edits:
        return gdf

    next_dtype = gdf[schema.next_river_id].dtype
    geom_for = gdf.set_index(schema.river_id)['geometry'].to_dict()
    area_for = gdf.set_index(schema.river_id)[schema.area] if schema.area in gdf.columns else None
    # the length columns follow the geometry: an outlet that now draws the lake's whole path is as
    # long as that path. TDX often ends a lake on a zero-length outlet, and left at zero the
    # zero-length cleanup that runs next deletes the outlet and with it the lake (Great Salt Lake
    # was one of a dozen), leaving its inlets as separate terminals
    length_fields = [f for f in (schema.length, schema.tdx_length_field) if f in gdf.columns]
    length_for = gdf.set_index(schema.river_id)[length_fields]
    added_length: dict = {}

    inlet_to_outlet = {}
    all_to_delete = set()
    added_area: dict = {}
    merged_geometries: dict = {}
    for outlet, edit in lake_edits.items():
        outlet = int(outlet)
        path = [int(p) for p in edit['geometry_path']]
        if len(path) > 1:
            merged_geometries[outlet] = linemerge([geom_for[p] for p in path if p in geom_for])
            added_length[outlet] = length_for.reindex(path).sum()
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

    if added_length and length_fields:
        lengths = pd.DataFrame(added_length).T
        outlet_rows = gdf[schema.river_id].isin(lengths.index)
        for field in length_fields:
            gdf.loc[outlet_rows, field] = (
                gdf.loc[outlet_rows, schema.river_id].map(lengths[field]).astype(gdf[field].dtype))

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
