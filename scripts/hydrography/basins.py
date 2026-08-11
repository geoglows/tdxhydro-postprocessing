"""
Pfafstetter-like basin codes: a nested partition of the network, one digit per level.

The point of the code is that **a prefix of length k names the level-k basin a reach belongs to**,
so the polygons for zoom band k are the catchments dissolved by that prefix. See
``scripts/catchment_tiling_design.md`` for why the tile pyramid needs this and how the levels map
to zooms.

The network is a *forest* of terminal watersheds, not a single basin, so a plain Pfafstetter
recursion cannot start: it has no root, and the top zooms need far fewer features than there are
terminal watersheds (17,448 globally against ~8 wanted at z0). The recursion here is therefore
uniform, and only the *split rule* switches on what the basin currently is:

    if the basin is already below the level's area target -> leaf; pad the rest of the code with 0
    elif it spans more than one terminal watershed        -> COASTAL rule
    else                                                 -> PFAFSTETTER rule

Both rules are the same digit convention, which is what makes the code readable as Pfafstetter:
**the four largest members take even digits 2, 4, 6, 8 in downstream/along-coast order, and the
runs between them take odd digits 1, 3, 5, 7, 9 with 1 the most downstream.** Only what counts as
a "member" differs:

===============  ==========================  ====================  ==============================
rule             member                      ranked by             ordered by
===============  ==========================  ====================  ==============================
coastal          terminal watershed          total watershed area  Hilbert index of its outlet
pfafstetter      tributary subtree           ``DSContArea``        junction position on main stem
===============  ==========================  ====================  ==============================

**Which basins are allowed to split is what makes this usable as a zoom pyramid.** Recursed
blindly, Pfafstetter is badly unbalanced -- an interbasin that is already tiny still spawns nine
children, eight of them trivial -- so a level-k prefix set mixes continent-scale basins with single
reaches and no zoom can be built from it.

Nor is it enough to stop at an area threshold. A zoom step quadruples the tile count, so each level
wants ~4x the features of the one above it, but a Pfafstetter split makes up to **nine** children at
once. Left to an area rule the top levels overshoot badly -- measured on 7020000010, the first four
levels stepped 8.8x, 7.9x, 7.5x, 4.4x and then stalled at 2.3x, 1.4x, so detail arrived in lurches
and then stopped arriving at all.

So the rule here is a **budget**, not a threshold: each level is given a target number of basins,
and basins are refined **largest first** until it is met. Everything else keeps digit 0 and waits
for a deeper level. That is the correct refinement order for a map -- the biggest thing on screen
is the one most worth subdividing -- and it makes the level counts follow whatever schedule the
zoom banding asks for instead of whatever the split radix happens to produce.

Two limits, both measured and both fine (see the design note):

- Pfafstetter only cuts at tributary junctions, so a basin that is a pure chain of reaches never
  subdivides. The recursion converges well short of one basin per reach. That is not a problem:
  the deepest zoom draws individual catchments keyed by ``riverId`` and needs no code at all.
- A handful of reaches have enormous catchments because step 2 collapses a lake's interior into
  its outlet (Superior, Baikal, Victoria, ...). Nothing can subdivide those, so a few basins stay
  large at every level.
"""
import numpy as np
import pandas as pd

from . import schema, topology

__all__ = ['level_targets', 'assign_basin_codes']


def level_targets(first_count: int, leaf_count: int, levels: list) -> dict:
    """How many basins each level is allowed, as a geometric ramp to the leaf count.

    The coarsest level is not free to choose its size - it is one basin split once, so it lands at
    whatever the split radix gives. From there the ramp is fixed at both ends: it starts at that
    realised count and has to arrive at ``leaf_count`` on the zoom *after* the last aggregate level,
    because that is where full resolution takes over. One step per level in between makes the growth
    factor ``(leaf/first) ** (1/len(levels))`` - 3.87x for 9 -> 117,948 over seven levels, which is
    the ~4x per zoom a tile pyramid wants.
    """
    growth = (leaf_count / first_count) ** (1 / len(levels))
    return {level: round(first_count * growth ** i) for i, level in enumerate(levels)}


def _mercator_area(area_m2: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """True area -> web mercator area. Mercator scales distance by 1/cos(lat), so area by 1/cos^2."""
    return area_m2 / np.cos(np.radians(np.clip(lat, -85.0, 85.0))) ** 2


def _even_odd_digits(weights: np.ndarray, order: np.ndarray) -> np.ndarray:
    """The shared digit convention, returned in the input's own row order.

    ``weights`` ranks the members, ``order`` sequences them downstream-to-upstream (or along the
    coast). The four heaviest take 2, 4, 6, 8 in that sequence; every other member takes the odd
    digit of the run it falls in -- 1 below the first heavy member, 9 above the last.
    """
    seq = np.argsort(order, kind='stable')                 # member indices in sequence order
    heavy = np.sort(np.argsort(weights[seq], kind='stable')[::-1][:4])   # positions within seq
    digits = 2 * np.searchsorted(heavy, np.arange(len(seq)), side='left') + 1
    for rank, position in enumerate(heavy):
        digits[position] = 2 * (rank + 1)
    out = np.empty(len(seq), dtype=np.int8)
    out[seq] = digits
    return out


def _coastal_digits(members, outlet_row, mercator_area, hilbert):
    """Split a basin that spans several terminal watersheds, by watershed, along the coast.

    HydroBASINS has the same problem above the watershed and solves it the same way: the coastline
    supplies the ordering that a main stem supplies inside a basin. The Hilbert index of the outlet
    point is that ordering -- it is already computed for the row sort, and it keeps neighbouring
    outlets adjacent, which is all the convention needs.
    """
    watershed = outlet_row[members]
    keys, inverse = np.unique(watershed, return_inverse=True)
    weights = np.bincount(inverse, weights=mercator_area[members])
    digits = _even_odd_digits(weights, hilbert[keys])
    return digits[inverse]


def _pfafstetter_digits(members, parent, children, ds_area, order_of_row):
    """Split a basin lying inside a single terminal watershed, by main stem tributaries.

    The main stem runs from the basin outlet up the largest-drainage branch at every junction. The
    four largest tributaries off it take the even digits; the stretches of stem between their
    junctions -- together with every tributary too small to be selected -- take the odd digits.
    """
    member_set = set(members.tolist())
    root = members[np.argmax(order_of_row[members])]      # downstream-most member
    stem, node = [root], root
    while True:
        upstream = [c for c in children.get(node, ()) if c in member_set]
        if not upstream:
            break
        # largest drainage wins; ties break on the id so the code is reproducible
        node = max(upstream, key=lambda c: (ds_area[c], -c))
        stem.append(node)
    stem_position = {reach: i for i, reach in enumerate(stem)}

    tributaries = [(c, ds_area[c], stem_position[s])
                   for s in stem
                   for c in children.get(s, ())
                   if c in member_set and c not in stem_position]

    digit = {}
    if tributaries:
        roots = np.array([t[0] for t in tributaries])
        digits = _even_odd_digits(np.array([t[1] for t in tributaries], dtype=float),
                                  np.array([t[2] for t in tributaries]))
        selected = {int(r): int(d) for r, d in zip(roots, digits) if d % 2 == 0}
        # the odd digits key off the same junctions the even ones were placed at. a stem reach *at*
        # a confluence is downstream of it, so it belongs to the lower interbasin -- 'left' does that
        cuts = np.sort(np.array([stem_position[parent[r]] for r in selected], dtype=np.int64))
        for reach, position in stem_position.items():
            digit[reach] = 2 * int(np.searchsorted(cuts, position, side='left')) + 1
        digit.update(selected)
    else:
        # a pure chain: no junction to cut at, so the whole basin is interbasin 1 and this level
        # makes no progress. see the module docstring -- expected, and harmless
        for reach in stem_position:
            digit[reach] = 1

    # everything off the stem inherits from the reach it drains into. members are in topological
    # order, so walking them downstream-first guarantees the parent is already resolved
    for reach in members[::-1]:
        if reach not in digit:
            digit[reach] = digit[parent[reach]]
    return np.array([digit[r] for r in members], dtype=np.int8)


def assign_basin_codes(df: pd.DataFrame, levels: list, log=None) -> pd.Series:
    """One digit per level for every reach, as a fixed-width string.

    ``df`` must hold riverId, nextRiverId, outletRiverId, areaM2, DSContArea, lat, lon and be in a
    topological order (upstream before downstream), which is what steps 2 and 3 write. Run it on one
    region at a time: the coastal rule ranks whole terminal watersheds against each other, which is
    a statement about one region's coast.

    The code is carried as a string rather than an integer because it passes 19 digits by level 12,
    so it does not fit int32 and must not become an int64 (see schema.py on why not).
    """
    n = len(df)
    river = df[schema.river_id].to_numpy()
    row_of = pd.Series(np.arange(n, dtype=np.int64), index=river)

    parent = row_of.reindex(df[schema.next_river_id].to_numpy()).to_numpy()
    parent = np.where(np.isnan(parent), -1, np.nan_to_num(parent, nan=-1)).astype(np.int64)
    outlet_row = row_of.reindex(df[schema.last_river_id].to_numpy()).to_numpy().astype(np.int64)
    ds_area = df[schema.tdx_ds_area_field].to_numpy()
    lat = df[schema.lat_field].to_numpy()
    merc_area = _mercator_area(df[schema.area].to_numpy(), lat)
    hilbert = topology.hilbert_index(df[schema.lon_field].to_numpy(), lat)

    flows = parent >= 0
    if (np.arange(n)[flows] >= parent[flows]).any():
        raise ValueError('rows are not in a topological order; assign_basin_codes needs step 2 order')
    order_of_row = np.arange(n, dtype=np.int64)

    children: dict = {}
    for child_row, parent_row in enumerate(parent):
        if parent_row >= 0:
            children.setdefault(int(parent_row), []).append(child_row)

    code = np.full(n, '', dtype=object)
    targets = None
    for level in levels:
        digit = np.zeros(n, dtype=np.int8)
        order = np.argsort(code, kind='stable')
        edges = np.flatnonzero(np.r_[True, code[order][1:] != code[order][:-1], True])
        basins = [np.sort(order[start:end]) for start, end in zip(edges[:-1], edges[1:])]
        basins.sort(key=lambda m: -merc_area[m].sum())
        count = len(basins)
        budget = float('inf') if targets is None else targets[level]

        n_coastal = n_pfaf = n_held = 0
        for members in basins:
            if count >= budget:
                n_held += 1
                continue
            if len(members) == 1:
                continue
            if len(np.unique(outlet_row[members])) > 1:
                digits = _coastal_digits(members, outlet_row, merc_area, hilbert)
                n_coastal += 1
            else:
                digits = _pfafstetter_digits(members, parent, children, ds_area, order_of_row)
                n_pfaf += 1
            grew = len(np.unique(digits)) - 1
            if grew <= 0:
                continue  # a pure chain: no junction to cut at, so do not burn a digit on it
            digit[members] = digits
            count += grew

        code = code + digit.astype(str).astype(object)
        realised = len(np.unique(code))
        if targets is None:
            targets = level_targets(realised, n, list(levels))
        if log is not None:
            log(f'level {level:>2}: {realised:>8,} basins '
                f'(target {"-" if budget == float("inf") else f"{int(budget):,}"})  '
                f'{n_coastal:,} coastal, {n_pfaf:,} pfafstetter, {n_held:,} held back')
    return pd.Series(code, index=df.index)
