"""
What survived simplification, reconstructed from the record simplify_streams.py leaves behind.

simplify_streams.py rewrites the network: it folds lake interiors into their outlet, dissolves headwaters,
prunes branches and consolidates short reaches. Every one of those edits removes reaches and
gives their drainage to a surviving *keeper* - except the orphaned coastal order-1s and the
zero-length reaches, which are dropped outright - and every one is written out under the region's
``mods/`` directory as it happens. This module replays those files into one question a caller
can ask: **given a source reach id, which published reach now carries its water?**

That question has to be answered at build time rather than written down anywhere. A source id
is stable - it comes from TDX-Hydro and the header arithmetic in 1_translate_tdxhydro.py - but which reach
survives is a property of one particular run of simplify_streams.py, and it moves whenever a threshold does.
Anything that names reaches by source id (``network_data/river_names.csv`` is the one in this
repo) therefore keeps naming the source id and resolves it here, instead of being edited every
time simplification changes its mind.

Three answers are possible, and the caller has to tell them apart:

    the id is published            it survived unchanged; nothing to resolve
    the id resolves to another     it was folded into that keeper, possibly through a chain
    the id resolves to nothing     it was removed outright - a dropped watershed, a sub-250 km2
                                   terminal, a zero-length reach, an orphaned coastal order-1
                                   (coastal_orphans.json) - and no reach carries it now

Only the last is an error, and only the caller knows whether it is a fatal one.

These files are published, under ``region=<id>/mods/`` beside the region's parquet, and that
is the only copy of them. They describe one run of simplify_streams.py, but so does every other published
file, and they are the only record of which source reach a published reach absorbed - a reader
holding the dataset can answer the question below without the build tree that produced it. A
caller still has to cope with them being absent, which is what ``mods_dirs`` reports by
returning nothing: a partly built release has them for some regions and not others.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

__all__ = ['mods_dirs', 'member_to_keeper', 'resolve_terminals', 'survivors', 'catchment_keepers']

# the edit files shaped {keeper: [members folded into keeper, ...]}, in no particular order:
# each reach is removed at most once across every edit, so the union cannot collide.
# coastal_orphans.json has the same shape but is not one: its members are dropped, not folded,
# so they resolve to nothing
MERGE_FILES = (
    'headwater_dissolves.json',
    'branches_to_prune.json',
    'short_consolidations.json',
)


def mods_dirs(publish_root: Path = None) -> list:
    """Every published region's ``mods/`` directory, in region order. Empty when nothing has been
    published yet, which a caller should report rather than read as "nothing was edited"."""
    root = paths.publish_root if publish_root is None else publish_root
    if not root.exists():
        return []
    return [d for d in sorted((p / 'mods' for p in root.iterdir()
                               if p.is_dir() and p.name.startswith('region=')), key=str)
            if d.is_dir()]


def member_to_keeper(dirs: list) -> dict:
    """Combine every region's edit files into one ``{member_id: keeper_id}`` map.

    A member is a reach that was removed and whose drainage now belongs to the keeper. Keepers
    are followed transitively by ``resolve_terminals``, because a keeper of one edit may itself
    be a member of a later one.
    """
    mapping: dict[int, int] = {}
    for mods_dir in dirs:
        lake_edits = mods_dir / 'lake_edits.json'
        if lake_edits.exists():
            # lake interior reaches fold into their lake outlet
            for outlet, edit in json.loads(lake_edits.read_text()).items():
                for deleted in edit.get('delete', []):
                    mapping[int(deleted)] = int(outlet)
        for name in MERGE_FILES:
            path = mods_dir / name
            if not path.exists():
                continue
            for keeper, members in json.loads(path.read_text()).items():
                for member in members:
                    mapping[int(member)] = int(keeper)
    return mapping


def resolve_terminals(mapping: dict) -> dict:
    """Follow each member's keeper chain to the terminal surviving reach.

    Edits only ever fold a reach downstream, so the chains form a DAG and the walk terminates.
    Memoized, so the whole map resolves in one pass over the edges however long the chains are.
    """
    terminal_of: dict[int, int] = {}
    for start in mapping:
        node, path = start, []
        while node in mapping and node not in terminal_of:
            path.append(node)
            node = mapping[node]
        terminal = terminal_of.get(node, node)
        for reach in path:
            terminal_of[reach] = terminal
    return terminal_of


def survivors(publish_root: Path = None) -> dict:
    """``{removed_id: surviving_id}`` for every reach simplify_streams.py folded into a keeper.

    Ids absent from the result were either never touched or removed outright; the caller
    distinguishes those two by checking the published network.
    """
    return resolve_terminals(member_to_keeper(mods_dirs(publish_root)))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def catchment_keepers(source_ids: np.ndarray, mods_dir: Path) -> tuple[np.ndarray, set]:
    """Which published reach each raw TDX-Hydro catchment is dissolved into, for one region.

    Returns the keeper id per entry of ``source_ids`` and the set of ids deleted outright (the
    zero-length reaches), whose catchments belong to no reach. A raw id that was never touched
    is its own keeper. This is the grouping create_catchments.py dissolves the catchment polygons by, and
    anything summarised per published catchment (the Koppen class in simplify_streams.py) must use the same
    one or it will describe a different polygon than the one published.

    coastal_orphans.json is left out, as in ``member_to_keeper``: those reaches are dropped
    with their catchments and their area, not folded into a neighbour.
    """
    redirect: dict = {}
    deleted: set = set()

    for outlet, edit in _load_json(mods_dir / 'lake_edits.json').items():
        outlet = int(outlet)
        for d in edit.get('delete', []):
            redirect[int(d)] = outlet

    zero_lengths = _load_json(mods_dir / 'zero_length_streams.json')
    for case in ('case1', 'case2', 'case3'):
        deleted.update(int(i) for i in zero_lengths.get(case, {}).get('ids', []))

    for fname in ('headwater_dissolves.json', 'branches_to_prune.json', 'short_consolidations.json'):
        for keeper, members in _load_json(mods_dir / fname).items():
            keeper = int(keeper)
            for member in members:
                redirect[int(member)] = keeper

    def resolve(rid: int) -> int:
        seen = set()
        while rid in redirect and rid not in seen:
            seen.add(rid)
            rid = redirect[rid]
        return rid

    keeper_map = {rid: resolve(int(rid)) for rid in np.unique(source_ids)}
    return pd.Series(source_ids).map(keeper_map).to_numpy(np.int64), deleted
