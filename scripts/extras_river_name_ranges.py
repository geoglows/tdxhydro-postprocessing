"""
Turn network_data/river_names.csv into the lookup the hydrography explorer paints from.

The naming rule the CSV encodes is: work each watershed from its mouth outward,
painting every name onto every reach upstream of its riverId and letting the later,
smaller rows overwrite the earlier ones. So the name on a reach is the one belonging
to the *smallest* named span containing it, and it reads as "the name of the exact
segment you clicked, or you are in an unnamed tributary of that name". Nothing here
reads the row order - the span sizes decide it - so the CSV's own sort is for the
person editing it, not for this script.

The CSV is the source of truth and this script primarily restructures it. Everything
descriptive - the name, the watershed, the country, the parent river, the bounding box -
is read from the CSV and republished verbatim; see extras_river_name_enrich.py, which is
what computes those columns into it. Only two things are worked out here, and both are
properties of the published network rather than of the river: the riverIndex spans, and
the colours.

Those descriptive columns are here because a name plus a watershed does not identify a
river. Ten names in the table are duplicated and three collide on the watershed name as
well, so a client listing search results has nothing to tell two rows apart with. A
country, the river it flows into, and an extent to frame are the smallest set that does.

Everything upstream of a reach is one contiguous run of riverIndex, and riverIndex
is unique across the whole published network, so a name is just an interval
[riverIndex - upstreamCount, riverIndex] on a single global axis. Named spans nest
or miss entirely - never partially overlap - which is what makes the overwrite rule
well defined, and is asserted here rather than assumed.

That lets the whole global assignment flatten to one sorted list of disjoint
intervals, which the app hands to MapLibre as a single `step` expression. A step
over ~800 boundaries is a binary search per feature; a `case` over 456 named spans
would have been a linear scan.

Colours are resolved here too, because which colour a river gets depends on the
whole arrangement, not on the river. Two spans that touch must not share one, so
this greedy-colours the adjacency graph (neighbours along the riverIndex axis, plus
the span each one nests inside, since a child is drawn over its parent). Three
colours suffice and the palette holds six, and the room that leaves is spent on two
things: the pairs the palette cannot separate never land on rivers that touch, and
each name takes the least-used colour still open to it, so the map does not come out
lopsided.

Publishes one JSON beside the global pmtiles, in group=0, because that is where the
explorer reads it from - the same origin and the same release as the tiles whose
riverIndex values it is written against. Bundling it into the app instead would let
the two drift apart silently: the spans would keep pointing at reach numbers the
published network no longer uses.

Cheap enough to rerun whenever a name is added or the network is rebuilt, and it
always overwrites - unlike the numbered pipeline steps, there is no output check
here, because being out of date is the only way this file can be wrong. Reads the
published group metadata, so run it after 5_concatenate_global.py.
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

REPO_ROOT = Path(__file__).resolve().parent.parent
NAMES_CSV = REPO_ROOT / 'network_data' / 'river_names.csv'
DEFAULT_OUT = hy.paths.global_root / 'riverNames.json'

# Four of a six-colour set specified by hand. The other two are spoken for: the blue is the
# rivers' own colour, which is what unnamed water keeps below, and the purple belongs to the
# group boundaries - so neither can also mean "a named river".
PALETTE = ['#FBBF24', '#34D399', '#FCA5A5', '#E5E7EB']
# Unnamed reaches keep exactly what the network already draws itself in, at its own width.
# Named water is the only thing the mode adds, which is what makes it read.
UNNAMED = '#3B82F6'

# One of the six pairs the palette can make is one the checker cannot separate: green and the
# salmon collapse under deuteranopia (dE 4.3). Rather than drop a colour, the two are simply
# never placed next to each other - the adjacency graph needs three colours and has four, so
# there is room to refuse a pairing. They can still appear on the same screen; they just
# cannot appear on two rivers that touch, which is where it would read as one river changing
# name.
CONFLICTS = [('#34D399', '#FCA5A5')]


def load_spans() -> pd.DataFrame:
    """Each named river as a half-open span on the global riverIndex axis."""
    names = pd.read_csv(NAMES_CSV)
    # The global concatenation rather than the 127 per-group files it is made of: this script
    # already has to run after 5_concatenate_global.py, and reading one file is both the same
    # rows and the same order that extras_river_name_enrich.py measured the bounding boxes on.
    meta = pd.read_parquet(
        hy.paths.global_root / 'metadata.parquet',
        columns=[hy.schema.river_id, hy.schema.river_index, hy.schema.upstream_count],
    ).set_index(hy.schema.river_id)

    for column in ('country', 'parentRiverId', 'bboxWest', 'bboxSouth', 'bboxEast', 'bboxNorth'):
        if column not in names.columns:
            raise SystemExit(f'{NAMES_CSV.name} has no {column} column; '
                             f'run extras_river_name_enrich.py to compute it')
    df = names.join(meta, on='riverId')
    missing = df[df[hy.schema.river_index].isna()]
    if len(missing):
        raise SystemExit(f'{len(missing)} rows in {NAMES_CSV.name} are not in the published network:\n'
                         f'{missing[["riverName", "riverId"]].to_string(index=False)}')
    df['lo'] = (df[hy.schema.river_index] - df[hy.schema.upstream_count]).astype(int)
    df['hi'] = df[hy.schema.river_index].astype(int)
    return df


def flatten(spans: list[dict]) -> list[tuple[int, int, int | None]]:
    """Disjoint intervals over riverIndex, each tagged with the row that wins on it.

    The winner on an interval is the smallest span containing it, which is exactly
    what "later rows overwrite earlier ones" comes to once the rows are sorted by
    area. Runs that agree and touch are merged, so the step expression carries a
    boundary only where the answer actually changes.
    """
    smallest_first = sorted(range(len(spans)), key=lambda i: spans[i]['hi'] - spans[i]['lo'])
    edges = sorted({e for s in spans for e in (s['lo'], s['hi'] + 1)})
    out: list[tuple[int, int, int | None]] = []
    for lo, end in zip(edges, edges[1:]):
        hi = end - 1
        win = next((i for i in smallest_first if spans[i]['lo'] <= lo and hi <= spans[i]['hi']), None)
        if out and out[-1][2] == win and out[-1][1] + 1 == lo:
            out[-1] = (out[-1][0], hi, win)
        else:
            out.append((lo, hi, win))
    return out


def check_nesting(spans: list[dict]) -> None:
    """A partial overlap would make the winner on the overlap depend on the order the
    rows happen to be in, so refuse to emit a lookup that has one."""
    smallest_first = sorted(range(len(spans)), key=lambda i: spans[i]['hi'] - spans[i]['lo'])
    for pos, i in enumerate(smallest_first):
        a = spans[i]
        for j in smallest_first[pos + 1:]:
            b = spans[j]
            if not (b['lo'] <= a['lo'] and a['hi'] <= b['hi']) and not (a['hi'] < b['lo'] or a['lo'] > b['hi']):
                raise SystemExit(f'{a["name"]} [{a["lo"]},{a["hi"]}] partially overlaps '
                                 f'{b["name"]} [{b["lo"]},{b["hi"]}]')


def colorize(spans: list[dict], flat: list[tuple[int, int, int | None]]) -> list[int]:
    """One palette slot per named river, differing from everything it touches."""
    adj: dict[int, set[int]] = {i: set() for i in range(len(spans))}
    for (_, _, a), (_, _, b) in zip(flat, flat[1:]):          # neighbours along the axis
        if a is not None and b is not None and a != b:
            adj[a].add(b)
            adj[b].add(a)
    smallest_first = sorted(range(len(spans)), key=lambda i: spans[i]['hi'] - spans[i]['lo'])
    for i, a in enumerate(spans):                              # and the span each nests in
        parent = next((k for k in smallest_first
                       if k != i and spans[k]['lo'] <= a['lo'] and a['hi'] <= spans[k]['hi']), None)
        if parent is not None:
            adj[i].add(parent)
            adj[parent].add(i)

    clash: dict[int, set[int]] = {c: set() for c in range(len(PALETTE))}
    for a, b in CONFLICTS:
        i, j = PALETTE.index(a), PALETTE.index(b)
        clash[i].add(j)
        clash[j].add(i)

    used = Counter({c: 0 for c in range(len(PALETTE))})
    color: dict[int, int] = {}
    for i in sorted(adj, key=lambda i: -len(adj[i])):           # hardest-constrained first
        taken = {color[n] for n in adj[i] if n in color}
        banned = set(taken).union(*(clash[c] for c in taken)) if taken else set()
        free = [c for c in range(len(PALETTE)) if c not in banned]
        if not free:
            # Never seen with this palette, but a denser arrangement could exhaust it. Keeping the
            # spans distinguishable at all beats keeping every neighbour pair well separated.
            free = [c for c in range(len(PALETTE)) if c not in taken]
        if not free:
            raise SystemExit(f'{spans[i]["name"]} touches all {len(PALETTE)} palette colours; '
                             f'the palette needs another entry')
        pick = min(free, key=lambda c: used[c])                 # keep the map from going lopsided
        color[i] = pick
        used[pick] += 1
    return [color[i] for i in range(len(spans))]


def resolve_parents(spans: list[dict], parent_ids: list) -> list[int | None]:
    """The CSV's parentRiverId as a position in `rivers`, checked against the spans.

    The parent is in the CSV because it is descriptive - it is what turns one of three
    rivers called Verde into "the Verde that flows into the Salt" - but it is also a fact
    about the network, so a rebuild can move a confluence and leave the column behind. The
    containment it claims is checkable here for nothing, so it is checked: a parent that no
    longer contains its child is a stale CSV, and shipping it would put a river under the
    wrong one silently.
    """
    position = {s['riverId']: i for i, s in enumerate(spans)}
    out: list[int | None] = []
    for i, (span, parent_id) in enumerate(zip(spans, parent_ids)):
        if pd.isna(parent_id):
            out.append(None)
            continue
        j = position.get(int(parent_id))
        if j is None:
            raise SystemExit(f'{span["name"]} names parent riverId {int(parent_id)}, '
                             f'which is not a named river')
        parent = spans[j]
        if not (parent['lo'] <= span['lo'] and span['hi'] <= parent['hi']):
            raise SystemExit(f'{span["name"]} [{span["lo"]},{span["hi"]}] is not inside its stated '
                             f'parent {parent["name"]} [{parent["lo"]},{parent["hi"]}]; rerun '
                             f'extras_river_name_enrich.py --overwrite against this network')
        out.append(j)
    return out


def main(out_path: Path) -> None:
    df = load_spans()
    spans = [dict(name=r.riverName, riverId=int(r.riverId), outletRiverId=int(r.outletRiverId),
                  watershed=r.watershedName, country=None if pd.isna(r.country) else r.country,
                  bbox=[round(float(v), 5) for v in (r.bboxWest, r.bboxSouth, r.bboxEast, r.bboxNorth)],
                  lo=int(r.lo), hi=int(r.hi))
             for r in df.itertuples()]
    check_nesting(spans)
    parents = resolve_parents(spans, list(df.parentRiverId))
    flat = flatten(spans)
    colors = colorize(spans, flat)

    # ['step', input, first, boundary, next, ...]: the value from `stops[k]` holds until
    # `bounds[k]`. -1 is unnamed. The app pairs these up; keeping them as two flat arrays
    # of numbers is what keeps the file small.
    bounds = [lo for lo, _, _ in flat[1:]]
    stops = [-1 if win is None else colors[win] for _, _, win in flat]

    named_reaches = sum(hi - lo + 1 for lo, hi, win in flat if win is not None)
    payload = {
        # Stamped so a client holding a cached copy can tell one release from another without
        # refetching the body. There is no schedule behind this file - names are added and
        # corrected as edits accumulate - so a consumer revalidates on a clock of its own and
        # compares this, rather than knowing when to expect a change.
        'generatedAt': datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'palette': PALETTE,
        'unnamed': UNNAMED,
        'namedReaches': named_reaches,
        'rivers': [dict(s, parent=p, color=c) for s, p, c in zip(spans, parents, colors)],
        'first': stops[0],
        'bounds': bounds,
        'stops': stops[1:],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(',', ':')))

    total = sum(hi - lo + 1 for lo, hi, _ in flat)
    print(f'{len(spans)} rivers in {df.outletRiverId.nunique()} watersheds, '
          f'{sum(p is not None for p in parents)} of them named tributaries of another')
    print(f'{df.country.nunique()} countries')
    print(f'{len(flat)} disjoint intervals, {len(bounds)} step boundaries')
    print(f'{named_reaches:,} named reaches of {total:,} spanned '
          f'({100 * named_reaches / total:.1f}% of the range the names cover)')
    print(f'colour usage: {Counter(colors).most_common()}')
    print(f'{out_path} ({out_path.stat().st_size / 1024:.1f} kB)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT, help=f'default: {DEFAULT_OUT}')
    main(ap.parse_args().out)
