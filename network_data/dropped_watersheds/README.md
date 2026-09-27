# dropped_watersheds

Watersheds excluded from the published network, keyed on the **raw** `outletRiverId` —
the id as it exists before `recompute_outlets` rewrites some of them, which is why
`3_simplify_streams.py` applies these lists immediately after `compute_topology`.

## Layout

    <level-2 region>/<reason>.csv      one column, header `drop`, ascending

One directory per HydroBASINS level-2 region, because a raw outletRiverId can only ever
name a watershed in one of them: the id is `LINKNO + header * 1e7`, and each of the 62
TDX headers belongs to exactly one level-2 region. Step 3 globs only its own region's
directory. A region with nothing dropped has no directory.

**Each id is listed at most once per region.** Deleting an id from the list that holds it
is then enough to bring the watershed back; step 3 refuses to run if an id is repeated.
Where two reasons both applied, the id was kept in the more specific one, in the order
manually_excluded, island_table, *_no_runoff, noncoastal, small_ocean — so regenerating
a list from its rule has to subtract the ids already listed elsewhere in the region.

These were six flat global files until the split. The flat form
made a region read 204k ids to use a few thousand, and gave no way to see what had been
decided for a region without filtering every file by id prefix — which is how region
1020027430 came to be dropped from `pipeline_env.sh` wholesale instead of by list.

## Reasons

| file | what it is |
|---|---|
| `small_ocean.csv` | tiny coastal watersheds, global policy |
| `island_table.csv` | islands |
| `sahara_no_runoff.csv` | hyper-arid endorheic, hourly-FDC Q80 < 5 m³/s or no v2 data, trimmed to the main contiguous blob |
| `gobi_no_runoff.csv` | same criterion, Gobi |
| `australia_no_runoff.csv` | same criterion, Australian interior incl. Kati Thanda–Lake Eyre |
| `noncoastal.csv` | outlet more than 2 km from the Natural Earth 10m Africa continent boundary |
| `manually_excluded.csv` | hand decisions, no shared criterion |

`gobi_no_runoff.csv` is currently inert: region 4020050290 is not in `REGIONS`.

## Hand decisions that a regeneration would revert

`noncoastal.csv` for region 1020027430 is generated from a geometric rule, with four
exceptions kept coastal by hand. Each reaches the sea but ends in a coastal sabkha a few
km past the line, and each is far too large to lose. Loosening the rule to 10 km instead
would sweep in fourteen more sabkhas nobody has looked at.

| outletRiverId | km² | outlet | where |
|---|---|---|---|
| 150889372 | 50,584 | 6.1 km out | 16.00E 31.23N, Gulf of Sirte |
| 151014345 | 32,556 | 4.7 km out | 15.61E 31.45N, Gulf of Sirte |
| 150811921 | 31,036 | 4.6 km out | 15.31E 32.10N, Gulf of Sirte |
| 150227960 | 29,488 | 8.7 km out | 16.39W 16.51N, Mauritanian shore |

The first and last of those were also lifted from `1020027430/sahara_no_runoff.csv`,
which had dropped them on Q80 < 5. That criterion knowingly sweeps basins that flood
rarely rather than never — 150889372 peaks at 633 m³/s in the v2 retrospective — so the
two decisions genuinely disagreed, and reaching the coast won.

Measuring the distance needs one thing the source does not advertise: TDX linestrings run
**downstream → upstream**, so an outlet is vertex 0, not the last vertex (`streams.py:91`
relies on the same). Measuring the wrong end puts the Medjerda, Chelif and Oum Er-Rbia
7–10 km inland and drops every major river in the Maghreb.
