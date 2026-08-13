#!/usr/bin/env bash
# Tile the catchments and the pfafstetter basin aggregations: one pmtiles per zoom band per
# geometry per region, joined directly into the single published global file. The inputs are the
# FlatGeobuf pairs step 7 writes - tippecanoe reads a named fgb in parallel, so there is no
# conversion step here.
# Usage: ./tile_catchments.sh [region ...]   (no arguments means every region)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env.sh"
select_regions "$@"

cd "$SCRIPT_DIR" || exit 1

# Only one of these at a time. Two runs share every path they touch - the per-region band files
# and the one global tileset - so an overlap has one run reading a band file while the other is
# writing it, and, worse, one run reaching the global join while the other is still producing the
# bands it is about to join. mkdir is the atomic test-and-set that is actually portable here;
# flock is not on macOS.
LOCK="$SCRATCH_ROOT/pmtiles/.tile_catchments.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "another tile_catchments.sh holds $LOCK - wait for it, or remove the directory if no such" \
         "run exists" >&2
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# The banding comes from 7_pfafstetter_basins.py, the same definition it built these polygon sets
# from, so the tiles and the polygons cannot disagree about what a band is. Edit it there. Read once
# rather than per region: it is the same answer every time. A bash array cannot be exported to the
# xargs children, so it travels as the raw newline-separated "level:minzoom:maxzoom" text and is
# split inside the functions below.
BANDS_RAW="$("$PYTHON" 7_pfafstetter_basins.py --bands)"
export BANDS_RAW

# jobs x threads = cores: tippecanoe genuinely uses its threads on an fgb input, and its RSS is
# small, so the count is core-bound rather than the memory-bound arithmetic the old ogr2ogr
# pipeline needed.
CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)"
TILE_JOBS="${TILE_JOBS:-8}"
export TIPPECANOE_MAX_THREADS="${TIPPECANOE_MAX_THREADS:-$(( CORES / TILE_JOBS > 1 ? CORES / TILE_JOBS : 2 ))}"

# tile-join prints "mismatched maxzooms: N ... vs previous M" every time an input's maxzoom differs
# from the running maximum, then takes the maximum. Every band here has a different maxzoom - that
# is the whole architecture, one tileset per zoom band - so it fires once per band per region in
# the global join and says nothing but that the banding is the banding. Verified on a joined
# region: the output is z0-z11 and every zoom carries both layers with the feature counts the
# banding predicts, so nothing is dropped at a boundary. Dropping the line keeps tile-join's real
# warnings visible.
drop_maxzoom_warning() {
    grep -v 'mismatched maxzooms' || true
}
export -f drop_maxzoom_warning

# The paths a band is known by, given a region and one "level:minzoom:maxzoom" line. Every phase
# below needs the same answer - what to tile, what to join, what to sweep up - so it is derived
# here once instead of being spelled out three times and drifting.
#
# The pmtiles carry the zoom range in their names, and one older than its fgb is rebuilt. Both are
# about the same failure: these files are the expensive half of the pipeline so they are cached,
# and a cache keyed on the band's name alone cannot tell that the band now means different zooms
# or that step 7 rewrote the polygons underneath it. It silently ships the old tiles instead, and
# tile-join stitches them into a tileset whose zooms do not line up with anything.
band_paths() {
    local region="$1" level="$2" minzoom="$3" maxzoom="$4"
    local region_dir="$SCRATCH_ROOT/regions/$region"
    local work="$SCRATCH_ROOT/pmtiles/catchment_levels_$region"
    if [ "$level" = leaf ]; then
        # the leaf band, not the published catchments: step 7 cuts it to what its band's zooms can
        # resolve, the same way it cuts every aggregate level to its own
        BAND_SRC="$region_dir/catchments_tile_${region}.fgb"
        BAND_POLY="$work/leaf.z$minzoom-$maxzoom.pmtiles"
        BAND_LINE="$work/leaf.z$minzoom-$maxzoom.lines.pmtiles"
    else
        BAND_SRC="$region_dir/basin_level${level}_${region}.fgb"
        BAND_POLY="$work/level${level}.z$minzoom-$maxzoom.pmtiles"
        BAND_LINE="$work/level${level}.z$minzoom-$maxzoom.lines.pmtiles"
    fi
    BAND_SRC_LINE="${BAND_SRC%.fgb}.lines.fgb"
}
export -f band_paths

# One queue entry: tile one band's fgb into one pmtiles. xargs -I hands the whole line over as a
# single argument, so the fields are split back apart here. The separator is a pipe rather than
# the obvious tab because BSD xargs rewrites tabs in a -I item to spaces, which silently glues
# every field into one. The item names the band rather than spelling out its paths, and calls
# band_paths to get them back: **BSD xargs refuses a -I item that expands past 255 bytes**, with
# "command line cannot be assembled, too long", and two absolute paths under $SCRATCH_ROOT is
# right at that edge - it went over the day the leaf band's source was renamed, and the run built
# one band of sixteen and then said nothing more about the rest.
tile_one() {
    set -eo pipefail
    local size region level minzoom maxzoom geom src out layer
    IFS='|' read -r size region level minzoom maxzoom geom <<< "$1"

    band_paths "$region" "$level" "$minzoom" "$maxzoom"
    if [ "$geom" = lines ]; then
        src="$BAND_SRC_LINE"
        out="$BAND_LINE"
        layer=catchment_lines
    else
        src="$BAND_SRC"
        out="$BAND_POLY"
        layer=catchments
    fi

    # --simplification=4 is a quarter of a tile pixel, matching the quarter pixel step 7 cut the
    # band's geometry at; 10, which this used to pass, is the top of what the tippecanoe manual
    # calls "probably no visible difference" and it composed with a coarse source into something
    # that was. -Sm 1 then leaves the band's deepest zoom - the one actually being looked at, and
    # the one clients overzoom from - at tippecanoe's own standard tolerance, so only the
    # intermediate zooms of a band pay for being cheap.
    #
    # -pn is the shared-edge flag. The input is an exact coverage, so a boundary between two
    # catchments is the same linework in both of them, and simplifying the two copies independently
    # pulls them apart into slivers. It replaces --detect-shared-borders, which is deprecated
    # ("faster and more correct", per the manual) and which only ever ran on the polygon layer -
    # the line layer, which is the one that is actually drawn, got no shared-edge handling at all.
    #
    # --projection=EPSG:3857 because step 7 writes the fgbs in the pipeline's own web mercator,
    # where every tolerance and snap lattice is defined - the old ogr2ogr stage reprojected to
    # 4326 only for tippecanoe to project straight back to mercator tiles. Without the flag
    # tippecanoe reads meters as degrees, clips nearly everything to the antimeridian, and ships
    # empty 4K tilesets.
    #
    # Written aside and moved into place, because the move is the only atomic step available and
    # this file is a cache entry keyed on being newer than its source. A tippecanoe killed partway
    # leaves a truncated pmtiles that is newer than its fgb, so the next run reads it as a hit
    # and joins a corrupt tileset into the published one. Interrupting this run cost 14 such files.
    # ".partial.pmtiles", not ".partial": both tippecanoe and tile-join pick the output format from
    # the extension, so a temp name that loses it writes MBTiles into a file the next step opens as
    # PMTiles, and the failure surfaces as a magic-number abort in tile-join rather than here.
    local partial="${out%.pmtiles}.partial.pmtiles"
    if ! tippecanoe -o "$partial" -Z"$minzoom" -z"$maxzoom" --layer "$layer" \
            --projection=EPSG:3857 \
            --simplification=4 --simplification-at-maximum-zoom=1 \
            --no-simplification-of-shared-nodes --no-progress-indicator --force "$src"; then
        rm -f "$partial"
        echo "  $(basename "$out"): tiling failed" >&2
        return 1
    fi
    mv -f "$partial" "$out"
    echo "  $(basename "$(dirname "$out")")/$(basename "$out") $(du -h "$out" | cut -f1)"
}
export -f tile_one

# --- what needs building ----------------------------------------------------------------------
# The whole queue is assembled before anything runs, so a missing input is a failure now rather
# than forty minutes in, and so the tasks can be ordered by size. Longest-first: the leaf bands
# are most of the work and starting them last would leave one of them running alone at the end.
# The queue is flat across regions rather than walked region by region - a region's eight bands
# are wildly uneven, and advancing regions in lockstep measured at ~56% of the cores.
TASKS=""
for region in "${REGIONS[@]}"; do
    mkdir -p "$SCRATCH_ROOT/pmtiles/catchment_levels_$region"
    while IFS=: read -r level minzoom maxzoom; do
        [ -n "$level" ] || continue
        band_paths "$region" "$level" "$minzoom" "$maxzoom"
        if [ ! -f "$BAND_SRC" ] || [ ! -f "$BAND_SRC_LINE" ]; then
            echo "  $region $level: $BAND_SRC missing, run 7_pfafstetter_basins.py first" >&2
            exit 1
        fi
        if [ ! -f "$BAND_POLY" ] || [ "$BAND_SRC" -nt "$BAND_POLY" ]; then
            size="$(stat -f%z "$BAND_SRC" 2>/dev/null || stat -c%s "$BAND_SRC")"
            TASKS+="$size|$region|$level|$minzoom|$maxzoom|polygons"$'\n'
        fi
        if [ ! -f "$BAND_LINE" ] || [ "$BAND_SRC_LINE" -nt "$BAND_LINE" ]; then
            size="$(stat -f%z "$BAND_SRC_LINE" 2>/dev/null || stat -c%s "$BAND_SRC_LINE")"
            TASKS+="$size|$region|$level|$minzoom|$maxzoom|lines"$'\n'
        fi
    done <<< "$BANDS_RAW"
done

if [ -n "$TASKS" ]; then
    echo "tiling $(printf '%s' "$TASKS" | wc -l | tr -d ' ') bands across ${#REGIONS[@]} regions," \
         "$TILE_JOBS at a time, $TIPPECANOE_MAX_THREADS tippecanoe threads each"
    printf '%s' "$TASKS" | sort -t'|' -rn -k1,1 | xargs -P "$TILE_JOBS" -I{} bash -c 'tile_one "$@"' _ {}
else
    echo "every band is newer than its fgb, nothing to tile"
fi

# --- per-region housekeeping ------------------------------------------------------------------
# What the current banding derives for a region, collected rather than globbed: with the aggregate
# levels off there are no level* files to match, and a stale one from an earlier run must not
# survive into a scratch directory the current banding does not include it in. A band that is no
# longer in the banding leaves its tiles behind, named for the zooms it used to cover - they are
# never joined, but they are the bulk of the scratch directory, so a run that supersedes them
# takes them with it.
sweep_region() {
    set -eo pipefail
    local region="$1"
    local work="$SCRATCH_ROOT/pmtiles/catchment_levels_$region"

    local -a KEEP=()
    local level minzoom maxzoom
    while IFS=: read -r level minzoom maxzoom; do
        [ -n "$level" ] || continue
        band_paths "$region" "$level" "$minzoom" "$maxzoom"
        KEEP+=("$BAND_POLY" "$BAND_LINE")
    done <<< "$BANDS_RAW"

    local file keep k
    for file in "$work"/*.pmtiles; do
        [ -e "$file" ] || continue
        keep=no
        for k in "${KEEP[@]}"; do
            if [ "$file" = "$k" ]; then keep=yes; break; fi
        done
        if [ "$keep" = no ]; then
            echo "  $(basename "$file"): not in the current banding, removing"
            rm -f "$file"
        fi
    done
}
export -f sweep_region

printf '%s\n' "${REGIONS[@]}" | xargs -P 8 -I{} bash -c 'sweep_region "$@"' _ {}

# --- the published tileset --------------------------------------------------------------------
GLOBAL_CATCHMENT_TILE="$GROUP_ROOT/group=0/catchments.pmtiles"

# The global tileset is named for every band of every region, not globbed. A glob is whatever
# happens to be on disk at the moment it expands, and this file is the published one: a run that
# covers a single region still reaches this join, and with a glob it would quietly republish the
# world from whatever had been tiled by then. That is not a hypothetical - it is how a
# catchments.pmtiles with no Americas in it got published, joined from a partial glob while the
# rest of the regions were still being written. Naming them turns that into the error below.
GLOBAL_INPUTS=()
MISSING_BANDS=()
for region in "${ALL_REGIONS[@]}"; do
    while IFS=: read -r level minzoom maxzoom; do
        [ -n "$level" ] || continue
        band_paths "$region" "$level" "$minzoom" "$maxzoom"
        for band_tile in "$BAND_POLY" "$BAND_LINE"; do
            if [ -f "$band_tile" ]; then
                GLOBAL_INPUTS+=("$band_tile")
            else
                MISSING_BANDS+=("$region/$(basename "$band_tile")")
            fi
        done
    done <<< "$BANDS_RAW"
done
if [ "${#MISSING_BANDS[@]}" -gt 0 ]; then
    echo "not publishing $GLOBAL_CATCHMENT_TILE: ${#MISSING_BANDS[@]} band tilesets missing" \
         "(${MISSING_BANDS[*]:0:8} ...). Tile them first - the global file has to cover every" \
         "region, whatever subset this run rebuilt." >&2
    exit 1
fi

# rebuilt whenever any band tileset is newer than it, for the same reason the bands are: an
# existence check alone would publish the previous banding. Tested with -nt rather than the
# obvious `ls -t | head -1`: 1600 paths overflow the 64KB pipe buffer, head exits after one
# line, ls dies of SIGPIPE, and pipefail turns that into killing the whole script - which is
# how the first full run died with every band built and nothing published.
REBUILD_GLOBAL=no
if [ ! -f "$GLOBAL_CATCHMENT_TILE" ]; then
    REBUILD_GLOBAL=yes
else
    for band_tile in "${GLOBAL_INPUTS[@]}"; do
        if [ "$band_tile" -nt "$GLOBAL_CATCHMENT_TILE" ]; then
            REBUILD_GLOBAL=yes
            break
        fi
    done
fi
if [ "$REBUILD_GLOBAL" = yes ]; then
    LAYER_NAME="River Forecast System v3 Catchments"
    GLOBAL_PARTIAL="${GLOBAL_CATCHMENT_TILE%.pmtiles}.partial.pmtiles"
    tile-join --force --no-tile-size-limit --name "$LAYER_NAME" -o "$GLOBAL_PARTIAL" \
        "${GLOBAL_INPUTS[@]}" 2> >(drop_maxzoom_warning >&2)
    mv -f "$GLOBAL_PARTIAL" "$GLOBAL_CATCHMENT_TILE"
    echo "joined ${#GLOBAL_INPUTS[@]} band tilesets -> $GLOBAL_CATCHMENT_TILE"
fi
