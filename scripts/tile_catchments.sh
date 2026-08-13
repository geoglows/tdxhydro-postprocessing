#!/usr/bin/env bash
# Tile the catchments and the pfafstetter basin aggregations: one pmtiles per zoom band per region,
# joined per region and then globally.
# Usage: ./tile_catchments.sh [region ...]   (no arguments means every region)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env.sh"
select_regions "$@"

cd "$SCRIPT_DIR" || exit 1

# Only one of these at a time. Two runs share every path they touch - the per-region band files,
# the per-region joins, and the one global tileset - so an overlap has one run reading a band file
# while the other is writing it, and, worse, one run reaching the global join while the other is
# still producing the regions it is about to join. mkdir is the atomic test-and-set that is actually
# portable here; flock is not on macOS.
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
BANDS_RAW="$("$PYTHON" 8_pfafstetter_basins.py --bands)"
export BANDS_RAW

# Every band of every region is an independent ogr2ogr|tippecanoe, so they all go in one queue
# rather than being walked region by region. A region has eight bands and they are wildly uneven -
# the leaf band is most of the work and the seven aggregate levels are a rounding error - so
# advancing eight regions in lockstep leaves most of the machine idle waiting on whichever region
# is still in its leaf band. Measured on this workload it sat at ~56% of the cores.
#
# TILE_JOBS is what the queue actually runs at once. A task is not worth a whole core - it is an
# ogr2ogr feeding a tippecanoe through a pipe, and the pipe means neither runs flat out - so the
# count has to overshoot the core count to fill the machine. Measured over a full run on 16 cores:
# six jobs held ~780% of 1600, nine holds ~1400%. It is bounded above by memory rather than by
# cores, because ogr2ogr peaks near 9GB on the largest leaf parquet: nine in flight touched 52GB of
# the 64GB here without swapping, and twelve would not fit.
CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)"
TILE_JOBS="${TILE_JOBS:-9}"
export TIPPECANOE_MAX_THREADS="${TIPPECANOE_MAX_THREADS:-$(( CORES / TILE_JOBS > 1 ? CORES / TILE_JOBS : 2 ))}"
JOIN_JOBS="${JOIN_JOBS:-8}"

# tile-join prints "mismatched maxzooms: N ... vs previous M" every time an input's maxzoom differs
# from the running maximum, then takes the maximum. Every band here has a different maxzoom - that
# is the whole architecture, one tileset per zoom band - so it fires once per band per region and
# says nothing but that the banding is the banding. Verified on a joined region: the output is
# z0-z11 and every zoom carries both layers with the feature counts the banding predicts, so
# nothing is dropped at a boundary. Dropping the line keeps tile-join's real warnings visible.
drop_maxzoom_warning() {
    grep -v 'mismatched maxzooms' || true
}
export -f drop_maxzoom_warning

# The paths a band is known by, given a region and one "level:minzoom:maxzoom" line. Every phase
# below needs the same answer - what to build, what to join, what to sweep up - so it is derived
# here once instead of being spelled out three times and drifting.
#
# The zoom range is in the name, and a tile older than its parquet is rebuilt. Both are about the
# same failure: these files are the expensive half of the pipeline so they are cached, and a cache
# keyed on the band's name alone cannot tell that the band now means different zooms or that step 8
# rewrote the polygons underneath it. It silently ships the old tiles instead, and tile-join
# stitches them into a tileset whose zooms do not line up with anything.
band_paths() {
    local region="$1" level="$2" minzoom="$3" maxzoom="$4"
    local region_dir="$SCRATCH_ROOT/regions/$region"
    local work="$SCRATCH_ROOT/pmtiles/catchment_levels_$region"
    if [ "$level" = leaf ]; then
        # the leaf band, not the published catchments: step 8 cuts it to what z11 can resolve, the
        # same way it cuts every aggregate level to its own band's zooms. The published file carries
        # the source DEM's resolution and is several times larger - ogr2ogr already peaks near 9GB
        # on this one, which is the memory bound TILE_JOBS above is set against.
        BAND_SRC="$region_dir/catchments_tile_${region}.geo.parquet"
        BAND_POLY="$work/leaf.z$minzoom-$maxzoom.pmtiles"
        BAND_LINE="$work/leaf.z$minzoom-$maxzoom.lines.pmtiles"
    else
        BAND_SRC="$region_dir/basin_level${level}_${region}.geo.parquet"
        BAND_POLY="$work/level${level}.z$minzoom-$maxzoom.pmtiles"
        BAND_LINE="$work/level${level}.z$minzoom-$maxzoom.lines.pmtiles"
    fi
}
export -f band_paths

# One queue entry: serialize one source to GeoJSON and tile it. xargs -I hands the whole line over
# as a single argument, so the fields are split back apart here. The separator is a pipe rather
# than the obvious tab because BSD xargs rewrites tabs in a -I item to spaces, which silently glues
# every field into one and leaves ogr2ogr printing its usage.
#
# The item names the band rather than spelling out its paths, and calls band_paths to get them back.
# That is not only about not repeating the derivation: **BSD xargs refuses a -I item that expands
# past 255 bytes**, with "command line cannot be assembled, too long", and two absolute paths under
# $SCRATCH_ROOT plus a layer name is right at that edge - it went over the day the leaf band's source
# was renamed from catchments_* to catchments_tile_*, and the run built one band of sixteen and then
# said nothing more about the rest. Naming the band costs about forty bytes and cannot drift there.
#
# --read-parallel splits the newline-delimited input across threads. It is the only flag here that
# touches the parse, and the parse is the wall clock on the leaf band. -nlt MULTILINESTRING hands
# ogr2ogr's boundary of each polygon to tippecanoe, so the strokes come from a line layer that
# clipping cannot fake an edge into.
tile_one() {
    set -eo pipefail
    local size region level minzoom maxzoom geom src out layer
    IFS='|' read -r size region level minzoom maxzoom geom <<< "$1"

    band_paths "$region" "$level" "$minzoom" "$maxzoom"
    src="$BAND_SRC"
    if [ "$geom" = lines ]; then
        out="$BAND_LINE"
        layer=catchment_lines
    else
        out="$BAND_POLY"
        layer=catchments
    fi

    # --simplification=4 is a quarter of a tile pixel, matching the quarter pixel step 8 cut the
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
    local -a ogr=(-f GeoJSONSeq -t_srs EPSG:4326 -lco COORDINATE_PRECISION=5)
    local -a tip=(--layer "$layer" --read-parallel
                  --simplification=4 --simplification-at-maximum-zoom=1
                  --no-simplification-of-shared-nodes --no-progress-indicator --force)
    if [ "$geom" = lines ]; then
        ogr+=(-nlt MULTILINESTRING)
    fi

    # Written aside and moved into place, because the move is the only atomic step available and
    # this file is a cache entry keyed on being newer than its source. A tippecanoe killed partway
    # leaves a truncated pmtiles that is newer than its parquet, so the next run reads it as a hit
    # and joins a corrupt tileset into the published one. Interrupting this run cost 14 such files.
    # ".partial.pmtiles", not ".partial": both tippecanoe and tile-join pick the output format from
    # the extension, so a temp name that loses it writes MBTiles into a file the next step opens as
    # PMTiles, and the failure surfaces as a magic-number abort in tile-join rather than here.
    local partial="${out%.pmtiles}.partial.pmtiles"
    if ! ogr2ogr "${ogr[@]}" /vsistdout/ "$src" \
        | tippecanoe -o "$partial" -Z"$minzoom" -z"$maxzoom" "${tip[@]}"; then
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
# than forty minutes in, and so the tasks can be ordered by size. Longest-first: the leaf bands are
# most of the work and starting them last would leave one of them running alone at the end.
TASKS=""
for region in "${REGIONS[@]}"; do
    mkdir -p "$SCRATCH_ROOT/pmtiles/catchment_levels_$region"
    while IFS=: read -r level minzoom maxzoom; do
        [ -n "$level" ] || continue
        band_paths "$region" "$level" "$minzoom" "$maxzoom"
        if [ ! -f "$BAND_SRC" ]; then
            echo "  $region $level: $BAND_SRC missing, run 4_create_catchments.py and" \
                 "8_pfafstetter_basins.py first" >&2
            exit 1
        fi
        size="$(stat -f%z "$BAND_SRC" 2>/dev/null || stat -c%s "$BAND_SRC")"
        if [ ! -f "$BAND_POLY" ] || [ "$BAND_SRC" -nt "$BAND_POLY" ]; then
            TASKS+="$size|$region|$level|$minzoom|$maxzoom|polygons"$'\n'
        fi
        if [ ! -f "$BAND_LINE" ] || [ "$BAND_SRC" -nt "$BAND_LINE" ]; then
            TASKS+="$size|$region|$level|$minzoom|$maxzoom|lines"$'\n'
        fi
    done <<< "$BANDS_RAW"
done

if [ -n "$TASKS" ]; then
    echo "tiling $(printf '%s' "$TASKS" | wc -l | tr -d ' ') bands across ${#REGIONS[@]} regions," \
         "$TILE_JOBS at a time, $TIPPECANOE_MAX_THREADS tippecanoe threads each"
    printf '%s' "$TASKS" | sort -t'|' -rn -k1,1 | xargs -P "$TILE_JOBS" -I{} bash -c 'tile_one "$@"' _ {}
else
    echo "every band is newer than its parquet, nothing to tile"
fi

# --- per-region joins -------------------------------------------------------------------------
join_region() {
    set -eo pipefail
    local region="$1"
    local work="$SCRATCH_ROOT/pmtiles/catchment_levels_$region"
    local out="$SCRATCH_ROOT/pmtiles/catchments_${region}.pmtiles"

    # what tile-join is handed, collected from the banding rather than globbed out of $work: with
    # the aggregate levels off there are no level*.pmtiles to match, and a stale one from an
    # earlier run must not be joined into a tileset the current banding does not include it in
    local -a JOIN_INPUTS=()
    local level minzoom maxzoom
    while IFS=: read -r level minzoom maxzoom; do
        [ -n "$level" ] || continue
        band_paths "$region" "$level" "$minzoom" "$maxzoom"
        JOIN_INPUTS+=("$BAND_POLY" "$BAND_LINE")
    done <<< "$BANDS_RAW"

    # a band that is no longer in the banding leaves its tiles behind, named for the zooms it used
    # to cover. They are never joined - JOIN_INPUTS is collected, not globbed - but they are the
    # bulk of the scratch directory, so a run that supersedes them takes them with it
    local file keep k
    for file in "$work"/*.pmtiles; do
        [ -e "$file" ] || continue
        keep=no
        for k in "${JOIN_INPUTS[@]}"; do
            if [ "$file" = "$k" ]; then keep=yes; break; fi
        done
        if [ "$keep" = no ]; then
            echo "  $(basename "$file"): not in the current banding, removing"
            rm -f "$file"
        fi
    done

    # A band the queue was supposed to produce and did not - said here rather than letting ls below
    # fail on it, because this is the shape the failure actually takes: the tiling phase reports the
    # band that failed and then this is the region that cannot be assembled from what is left.
    local missing=()
    for k in "${JOIN_INPUTS[@]}"; do
        [ -f "$k" ] || missing+=("$(basename "$k")")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        echo "region $region: not joined, ${#missing[@]} bands missing (${missing[*]})" >&2
        return 1
    fi

    # only when a band moved under it - the joins are cheap next to the tiling but there are fifty
    # of them, and a rerun that rebuilt one region should not rewrite the other forty-nine
    local newest
    newest="$(ls -t "${JOIN_INPUTS[@]}" | head -1)"
    if [ -f "$out" ] && [ ! "$newest" -nt "$out" ]; then
        return 0
    fi
    tile-join --force --no-tile-size-limit \
        --name "River Forecast System v3 Catchments $region" \
        -o "${out%.pmtiles}.partial.pmtiles" "${JOIN_INPUTS[@]}" 2> >(drop_maxzoom_warning >&2)
    mv -f "${out%.pmtiles}.partial.pmtiles" "$out"
    echo "region $region: joined -> $out ($(du -h "$out" | cut -f1))"
}
export -f join_region

printf '%s\n' "${REGIONS[@]}" | xargs -P "$JOIN_JOBS" -I{} bash -c 'join_region "$@"' _ {}

# --- the published tileset --------------------------------------------------------------------
GLOBAL_CATCHMENT_TILE="$GROUP_ROOT/group=0/catchments.pmtiles"

# The global tileset is named for every region, not globbed. A glob is whatever happens to be on
# disk at the moment it expands, and this file is the published one: a run that covers a single
# region still reaches this join, and with a glob it would quietly republish the world from the
# regions that had been tiled by then. That is not a hypothetical - it is how a catchments.pmtiles
# with no Americas in it got published, joined from a partial glob while the rest of the regions
# were still being written. Naming them turns that into the error below.
GLOBAL_INPUTS=()
MISSING_REGIONS=()
for region in "${ALL_REGIONS[@]}"; do
    region_tile="$SCRATCH_ROOT/pmtiles/catchments_${region}.pmtiles"
    if [ -f "$region_tile" ]; then
        GLOBAL_INPUTS+=("$region_tile")
    else
        MISSING_REGIONS+=("$region")
    fi
done
if [ "${#MISSING_REGIONS[@]}" -gt 0 ]; then
    echo "not publishing $GLOBAL_CATCHMENT_TILE: ${#MISSING_REGIONS[@]} of ${#ALL_REGIONS[@]}" \
         "regions have no tileset (${MISSING_REGIONS[*]}). Tile them first - the global file has to" \
         "cover every region, whatever subset this run rebuilt." >&2
    exit 1
fi

# rebuilt whenever any region's tileset is newer than it, for the same reason the bands are: an
# existence check alone would publish the previous banding
NEWEST_REGION_TILE="$(ls -t "${GLOBAL_INPUTS[@]}" | head -1)"
if [ ! -f "$GLOBAL_CATCHMENT_TILE" ] || [ "$NEWEST_REGION_TILE" -nt "$GLOBAL_CATCHMENT_TILE" ]; then
    LAYER_NAME="River Forecast System v3 Catchments"
    GLOBAL_PARTIAL="${GLOBAL_CATCHMENT_TILE%.pmtiles}.partial.pmtiles"
    tile-join --force --no-tile-size-limit --name "$LAYER_NAME" -o "$GLOBAL_PARTIAL" \
        "${GLOBAL_INPUTS[@]}" 2> >(drop_maxzoom_warning >&2)
    mv -f "$GLOBAL_PARTIAL" "$GLOBAL_CATCHMENT_TILE"
    echo "joined ${#GLOBAL_INPUTS[@]} region tilesets -> $GLOBAL_CATCHMENT_TILE"
fi
