#!/usr/bin/env bash
# Tile the catchment map: one pmtiles per zoom band, joined into the single published
# group=0/catchments.pmtiles with the same two layers the map has always drawn - `catchments`
# (polygons) and `catchment_lines` (boundaries).
#
# The feed is split by cadence. The basin bands (levels 3-8, z0-9) are GLOBAL fgb pairs written by
# 6_publish_basins.py from the frozen basins stamped with this release's ids - attributes embedded,
# excluded regions absent - so they are tiled here as six whole-planet tippecanoe runs, not per
# region. Level 2 contributes a lines-only band: the region divides ride in catchment_lines at
# every zoom, but region polygons are not a drawable band. The leaf band (z10) is per-region,
# written by 5_concatenate_global.py beside the region catchments, tiled per region and joined
# with everything else at the end.
# Usage: ./tile_catchments.sh [region ...]   (no arguments means every region's leaf band)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env.sh"
pipeline_banner "Tile catchment vectors"
select_regions "$@"

cd "$SCRIPT_DIR" || exit 1

# Only one of these at a time. Two runs share every path they touch - the band tilesets and the
# one global join - so an overlap has one run joining while the other is still tiling. mkdir is
# the atomic test-and-set that is actually portable here; flock is not on macOS.
LOCK="$SCRATCH_ROOT/pmtiles/.tile_catchments.lock"
mkdir -p "$SCRATCH_ROOT/pmtiles"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "another tile_catchments.sh holds $LOCK - wait for it, or remove the directory if no such" \
         "run exists" >&2
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# The banding comes from the shared definition in hydrography/basins.py, printed by step 2, so the
# tiles and the polygons cannot disagree about what a band is. Read once: same answer every time.
BANDS_RAW="$("$PYTHON" 2_global_basins.py --bands)"
export BANDS_RAW
# the level-2 lines band spans the full pyramid, z0 to the leaf's deepest zoom
LEAF_MAXZOOM="$(printf '%s\n' "$BANDS_RAW" | awk -F: '$1 == "leaf" {print $3}')"
export LEAF_MAXZOOM

# jobs x threads = cores: tippecanoe genuinely uses its threads on an fgb input, and its RSS is
# small, so the count is core-bound.
CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)"
TILE_JOBS="${TILE_JOBS:-8}"
export TIPPECANOE_MAX_THREADS="${TIPPECANOE_MAX_THREADS:-$(( CORES / TILE_JOBS > 1 ? CORES / TILE_JOBS : 2 ))}"

# tile-join warns about every input whose maxzoom differs from the running maximum - one tileset
# per zoom band is the whole architecture, so it fires once per band and says nothing. Dropping
# the line keeps tile-join's real warnings visible.
drop_maxzoom_warning() {
    grep -v 'mismatched maxzooms' || true
}
export -f drop_maxzoom_warning

# The paths a band is known by, given "<region>|<level>|<minzoom>|<maxzoom>". Region is "-" for
# the global basin bands. The pmtiles carry the zoom range in their names so a band whose zooms
# change cannot be shipped stale from a cache keyed on its name alone; one older than its fgb is
# rebuilt for the same reason.
band_paths() {
    local region="$1" level="$2" minzoom="$3" maxzoom="$4"
    if [ "$level" = leaf ]; then
        local work="$SCRATCH_ROOT/pmtiles/catchment_leaf"
        BAND_SRC="$SCRATCH_ROOT/regions/$region/catchments_tile_${region}.fgb"
        BAND_POLY="$work/leaf_${region}.z$minzoom-$maxzoom.pmtiles"
        BAND_LINE="$work/leaf_${region}.z$minzoom-$maxzoom.lines.pmtiles"
        BAND_MAKER="5_concatenate_global.py"
    else
        local work="$SCRATCH_ROOT/pmtiles/catchment_bands"
        BAND_SRC="$SCRATCH_ROOT/pmtiles/basin_bands/basin_level${level}.fgb"
        BAND_POLY="$work/level${level}.z$minzoom-$maxzoom.pmtiles"
        BAND_LINE="$work/level${level}.z$minzoom-$maxzoom.lines.pmtiles"
        BAND_MAKER="6_publish_basins.py"
    fi
    BAND_SRC_LINE="${BAND_SRC%.fgb}.lines.fgb"
}
export -f band_paths

# One queue entry: tile one band's fgb into one pmtiles. Fields travel pipe-separated because BSD
# xargs rewrites tabs in a -I item to spaces, and the item stays short of BSD xargs' 255-byte -I
# limit by naming the band rather than spelling out its paths.
tile_one() {
    set -eo pipefail
    local size region level minzoom maxzoom geom src out layer order
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

    # --simplification=4 is a quarter of a tile pixel, matching the quarter pixel the band's
    # geometry was cut at; -Sm 1 leaves the band's deepest zoom - the one actually looked at, and
    # overzoomed from - at tippecanoe's own standard tolerance.
    #
    # -pn is the shared-edge flag: the input is an exact coverage and simplifying the two copies
    # of a shared boundary independently pulls them apart into slivers.
    #
    # --preserve-input-order on the basin polygon bands only: 6_publish_basins.py writes them
    # largest-first so a pinprick enclave draws above the solid basin whose filled hole it sits
    # in, and tippecanoe must not reorder that. The leaf is a coverage with no overlaps.
    #
    # --projection=EPSG:3857 because the fgbs are in the pipeline's own web mercator; without it
    # tippecanoe reads meters as degrees and ships empty tilesets.
    order=""
    if [ "$level" != leaf ] && [ "$geom" = polygons ]; then
        order="--preserve-input-order"
    fi
    local partial="${out%.pmtiles}.partial.pmtiles"
    if ! tippecanoe -o "$partial" -Z"$minzoom" -z"$maxzoom" --layer "$layer" \
            --projection=EPSG:3857 $order \
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

# Every band a full map needs, as "<region>|<level>|<minzoom>|<maxzoom>|<geom>" lines. The basin
# bands are global (region "-"): polygons and lines for levels 3-8, lines only for level 2. The
# leaf polygons and lines are per region.
all_bands() {
    local scope="$1"    # "selected" for this run's regions, "all" for the published join
    echo "-|2|0|$LEAF_MAXZOOM|lines"
    while IFS=: read -r level minzoom maxzoom; do
        [ -n "$level" ] || continue
        if [ "$level" = leaf ]; then
            local -a regions
            if [ "$scope" = all ]; then regions=("${ALL_REGIONS[@]}"); else regions=("${REGIONS[@]}"); fi
            local region
            for region in "${regions[@]}"; do
                echo "$region|leaf|$minzoom|$maxzoom|polygons"
                echo "$region|leaf|$minzoom|$maxzoom|lines"
            done
        else
            echo "-|$level|$minzoom|$maxzoom|polygons"
            echo "-|$level|$minzoom|$maxzoom|lines"
        fi
    done <<< "$BANDS_RAW"
}

# --- what needs building ----------------------------------------------------------------------
# The whole queue is assembled before anything runs, so a missing input fails now rather than
# forty minutes in, and the tasks run longest-first so a big band never runs alone at the end.
mkdir -p "$SCRATCH_ROOT/pmtiles/catchment_bands" "$SCRATCH_ROOT/pmtiles/catchment_leaf"
TASKS=""
while IFS='|' read -r region level minzoom maxzoom geom; do
    [ -n "$level" ] || continue
    band_paths "$region" "$level" "$minzoom" "$maxzoom"
    if [ "$geom" = lines ]; then src="$BAND_SRC_LINE"; out="$BAND_LINE"; else src="$BAND_SRC"; out="$BAND_POLY"; fi
    if [ ! -f "$src" ]; then
        echo "  $level ($region): $src missing, run $BAND_MAKER first" >&2
        exit 1
    fi
    if [ ! -f "$out" ] || [ "$src" -nt "$out" ]; then
        size="$(stat -f%z "$src" 2>/dev/null || stat -c%s "$src")"
        TASKS+="$size|$region|$level|$minzoom|$maxzoom|$geom"$'\n'
    fi
done <<< "$(all_bands selected)"

if [ -n "$TASKS" ]; then
    echo "tiling $(printf '%s' "$TASKS" | wc -l | tr -d ' ') bands," \
         "$TILE_JOBS at a time, $TIPPECANOE_MAX_THREADS tippecanoe threads each"
    printf '%s' "$TASKS" | sort -t'|' -rn -k1,1 | xargs -P "$TILE_JOBS" -I{} bash -c 'tile_one "$@"' _ {}
else
    echo "every band tileset is newer than its fgb, nothing to tile"
fi

# --- housekeeping ------------------------------------------------------------------------------
# Anything in the band scratch dirs the current banding does not name is a relic of an earlier
# banding or regime - including the retired per-region catchment_levels_* directories - and a
# stale tileset must never survive to be joined.
KEEP_LIST="$(all_bands all | while IFS='|' read -r region level minzoom maxzoom geom; do
    [ -n "$level" ] || continue
    band_paths "$region" "$level" "$minzoom" "$maxzoom"
    if [ "$geom" = lines ]; then echo "$BAND_LINE"; else echo "$BAND_POLY"; fi
done)"
for file in "$SCRATCH_ROOT"/pmtiles/catchment_bands/*.pmtiles \
            "$SCRATCH_ROOT"/pmtiles/catchment_leaf/*.pmtiles; do
    [ -e "$file" ] || continue
    if ! grep -qxF "$file" <<< "$KEEP_LIST"; then
        echo "  $(basename "$file"): not in the current banding, removing"
        rm -f "$file"
    fi
done
for legacy in "$SCRATCH_ROOT"/pmtiles/catchment_levels_*; do
    [ -e "$legacy" ] || continue
    echo "  $(basename "$legacy"): retired per-region band directory, removing"
    rm -rf "$legacy"
done

# --- the published tileset --------------------------------------------------------------------
# Named for every band of every region, never globbed: a run that covers one region still reaches
# this join, and a glob would quietly republish the world from whatever happened to be on disk.
GLOBAL_CATCHMENT_TILE="$GROUP_ROOT/group=0/catchments.pmtiles"
GLOBAL_INPUTS=()
MISSING_BANDS=()
while IFS='|' read -r region level minzoom maxzoom geom; do
    [ -n "$level" ] || continue
    band_paths "$region" "$level" "$minzoom" "$maxzoom"
    if [ "$geom" = lines ]; then band_tile="$BAND_LINE"; else band_tile="$BAND_POLY"; fi
    if [ -f "$band_tile" ]; then
        GLOBAL_INPUTS+=("$band_tile")
    else
        MISSING_BANDS+=("$level/$region/$(basename "$band_tile")")
    fi
done <<< "$(all_bands all)"
if [ "${#MISSING_BANDS[@]}" -gt 0 ]; then
    echo "not publishing $GLOBAL_CATCHMENT_TILE: ${#MISSING_BANDS[@]} band tilesets missing" \
         "(${MISSING_BANDS[*]:0:8} ...). Tile them first - the global file has to cover every" \
         "region, whatever subset this run rebuilt." >&2
    exit 1
fi

# rebuilt whenever any band tileset is newer than it: an existence check alone would publish the
# previous banding
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
