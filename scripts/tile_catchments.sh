#!/usr/bin/env bash
# Usage: ./tile_catchments.sh [region ...]   (no arguments means every region's leaf band)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env.sh"
select_regions "$@"
cd "$SCRIPT_DIR" || exit 1

BANDS_RAW="$("$PYTHON" 2_global_basins.py --bands)"
export BANDS_RAW
LEAF_MAXZOOM="$(printf '%s\n' "$BANDS_RAW" | awk -F: '$1 == "leaf" {print $3}')"
export LEAF_MAXZOOM
GLOBAL_CATCHMENT_TILE="$GROUP_ROOT/group=0/catchments.pmtiles"

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

all_bands() {
    local scope="$1"
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

TASKS=""
MISSING_SOURCES=()
while IFS='|' read -r region level minzoom maxzoom geom; do
    [ -n "$level" ] || continue
    band_paths "$region" "$level" "$minzoom" "$maxzoom"
    if [ "$geom" = lines ]; then src="$BAND_SRC_LINE"; out="$BAND_LINE"; else src="$BAND_SRC"; out="$BAND_POLY"; fi
    [ -f "$out" ] && continue
    if [ ! -f "$src" ]; then
        MISSING_SOURCES+=("$src (run $BAND_MAKER)")
        continue
    fi
    size="$(stat -f%z "$src" 2>/dev/null || stat -c%s "$src")"
    TASKS+="$size|$region|$level|$minzoom|$maxzoom|$geom"$'\n'
done <<< "$(all_bands selected)"

if [ -z "$TASKS" ] && [ "${#MISSING_SOURCES[@]}" -eq 0 ] && [ -f "$GLOBAL_CATCHMENT_TILE" ]; then
    echo "every output exists, nothing to do"
    exit 0
fi
if [ "${#MISSING_SOURCES[@]}" -gt 0 ]; then
    printf '%s\n' "missing tiling sources:" "${MISSING_SOURCES[@]}" >&2
    exit 1
fi

LOCK="$SCRATCH_ROOT/pmtiles/.tile_catchments.lock"
mkdir -p "$SCRATCH_ROOT/pmtiles/catchment_bands" "$SCRATCH_ROOT/pmtiles/catchment_leaf"
if ! mkdir "$LOCK" 2>/dev/null; then
    echo "another tile_catchments.sh holds $LOCK - wait for it, or remove the directory if no such" \
         "run exists" >&2
    exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)"
TILE_JOBS="${TILE_JOBS:-8}"
export TIPPECANOE_MAX_THREADS="${TIPPECANOE_MAX_THREADS:-$(( CORES / TILE_JOBS > 1 ? CORES / TILE_JOBS : 2 ))}"

drop_maxzoom_warning() {
    grep -v 'mismatched maxzooms' || true
}
export -f drop_maxzoom_warning

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

if [ -n "$TASKS" ]; then
    echo "tiling $(printf '%s' "$TASKS" | wc -l | tr -d ' ') bands," \
         "$TILE_JOBS at a time, $TIPPECANOE_MAX_THREADS tippecanoe threads each"
    printf '%s' "$TASKS" | sort -t'|' -rn -k1,1 | xargs -P "$TILE_JOBS" -I{} bash -c 'tile_one "$@"' _ {}
fi

if [ -f "$GLOBAL_CATCHMENT_TILE" ]; then
    exit 0
fi

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
         "(${MISSING_BANDS[*]:0:8} ...)" >&2
    exit 1
fi

GLOBAL_PARTIAL="${GLOBAL_CATCHMENT_TILE%.pmtiles}.partial.pmtiles"
tile-join --force --no-tile-size-limit --name "River Forecast System v3 Catchments" \
    -o "$GLOBAL_PARTIAL" "${GLOBAL_INPUTS[@]}" 2> >(drop_maxzoom_warning >&2)
mv -f "$GLOBAL_PARTIAL" "$GLOBAL_CATCHMENT_TILE"
echo "joined ${#GLOBAL_INPUTS[@]} band tilesets -> $GLOBAL_CATCHMENT_TILE"
