#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env.sh"

OUTPUT_GROUPS=($(find "$GROUP_ROOT" -maxdepth 1 -type d -name "group=*" ! -name "group=0" \
    | sed 's|.*/group=||' | sort -n))
if [ "${#OUTPUT_GROUPS[@]}" -eq 0 ]; then
    echo "no group=* directories under $GROUP_ROOT - run step 5 first" >&2
    exit 1
fi

GLOBAL_TILE="$GROUP_ROOT/group=0/streams.pmtiles"
GROUP_TILES=()
for group in "${OUTPUT_GROUPS[@]}"; do
    GROUP_TILES+=("$SCRATCH_ROOT/pmtiles/streams_${group}.pmtiles")
done

TO_TILE=""
MISSING=()
for group in "${OUTPUT_GROUPS[@]}"; do
    streams="$GROUP_ROOT/group=$group/streams_${group}.geo.parquet"
    tile="$SCRATCH_ROOT/pmtiles/streams_${group}.pmtiles"
    [ -f "$tile" ] && continue
    if [ ! -f "$streams" ]; then
        MISSING+=("$group")
    else
        size="$(stat -f%z "$streams" 2>/dev/null || stat -c%s "$streams")"
        TO_TILE+="$size|$group"$'\n'
    fi
done

if [ -z "$TO_TILE" ] && [ "${#MISSING[@]}" -eq 0 ] && [ -f "$GLOBAL_TILE" ]; then
    echo "every output exists, nothing to do"
    exit 0
fi

if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "no streams parquet for group(s) ${MISSING[*]} - run step 5 first" >&2
    exit 1
fi

export STREAM_TIER_FILTER="$(cat "$SCRIPT_DIR/pmtile_filters/z4_delayed.json")"
export TIPPECANOE_MAX_THREADS="${TIPPECANOE_MAX_THREADS:-$(( PIPELINE_CORES / STREAM_JOBS > 1 ? PIPELINE_CORES / STREAM_JOBS : 1 ))}"
mkdir -p "$SCRATCH_ROOT/pmtiles"

tile_group() {
    set -eo pipefail
    local group="$1"
    local streams="$GROUP_ROOT/group=$group/streams_${group}.geo.parquet"
    local tile="$SCRATCH_ROOT/pmtiles/streams_${group}.pmtiles"
    local fgb="$SCRATCH_ROOT/pmtiles/streams_${group}.fgb"
    local partial="${tile%.pmtiles}.partial.pmtiles"

    rm -f "$fgb"
    if ! ogr2ogr -f FlatGeobuf -lco SPATIAL_INDEX=NO -nlt MULTILINESTRING -mapFieldType Integer=Real,Integer64=Real "$fgb" "$streams"; then
        rm -f "$fgb"
        echo "group $group: conversion failed" >&2
        return 1
    fi
    if ! tippecanoe -o "$partial" -Z0 -z11 --layer streams -j "$STREAM_TIER_FILTER" \
            --projection=EPSG:3857 \
            --exclude musk_k --exclude musk_x --exclude velocity_factor --exclude USContArea \
            --no-simplification-of-shared-nodes \
            --drop-densest-as-needed --no-progress-indicator --force "$fgb"; then
        rm -f "$partial" "$fgb"
        echo "group $group: tiling failed" >&2
        return 1
    fi
    rm -f "$fgb"
    mv -f "$partial" "$tile"
    echo "group $group: tiled -> $(basename "$tile")"
}
export -f tile_group

if [ -n "$TO_TILE" ]; then
    N_TILE="$(printf '%s' "$TO_TILE" | wc -l | tr -d ' ')"
    echo "${#OUTPUT_GROUPS[@]} groups: $(( ${#OUTPUT_GROUPS[@]} - N_TILE )) present," \
         "$N_TILE to tile largest-first, $STREAM_JOBS jobs x $TIPPECANOE_MAX_THREADS threads"
    printf '%s' "$TO_TILE" | sort -t'|' -rn -k1,1 | cut -d'|' -f2 \
        | xargs -P "$STREAM_JOBS" -I{} bash -c 'tile_group "$@"' _ {}
fi

if [ ! -f "$GLOBAL_TILE" ]; then
    GLOBAL_PARTIAL="${GLOBAL_TILE%.pmtiles}.partial.pmtiles"
    tile-join --force --no-tile-size-limit --name "River Forecast System v3 Streams" \
        -o "$GLOBAL_PARTIAL" "${GROUP_TILES[@]}"
    mv -f "$GLOBAL_PARTIAL" "$GLOBAL_TILE"
    echo "joined ${#GROUP_TILES[@]} group tilesets -> $GLOBAL_TILE"
fi
