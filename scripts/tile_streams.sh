#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env.sh"

# a published region is one step 5 wrote metadata for, not merely one with a directory:
# step 3 publishes mods/ into region=<id>/ long before step 5 fills it, and that half-made
# directory is a normal mid-pipeline state rather than something to tile or complain about.
# A region WITH metadata but without streams is still an error - see MISSING below.
OUTPUT_REGIONS=($(find "$PUBLISH_ROOT" -maxdepth 2 -name "metadata_*.parquet" -path "*/region=*" \
    | sed 's|.*/region=\([0-9]*\)/.*|\1|' | sort -n))
if [ "${#OUTPUT_REGIONS[@]}" -eq 0 ]; then
    echo "no published regions under $PUBLISH_ROOT - run step 5 first" >&2
    exit 1
fi

GLOBAL_TILE="$PUBLISH_ROOT/global/streams.pmtiles"
REGION_TILES=()
for region in "${OUTPUT_REGIONS[@]}"; do
    REGION_TILES+=("$SCRATCH_ROOT/pmtiles/streams_${region}.pmtiles")
done

TO_TILE=""
MISSING=()
for region in "${OUTPUT_REGIONS[@]}"; do
    streams="$PUBLISH_ROOT/region=$region/streams_${region}.geo.parquet"
    tile="$SCRATCH_ROOT/pmtiles/streams_${region}.pmtiles"
    [ -f "$tile" ] && continue
    if [ ! -f "$streams" ]; then
        MISSING+=("$region")
    else
        size="$(stat -f%z "$streams" 2>/dev/null || stat -c%s "$streams")"
        TO_TILE+="$size|$region"$'\n'
    fi
done

if [ -z "$TO_TILE" ] && [ "${#MISSING[@]}" -eq 0 ] && [ -f "$GLOBAL_TILE" ]; then
    echo "every output exists, nothing to do"
    exit 0
fi

if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "no streams parquet for region(s) ${MISSING[*]} - run step 5 first" >&2
    exit 1
fi

export STREAM_TIER_FILTER="$(cat "$SCRIPT_DIR/pmtile_filters/z4_delayed.json")"
export TIPPECANOE_MAX_THREADS="${TIPPECANOE_MAX_THREADS:-$(( PIPELINE_CORES / STREAM_JOBS > 1 ? PIPELINE_CORES / STREAM_JOBS : 1 ))}"
mkdir -p "$SCRATCH_ROOT/pmtiles"

tile_region() {
    set -eo pipefail
    local region="$1"
    local streams="$PUBLISH_ROOT/region=$region/streams_${region}.geo.parquet"
    local tile="$SCRATCH_ROOT/pmtiles/streams_${region}.pmtiles"
    local fgb="$SCRATCH_ROOT/pmtiles/streams_${region}.fgb"
    local partial="${tile%.pmtiles}.partial.pmtiles"

    rm -f "$fgb"
    if ! ogr2ogr -f FlatGeobuf -lco SPATIAL_INDEX=NO -nlt MULTILINESTRING -mapFieldType Integer=Real,Integer64=Real "$fgb" "$streams"; then
        rm -f "$fgb"
        echo "region $region: conversion failed" >&2
        return 1
    fi
    if ! tippecanoe -o "$partial" -Z0 -z11 --layer streams -j "$STREAM_TIER_FILTER" \
            --projection=EPSG:3857 \
            --exclude musk_k --exclude musk_x --exclude velocity_factor --exclude USContArea \
            --no-simplification-of-shared-nodes \
            --drop-densest-as-needed --no-progress-indicator --force "$fgb"; then
        rm -f "$partial" "$fgb"
        echo "region $region: tiling failed" >&2
        return 1
    fi
    rm -f "$fgb"
    mv -f "$partial" "$tile"
    echo "region $region: tiled -> $(basename "$tile")"
}
export -f tile_region

if [ -n "$TO_TILE" ]; then
    N_TILE="$(printf '%s' "$TO_TILE" | wc -l | tr -d ' ')"
    echo "${#OUTPUT_REGIONS[@]} regions: $(( ${#OUTPUT_REGIONS[@]} - N_TILE )) present," \
         "$N_TILE to tile largest-first, $STREAM_JOBS jobs x $TIPPECANOE_MAX_THREADS threads"
    printf '%s' "$TO_TILE" | sort -t'|' -rn -k1,1 | cut -d'|' -f2 \
        | xargs -P "$STREAM_JOBS" -I{} bash -c 'tile_region "$@"' _ {}
fi

if [ ! -f "$GLOBAL_TILE" ]; then
    GLOBAL_PARTIAL="${GLOBAL_TILE%.pmtiles}.partial.pmtiles"
    tile-join --force --no-tile-size-limit --name "River Forecast System v3 Streams" \
        -o "$GLOBAL_PARTIAL" "${REGION_TILES[@]}"
    mv -f "$GLOBAL_PARTIAL" "$GLOBAL_TILE"
    echo "joined ${#REGION_TILES[@]} region tilesets -> $GLOBAL_TILE"
fi
