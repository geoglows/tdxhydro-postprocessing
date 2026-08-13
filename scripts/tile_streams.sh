#!/usr/bin/env bash
# Tile the simplified stream network: one pmtiles per region, then a single global join.
# Usage: ./tile_streams.sh [region ...]   (no arguments means every region)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env.sh"
select_regions "$@"

export STREAM_TIER_FILTER="$(cat "$SCRIPT_DIR/pmtile_filters/z4_delayed.json")"

tile_region() {
    set -eo pipefail
    local region="$1"
    local streams="$SCRATCH_ROOT/regions/$region/streams_${region}.geo.parquet"
    local tile="$SCRATCH_ROOT/pmtiles/streams_${region}.pmtiles"
    if [ ! -f "$streams" ]; then
        echo "region $region: no streams, run step 2 first, skipping"
        return
    fi
    # newer-than, not exists: step 2 rewriting the parquet has to retile the region, and an
    # existence check reads the old tiles as done and silently ships them
    if [ -f "$tile" ] && [ ! "$streams" -nt "$tile" ]; then
        echo "region $region: pmtiles newer than streams, skipping"
        return
    fi

    # tippecanoe cannot read geoparquet, and GeoJSON text - which this used to pipe through - is
    # parsed on a single thread when it comes from a pipe, no matter how many threads tippecanoe
    # is given. A named FlatGeobuf file is mmapped and parsed by every thread at once, so the
    # conversion lands on disk first and is deleted after the one consumer has read it.
    # SPATIAL_INDEX=NO skips an indexing pass that buffers every feature and that tippecanoe never
    # reads; -nlt MULTILINESTRING because a FlatGeobuf layer holds one geometry type and the
    # parquet is free to mix LineString and MultiLineString.
    #
    # -mapFieldType is load-bearing: tippecanoe's FlatGeobuf reader hands integer attributes to
    # the -j filter's numeric comparison as a type it refuses ("mismatched type in comparison"),
    # and a refused comparison drops the feature - every strahlerOrder test failed and the tiles
    # came out empty. Casting to Real reproduces the GeoJSON path exactly, where every number is
    # a double, and tippecanoe writes integral doubles back as integers, so the published tiles
    # are unchanged. Nothing tiled here carries a boolean, so the blanket Integer cast is safe.
    # The parquet's own EPSG:3857 goes straight through - tippecanoe takes mercator input with
    # --projection below, and mercator to tile coordinates is linear, so nothing reprojects.
    local fgb="$SCRATCH_ROOT/pmtiles/streams_${region}.fgb"
    rm -f "$fgb"
    if ! ogr2ogr -f FlatGeobuf -lco SPATIAL_INDEX=NO -nlt MULTILINESTRING \
            -mapFieldType Integer=Real,Integer64=Real "$fgb" "$streams"; then
        rm -f "$fgb"
        echo "region $region: conversion failed" >&2
        return 1
    fi

    # Step 2 writes the geometry at source resolution and does not generalize it, so all of the
    # simplification happens here, per zoom, where it can be undone by asking for a deeper one.
    # -pn holds the nodes where reaches meet: a confluence is one point shared by three features,
    # and simplifying each of them independently is free to move it three different ways and open
    # a gap in the network. It also makes a stretch shared by two features simplify identically in
    # both. tippecanoe's default tolerance - within one tile unit - is left alone.
    #
    # Written aside and moved into place: a tippecanoe killed partway leaves a truncated pmtiles,
    # and the mtime check above reads it as done forever after. The extension survives the temp
    # name because tippecanoe picks its output format from it.
    local partial="${tile%.pmtiles}.partial.pmtiles"
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
    echo "region $region: tiled -> $tile"
}
export -f tile_region

# jobs x threads = cores. The old pipe fed a single sequential parser, so extra tippecanoe threads
# mostly waited and the numbers here could pretend to be anything; a FlatGeobuf input actually
# uses its threads, and oversubscribing 16 cores with 90 of them - which 15 threads x 6 jobs was -
# just trades throughput for context switching.
CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)"
STREAM_JOBS="${STREAM_JOBS:-8}"
export TIPPECANOE_MAX_THREADS="${TIPPECANOE_MAX_THREADS:-$(( CORES / STREAM_JOBS > 1 ? CORES / STREAM_JOBS : 2 ))}"
printf '%s\n' "${REGIONS[@]}" | xargs -P "$STREAM_JOBS" -I{} bash -c 'tile_region "$@"' _ {}

# The global tileset is named for every region, not globbed: a glob is whatever happens to be on
# disk when it expands, and a run covering one region would quietly republish the world from
# whatever subset existed at that moment. Missing regions are an error, and the join reruns
# whenever any region's tiles are newer than the published file - an existence check alone would
# ship stale regions forever.
GLOBAL_TILE="$GROUP_ROOT/group=0/streams.pmtiles"
GLOBAL_INPUTS=()
MISSING_REGIONS=()
for region in "${ALL_REGIONS[@]}"; do
    region_tile="$SCRATCH_ROOT/pmtiles/streams_${region}.pmtiles"
    if [ -f "$region_tile" ]; then
        GLOBAL_INPUTS+=("$region_tile")
    else
        MISSING_REGIONS+=("$region")
    fi
done
if [ "${#MISSING_REGIONS[@]}" -gt 0 ]; then
    echo "not publishing $GLOBAL_TILE: ${#MISSING_REGIONS[@]} of ${#ALL_REGIONS[@]} regions have" \
         "no tileset (${MISSING_REGIONS[*]}). Tile them first - the global file has to cover" \
         "every region, whatever subset this run rebuilt." >&2
    exit 1
fi

# -nt per input rather than `ls -t | head -1`, which dies of SIGPIPE under pipefail once the
# path list outgrows the pipe buffer - see the same gate in tile_catchments.sh
REBUILD_GLOBAL=no
if [ ! -f "$GLOBAL_TILE" ]; then
    REBUILD_GLOBAL=yes
else
    for region_tile in "${GLOBAL_INPUTS[@]}"; do
        if [ "$region_tile" -nt "$GLOBAL_TILE" ]; then
            REBUILD_GLOBAL=yes
            break
        fi
    done
fi
if [ "$REBUILD_GLOBAL" = yes ]; then
    LAYER_NAME="River Forecast System v3 Streams"
    GLOBAL_PARTIAL="${GLOBAL_TILE%.pmtiles}.partial.pmtiles"
    tile-join --force --no-tile-size-limit --name "$LAYER_NAME" -o "$GLOBAL_PARTIAL" \
        "${GLOBAL_INPUTS[@]}"
    mv -f "$GLOBAL_PARTIAL" "$GLOBAL_TILE"
    echo "joined ${#GLOBAL_INPUTS[@]} region tilesets -> $GLOBAL_TILE"
fi
