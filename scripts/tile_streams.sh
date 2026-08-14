#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env.sh"
pipeline_banner "Tile stream vectors"
export STREAM_TIER_FILTER="$(cat "$SCRIPT_DIR/pmtile_filters/z4_delayed.json")"

# The caller decides which groups need tiling - see the work list below - so this does no
# existence or staleness checking of its own. Its output is one line per group actually tiled.
tile_group() {
    set -eo pipefail
    local group="$1"
    local streams="$GROUP_ROOT/group=$group/streams_${group}.geo.parquet"
    local tile="$SCRATCH_ROOT/pmtiles/streams_${group}.pmtiles"

    # tippecanoe cannot read geoparquet, and GeoJSON text - which this used to pipe through - is
    # parsed on a single thread when it comes from a pipe. A named FlatGeobuf file is mmapped and
    # parsed by every thread at once, so the conversion lands on disk first and is deleted after
    # its one consumer has read it. SPATIAL_INDEX=NO skips an indexing pass tippecanoe never
    # reads; -nlt MULTILINESTRING because a FlatGeobuf layer holds one geometry type.
    #
    # -mapFieldType is load-bearing: tippecanoe's FlatGeobuf reader hands integer attributes to
    # the -j filter's numeric comparison as a type it refuses, and a refused comparison drops the
    # feature - every strahlerOrder test failed and the tiles came out empty. Casting to Real
    # reproduces the GeoJSON path, where every number is a double, and tippecanoe writes integral
    # doubles back as integers, so the published tiles are unchanged.
    local fgb="$SCRATCH_ROOT/pmtiles/streams_${group}.fgb"
    rm -f "$fgb"
    if ! ogr2ogr -f FlatGeobuf -lco SPATIAL_INDEX=NO -nlt MULTILINESTRING -mapFieldType Integer=Real,Integer64=Real "$fgb" "$streams"; then
        rm -f "$fgb"
        echo "group $group: conversion failed" >&2
        return 1
    fi

    local partial="${tile%.pmtiles}.partial.pmtiles"
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

# All parallelism decisions come from pipeline_env.sh; strict respect of those values here.
# STREAM_JOBS is set in environment; TIPPECANOE_MAX_THREADS is derived from it.
export TIPPECANOE_MAX_THREADS="${TIPPECANOE_MAX_THREADS:-$(( PIPELINE_CORES / STREAM_JOBS > 1 ? PIPELINE_CORES / STREAM_JOBS : 1 ))}"
mkdir -p "$SCRATCH_ROOT/pmtiles"
# list the folders in group root, except for group=0, split by = to get the group number
OUTPUT_GROUPS=($(find "$GROUP_ROOT" -maxdepth 1 -type d -name "group=*" ! -name "group=0"))
if [ "${#OUTPUT_GROUPS[@]}" -eq 0 ]; then
    echo "no group=* directories under $GROUP_ROOT - run step 5 first" >&2
    exit 1
fi
OUTPUT_GROUPS=($(printf '%s\n' "${OUTPUT_GROUPS[@]}" | sed 's|.*/group=||' | sort -n))

# The work list, decided here rather than inside tile_group, for two reasons: a group is skipped
# without printing anything (121 "tile exists" lines said nothing a single count does not), and
# the test is mtime, not existence. Existence alone was shipping stale tiles - step 5 rewrites
# every group's streams parquet, and a tileset built from the previous release survived it and
# went straight into the global join. Same rule as tile_catchments.sh: rebuild what is older
# than its source.
#
# TO_TILE is "<source bytes>|<group>" lines rather than an array of group numbers, because the
# order the groups are handed to xargs decides the wall clock. Group sizes span 127 MB to
# effectively nothing, and in group-id order the two largest happened to start last: the run spent
# its final 2.5 minutes on 3 jobs, 9 of 32 cores. Longest-processing-time-first is the standard
# fix, and simulating both orderings over the measured 121 sizes puts it 15% faster (537s -> 456s)
# and within 3s of the perfect-balance floor. Source bytes stand in for tiling time; the tileset
# is not there to measure until after the work. Same shape as tile_catchments.sh, which has
# always sorted its bands this way.
GROUP_TILES=()
TO_TILE=""
MISSING=()
for group in "${OUTPUT_GROUPS[@]}"; do
    streams="$GROUP_ROOT/group=$group/streams_${group}.geo.parquet"
    tile="$SCRATCH_ROOT/pmtiles/streams_${group}.pmtiles"
    # named for every group, never globbed: a streams_<g>.partial.pmtiles left by a killed
    # tippecanoe matches streams_*.pmtiles and would otherwise be joined into the published file
    GROUP_TILES+=("$tile")
    if [ ! -f "$streams" ]; then
        MISSING+=("$group")
    elif [ ! -f "$tile" ] || [ "$streams" -nt "$tile" ]; then
        size="$(stat -f%z "$streams" 2>/dev/null || stat -c%s "$streams")"
        TO_TILE+="$size|$group"$'\n'
    fi
done

# Fatal, not skipped: the global join names every group, so tiling around a missing one publishes
# a tileset with a hole in it and says only that it skipped something.
if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "no streams parquet for group(s) ${MISSING[*]} - run step 5 first" >&2
    exit 1
fi

if [ -n "$TO_TILE" ]; then
    N_TILE="$(printf '%s' "$TO_TILE" | wc -l | tr -d ' ')"
    echo "${#OUTPUT_GROUPS[@]} groups: $(( ${#OUTPUT_GROUPS[@]} - N_TILE )) current," \
         "$N_TILE to tile largest-first, $STREAM_JOBS jobs x $TIPPECANOE_MAX_THREADS threads =" \
         "$(( STREAM_JOBS * TIPPECANOE_MAX_THREADS )) cores"
    printf '%s' "$TO_TILE" | sort -t'|' -rn -k1,1 | cut -d'|' -f2 \
        | xargs -P "$STREAM_JOBS" -I{} bash -c 'tile_group "$@"' _ {}
else
    echo "${#OUTPUT_GROUPS[@]} groups: every tileset is newer than its streams parquet, nothing to tile"
fi

# The global tileset is joined from the group tiles named above, and the join reruns whenever any
# of them is newer than the published file - an existence check alone would ship stale groups
# forever, which is exactly what the "will not be overwritten" message here used to claim while
# the join ran unconditionally underneath it.
GLOBAL_TILE="$GROUP_ROOT/group=0/streams.pmtiles"
GLOBAL_PARTIAL="${GLOBAL_TILE%.pmtiles}.partial.pmtiles"
LAYER_NAME="River Forecast System v3 Streams"

REBUILD_GLOBAL=no
if [ ! -f "$GLOBAL_TILE" ]; then
    REBUILD_GLOBAL=yes
else
    for tile in "${GROUP_TILES[@]}"; do
        if [ "$tile" -nt "$GLOBAL_TILE" ]; then
            REBUILD_GLOBAL=yes
            break
        fi
    done
fi

if [ "$REBUILD_GLOBAL" = yes ]; then
    tile-join --force --no-tile-size-limit --name "$LAYER_NAME" -o "$GLOBAL_PARTIAL" "${GROUP_TILES[@]}"
    mv -f "$GLOBAL_PARTIAL" "$GLOBAL_TILE"
    echo "joined ${#GROUP_TILES[@]} group tilesets -> $GLOBAL_TILE"
else
    echo "$(basename "$GLOBAL_TILE") is newer than every group tileset, nothing to join"
fi
