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
    if [ -f "$tile" ]; then
        echo "region $region: pmtiles already exists, skipping"
        return
    fi
    if [ ! -f "$streams" ]; then
        echo "region $region: no streams, run step 2 first, skipping"
        return
    fi
    # Step 2 writes the geometry at source resolution and does not generalize it, so all of the
    # simplification happens here, per zoom, where it can be undone by asking for a deeper one.
    # -pn holds the nodes where reaches meet: a confluence is one point shared by three features,
    # and simplifying each of them independently is free to move it three different ways and open
    # a gap in the network. It also makes a stretch shared by two features simplify identically in
    # both. tippecanoe's default tolerance - within one tile unit - is left alone.
    ogr2ogr -f GeoJSONSeq -t_srs EPSG:4326 /vsistdout/ "$streams" \
        | tippecanoe -o "$tile" -Z0 -z11 --layer streams -j "$STREAM_TIER_FILTER" \
            --exclude musk_k --exclude musk_x --exclude velocity_factor --exclude USContArea \
            --no-simplification-of-shared-nodes \
            --drop-densest-as-needed --no-progress-indicator --force
    echo "region $region: tiled -> $tile"
}
export -f tile_region

export TIPPECANOE_MAX_THREADS=15
printf '%s\n' "${REGIONS[@]}" | xargs -P 6 -I{} bash -c 'tile_region "$@"' _ {}

GLOBAL_TILE="$GROUP_ROOT/group=0/streams.pmtiles"
if [ ! -f "$GLOBAL_TILE" ]; then
    LAYER_NAME="River Forecast System v3 Streams"
    tile-join --force --no-tile-size-limit --name "$LAYER_NAME" -o "$GLOBAL_TILE" "$SCRATCH_ROOT"/pmtiles/streams_*.pmtiles
    echo "joined stream tiles -> $GLOBAL_TILE"
fi
