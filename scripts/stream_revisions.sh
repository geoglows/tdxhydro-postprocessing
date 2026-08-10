#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/../.venv/bin/python"

export RFS_DATA_ROOT="/Users/rchales/data/rfsv3"
export TDXHYDRO_ROOT="/Users/rchales/data/TDXHydroGeoParquet"
# not GROUPS: that is a bash special variable (the user's gids) and assignments to it
# are silently ignored, so the paths below would resolve relative to the cwd
export GROUP_ROOT="$RFS_DATA_ROOT/hydrography"
export SCRATCH="$RFS_DATA_ROOT/hydrography-scratchfiles"

mkdir -p "$GROUP_ROOT/group=0"
mkdir -p "$SCRATCH/pmtiles"
mkdir -p "$SCRATCH/regions"

cd "$SCRIPT_DIR" || exit 1

REGIONS=(
    1020000010
    1020011530
    1020018110
    1020021940
    1020027430
    1020034170
    1020035180
    1020040190
    2020000010
    2020003440
    2020018240
    2020024230
    2020033490
    2020065840
    2020071190
    3020000010
    3020003790
    3020008670
    3020024310
    4020000010
    4020006940
    4020015090
    4020024190
    4020034510
    4020050210
    4020050220
    4020050290
    4020050470
    5020000010
    5020015660
    5020037270
    5020049720
    5020082270
    6020000010
    6020006540
    6020008320
    6020014330
    6020017370
    6020021870
    6020029280
    7020000010
    7020014250
    7020021430
    7020024600
    7020038340
    7020046750
    7020047840
    7020065090
    8020000010
    8020008900
)

# prepare streams
printf '%s\n' "${REGIONS[@]}" | xargs -P 12 -I{} "$PYTHON" 2_simplify_streams.py {}
"$PYTHON" 3_global_stream_attributes.py 8

# prepare catchments
printf '%s\n' "${REGIONS[@]}" | xargs -P 2 -I{} "$PYTHON" 4_create_catchments.py {}

# subdivide regions to groups
printf '%s\n' "${REGIONS[@]}" | xargs -P 5 -I{} "$PYTHON" 5_generate_groups.py {}

# create global files
"$PYTHON" 6_concatenate_global.py

# map every original tdx-hydro reach to its id in the v3 stream set
"$PYTHON" 7_identify_id_map.py

export STREAM_TIER_FILTER="$(cat "$SCRIPT_DIR/pmtile_filters/z4_delayed.json")"

tile_region() {
    set -eo pipefail
    local region="$1"
    local streams="$SCRATCH/regions/$region/streams_${region}.geo.parquet"
    local tile="$SCRATCH/pmtiles/streams_${region}.pmtiles"
    if [ -f "$tile" ]; then
        echo "region $region: pmtiles already exists, skipping"
        return
    fi
    if [ ! -f "$streams" ]; then
        echo "region $region: no streams, run step 2 first, skipping"
        return
    fi
    # the streams are stored in EPSG:3857; GeoJSON is defined in lon/lat, so reproject on the way in
    ogr2ogr -f GeoJSONSeq -t_srs EPSG:4326 /vsistdout/ "$streams" \
        | tippecanoe -o "$tile" -Z0 -z11 --layer streams -j "$STREAM_TIER_FILTER" \
            --exclude musk_k --exclude musk_x --exclude velocity_factor --exclude USContArea \
            --drop-densest-as-needed --simplification=10 --no-progress-indicator --force
    echo "region $region: tiled -> $tile"
}
export -f tile_region
export TIPPECANOE_MAX_THREADS=15
printf '%s\n' "${REGIONS[@]}" | xargs -P 6 -I{} bash -c 'tile_region "$@"' _ {}
GLOBAL_TILE="$GROUP_ROOT/group=0/streams.pmtiles"
if [ ! -f "$GLOBAL_TILE" ]; then
  tile-join --force --no-tile-size-limit --name 'River Forecast System v3 Streams' \
      -o "$GLOBAL_TILE" "$SCRATCH"/pmtiles/streams_*.pmtiles
fi
