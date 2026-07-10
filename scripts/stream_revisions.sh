#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/../.venv/bin/python"
DATA="$SCRIPT_DIR/../data"

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

# prepare tdx regions
printf '%s\n' "${REGIONS[@]}" | xargs -P 12 -I{} "$PYTHON" 2_simplify_streams.py {}
#printf '%s\n' "${REGIONS[@]}" | xargs -P 1 -I{} "$PYTHON" 3_create_catchments.py {}
## group subdividing
#printf '%s\n' "${REGIONS[@]}" | xargs -P 3 -I{} "$PYTHON" 4_generate_groups.py {}
## global files
#"$PYTHON" 5_concatenate_global.py

# tiling
mkdir -p "$DATA/global"

STREAM_TIER_FILTER='{"*":["any",[">=","strahlerOrder",6],["all",[">=","strahlerOrder",4],[">=","$zoom",6]],[">=","$zoom",8]]}'
export DATA STREAM_TIER_FILTER

tile_region() {
    set -eo pipefail
    local region="$1"
    local mapping="$DATA/regions/$region/streams_mapping_${region}.geo.parquet"
    local tile="$DATA/regions/$region/streams_${region}.pmtiles"
    if [ ! -f "$mapping" ]; then
        echo "region $region: no streams_mapping, run step 2 first, skipping"
        return
    fi
    if [ -f "$tile" ]; then
        echo "region $region: tile already exists, skipping"
        return
    fi
    ogr2ogr -f GeoJSONSeq -t_srs EPSG:4326 /vsistdout/ "$mapping" \
        | tippecanoe -o "$tile" -Z0 -z12 --layer streams --include riverId --include strahlerOrder \
            --drop-densest-as-needed --simplification=10 --no-simplification-of-shared-nodes \
            -j "$STREAM_TIER_FILTER" --no-progress-indicator --force
    echo "region $region: tiled -> $tile"
}
export -f tile_region
export TIPPECANOE_MAX_THREADS=12
printf '%s\n' "${REGIONS[@]}" | xargs -P 6 -I{} bash -c 'tile_region "$@"' _ {}
tile-join --force --no-tile-size-limit --name 'GEOGLOWS RFS Streams (global)' \
    -o "$DATA/global/streams_global.pmtiles" "$DATA"/regions/*/streams_*.pmtiles
