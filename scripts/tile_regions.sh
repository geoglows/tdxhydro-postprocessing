#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env.sh"

REGIONS_PARQUET="$PUBLISH_ROOT/global/regions.geo.parquet"
REGIONS_TILE="$PUBLISH_ROOT/global/regions.pmtiles"
REGIONS_PARTIAL="${REGIONS_TILE%.pmtiles}.partial.pmtiles"

if [ -f "$REGIONS_TILE" ]; then
    echo "$REGIONS_TILE exists, nothing to do"
    exit 0
fi
if [ ! -f "$REGIONS_PARQUET" ]; then
    echo "no $REGIONS_PARQUET - run steps 5 and 6 first" >&2
    exit 1
fi

ogr2ogr -f GeoJSONSeq -t_srs EPSG:4326 /vsistdout/ "$REGIONS_PARQUET" \
    | tippecanoe -o "$REGIONS_PARTIAL" -Z0 -z12 --layer regions --name 'River Forecast System v3 Regions' \
        --no-tile-size-limit --simplification=4 --simplification-at-maximum-zoom=1 \
        --no-simplification-of-shared-nodes --no-progress-indicator --force
mv -f "$REGIONS_PARTIAL" "$REGIONS_TILE"
echo "tiled region boundaries -> $REGIONS_TILE"
