#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env.sh"

GROUPS_PARQUET="$GROUP_ROOT/group=0/groups.geo.parquet"
GROUPS_TILE="$GROUP_ROOT/group=0/groups.pmtiles"
GROUPS_PARTIAL="${GROUPS_TILE%.pmtiles}.partial.pmtiles"

if [ -f "$GROUPS_TILE" ]; then
    echo "$GROUPS_TILE exists, nothing to do"
    exit 0
fi
if [ ! -f "$GROUPS_PARQUET" ]; then
    echo "no $GROUPS_PARQUET - run steps 5 and 6 first" >&2
    exit 1
fi

ogr2ogr -f GeoJSONSeq -t_srs EPSG:4326 /vsistdout/ "$GROUPS_PARQUET" \
    | tippecanoe -o "$GROUPS_PARTIAL" -Z0 -z12 --layer groups --name 'River Forecast System v3 Groups' \
        --no-tile-size-limit --simplification=4 --simplification-at-maximum-zoom=1 \
        --no-simplification-of-shared-nodes --no-progress-indicator --force
mv -f "$GROUPS_PARTIAL" "$GROUPS_TILE"
echo "tiled group boundaries -> $GROUPS_TILE"
