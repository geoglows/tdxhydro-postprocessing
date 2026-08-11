#!/usr/bin/env bash
# Tile the group boundaries. One global file, written by 6_concatenate_global.py, so there is
# nothing per region to fan out over.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env.sh"

GROUPS_PARQUET="$GROUP_ROOT/group=0/groups.geo.parquet"
GROUPS_TILE="$GROUP_ROOT/group=0/groups.pmtiles"
if [ ! -f "$GROUPS_PARQUET" ]; then
    echo "no $GROUPS_PARQUET, skipping group boundary tiles (run steps 4, 5 and 6)"
    exit 0
fi
if [ -f "$GROUPS_TILE" ]; then
    echo "$GROUPS_TILE already exists, skipping"
    exit 0
fi

ogr2ogr -f GeoJSONSeq -t_srs EPSG:4326 /vsistdout/ "$GROUPS_PARQUET" \
    | tippecanoe -o "$GROUPS_TILE" -Z0 -z12 --layer groups --name 'River Forecast System v3 Groups' \
        --no-tile-size-limit --simplification=4 --no-progress-indicator --force
echo "tiled group boundaries -> $GROUPS_TILE"
