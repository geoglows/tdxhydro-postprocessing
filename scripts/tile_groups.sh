#!/usr/bin/env bash
# Tile the group boundaries. One global file, written by 6_publish_basins.py from the stamped level-8 basins, so there is
# nothing per region to fan out over.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env.sh"
pipeline_banner "Tile group boundary vectors"

GROUPS_PARQUET="$GROUP_ROOT/group=0/groups.geo.parquet"
GROUPS_TILE="$GROUP_ROOT/group=0/groups.pmtiles"
if [ ! -f "$GROUPS_PARQUET" ]; then
    echo "no $GROUPS_PARQUET, skipping group boundary tiles (run steps 5 and 6)"
    exit 0
fi
# newer-than, not exists: step 6 rewriting the parquet has to retile, and an existence check
# reads the old tiles as done and silently ships them
if [ -f "$GROUPS_TILE" ] && [ ! "$GROUPS_PARQUET" -nt "$GROUPS_TILE" ]; then
    echo "$GROUPS_TILE newer than $GROUPS_PARQUET, skipping"
    exit 0
fi

# The outlines go in exact rather than simplified - step 6 dissolves them from the frozen level-8 basins (band-resolution, straddlers patched exactly)
# so the deepest zoom carries what the band resolution has. -pn is
# what makes that safe: adjacent groups share their whole dividing edge, and simplifying the two
# copies of it independently pulls them apart into slivers at every zoom above the deepest.
#
# Written aside and moved into place: a tippecanoe killed partway leaves a truncated pmtiles that
# is newer than its parquet, and the mtime check above reads it as done forever after. This one
# file is small enough that the GeoJSON pipe is not worth replacing with a FlatGeobuf detour.
GROUPS_PARTIAL="${GROUPS_TILE%.pmtiles}.partial.pmtiles"
ogr2ogr -f GeoJSONSeq -t_srs EPSG:4326 /vsistdout/ "$GROUPS_PARQUET" \
    | tippecanoe -o "$GROUPS_PARTIAL" -Z0 -z12 --layer groups --name 'River Forecast System v3 Groups' \
        --no-tile-size-limit --simplification=4 --simplification-at-maximum-zoom=1 \
        --no-simplification-of-shared-nodes --no-progress-indicator --force
mv -f "$GROUPS_PARTIAL" "$GROUPS_TILE"
echo "tiled group boundaries -> $GROUPS_TILE"
