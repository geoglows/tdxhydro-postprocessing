#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env.sh"

cd "$SCRIPT_DIR" || exit 1

printf '%s\n' "${REGIONS[@]}" | xargs -P 12 -I{} "$PYTHON" 2_simplify_streams.py {}
"$PYTHON" 3_global_stream_attributes.py 8

printf '%s\n' "${REGIONS[@]}" | xargs -P 2 -I{} "$PYTHON" 4_create_catchments.py {}

# subdivide regions to groups
printf '%s\n' "${REGIONS[@]}" | xargs -P 5 -I{} "$PYTHON" 5_generate_groups.py {}

# create global files
"$PYTHON" 6_concatenate_global.py

# map every original tdx-hydro reach to its id in the v3 stream set
"$PYTHON" 7_identify_id_map.py

printf '%s\n' "${REGIONS[@]}" | xargs -P 4 -I{} "$PYTHON" 8_pfafstetter_basins.py {}

./tile_streams.sh
./tile_catchments.sh
./tile_groups.sh
