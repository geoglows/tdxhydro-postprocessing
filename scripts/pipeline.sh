#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_env.sh"

cd "$SCRIPT_DIR" || exit 1

################## prepare global tdxhydro baseline
#"$PYTHON" 1_translate_tdxhydro.py

################## analyze streams to correct and simplify representation
printf '%s\n' "${REGIONS[@]}" | xargs -P 12 -I{} "$PYTHON" 2_simplify_streams.py {}
"$PYTHON" 3_global_stream_attributes.py 8

################## apply the same changes to the catchments
printf '%s\n' "${REGIONS[@]}" | xargs -P 3 -I{} "$PYTHON" 4_create_catchments.py {}

################## generate group and global level files from regional files
printf '%s\n' "${REGIONS[@]}" | xargs -P 5 -I{} "$PYTHON" 5_generate_groups.py {}
"$PYTHON" 6_concatenate_global.py

################## create nested pfafstetter style basins from catchments
printf '%s\n' "${REGIONS[@]}" | xargs -P 4 -I{} "$PYTHON" 8_pfafstetter_basins.py {}

./tile_streams.sh
./tile_catchments.sh
./tile_groups.sh

################## Optional extra things to generate
#"$PYTHON" extras_identify_id_map.py
