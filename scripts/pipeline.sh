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
# One at a time. Step 4 holds a whole region's raw dissolve in GEOS objects at ~78 bytes a vertex,
# and now that its coverage pass runs near the source's own resolution the simplified copy it holds
# alongside is nearly as large: measured 37.7 GB peak on 7020000010, which is not the biggest region
# (205 M vertices against 1020000010's 257 M). Two of those do not fit in 64 GB, and the run degrades
# into swap rather than failing. The parallelism that matters is inside the step anyway - the
# dissolve ahead of the coverage pass is already threaded over every core.
printf '%s\n' "${REGIONS[@]}" | xargs -P 1 -I{} "$PYTHON" 4_create_catchments.py {}

################## generate group and global level files from regional files
printf '%s\n' "${REGIONS[@]}" | xargs -P 5 -I{} "$PYTHON" 5_generate_groups.py {}
"$PYTHON" 6_concatenate_global.py
"$PYTHON" 7_identify_id_map.py

################## create nested pfafstetter style basins from catchments
printf '%s\n' "${REGIONS[@]}" | xargs -P 4 -I{} "$PYTHON" 8_pfafstetter_basins.py {}

./tile_streams.sh
./tile_catchments.sh
./tile_groups.sh
