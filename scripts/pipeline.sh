#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

source "pipeline_env.sh"

################## prepare global tdxhydro baseline, basins - one-time steps, not per build
"$PYTHON" 1_translate_tdxhydro.py "${REGIONS[@]}"
"$PYTHON" 2_global_basins.py

################## analyze streams to correct and simplify representation (region-local outputs)
################## populates the scratch/regions directory with regionally unique indices and ids
printf '%s\n' "${REGIONS[@]}" | xargs -P "$SIMPLIFY_JOBS" -I{} "$PYTHON" 3_simplify_streams.py {}
printf '%s\n' "${REGIONS[@]}" | xargs -P "$CATCHMENT_JOBS" -I{} "$PYTHON" 4_create_catchments.py {}

################## Apply global indices in ascending region order and publish one directory per region
"$PYTHON" 5_concatenate_global.py "$CONCAT_WORKERS"

################## Dissolve each region's published catchments into its outline. Not the same thing
################## as basins_level2: this one has holes exactly where the release dropped watersheds
"$PYTHON" 6_publish_regions.py "$BOUNDARY_JOBS"

./tile_streams.sh
#./tile_catchments.sh
#./tile_regions.sh

################## Optional extra things to generate
#"$PYTHON" extras_identify_id_map.py
"$PYTHON" extras_river_name_ranges.py
