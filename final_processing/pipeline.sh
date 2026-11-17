#!/usr/bin/env bash

# this script consumes some already partially processed and prepared global tdxhydro datasets to generate a specially tailored final output set.
# the results of the ../raw_processing/pipeline.sh are required for this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

source "../pipeline_env.sh"
select_regions "$@"

##### Regional intermediate outputs for streams and catchments
printf '%s\n' "${REGIONS[@]}" | xargs -P "$SIMPLIFY_JOBS" -I{} "$PYTHON" simplify_streams.py {}
printf '%s\n' "${REGIONS[@]}" | xargs -P "$CATCHMENT_JOBS" -I{} "$PYTHON" create_catchments.py {}

##### Modify regional outputs. Stamps in a global index and sort, publishes the final one directory per region outputs plus a global output
"$PYTHON" concatenate_global.py "$CONCAT_WORKERS"

##### Dissolve each region's published catchments into its outline.
"$PYTHON" publish_regions.py "$BOUNDARY_JOBS"

./tile_streams.sh
./tile_catchments.sh
./tile_regions.sh

##### Optional extra things to generate
"$PYTHON" extras_identify_id_map.py
"$PYTHON" extras_river_name_ranges.py
