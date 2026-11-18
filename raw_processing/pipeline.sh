#!/usr/bin/env bash
# One-time processing of the raw TDX-Hydro data, run before (and much less often than) the build in
# ../final_processing/pipeline.sh. Everything here is derived from the raw data and a fixed external source, and is
# written beside the raw data under $TDXHYDRO_GEOPARQUET_FINISHED so it outlasts every rebuild of the published
# hydrography (final_processing/pipeline.sh). Each product is keyed by the raw TDXHydroLinkNo; the build joins in the ones it
# publishes, and the rest stay supplemental tables joined by whoever wants them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

source "../pipeline_env.sh"
# no arguments: every region (the build's list in pipeline_env.sh for the translation, and every
# region under $TDXHYDRO_GEOPARQUET_FINISHED for the attributes, whose global tables need all of them).
# Region arguments restrict both, e.g. to redo one region.
select_regions "$@"

################## TDX-Hydro geopackages -> raw geoparquet beside them in $TDXHYDRO_GEOPARQUET_RAW -> conditioned geoparquet in $TDXHYDRO_GEOPARQUET_FINISHED: globally
################## unique ids, geodesic lengths, duplicated watersheds dropped, basin coverage noded
"$PYTHON" 1_translate_tdxhydro.py "${REGIONS[@]}"

"$PYTHON" koppen_classifications.py "$@"

"$PYTHON" country_classifications.py "$@"
