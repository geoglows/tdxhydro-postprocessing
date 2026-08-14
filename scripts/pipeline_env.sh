#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/../.venv/bin/python"

export RFS_DATA_ROOT="/Users/rchales/data/rfsv3"
export TDXHYDRO_ROOT="/Users/rchales/data/TDXHydroGeoParquet"
export GROUP_ROOT="$RFS_DATA_ROOT/hydrography"
export SCRATCH_ROOT="$RFS_DATA_ROOT/hydrography-scratchfiles"

mkdir -p "$GROUP_ROOT/group=0"
mkdir -p "$SCRATCH_ROOT/pmtiles"
mkdir -p "$SCRATCH_ROOT/regions"

# ---------------------------------------------------------------------------------------------
# Every step is `jobs x threads = cores`
export PIPELINE_CORES="${PIPELINE_CORES:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)}"
# step 3: one core per process
export SIMPLIFY_JOBS="${SIMPLIFY_JOBS:-24}"
# step 4: jobs x chunk threads = cores
export CATCHMENT_JOBS="${CATCHMENT_JOBS:-8}"
export CATCHMENT_CHUNK_THREADS="${CATCHMENT_CHUNK_THREADS:-4}"
# step 5: regions in flight at once inside the one concatenate process; each holds a region's
# streams and catchments, so the ceiling is memory rather than cores
export CONCAT_WORKERS="${CONCAT_WORKERS:-8}"
# step 1: one worker per gpkg, end to end. The dial is memory, not cores - a basins gpkg is tens
# of GB as a GeoDataFrame - so this stays well under the core count.
export TRANSLATE_JOBS="${TRANSLATE_JOBS:-6}"
# step 2 takes no jobs dial: it runs one region at a time and the dissolve inside it already
# spreads across every core (2_global_basins.py UNION_THREADS, threads because GEOS drops the GIL).
# tiling: tile_*.sh derive TIPPECANOE_MAX_THREADS as cores/jobs, so these already scaled with the
# machine. 10 jobs x 3 threads = 30 cores balances per-job optimization with I/O parallelism.
export STREAM_JOBS="${STREAM_JOBS:-10}"
export TILE_JOBS="${TILE_JOBS:-10}"
# ---------------------------------------------------------------------------------------------

# The dashed step marker the shell steps print, matched to hydrography/console.py so a tiling step
# and a python step read as the same run in the terminal. One printf, as there.
pipeline_banner() {
    local rule
    rule="$(printf '%.0s-' {1..78})"
    printf '\n%s\n%s\n%s\n' "$rule" "$1" "$rule"
}

REGIONS=(
    1020000010
    1020011530
    1020018110
    1020021940
    1020027430
    1020034170
    1020035180
    1020040190
    2020000010
    2020003440
    2020018240
    2020024230
    2020033490
    2020065840
    2020071190
    3020000010
    3020003790
    3020008670
    3020024310
    4020000010
    4020006940
    4020015090
    4020024190
    4020034510
    4020050210
    4020050220
    4020050290
    4020050470
    5020000010
    5020015660
    5020037270
    5020049720
    5020082270
    6020000010
    6020006540
    6020008320
    6020014330
    6020017370
    6020021870
    6020029280
    7020000010
    7020014250
    7020021430
    7020024600
    7020038340
    7020046750
    7020047840
    7020065090
    8020000010
    8020008900
)

select_regions() {
    ALL_REGIONS=("${REGIONS[@]}")
    if [ "$#" -gt 0 ]; then
        REGIONS=("$@")
    fi
}
