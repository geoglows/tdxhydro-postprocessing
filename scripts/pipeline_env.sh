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

# Every tile_*.sh takes an optional list of regions on the command line and falls back to the full
# set, so a single region can be retiled without editing anything. ALL_REGIONS keeps the full set
# either way: what a run rebuilds is a subset, but what a global product has to cover never is, and
# a step that publishes one needs to be able to tell the difference.
select_regions() {
    ALL_REGIONS=("${REGIONS[@]}")
    if [ "$#" -gt 0 ]; then
        REGIONS=("$@")
    fi
}
