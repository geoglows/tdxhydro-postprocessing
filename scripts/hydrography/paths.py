"""
Filesystem layout for the pipeline.

Two directories, two environment variables, both exported by
pipeline.sh before it runs any step:

    RFS_DATA_ROOT    everything the pipeline writes
    TDXHYDRO_ROOT    the raw TDX-Hydro geoparquet it reads, which is kept
                     separately from the outputs

Every step derives its paths from those roots, so relocating either one is a
matter of changing the one export in that script.

Under the data root, the published dataset and the working files are kept apart:

    hydrography/group=<id>/     the deliverable - one directory per group, in the
                                hive-style naming the published dataset uses
    hydrography-scratchfiles/   per-region intermediates, per-region pmtiles and
                                logs - everything the group files are built from,
                                which no consumer of the dataset needs

There are deliberately no defaults here: the orchestrator owns the values, and a
step run without them should say so rather than quietly build a second copy of
the dataset somewhere else. To run a step by hand, set the variables first.

Inputs that are version controlled alongside the code - the lookup tables and
drop lists in network_data/ - are found relative to the repository instead, so
they follow the checkout rather than either root.
"""
import os
from pathlib import Path

DATA_ROOT_VAR = 'RFS_DATA_ROOT'
TDX_ROOT_VAR = 'TDXHYDRO_ROOT'

# version controlled inputs: groupIds_table.csv, lake_table.csv, dropped_watersheds/, tdxhydro_splits/
repo_root = Path(__file__).resolve().parents[2]
network_data_root = repo_root / 'network_data'


def _root_from_env(var: str) -> Path:
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(
            f'${var} is not set. It is exported by scripts/pipeline.sh, which is how these '
            f'steps are normally run. To run one on its own, set ${DATA_ROOT_VAR} and ${TDX_ROOT_VAR} '
            f'first, e.g. {DATA_ROOT_VAR}=/Users/rchales/data/rfsv3 '
            f'{TDX_ROOT_VAR}=/Users/rchales/data/TDXHydroGeoParquet python 2_simplify_streams.py <region>'
        )
    return Path(value).expanduser()


data_root = _root_from_env(DATA_ROOT_VAR)

# the raw TDX-Hydro geoparquet is this pipeline's input, not its output, so it lives
# outside the data root and moves independently of it
tdx_root = _root_from_env(TDX_ROOT_VAR)

# the published dataset: one hive-style directory per group
group_root = data_root / 'hydrography'
global_root = group_root / 'group=0'  # global files are published as the group 0 dataset


def group_dir(group_id) -> Path:
    """The directory a group's files are published in. Hive-style so a reader can
    partition on groupId without the column being written into the files."""
    return group_root / f'group={group_id}'


# working files: everything the group files are built from, kept out of the deliverable
scratch_root = data_root / 'hydrography-scratchfiles'
region_root = scratch_root / 'regions'
pmtiles_root = scratch_root / 'pmtiles'
logs_root = scratch_root / 'logs'
