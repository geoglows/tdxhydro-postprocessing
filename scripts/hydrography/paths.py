"""
Filesystem layout for the pipeline.

Two directories, two environment variables, both exported by
pipeline.sh before it runs any step:

    RFS_DATA_ROOT    everything the pipeline writes per release
    TDXHYDRO_ROOT    the raw TDX-Hydro geoparquet the releases are built from, kept separately
                     from the outputs, plus the one-time global basins derived from it
                     (global_basins/) - generated once, consumed by every release

Every step derives its paths from those roots, so relocating either one is a
matter of changing the one export in that script.

Under the data root, the published dataset and the working files are kept apart:

    hydrography/region=<id>/        the deliverable - one directory per HydroBASINS level-2
                                    region, in the hive-style naming the published dataset uses
    hydrography/region=<id>/mods/   the edits step 3 made to the source TDX-Hydro, published
                                    rather than kept as scratch: they are the provenance of a
                                    network that is a modification of a previous dataset, and
                                    they are the only record of which source reach a published
                                    reach absorbed. Written once, where they are published - see
                                    mods_dir() below.
    hydrography/global/             the products that span every region - metadata.parquet,
                                    metadata.zarr, the joined pmtiles
    hydrography-scratchfiles/       per-region intermediates, per-region pmtiles and
                                    logs - everything the published files are built from,
                                    which no consumer of the dataset needs

The level-2 region is the only partition the dataset has. It is the unit the raw TDX-Hydro
comes in, the unit every step processes, and the unit the release publishes, so a reach's
region can be read off its directory, its filename and its TDXHydroRegion column alike, and
no reach ever drains out of the region it is filed under.

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

# version controlled inputs: lake_table.csv, dropped_watersheds/, tdxhydro_splits/
repo_root = Path(__file__).resolve().parents[2]
network_data_root = repo_root / 'network_data'


def _root_from_env(var: str) -> Path:
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(
            f'${var} is not set. It is exported by scripts/pipeline.sh, which is how these '
            f'steps are normally run. To run one on its own, set ${DATA_ROOT_VAR} and ${TDX_ROOT_VAR} '
            f'first, e.g. {DATA_ROOT_VAR}=/Users/rchales/data/rfsv3 '
            f'{TDX_ROOT_VAR}=/Users/rchales/data/TDXHydroGeoParquet python 3_simplify_streams.py <region>'
        )
    return Path(value).expanduser()


data_root = _root_from_env(DATA_ROOT_VAR)

# the raw TDX-Hydro geoparquet is this pipeline's input, not its output, so it lives
# outside the data root and moves independently of it
tdx_root = _root_from_env(TDX_ROOT_VAR)

# the published dataset: one hive-style directory per level-2 region, plus the global products
publish_root = data_root / 'hydrography'
global_root = publish_root / 'global'


def publish_dir(region) -> Path:
    """The directory a region's published files go in. Hive-style so a reader can partition
    on the level-2 region without reading a column to do it."""
    return publish_root / f'region={region}'


def mods_dir(region) -> Path:
    """Where step 3 records the edits it made to the source TDX-Hydro, and the only copy of
    them there is.

    They are written straight into the published region directory instead of into scratch and
    copied out later, because two copies of a provenance record is one copy too many - the
    question "which source reach does this published reach stand for" has to have a single
    answer. Step 4 reads them back from here, so the published tree does hold one build input;
    that is the price of not duplicating them, and it is the reason step 4 refuses to run when
    they are missing rather than treating an absent file as "nothing was edited".
    """
    return publish_dir(region) / 'mods'


# working files: everything the published files are built from, kept out of the deliverable
scratch_root = data_root / 'hydrography-scratchfiles'
region_root = scratch_root / 'regions'
pmtiles_root = scratch_root / 'pmtiles'
logs_root = scratch_root / 'logs'

# The frozen basin definition 2_global_basins.py generates once from the raw inputs. It lives
# INSIDE the raw tree, not under the data root, because it shares the raw data's lifecycle
# exactly: derived from nothing but those files, consumed by every release, regenerated only if
# the raw data itself changes. Keeping it beside its source is also what keeps it safe - the
# scratch tree is wiped and rebuilt as a matter of course, and a permanent artifact parked there
# was destroyed by exactly such a wipe once.
global_basins_root = tdx_root / 'global_basins'
