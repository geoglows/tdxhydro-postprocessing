"""
Filesystem layout for the pipeline.

Three directories, three environment variables, all exported by pipeline_env.sh, which both
stage drivers source before any step:

    OUTPUTS_ROOT                    everything the pipeline writes per release
    TDXHYDRO_GEOPARQUET_FINISHED    the conditioned TDX-Hydro geoparquet every release is built
                                    from (globally unique ids, duplicated watersheds dropped,
                                    basin coverage noded), plus the one-time products derived
                                    from it (global_basins/, global_koppen/, global_countries/)
    TDXHYDRO_GEOPARQUET_RAW         TDX-Hydro as published, untouched except converted from
                                    geopackage to geoparquet - by raw_processing/
                                    1_translate_tdxhydro.py, which reads the geopackages put here
                                    and then writes the finished set from the geoparquet

Every step derives its paths from those roots, so relocating either one is a
matter of changing the one export in that script.

Under the outputs root, the published dataset and the working files are kept apart:

    hydrography/region=<id>/        the deliverable - one directory per HydroBASINS level-2
                                    region, in the hive-style naming the published dataset uses
    hydrography/region=<id>/mods/   the edits simplify_streams.py made to the source TDX-Hydro, published
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

OUTPUTS_ROOT_VAR = 'OUTPUTS_ROOT'
TDX_ROOT_VAR = 'TDXHYDRO_GEOPARQUET_FINISHED'
TDX_RAW_ROOT_VAR = 'TDXHYDRO_GEOPARQUET_RAW'

# version controlled inputs: lake_table.csv, dropped_watersheds/, tdxhydro_splits/
repo_root = Path(__file__).resolve().parents[1]
network_data_root = repo_root / 'network_data'


def _root_from_env(var: str) -> Path:
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(
            f'${var} is not set. It is exported by pipeline_env.sh at the repository root, which is how these '
            f'steps are normally run. To run one on its own, set ${OUTPUTS_ROOT_VAR} and ${TDX_ROOT_VAR} '
            f'first, e.g. {OUTPUTS_ROOT_VAR}=/Users/rchales/data/rfsv3 '
            f'{TDX_ROOT_VAR}=/Users/rchales/data/tdxhydro_geoparquet_finished python simplify_streams.py <region>'
        )
    return Path(value).expanduser()


outputs_root = _root_from_env(OUTPUTS_ROOT_VAR)

# the conditioned TDX-Hydro geoparquet is this pipeline's input, not its output, so it lives
# outside the outputs root and moves independently of it
tdx_root = _root_from_env(TDX_ROOT_VAR)


def tdx_raw_root() -> Path:
    """TDX-Hydro as published. A function, not a constant, because only 1_translate_tdxhydro.py reads it: every
    other step runs without the raw set present or its variable set."""
    return _root_from_env(TDX_RAW_ROOT_VAR)

# the published dataset: one hive-style directory per level-2 region, plus the global products
publish_root = outputs_root / 'hydrography'
global_root = publish_root / 'global'


def publish_dir(region) -> Path:
    """The directory a region's published files go in. Hive-style so a reader can partition
    on the level-2 region without reading a column to do it."""
    return publish_root / f'region={region}'


def mods_dir(region) -> Path:
    """Where simplify_streams.py records the edits it made to the source TDX-Hydro, and the only copy of
    them there is.

    They are written straight into the published region directory instead of into scratch and
    copied out later, because two copies of a provenance record is one copy too many - the
    question "which source reach does this published reach stand for" has to have a single
    answer. create_catchments.py reads them back from here, so the published tree does hold one build input;
    that is the price of not duplicating them, and it is the reason create_catchments.py refuses to run when
    they are missing rather than treating an absent file as "nothing was edited".
    """
    return publish_dir(region) / 'mods'


# working files: everything the published files are built from, kept out of the deliverable
scratch_root = outputs_root / 'hydrography-scratchfiles'
region_root = scratch_root / 'regions'
pmtiles_root = scratch_root / 'pmtiles'
logs_root = scratch_root / 'logs'

# The frozen basin definition raw_processing/global_basins.py generates once from the raw inputs. It lives
# INSIDE the raw tree, not under the outputs root, because it shares the raw data's lifecycle
# exactly: derived from nothing but those files, consumed by every release, regenerated only if
# the raw data itself changes. Keeping it beside its source is also what keeps it safe - the
# scratch tree is wiped and rebuilt as a matter of course, and a permanent artifact parked there
# was destroyed by exactly such a wipe once.
global_basins_root = tdx_root / 'global_basins'

# The Koppen-Geiger class areas raw_processing/koppen_classifications.py derives from the raw catchments. Like
# the global basins they are derived from nothing but the raw data (plus the climate raster), so
# they live beside it and outlast every rebuild of the scratch tree. Two products per period:
#   parts/koppen_<period>_<region>.parquet           each raw catchment's land area per class -
#                                                     working data simplify_streams.py sums into koppenClass
#   koppen_upstream_<period>.parquet                 supplemental, never in the GIS files: the
#                                                     upstream climate of every raw reach, for
#                                                     anyone who wants to join it on riverId
#                                                     (assembled from parts/ once every region is done)
koppen_root = tdx_root / 'global_koppen'

# the climate period the published koppen columns describe. simplify_streams.py reads the labels for this
# period, so the classifier and simplify_streams.py must agree on it - pipeline_env.sh exports it for both
KOPPEN_PERIOD_VAR = 'KOPPEN_PERIOD'
default_koppen_period = '1991_2020'


def koppen_period() -> str:
    return os.environ.get(KOPPEN_PERIOD_VAR) or default_koppen_period


def koppen_part(region, period: str | None = None) -> Path:
    """One region's raw-catchment land area per Koppen class, keyed by TDXHydroLinkNo."""
    return koppen_root / 'parts' / f'koppen_{period or koppen_period()}_{region}.parquet'


def koppen_upstream_part(region, period: str | None = None) -> Path:
    """One region's supplemental upstream climate, keyed by TDXHydroLinkNo."""
    return koppen_root / 'parts' / f'koppen_upstream_{period or koppen_period()}_{region}.parquet'


def koppen_upstream_global(period: str | None = None) -> Path:
    return koppen_root / f'koppen_upstream_{period or koppen_period()}.parquet'


# The country areas raw_processing/country_classifications.py derives from the raw catchments and
# the geoBoundaries CGAZ ADM0 boundaries, beside the raw data like the Koppen ones:
#   parts/country_<region>.parquet          each raw catchment's land per country, long format
#                                           (TDXHydroLinkNo, country, km2) - simplify_streams.py sums these
#                                           into the published country column
#   country_upstream.parquet                supplemental, never in the GIS files: per raw reach,
#                                           its country with ISO3 and disputed flag, its outlet's
#                                           country, the countries upstream with their shares,
#                                           and a transboundary flag
#   countries.parquet                       the label table (country, countryIso3, countryDisputed)
#   country_pieces.parquet                  the boundaries cut small for overlaying, a cache
country_root = tdx_root / 'global_countries'


def country_part(region) -> Path:
    """One region's raw-catchment land per country, keyed by TDXHydroLinkNo."""
    return country_root / 'parts' / f'country_{region}.parquet'


def country_upstream_part(region) -> Path:
    return country_root / 'parts' / f'country_upstream_{region}.parquet'


country_upstream_global = country_root / 'country_upstream.parquet'
country_table = country_root / 'countries.parquet'
country_pieces = country_root / 'country_pieces.parquet'
