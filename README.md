# tdxhydro-postprocessing

The pipeline that builds the RFS v3 hydrography from TDX-Hydro. What it publishes, the schema and layout of every file,
is in the [RFS v3 specification](https://github.com/river-forecast-system/rfs-specification-documents/blob/main/docs/specs/rfs-v3.md#hydrography).

## Hydrography preparation

The hydrography is built once per release, not daily, by the `tdxhydro-postprocessing` repository, in two stages: `raw_processing/pipeline.sh` once, then `final_processing/pipeline.sh` per build. Both read their settings from `pipeline_env.sh`. It turns the raw TDX-Hydro stream and basin geopackages into the products
in [Hydrography](https://github.com/river-forecast-system/rfs-specification-documents/blob/main/docs/specs/rfs-v3.md#hydrography). Three directories, three environment variables:

```text
$TDXHYDRO_GEOPARQUET_RAW/               TDX-Hydro as published: put the geopackages here, and raw_processing 1_translate_tdxhydro.py writes each,
    TDX_streamnet_<region>_01.gpkg      untouched, as a geoparquet beside it (the geopackage may then be deleted)
    TDX_streamnet_<region>_01.parquet
    TDX_streamreach_basins_<region>_01.parquet
$TDXHYDRO_GEOPARQUET_FINISHED/          the conditioned TDX-Hydro every build reads, plus global_koppen/, global_countries/
    TDX_streamnet_<region>_01.parquet
    TDX_streamreach_basins_<region>_01.parquet
$OUTPUTS_ROOT/
    hydrography/region=<id>/            the published dataset, uploaded as is to s3://river-forecast-system-v3/hydrography/
    hydrography/global/                 the products that span every region
    hydrography-scratchfiles/           regions/, pmtiles/, logs/ — per region intermediates no consumer needs
```

The code is three directories at the repository root: `raw_processing/` and `final_processing/` hold the two stages' scripts and drivers, and `hydrography/` is the library they share. `uv sync` installs `hydrography` into `.venv` in editable mode, so every script imports it directly and edits to it take effect without reinstalling.

Version controlled inputs live in the repository's `network_data/`: `lake_table.csv` (inlet, outlet, lake id, region, endorheic flag, trace flag, HydroLAKES and GDW ids; every inlet drains to its outlet in raw TDX-Hydro), `lake_outlet_links.csv` (riverId, nextRiverId, note: the connectivity overrides, all explicit, each repointing a lake's terminal arm at the lake's outlet; the build stops if any `lake_id` still has more than one outlet once they are applied), `dropped_watersheds/*.csv` (outlets to remove),
and `tdxhydro_splits/` (the per region id offsets and the duplicated watersheds to drop where regions overlap).

The published tree holds one build input: `region=<id>/mods/`, which simplify_streams.py writes and create_catchments.py reads back. That is the price of the edit records having exactly one copy rather than a scratch original
and a published duplicate, and it is why create_catchments.py refuses to run when they are absent rather than treating a missing file as "nothing was edited".

### One-time raw processing

`raw_processing/pipeline.sh` turns the raw TDX-Hydro into the inputs every build reads, and computes the per-reach attributes that depend only on the raw data and a fixed external
source. It runs once (or once per new input, such as a climate period), not per build, and writes beside the raw data under `$TDXHYDRO_GEOPARQUET_FINISHED` so its products outlast every rebuild of the
published tree. Every product is keyed by the raw `TDXHydroLinkNo`, which is the published `riverId` of the reach that kept it. The build joins in the few attributes it publishes; the rest stay
supplemental tables, never part of the GIS files, for whoever wants to join them.

| # | Script                           | Produces                                                                                                                                                                                                                                                                                       |
|---|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | `1_translate_tdxhydro.py`        | TDX-Hydro geopackages to raw geoparquet, then the finished set. Offsets `LINKNO` by a per region constant so ids are globally unique, adds the geodesic length, region id, and outlet lon/lat, drops watersheds duplicated between overlapping regions, and nodes the source basin coverage so later dissolves are exact          |
| - | `koppen_classifications.py`    | `global_koppen/parts/koppen_<period>_<region>.parquet`: each raw catchment's land area per Koppen-Geiger class, which `simplify_streams.py` sums into the published `koppenClass`. And `global_koppen/koppen_upstream_<period>.parquet`, supplemental: each reach's dominant upstream class and main-group fractions |
| - | `country_classifications.py`   | `global_countries/parts/country_<region>.parquet`: each raw catchment's land per country (geoBoundaries CGAZ ADM0, CC BY 4.0; disputed areas by their own names), which `simplify_streams.py` sums into the published `country`. And `global_countries/country_upstream.parquet`, supplemental: ISO3, disputed flag, outlet country, upstream countries and shares, transboundary flag |

### Build

`final_processing/pipeline.sh` polishes those inputs into the published product. simplify_streams.py and create_catchments.py run per region in parallel; everything else is global.

| # | Script                    | Scope                 | Produces                                                                                                                                                                                                                                                                                                                                    |
|---|---------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| - | `simplify_streams.py`   | per region            | `streams_*`, `metadata_*`, `confluences_*` in the region's scratch directory. Applies the [network simplification](https://github.com/river-forecast-system/rfs-specification-documents/blob/main/docs/specs/rfs-v3.md#network-simplification) and [lake edits](https://github.com/river-forecast-system/rfs-specification-documents/blob/main/docs/specs/rfs-v3.md#lakes-and-reservoirs), sets the nested-set row order, and writes every edit to the region's **published** `mods/*.json`                    |
| - | `create_catchments.py`  | per region            | `catchments_*`: the source basins redirected along simplify_streams.py's edits and dissolved into one polygon per surviving reach, projected, coverage simplified at 20 m, snapped to 1 m, in the published row order                                                                                                                                    |
| - | `concatenate_global.py` | all regions           | Concatenates the regions in ascending region number, stamps the global `riverIndex` by offset arithmetic, checks that no reach drains across a region, that `riverId` is globally unique, and that the nested-set property holds; writes `global/metadata.parquet` and `metadata.zarr`; publishes every region's tables into `region=<id>/` |
| - | `publish_regions.py`    | all regions           | Dissolves each region's published catchments into `boundary_<id>.geo.parquet`                                                                                                                                                                                                                                                               |
| - | `tile_streams.sh`         | per region, then join | `streams.pmtiles`: one tippecanoe run per region, largest first, joined with `tile-join`                                                                                                                                                                                                                                                    |
| - | `tile_regions.sh`         | global                | `regions.pmtiles`. **Not currently run**                                                                                                                                                                                                                                                                                                    |

Every step is idempotent at the granularity of its output files: a step whose outputs exist exits successfully without rewriting them, so a rebuild after a change touches only what depends on it.
Reordering the rows (concatenate_global.py) never requires rerunning the simplification (simplify_streams.py) or the catchments (create_catchments.py); it permutes rows and derives integers.

Then, outside the numbered steps: optionally, `extras_identify_id_map.py` writes `global/tdxhydro_to_v3_id_map.parquet`, a two column table mapping every original TDX-Hydro
reach in every region, about 16 million, to the v3 `riverId` that now represents it, or null when it was dropped or is in a region v3 does not cover.

The pipeline writes no routing configuration, the Muskingum k and x included. `routing.parquet` and the grid weights are made from its published output afterwards, by the
Calibration Scripts, into `routing/region=<id>/` beside `hydrography/`, see [Routing Configurations](https://github.com/river-forecast-system/rfs-specification-documents/blob/main/docs/specs/rfs-v3.md#routing-configurations).
