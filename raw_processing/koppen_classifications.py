#!/usr/bin/env python
"""Measure the Koppen-Geiger climate of every raw TDX-Hydro catchment and of its upstream area.

Two products per climate period, both under $TDXHYDRO_GEOPARQUET_FINISHED/global_koppen/ (hy.paths.koppen_*):

1. parts/koppen_<period>_<region>.parquet - working data, read by simplify_streams.py. One row per raw
   catchment: TDXHydroLinkNo, TDXHydroRegion, and its land area in km2 in each of the 30 classes
   (km2_Af ... km2_EF, hy.koppen.AREA_COLUMNS). simplify_streams.py sums these rows over the raw catchments
   that make up each published reach's catchment and publishes the winner as koppenClass - the
   only climate attribute in the GIS files.

2. koppen_upstream_<period>.parquet - a supplemental table, not part of the GIS files, for
   analyses that want to know the climate a reach's water comes from (e.g. evaluating a reach or
   a gauge against how arid its basin is). Written per region under parts/ and combined once
   every region is done. One row per raw reach, joined to the published streams on
   riverId = TDXHydroLinkNo:

    TDXHydroLinkNo        : int64    the raw, globally unique reach id
    TDXHydroRegion        : string   the HydroBASINS level-2 region
    koppenUpstreamClass   : string   the class with the most land over the reach's whole upstream
                                     area - its own catchment plus every catchment that drains
                                     into it - as letters (Af, BWh, Cfb, ...); null with no land
    koppenUpstreamFracA   : float32  the fraction of that upstream land in each main group
    ... FracB, FracC, FracD, FracE   (A tropical, B arid, C temperate, D cold, E polar); they sum
                                     to 1, NaN with no land. A single dominant class hides a
                                     basin that is 45% arid and 40% temperate; these do not

Only catchments with a reach in the 1_translate_tdxhydro.py streamnet are kept. 1_translate_tdxhydro.py drops the duplicated
watersheds (network_data/tdxhydro_splits/duplicated_watersheds.csv - small watersheds on a region
edge that TDX-Hydro delineated in both neighbouring regions) from the streamnet but not from the
basins, so their leftover catchments are dropped here too; the twin in the region that kept the
watershed is labelled there, and no land is counted twice.

A catchment's area in a class is its class fraction x its covered pixels x the ground area of a
pixel at its latitude (a pixel shrinks toward the poles, and an upstream area or a merged
catchment can span many degrees). Every label is a vote by that land area: a catchment that is
60% Cfb and 40% Dfb is Cfb, and a tie goes to the lower class number in legend.txt.

Upstream areas are summed downstream along the 1_translate_tdxhydro.py streamnet's LINKNO -> DSLINKNO. A TDX-Hydro
region is closed - no reach drains out of it - so summing within a region is complete.

Environment, in addition to what pipeline_env.sh already exports:

    KOPPEN_TIF          the raster (required), e.g.
                        .../koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif
    KOPPEN_PERIOD       the label the outputs are named with; defaults to the raster's folder
                        name (1991_2020 above). Set it for a projected period - see below
    KOPPEN_JOBS         regions labelled at once (default 6)
    KOPPEN_BATCH_SIZE   catchments read and overlaid at a time (default 100000); lower it to
                        use less memory on a large region

Usage: koppen_classifications.py [region ...]
       (no arguments means every region under TDXHYDRO_GEOPARQUET_FINISHED)

Data source: the Beck et al. (2023) Koppen-Geiger maps, version 3, downloaded from
https://www.gloh2o.org/koppen/ under "Data access". That link downloads one zip of everything,
koppen_geiger_tif.zip, which unpacks to a koppen_geiger_tif/ folder holding legend.txt and one
folder per period: 1901_1930, 1931_1960, 1961_1990 and 1991_2020 (observed), and 2041_2070 and
2071_2099 (projected), each of those split into ssp119, ssp126, ssp245, ssp370, ssp434, ssp460
and ssp585. Every period folder holds the same global map at four resolutions:
koppen_geiger_0p00833333.tif (1 km), _0p1.tif, _0p5.tif and _1p0.tif (0.1, 0.5 and 1 degree).
The file used is the 1 km map for the present-day climate:

    koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif

For a projected period, set KOPPEN_PERIOD (e.g. 2041_2070_ssp245): the raster's folder name is
then only the scenario (ssp245), so without it 2041_2070 and 2071_2099 would write the same file.

Citation for Koppen map:

    Beck, H. E., T. R. McVicar, N. Vergopolan, A. Berg, N. J. Lutsko, A. Dufour, Z. Zeng,
    X. Jiang, A. I. J. M. van Dijk, and D. G. Miralles. High-resolution (1 km) Koppen-Geiger maps
    for 1901-2099 based on constrained CMIP6 projections. Scientific Data 10, 724 (2023).
    https://www.nature.com/articles/s41597-023-02549-6
"""
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
from exactextract import exact_extract

import hydrography as hy

CLASS_COUNT = hy.koppen.CLASS_COUNT
BATCH_SIZE = int(os.environ.get('KOPPEN_BATCH_SIZE', 100_000))

upstream_class = 'koppenUpstreamClass'
upstream_fractions = tuple(f'koppenUpstreamFrac{g}' for g in hy.koppen.GROUPS)

EARTH_RADIUS_KM = 6371.0088


def raster_path() -> Path:
    value = os.environ.get('KOPPEN_TIF')
    if not value:
        sys.exit('$KOPPEN_TIF is not set - point it at a Koppen-Geiger GeoTIFF, e.g. '
                 '.../1991_2020/koppen_geiger_0p00833333.tif')
    path = Path(value).expanduser()
    if not path.exists():
        sys.exit(f'{path} not found')
    return path


def period_label(raster: Path) -> str:
    return os.environ.get('KOPPEN_PERIOD') or raster.parent.name


def pixel_area_km2(latitude: np.ndarray, res_x: float, res_y: float) -> np.ndarray:
    """The ground area of one raster pixel centred on each latitude, on a spherical earth."""
    return (EARTH_RADIUS_KM ** 2 * np.radians(res_x) * np.radians(res_y)
            * np.cos(np.radians(latitude)))


def class_areas(stats: pd.DataFrame, pixel_area: np.ndarray, region: str) -> np.ndarray:
    """Each catchment's land area in each class, km2, one column per raster value (0 unused).

    exactextract's unique values are not in any particular order, so they are scattered into
    fixed columns here, and the column index is what makes a tie go to the lower class number.
    """
    areas = np.zeros((len(stats), CLASS_COUNT + 1))
    for row, (values, fractions, count) in enumerate(zip(stats['unique'], stats['frac'],
                                                         stats['count'])):
        if len(values):
            values = np.asarray(values, dtype=np.int64)
            if values.max() > CLASS_COUNT:
                raise RuntimeError(f'{region}: a class above {CLASS_COUNT} - is this a '
                                   f'Koppen-Geiger raster?')
            areas[row, values] = np.asarray(fractions) * count * pixel_area[row]
    return areas


def label_region(basins: Path, raster: Path, period: str) -> str:
    region = hy.raw.region_of(basins)
    outputs = (hy.paths.koppen_part(region, period), hy.paths.koppen_upstream_part(region, period))
    if all(o.exists() for o in outputs):
        return f'{region}: already labelled, skipped'
    started = time.time()
    if not hy.coverage.is_clean(basins):
        logging.warning(f'{region}: {basins.name} has not been coverage-cleaned by 1_translate_tdxhydro.py; the '
                        f'labels are unaffected in practice, but create_catchments.py builds from the cleaned '
                        f'polygons')

    streamnet = hy.raw.streamnet_of(basins)
    if not streamnet.exists():
        raise FileNotFoundError(f'{streamnet} not found; the upstream label needs the network')

    ids = hy.raw.id_column(basins)
    crs = hy.raw.source_crs(basins)
    with rasterio.open(raster) as src:
        raster_crs = pyproj.CRS.from_user_input(src.crs)
        res_x, res_y = src.res
    if not raster_crs.is_geographic:
        raise RuntimeError(f'{raster.name} is not in degrees; pixel_area_km2 assumes it is')
    reproject = not crs.equals(raster_crs, ignore_axis_order=True)

    link_ids, area_rows = [], []
    for chunk in hy.raw.batches(basins, ids, crs, BATCH_SIZE):
        if reproject:
            chunk = chunk.to_crs(raster_crs)
        stats = exact_extract(str(raster), chunk, ['unique', 'frac', 'count'], output='pandas')
        if len(stats) != len(chunk):
            raise RuntimeError(f'{region}: exactextract returned {len(stats):,} rows for '
                               f'{len(chunk):,} catchments')
        bounds = chunk.geometry.bounds
        latitude = ((bounds['miny'] + bounds['maxy']) / 2).to_numpy()
        link_ids.append(chunk[ids].to_numpy())
        area_rows.append(class_areas(stats, pixel_area_km2(latitude, res_x, res_y), region))
    link_ids = np.concatenate(link_ids)
    # column 0 is the raster's ocean/nodata value: only land votes
    areas = np.concatenate(area_rows)[:, 1:]
    # some regions (1020018110, 5020049720) store a catchment as several rows - disjoint pieces,
    # mostly slivers, not copies - so a catchment's land is the sum over its rows, as create_catchments.py's
    # dissolve unions them
    if pd.Index(link_ids).has_duplicates:
        pieces = len(link_ids)
        summed = pd.DataFrame(areas).groupby(link_ids, sort=False).sum()
        link_ids, areas = summed.index.to_numpy(np.int64), summed.to_numpy()
        logging.info(f'{region}: {pieces:,} basin rows summed into {len(link_ids):,} catchments')

    # sum the catchments' areas down the network, reaches with no catchment adding nothing
    network = pd.read_parquet(streamnet, columns=[hy.schema.tdx_link_field, hy.schema.tdx_ds_link_field])
    reach_ids = network[hy.schema.tdx_link_field].to_numpy(np.int64)
    catchment_row = pd.Index(link_ids).get_indexer(reach_ids)
    reach_areas = np.zeros((len(reach_ids), CLASS_COUNT))
    has_catchment = catchment_row >= 0
    reach_areas[has_catchment] = areas[catchment_row[has_catchment]]
    upstream_areas = hy.raw.accumulate(reach_ids, network[hy.schema.tdx_ds_link_field].to_numpy(np.int64),
                                reach_areas)

    # a catchment with no reach is one of the duplicated watersheds 1_translate_tdxhydro.py dropped from the
    # streamnet but not from the basins: its twin is labelled in the region that kept it
    reach_row = pd.Index(reach_ids).get_indexer(link_ids)
    in_network = reach_row >= 0
    dropped = int((~in_network).sum())
    link_ids = link_ids[in_network]
    areas = areas[in_network]
    upstream_areas = upstream_areas[reach_row[in_network]]

    catchments = pd.DataFrame(areas.astype(np.float32), columns=list(hy.koppen.AREA_COLUMNS))
    catchments.insert(0, hy.schema.tdx_link_no_field, link_ids)
    catchments.insert(1, hy.schema.tdx_region_field, region)

    upstream = pd.DataFrame({
        hy.schema.tdx_link_no_field: link_ids,
        hy.schema.tdx_region_field: region,
        upstream_class: hy.koppen.dominant(upstream_areas),
    })
    for column, values in zip(upstream_fractions, hy.koppen.group_fractions(upstream_areas).T):
        upstream[column] = values.astype(np.float32)

    for table, output in zip((catchments, upstream), outputs):
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f'{output.name}.partial')
        hy.parquet.write_parquet(table, partial, index=False, row_group_size=None)
        partial.replace(output)

    own_class = pd.Series(hy.koppen.dominant(areas))
    unlabelled = int(own_class.isna().sum())
    classes = int(own_class.nunique())
    differs = int((upstream[upstream_class] != own_class).fillna(False).sum())
    return (f'{region}: {len(link_ids):,} catchments, {classes} classes, {unlabelled:,} with no '
            f'land pixel, {differs:,} with a different upstream class, {dropped:,} duplicated '
            f'catchments dropped, {time.time() - started:.0f}s')


if __name__ == '__main__':
    hy.console.banner('Label raw TDX-Hydro catchments with Koppen-Geiger classes')
    raster = raster_path()
    period = period_label(raster)
    final = hy.paths.koppen_upstream_global(period)
    if final.exists():
        # exit 0, not 1: pipeline.sh runs under `set -e`, so a step with nothing to do must report
        # success or it aborts the whole run
        print(f'{final.name} exists, nothing to do')
        sys.exit(0)

    basins = sorted(hy.paths.tdx_root.glob('TDX_streamreach_basins_*_01.parquet'),
                    key=lambda p: p.stat().st_size, reverse=True)
    if not basins:
        sys.exit(f'no TDX_streamreach_basins parquet under {hy.paths.tdx_root} - run raw_processing/1_translate_tdxhydro.py first')
    wanted = set(sys.argv[1:])
    if wanted:
        basins = [p for p in basins if hy.raw.region_of(p) in wanted]
        print(f'{len(basins)} region(s) selected by argument')

    hy.paths.logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=hy.paths.logs_root / f'koppen_{period}.log', filemode='w',
                        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.info(f'raster {raster}, period {period}, {len(basins)} region(s)')

    started = time.time()
    workers = max(1, int(os.environ.get('KOPPEN_JOBS', 6)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for message in pool.map(label_region, basins, [raster] * len(basins),
                                [period] * len(basins)):
            logging.info(message)
            print(message)

    # the global file is written only from a complete set, so a partial run can never be mistaken
    # for the whole world once it exists
    every_region = sorted(hy.raw.region_of(p) for p in
                          hy.paths.tdx_root.glob('TDX_streamreach_basins_*_01.parquet'))
    missing = [r for r in every_region if not hy.paths.koppen_upstream_part(r, period).exists()]
    if missing:
        print(f'{len(missing)} region(s) not labelled yet, not writing {final.name}')
        sys.exit(0)
    table = pd.concat([pd.read_parquet(hy.paths.koppen_upstream_part(r, period)) for r in every_region],
                      ignore_index=True)
    if table[hy.schema.tdx_link_no_field].duplicated().any():
        raise RuntimeError('raw ids are not globally unique across regions')
    partial = final.with_name(f'{final.name}.partial')
    hy.parquet.write_parquet(table, partial, index=False, row_group_size=None)
    partial.replace(final)
    print(f'{len(table):,} reaches, upstream climate -> {final}, {time.time() - started:.0f}s')
