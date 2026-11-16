#!/usr/bin/env python
"""Label every raw TDX-Hydro catchment with its dominant Koppen-Geiger climate class.

It reads the raw step-1 inputs and writes one parquet per region to
$TDXHYDRO_ROOT/global_koppen/parts/koppen_<period>_<region>.parquet, then, once every region
under $TDXHYDRO_ROOT is labelled, combines them into
$TDXHYDRO_ROOT/global_koppen/koppen_<period>.parquet.

Four columns, one row per raw catchment:

    TDXHydroLinkNo : int64    the raw, globally unique reach id
    TDXHydroRegion : string   the HydroBASINS level-2 region the catchment belongs to
    koppen         : string   the winning class as its Koppen-Geiger letters (Af, BWh, Cfb, ...),
                              one of the 30 in the Beck et al. (2023) legend.txt; null when the
                              catchment covers no land pixel at all.
    koppenUpstream : string   the winning class over the reach's whole upstream area - its own
                              catchment plus every catchment that drains into it - in the same
                              letters; null when there is no land upstream

Only catchments with a reach in the step-1 streamnet are kept. Step 1 drops the duplicated
watersheds (network_data/tdxhydro_splits/duplicated_watersheds.csv - small watersheds on a region
edge that TDX-Hydro delineated in both neighbouring regions) from the streamnet but not from the
basins, so their leftover catchments are dropped here too; the twin in the region that kept the
watershed is labelled there, and no land is counted twice.

A catchment's class is chosen by the fraction of its area covered by each class, so a catchment
that is 60% Cfb and 40% Dfb is labelled Cfb. A tie goes to the lower class number in legend.txt.

The upstream label is a vote by land area, not by catchment: each catchment's area in every class
(its class fractions x its covered pixels x the ground area of a pixel at its latitude, since a
pixel shrinks toward the poles and an upstream area can span many degrees) is summed downstream
along the step-1 streamnet's LINKNO -> DSLINKNO, and the class with the most area wins. A
TDX-Hydro region is closed - no reach drains out of it - so summing within a region is complete.

Environment, in addition to what pipeline_env.sh already exports:

    KOPPEN_TIF          the raster (required), e.g.
                        .../koppen_geiger_tif/1991_2020/koppen_geiger_0p00833333.tif
    KOPPEN_PERIOD       the label the outputs are named with; defaults to the raster's folder
                        name (1991_2020 above). Set it for a projected period - see below
    KOPPEN_JOBS         regions labelled at once (default 6)
    KOPPEN_BATCH_SIZE   catchments read and overlaid at a time (default 100000); lower it to
                        use less memory on a large region

Usage: koppen_classifications.py [region ...]
       (no arguments means every region under TDXHYDRO_ROOT)

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
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyproj
import rasterio
import shapely
from exactextract import exact_extract

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

# the raster's values 1-30 in legend.txt order; index 0 is the raster's ocean/nodata value
CLASS_CODES = (None,
               'Af', 'Am', 'Aw', 'BWh', 'BWk', 'BSh', 'BSk', 'Csa', 'Csb', 'Csc',
               'Cwa', 'Cwb', 'Cwc', 'Cfa', 'Cfb', 'Cfc', 'Dsa', 'Dsb', 'Dsc', 'Dsd',
               'Dwa', 'Dwb', 'Dwc', 'Dwd', 'Dfa', 'Dfb', 'Dfc', 'Dfd', 'ET', 'EF')
CLASS_COUNT = len(CLASS_CODES) - 1
BATCH_SIZE = int(os.environ.get('KOPPEN_BATCH_SIZE', 100_000))

koppen_code = 'koppen'
koppen_upstream = 'koppenUpstream'

EARTH_RADIUS_KM = 6371.0088

out_dir = hy.paths.tdx_root / 'global_koppen'
parts_dir = out_dir / 'parts'


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


def part_output(period: str, region: str) -> Path:
    return parts_dir / f'koppen_{period}_{region}.parquet'


def global_output(period: str) -> Path:
    return out_dir / f'koppen_{period}.parquet'


def region_of(basins: Path) -> str:
    return basins.name.split('_')[-2]


def source_crs(path: Path) -> pyproj.CRS:
    """The CRS as the file's GeoParquet metadata states it; a null CRS is the spec's CRS84."""
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    geo = json.loads(metadata.get(b'geo', b'{}'))
    column = geo.get('primary_column', hy.schema.geometry)
    crs = geo.get('columns', {}).get(column, {}).get('crs', 'missing')
    if crs == 'missing':
        raise RuntimeError(f'{path.name} carries no GeoParquet CRS metadata')
    return pyproj.CRS.from_json_dict(crs) if crs is not None else pyproj.CRS.from_user_input('OGC:CRS84')


def id_column(path: Path) -> str:
    names = pq.read_schema(path).names
    return hy.schema.tdx_link_no_field if hy.schema.tdx_link_no_field in names \
        else hy.schema.tdx_link_field


def batches(path: Path, ids: str, crs: pyproj.CRS):
    """The file as GeoDataFrames of BATCH_SIZE rows, so a large region is never held whole."""
    parquet = pq.ParquetFile(path)
    geoarrow = parquet.schema_arrow.field(hy.schema.geometry).metadata is not None
    for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=[ids, hy.schema.geometry]):
        if geoarrow:
            geometry = gpd.GeoDataFrame.from_arrow(batch).geometry.values
        else:
            geometry = shapely.from_wkb(batch.column(hy.schema.geometry).to_numpy(zero_copy_only=False))
        yield gpd.GeoDataFrame({ids: batch.column(ids).to_numpy().astype(np.int64)},
                               geometry=geometry, crs=crs)


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


def winners(areas: np.ndarray) -> np.ndarray:
    """The winning raster value per row (0 where there is no land); argmax keeps the lower code on a tie."""
    return np.where(areas.sum(axis=1) > 0, np.argmax(areas, axis=1), 0).astype(np.uint8)


def accumulate(link: np.ndarray, downstream: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Each reach's areas plus those of every reach upstream of it.

    Processed in waves from the headwaters: a reach is added into the one below it only once
    everything above it has been added into it. A downstream id that is not in the table (-1 at
    an outlet) ends the path.
    """
    down = pd.Index(link).get_indexer(downstream)
    total = areas.copy()
    waiting = np.bincount(down[down >= 0], minlength=len(link))
    frontier = np.flatnonzero(waiting == 0)
    done = 0
    while frontier.size:
        done += frontier.size
        frontier = frontier[down[frontier] >= 0]
        targets = down[frontier]
        np.add.at(total, targets, total[frontier])
        np.subtract.at(waiting, targets, 1)
        targets = np.unique(targets)
        frontier = targets[waiting[targets] == 0]
    if done != len(link):
        raise RuntimeError(f'{len(link) - done:,} reaches never drained - the network has a cycle')
    return total


def label_region(basins: Path, raster: Path, period: str) -> str:
    region = region_of(basins)
    output = part_output(period, region)
    if output.exists():
        return f'{region}: already labelled, skipped'
    started = time.time()
    if not hy.coverage.is_clean(basins):
        logging.warning(f'{region}: {basins.name} has not been coverage-cleaned by step 1; the '
                        f'labels are unaffected in practice, but step 4 builds from the cleaned '
                        f'polygons')

    streamnet = basins.with_name(basins.name.replace('streamreach_basins', 'streamnet'))
    if not streamnet.exists():
        raise FileNotFoundError(f'{streamnet} not found; the upstream label needs the network')

    ids = id_column(basins)
    crs = source_crs(basins)
    with rasterio.open(raster) as src:
        raster_crs = pyproj.CRS.from_user_input(src.crs)
        res_x, res_y = src.res
    if not raster_crs.is_geographic:
        raise RuntimeError(f'{raster.name} is not in degrees; pixel_area_km2 assumes it is')
    reproject = not crs.equals(raster_crs, ignore_axis_order=True)

    link_ids, area_rows = [], []
    for chunk in batches(basins, ids, crs):
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
    areas = np.concatenate(area_rows)
    if pd.Index(link_ids).has_duplicates:
        raise RuntimeError(f'{region}: a catchment id appears more than once in {basins.name}')

    # sum the catchments' areas down the network, reaches with no catchment adding nothing
    network = pd.read_parquet(streamnet, columns=[hy.schema.tdx_link_field, hy.schema.tdx_ds_link_field])
    reach_ids = network[hy.schema.tdx_link_field].to_numpy(np.int64)
    catchment_row = pd.Index(link_ids).get_indexer(reach_ids)
    reach_areas = np.zeros((len(reach_ids), CLASS_COUNT + 1))
    has_catchment = catchment_row >= 0
    reach_areas[has_catchment] = areas[catchment_row[has_catchment]]
    upstream_areas = accumulate(reach_ids, network[hy.schema.tdx_ds_link_field].to_numpy(np.int64),
                                reach_areas)

    # a catchment with no reach is one of the duplicated watersheds step 1 dropped from the
    # streamnet but not from the basins: its twin is labelled in the region that kept it
    reach_row = pd.Index(reach_ids).get_indexer(link_ids)
    in_network = reach_row >= 0
    dropped = int((~in_network).sum())
    codes = np.array(CLASS_CODES, dtype=object)
    table = pd.DataFrame({
        hy.schema.tdx_link_no_field: link_ids[in_network],
        hy.schema.tdx_region_field: region,
        koppen_code: pd.array(codes[winners(areas[in_network])], dtype='string'),
        koppen_upstream: pd.array(codes[winners(upstream_areas[reach_row[in_network]])],
                                  dtype='string'),
    })

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f'{output.name}.partial')
    hy.parquet.write_parquet(table, partial, index=False, row_group_size=None)
    partial.replace(output)

    unlabelled = int(table[koppen_code].isna().sum())
    classes = int(table[koppen_code].nunique())
    differs = int((table[koppen_upstream] != table[koppen_code]).fillna(False).sum())
    return (f'{region}: {len(table):,} catchments, {classes} classes, {unlabelled:,} with no '
            f'land pixel, {differs:,} with a different upstream class, {dropped:,} duplicated '
            f'catchments dropped, {time.time() - started:.0f}s')


if __name__ == '__main__':
    hy.console.banner('Label raw TDX-Hydro catchments with Koppen-Geiger classes')
    raster = raster_path()
    period = period_label(raster)
    final = global_output(period)
    if final.exists():
        # exit 0, not 1: pipeline.sh runs under `set -e`, so a step with nothing to do must report
        # success or it aborts the whole run
        print(f'{final.name} exists, nothing to do')
        sys.exit(0)

    basins = sorted(hy.paths.tdx_root.glob('TDX_streamreach_basins_*_01.parquet'),
                    key=lambda p: p.stat().st_size, reverse=True)
    if not basins:
        sys.exit(f'no TDX_streamreach_basins parquet under {hy.paths.tdx_root} - run step 1 first')
    wanted = set(sys.argv[1:])
    if wanted:
        basins = [p for p in basins if region_of(p) in wanted]
        print(f'{len(basins)} region(s) selected by argument')

    hy.paths.logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=hy.paths.logs_root / f'koppen_classifications_{period}.log', filemode='w',
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
    every_region = sorted(region_of(p) for p in
                          hy.paths.tdx_root.glob('TDX_streamreach_basins_*_01.parquet'))
    missing = [r for r in every_region if not part_output(period, r).exists()]
    if missing:
        print(f'{len(missing)} region(s) not labelled yet, not writing {final.name}')
        sys.exit(0)
    table = pd.concat([pd.read_parquet(part_output(period, r)) for r in every_region],
                      ignore_index=True)
    if table[hy.schema.tdx_link_no_field].duplicated().any():
        raise RuntimeError('raw ids are not globally unique across regions')
    partial = final.with_name(f'{final.name}.partial')
    hy.parquet.write_parquet(table, partial, index=False, row_group_size=None)
    partial.replace(final)
    print(f'{len(table):,} catchments labelled -> {final}, {time.time() - started:.0f}s')
