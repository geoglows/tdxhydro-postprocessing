#!/usr/bin/env python
"""Measure which country every raw TDX-Hydro catchment lies in, and the countries upstream of it.

Products, under $TDXHYDRO_GEOPARQUET_FINISHED/global_countries/ (hy.paths.country_*):

1. parts/country_<region>.parquet - working data, read by simplify_streams.py. Long format, one row per raw
   catchment per country it touches: TDXHydroLinkNo, country, km2. simplify_streams.py sums these over the raw
   catchments that make up each published reach's catchment and publishes the winner as
   ``country`` - the only country attribute in the GIS files.

2. country_upstream.parquet - a supplemental table, not part of the GIS files, assembled from
   parts/country_upstream_<region>.parquet once every region is done. One row per raw reach,
   joined to the published streams on riverId = TDXHydroLinkNo:

    TDXHydroLinkNo           : int64
    TDXHydroRegion           : string
    country                  : string    the label with the most land in the reach's own catchment
    countryIso3              : string    its ISO 3166 alpha-3 code; null for a disputed area
                                         without one
    countryDisputed          : bool      the label is a geoBoundaries disputed area
    outletCountry            : string    the country of the reach's watershed outlet, the last
                                         reach on its path downstream
    upstreamCountries        : list<str> every label with land upstream (the reach's own catchment
                                         plus everything draining into it), largest share first
    upstreamCountryFractions : list<f32> those shares, summing to 1
    transboundary            : bool      two or more labels each hold at least 1% of the upstream
                                         land. The threshold keeps the slivers a catchment picks
                                         up where TDX and geoBoundaries draw a border slightly
                                         differently from calling a basin shared

3. countries.parquet - the labels: country, countryIso3, countryDisputed.

Labels and the vote are in hydrography/countries.py: 198 countries by name, the 20 geoBoundaries
disputed areas by their own names (flagged), and No Man's Land labelled Israel. A catchment with
no land in any label (offshore slivers) gets null.

Only catchments with a reach in the 1_translate_tdxhydro.py streamnet are kept, as in koppen_classifications.py,
and a catchment stored as several rows is summed over them.

Environment, in addition to what pipeline_env.sh already exports:

    COUNTRIES_GPKG      the geoBoundaries CGAZ ADM0 geopackage (required)
    COUNTRY_JOBS        regions labelled at once (default 6)
    COUNTRY_BATCH_SIZE  catchments read and overlaid at a time (default 100000)

Usage: country_classifications.py [region ...]   (no arguments means every region)

Data source: geoBoundaries CGAZ ADM0, https://www.geoboundaries.org/globalDownloads.html, licensed
CC BY 4.0. Citation:

    Runfola, D. et al. (2020) geoBoundaries: A global database of political administrative
    boundaries. PLoS ONE 15(4): e0231866. https://doi.org/10.1371/journal.pone.0231866
"""
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely

import hydrography as hy

C = hy.countries
BATCH_SIZE = int(os.environ.get('COUNTRY_BATCH_SIZE', 100_000))
TRANSBOUNDARY_SHARE = 0.01

OUTLET = 'outletCountry'
UP_NAMES = 'upstreamCountries'
UP_FRACTIONS = 'upstreamCountryFractions'
TRANSBOUNDARY = 'transboundary'

# per worker process, loaded once by _init
_tree = None
_piece_label = None
_names = None


def boundaries_path() -> Path:
    value = os.environ.get('COUNTRIES_GPKG')
    if not value:
        sys.exit('$COUNTRIES_GPKG is not set - point it at geoBoundariesCGAZ_ADM0.gpkg')
    path = Path(value).expanduser()
    if not path.exists():
        sys.exit(f'{path} not found')
    return path


def prepare(gpkg: Path) -> None:
    """Write the label table and the cut-up boundaries once; every worker reads them."""
    if hy.paths.country_table.exists() and hy.paths.country_pieces.exists():
        return
    started = time.time()
    boundaries = C.load_boundaries(gpkg)
    pieces = C.split_pieces(boundaries)
    hy.paths.country_root.mkdir(parents=True, exist_ok=True)
    pieces.to_parquet(hy.paths.country_pieces)
    hy.parquet.write_parquet(boundaries.drop(columns='geometry'), hy.paths.country_table,
                             index=False, row_group_size=None)
    print(f'{len(boundaries)} labels cut into {len(pieces):,} pieces, {time.time() - started:.0f}s')


def _init() -> None:
    global _tree, _piece_label, _names
    pieces = gpd.read_parquet(hy.paths.country_pieces)
    _piece_label = pieces['label'].to_numpy()
    _tree = shapely.STRtree(pieces.geometry.values)
    _names = pd.read_parquet(hy.paths.country_table)[C.NAME].to_numpy(dtype=object)


def overlay(geometries: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row, label, km2) for every piece of land each geometry has in each label."""
    pieces = _tree.geometries
    area = C.planar_km2(geometries)
    # a catchment within a single piece is all that piece's label, no intersection needed
    inside_row, inside_piece = _tree.query(geometries, predicate='within')
    inside_row, first = np.unique(inside_row, return_index=True)
    inside_piece = inside_piece[first]
    rest = np.setdiff1d(np.arange(len(geometries)), inside_row)
    row, piece = _tree.query(geometries[rest], predicate='intersects')
    row = rest[row]
    clipped = shapely.intersection(geometries[row], pieces[piece])
    clipped_km2 = C.planar_km2(clipped)
    keep = clipped_km2 > 0
    return (np.concatenate([inside_row, row[keep]]),
            np.concatenate([_piece_label[inside_piece], _piece_label[piece[keep]]]),
            np.concatenate([area[inside_row], clipped_km2[keep]]))


def upstream_lists(totals: np.ndarray, names: np.ndarray) -> tuple[list, list, np.ndarray]:
    """Per row of a (reaches x labels) area matrix: names by share, shares, transboundary."""
    land = totals.sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        share = np.where(land[:, None] > 0, totals / land[:, None], 0.0)
    order = np.argsort(-share, axis=1, kind='stable')
    ranked = np.take_along_axis(share, order, axis=1)
    counts = (ranked > 0).sum(axis=1)
    name_lists = [names[o[:n]].tolist() for o, n in zip(order, counts)]
    share_lists = [r[:n].astype(np.float32).tolist() for r, n in zip(ranked, counts)]
    transboundary = (share >= TRANSBOUNDARY_SHARE).sum(axis=1) >= 2
    return name_lists, share_lists, transboundary


def label_region(basins: Path) -> str:
    region = hy.raw.region_of(basins)
    outputs = (hy.paths.country_part(region), hy.paths.country_upstream_part(region))
    if all(o.exists() for o in outputs):
        return f'{region}: already labelled, skipped'
    started = time.time()
    streamnet = hy.raw.streamnet_of(basins)
    if not streamnet.exists():
        raise FileNotFoundError(f'{streamnet} not found; the upstream labels need the network')

    ids = hy.raw.id_column(basins)
    crs = hy.raw.source_crs(basins)
    reproject = not crs.equals(pyproj.CRS.from_epsg(4326), ignore_axis_order=True)

    link, label, km2 = [], [], []
    for chunk in hy.raw.batches(basins, ids, crs, BATCH_SIZE):
        if reproject:
            chunk = chunk.to_crs(4326)
        geometries = chunk.geometry.values
        invalid = ~shapely.is_valid(geometries)
        if invalid.any():
            geometries = geometries.copy()
            geometries[invalid] = shapely.make_valid(geometries[invalid])
        row, lab, area = overlay(np.asarray(geometries))
        link.append(chunk[ids].to_numpy()[row])
        label.append(lab)
        km2.append(area)
    long = pd.DataFrame({hy.schema.tdx_link_no_field: np.concatenate(link),
                         'label': np.concatenate(label), 'km2': np.concatenate(km2)})
    # several basin rows for one catchment, or one catchment over several pieces of a label
    long = long.groupby([hy.schema.tdx_link_no_field, 'label'], as_index=False)['km2'].sum()

    network = pd.read_parquet(streamnet, columns=[hy.schema.tdx_link_field, hy.schema.tdx_ds_link_field])
    reach_ids = network[hy.schema.tdx_link_field].to_numpy(np.int64)
    downstream = network[hy.schema.tdx_ds_link_field].to_numpy(np.int64)
    in_network = long[hy.schema.tdx_link_no_field].isin(reach_ids)
    dropped = int(long.loc[~in_network, hy.schema.tdx_link_no_field].nunique())
    long = long[in_network]

    part = pd.DataFrame({
        hy.schema.tdx_link_no_field: long[hy.schema.tdx_link_no_field].to_numpy(np.int64),
        C.NAME: pd.array(_names[long['label'].to_numpy()], dtype='string'),
        'km2': long['km2'].to_numpy(np.float32),
    })

    # the region's labels as dense columns, so the upstream sums are one matrix walk
    present = np.unique(long['label'].to_numpy())
    column = pd.Series(np.arange(len(present)), index=present)
    own = np.zeros((len(reach_ids), len(present)))
    row = pd.Index(reach_ids).get_indexer(long[hy.schema.tdx_link_no_field].to_numpy())
    np.add.at(own, (row, column[long['label'].to_numpy()].to_numpy()), long['km2'].to_numpy())
    upstream = hy.raw.accumulate(reach_ids, downstream, own)
    outlet_row = hy.raw.outlets(reach_ids, downstream)

    names = _names[present]
    table = pd.read_parquet(hy.paths.country_table).set_index(C.NAME)
    own_name = pd.Series(C.dominant(part[hy.schema.tdx_link_no_field].to_numpy(),
                                    part[C.NAME].to_numpy(), part['km2'].to_numpy()))
    own_name = own_name.reindex(reach_ids).to_numpy(dtype=object)
    up_names, up_shares, transboundary = upstream_lists(upstream, names)
    supplemental = pd.DataFrame({
        hy.schema.tdx_link_no_field: reach_ids,
        hy.schema.tdx_region_field: region,
        C.NAME: pd.array(own_name, dtype='string'),
        C.ISO3: pd.array(table[C.ISO3].reindex(own_name).to_numpy(dtype=object), dtype='string'),
        C.DISPUTED: table[C.DISPUTED].reindex(own_name).fillna(False).to_numpy(bool),
        OUTLET: pd.array(own_name[outlet_row], dtype='string'),
        UP_NAMES: up_names,
        UP_FRACTIONS: up_shares,
        TRANSBOUNDARY: transboundary,
    })

    for frame, output in zip((part, supplemental), outputs):
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f'{output.name}.partial')
        hy.parquet.write_parquet(frame, partial, index=False, row_group_size=None)
        partial.replace(output)

    unlabelled = int(supplemental[C.NAME].isna().sum())
    return (f'{region}: {len(reach_ids):,} reaches, {len(present)} labels, {unlabelled:,} with no '
            f'land, {int(transboundary.sum()):,} transboundary, {dropped:,} duplicated catchments '
            f'dropped, {time.time() - started:.0f}s')


if __name__ == '__main__':
    hy.console.banner('Label raw TDX-Hydro catchments with countries')
    final = hy.paths.country_upstream_global
    if final.exists():
        # exit 0: pipeline.sh runs under `set -e`, so a step with nothing to do must succeed
        print(f'{final.name} exists, nothing to do')
        sys.exit(0)
    gpkg = boundaries_path()

    basins = sorted(hy.paths.tdx_root.glob('TDX_streamreach_basins_*_01.parquet'),
                    key=lambda p: p.stat().st_size, reverse=True)
    if not basins:
        sys.exit(f'no TDX_streamreach_basins parquet under {hy.paths.tdx_root} - run '
                 f'raw_processing/1_translate_tdxhydro.py first')
    wanted = set(sys.argv[1:])
    if wanted:
        basins = [p for p in basins if hy.raw.region_of(p) in wanted]
        print(f'{len(basins)} region(s) selected by argument')

    hy.paths.logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=hy.paths.logs_root / 'country_classifications.log', filemode='w',
                        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    started = time.time()
    prepare(gpkg)

    workers = max(1, int(os.environ.get('COUNTRY_JOBS', 6)))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        for message in pool.map(label_region, basins):
            logging.info(message)
            print(message)

    every_region = sorted(hy.raw.region_of(p) for p in
                          hy.paths.tdx_root.glob('TDX_streamreach_basins_*_01.parquet'))
    missing = [r for r in every_region if not hy.paths.country_upstream_part(r).exists()]
    if missing:
        print(f'{len(missing)} region(s) not labelled yet, not writing {final.name}')
        sys.exit(0)
    table = pd.concat([pd.read_parquet(hy.paths.country_upstream_part(r)) for r in every_region],
                      ignore_index=True)
    if table[hy.schema.tdx_link_no_field].duplicated().any():
        raise RuntimeError('raw ids are not globally unique across regions')
    partial = final.with_name(f'{final.name}.partial')
    hy.parquet.write_parquet(table, partial, index=False, row_group_size=None)
    partial.replace(final)
    print(f'{len(table):,} reaches, upstream countries -> {final}, {time.time() - started:.0f}s')
