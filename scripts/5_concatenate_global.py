#!/usr/bin/env python
"""Assemble the release: global ordering, published region files, global products, leaf tile band.

The release has exactly one partition, the HydroBASINS level-2 region, and it is not a choice this
step makes - it is the unit the raw TDX-Hydro arrives in and the unit every step before this one
processes. Nothing here splits a region further. The global ordering is the regions concatenated in
ascending region number, so each region's published files are one unbroken run of riverIndex and a
reader can treat a region file as a dense array whose local index is ``riverIndex - riverIndexStart``.

What replaces the old group partition is guidance rather than geometry: ``watersheds.parquet``
names every terminal watershed and the contiguous riverIndex range it occupies. A watershed is a
whole connected drainage network with no edge leaving it, so ANY bundle of watersheds is a valid
independent unit of parallel computation. A consumer that wants sixteen balanced workers bin-packs
the rows by reachCount; one that wants a hundred packs them a hundred ways. The pipeline does not
have to guess the worker count, and no file has to be cut to match it.
"""
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyogrio
import shapely
from natsort import natsorted
from zarr.codecs import BloscCodec

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

region_root = hy.paths.region_root
publish_root = hy.paths.publish_root
global_root = hy.paths.global_root
logs_root = hy.paths.logs_root
pmtiles_root = hy.paths.pmtiles_root

WORKERS = max(1, int(os.environ.get('CONCAT_WORKERS', 8)))

LARGE_GEOMETRY_KINDS = {'streams', 'catchments'}
GEOMETRY_KINDS = LARGE_GEOMETRY_KINDS | {'confluences'}
SUFFIXES = {'metadata': '.parquet', 'watersheds': '.parquet'}
REGION_KINDS = ['metadata', 'streams', 'confluences', 'watersheds']

LEAF_TOLERANCE = max(hy.basins.zoom_tolerance(hy.basins.LEAF_ZOOMS[1]), 1.0)
LEAF_CHUNK = 20_000

# watersheds.parquet column names. riverIndexStart/riverIndexEnd are inclusive bounds in the
# GLOBAL riverIndex, which is what the published files carry.
index_start = 'riverIndexStart'
index_end = 'riverIndexEnd'
reach_count = 'reachCount'


def leaf_band_path(region: str) -> Path:
    """The leaf band is a tiler input, not a deliverable, so it lands beside every other
    tiling intermediate rather than in the region's scratch directory."""
    return pmtiles_root / f'catchments_tile_{region}.fgb'


def out_name(kind: str, region: str) -> str:
    return f'{kind}_{region}{SUFFIXES.get(kind, ".geo.parquet")}'


def lines_path(path: Path) -> Path:
    return path.with_name(path.name.replace('.fgb', '.lines.fgb'))


def write_band(gdf: gpd.GeoDataFrame, path: Path) -> None:
    frame = gdf.copy()
    for column in frame.columns:
        if pd.api.types.is_integer_dtype(frame[column]):
            frame[column] = frame[column].astype('float64')
    options = dict(driver='FlatGeobuf', promote_to_multi=True, SPATIAL_INDEX='NO')
    for target, geometry_type in ((path, 'MultiPolygon'), (lines_path(path), 'MultiLineString')):
        partial = target.with_name(target.name.replace('.fgb', '.partial.fgb'))
        pyogrio.write_dataframe(frame, partial, geometry_type=geometry_type, **options)
        partial.replace(target)
        frame = frame.set_geometry(frame.geometry.boundary)


def cut_leaf_band(catchments: gpd.GeoDataFrame, path: Path) -> None:
    started = time.time()
    parts = catchments.geometry.to_numpy()
    before = after = held = 0
    for start in range(0, len(parts), LEAF_CHUNK):
        block = slice(start, start + LEAF_CHUNK)
        raw = parts[block]
        before += int(shapely.get_num_coordinates(raw).sum())
        clean, lost = hy.geometry.repair(
            hy.geometry.simplify_coverage(raw, LEAF_TOLERANCE), fallback=raw)
        snapped, collapsed = hy.geometry.snap(clean, LEAF_TOLERANCE)
        clean, lost_again = hy.geometry.repair(snapped, fallback=clean)
        after += int(shapely.get_num_coordinates(clean).sum())
        held += lost + collapsed + lost_again
        parts[block] = clean
    leaf = gpd.GeoDataFrame(
        catchments[[hy.schema.river_id, hy.schema.river_index]].copy(),
        geometry=parts, crs=catchments.crs)
    write_band(leaf, path)
    lo, hi = hy.basins.LEAF_ZOOMS
    logging.info(f'leaf band (z{lo}-{hi}): {len(leaf):,} catchments, {before:,} -> {after:,} '
                 f'vertices at {LEAF_TOLERANCE:,.0f} m, {held:,} kept an earlier geometry, '
                 f'{time.time() - started:.0f}s -> {path.name}')


def check_region_ordering(meta: pd.DataFrame, region: str) -> None:
    """Step 3 leaves the region-local riverIndex equal to the row position; everything below is
    offset arithmetic on that, so it is checked rather than assumed."""
    local = meta[hy.schema.river_index].to_numpy()
    if not np.array_equal(local, np.arange(len(meta), dtype=local.dtype)):
        raise ValueError(f'{region}: the region-local riverIndex is not the row position; '
                         f'rerun step 3')


def watershed_table(meta: pd.DataFrame) -> pd.DataFrame:
    """One row per terminal watershed: the contiguous riverIndex range it occupies, and enough
    about it to bin-pack the rows into parallel computation groups.

    The range is exact, not an estimate. ``nested_set_order`` emits every watershed in depth-first
    post-order, so its reaches occupy ``[position - upstreamCount, position]`` with the outlet
    last, and no edge crosses out of that interval - that is the same property the whole nested-set
    scheme rests on and ``assert_nested_set_is_valid`` checks it on the assembled network below.
    """
    outlets = meta[meta[hy.schema.next_river_id] == -1]
    end = outlets[hy.schema.river_index].to_numpy()
    count = outlets[hy.schema.upstream_count].to_numpy() + 1
    table = pd.DataFrame({
        hy.schema.river_id: outlets[hy.schema.river_id].to_numpy(),
        hy.schema.tdx_region_field: outlets[hy.schema.tdx_region_field].to_numpy(),
        index_start: (end - count + 1).astype(np.int32),
        index_end: end.astype(np.int32),
        reach_count: count.astype(np.int32),
        hy.schema.tdx_ds_area_field: outlets[hy.schema.tdx_ds_area_field].to_numpy(),
        hy.schema.lat_field: outlets[hy.schema.lat_field].to_numpy(),
        hy.schema.lon_field: outlets[hy.schema.lon_field].to_numpy(),
    })
    return table.sort_values(index_start, ignore_index=True)


def region_outputs(region: str, has_catchments: bool) -> list:
    kinds = REGION_KINDS + (['catchments'] if has_catchments else [])
    paths = [hy.paths.publish_dir(region) / out_name(kind, region) for kind in kinds]
    if has_catchments:
        leaf = leaf_band_path(region)
        paths += [leaf, lines_path(leaf)]
    return paths


def publish_region(region: str, meta: pd.DataFrame, has_catchments: bool) -> bool:
    """Write one region's published files, whole. ``meta`` already carries the global riverIndex."""
    region_dir = region_root / region
    out_dir = hy.paths.publish_dir(region)
    out_dir.mkdir(parents=True, exist_ok=True)

    streams = gpd.read_parquet(region_dir / f'streams_{region}.geo.parquet')
    if not np.array_equal(streams[hy.schema.river_id].to_numpy(),
                          meta[hy.schema.river_id].to_numpy()):
        raise ValueError(f'{region}: streams and metadata are not in the same row order; '
                         f'rerun step 3')
    streams[hy.schema.river_index] = meta[hy.schema.river_index].to_numpy()
    streams = hy.schema.enforce_int32(streams)

    confluences = gpd.read_parquet(region_dir / f'confluences_{region}.geo.parquet')
    position = pd.Series(np.arange(len(meta)), index=meta[hy.schema.river_id].to_numpy())
    conf_pos = position.reindex(confluences[hy.schema.river_id].to_numpy()).to_numpy()
    keep = ~np.isnan(conf_pos)
    if (~keep).any():
        logging.info(f'{region}: dropped {int((~keep).sum())} confluence row(s) with no reach')
    confluences = confluences.loc[keep].assign(_pos=conf_pos[keep].astype(np.int64))
    confluences = (confluences.sort_values('_pos', kind='stable')
                   .drop(columns=['_pos']).reset_index(drop=True))

    parts = {
        'metadata': meta,
        'streams': streams,
        'confluences': confluences,
        'watersheds': watershed_table(meta),
    }
    if has_catchments:
        catchments = gpd.read_parquet(region_dir / f'catchments_{region}.geo.parquet')
        if not np.array_equal(catchments[hy.schema.river_id].to_numpy(),
                              meta[hy.schema.river_id].to_numpy()):
            raise ValueError(f'{region}: catchments and metadata are not in the same row order; '
                             f'rerun step 4')
        catchments[hy.schema.river_index] = meta[hy.schema.river_index].to_numpy()
        parts['catchments'] = hy.schema.enforce_int32(catchments)

    counts = {}
    for kind, part in parts.items():
        if hy.schema.river_index in part.columns and len(part):
            stamped = part[hy.schema.river_index].to_numpy()
            if not (np.diff(stamped) == 1).all():
                raise ValueError(f'{kind} for region {region} is not one unbroken run of '
                                 f'riverIndex')
        out_path = out_dir / out_name(kind, region)
        # Every published per-region file is written to be range-fetched: a client turns a
        # riverIndex range into row groups off the footer statistics and pulls only those. The
        # geometry tables chunk finer than the attribute tables only because their rows are an
        # order of magnitude larger - the boundaries still line up. See hydrography/parquet.py.
        if kind in GEOMETRY_KINDS:
            row_group_size = hy.parquet.GEOMETRY_ROW_GROUP_SIZE if kind in LARGE_GEOMETRY_KINDS \
                else hy.parquet.ATTRIBUTE_ROW_GROUP_SIZE
            hy.parquet.write_geoparquet(part, out_path, row_group_size=row_group_size)
        else:
            hy.parquet.write_parquet(part, out_path)
        counts[kind] = len(part)

    logging.info(f'region {region}: ' + ', '.join(f'{n:,} {k}' for k, n in counts.items()))

    if has_catchments:
        pmtiles_root.mkdir(parents=True, exist_ok=True)
        cut_leaf_band(parts['catchments'], leaf_band_path(region))
    print(f'region {region}: {counts["metadata"]:,} reaches, {counts["watersheds"]:,} watersheds'
          + ('' if not has_catchments else ' + leaf band'))
    return True


SUMMARY_KINDS = ['metadata', 'streams', 'confluences', 'catchments', 'watersheds']


def dataset_summary(regions: list, metadata: pd.DataFrame, watersheds: pd.DataFrame) -> None:
    counts = dict.fromkeys(SUMMARY_KINDS, 0)
    sizes = dict.fromkeys(SUMMARY_KINDS, 0)
    absent = dict.fromkeys(SUMMARY_KINDS, 0)
    for region in regions:
        for kind in SUMMARY_KINDS:
            path = hy.paths.publish_dir(region) / out_name(kind, region)
            if not path.exists():
                absent[kind] += 1
                continue
            counts[kind] += pq.read_metadata(path).num_rows
            sizes[kind] += path.stat().st_size

    metadata_out = global_root / 'metadata.parquet'
    zarr_out = global_root / 'metadata.zarr'
    watersheds_out = global_root / 'watersheds.parquet'
    streams = pq.read_metadata(metadata_out).num_rows
    zarr_size = sum(p.stat().st_size for p in zarr_out.rglob('*') if p.is_file())
    published = (sum(sizes.values()) + metadata_out.stat().st_size + zarr_size
                 + watersheds_out.stat().st_size)

    largest = int(metadata[hy.schema.upstream_count].max()) + 1
    biggest = watersheds[reach_count].sort_values(ascending=False)

    rows = [
        ('regions', f'{len(regions):,}'),
        ('streams', f'{streams:,}'),
        ('  watersheds (parallel units)', f'{len(watersheds):,}'),
        ('  reaches in the largest', f'{largest:,}'),
        ('  reaches in the largest 8', f'{int(biggest.head(8).sum()):,} '
                                       f'({biggest.head(8).sum() / streams:.1%} of the network)'),
    ]
    for kind in SUMMARY_KINDS:
        if absent[kind] == len(regions):
            rows.append((f'{kind} in region files', 'not built'))
            continue
        short = f'  ({absent[kind]} region(s) missing)' if absent[kind] else ''
        rows.append((f'{kind} in region files',
                     f'{counts[kind]:,} rows, {hy.console.humanize_bytes(sizes[kind])}{short}'))
    rows += [
        ('metadata.parquet', hy.console.humanize_bytes(metadata_out.stat().st_size)),
        ('metadata.zarr', hy.console.humanize_bytes(zarr_size)),
        ('watersheds.parquet', hy.console.humanize_bytes(watersheds_out.stat().st_size)),
        ('published total', hy.console.humanize_bytes(published)),
    ]

    notes = []
    if counts['metadata'] != streams and not absent['metadata']:
        notes.append(f'WARNING: region metadata totals {counts["metadata"]:,} rows against '
                     f'{streams:,} in metadata.parquet')
    for kind in ('streams', 'catchments'):
        if not absent[kind] and counts[kind] != streams:
            notes.append(f'WARNING: {kind} totals {counts[kind]:,} rows against {streams:,} '
                         f'reaches')
    notes.append(f'{hy.paths.publish_root}')
    hy.console.summary('RELEASE SUMMARY', rows, notes)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if args:
        WORKERS = max(1, int(args[0]))
    hy.console.banner('Assemble the release')

    logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=logs_root / 'concatenate_global.log', filemode='w',
                        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    started = time.time()

    metadata_paths = natsorted(region_root.glob('*/metadata_*.parquet'), key=str)
    regions = [p.parent.name for p in metadata_paths]
    if len(set(regions)) != len(regions):
        raise RuntimeError('two scratch directories claim the same region')
    # ascending level-2 region number IS the global ordering; nothing else decides it
    order = np.argsort([int(r) for r in regions])
    regions = [regions[i] for i in order]
    metadata_paths = [metadata_paths[i] for i in order]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        frames = dict(zip(regions, pool.map(pd.read_parquet, metadata_paths)))
    for region in regions:
        check_region_ordering(frames[region], region)
    sizes = np.array([len(frames[r]) for r in regions], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    logging.info(f'{len(regions)} regions, {int(sizes.sum()):,} reaches; global riverIndex is '
                 f'offset arithmetic on the ascending region order')

    metadata_out = global_root / 'metadata.parquet'
    zarr_out = global_root / 'metadata.zarr'
    watersheds_out = global_root / 'watersheds.parquet'
    catchments_present = {r: (region_root / r / f'catchments_{r}.geo.parquet').exists()
                          for r in regions}
    outputs = [metadata_out, zarr_out, watersheds_out]
    for region in regions:
        outputs += region_outputs(region, catchments_present[region])
    if all(path.exists() for path in outputs):
        # exit 0, not 1: pipeline.sh runs under `set -e`, so a step with nothing to do must report
        # success or it aborts the whole run. sys.exit(<string>) prints to stderr and exits 1.
        print(f'all {len(outputs)} outputs exist, nothing to do')
        sys.exit(0)

    for region, start, size in zip(regions, starts, sizes):
        frames[region][hy.schema.river_index] = np.arange(start, start + size, dtype=np.int32)

    metadata = pd.concat([frames[r] for r in regions], ignore_index=True)
    if not np.array_equal(metadata[hy.schema.river_index].to_numpy(),
                          np.arange(len(metadata), dtype=np.int32)):
        raise RuntimeError('the stamped global riverIndex is not the row position - the offset '
                           'arithmetic is wrong')
    if not metadata[hy.schema.river_id].is_unique:
        raise RuntimeError('riverId is not globally unique across regions')
    region_of = pd.Series(metadata[hy.schema.tdx_region_field].to_numpy(),
                          index=metadata[hy.schema.river_id].to_numpy())
    flowing = metadata[metadata[hy.schema.next_river_id] != -1]
    downstream_region = region_of.reindex(flowing[hy.schema.next_river_id].to_numpy()).to_numpy()
    crossing = int((downstream_region != flowing[hy.schema.tdx_region_field].to_numpy()).sum())
    if crossing or pd.isna(downstream_region).any():
        raise RuntimeError(f'{crossing:,} reach(es) drain into another region and '
                           f'{int(pd.isna(downstream_region).sum()):,} into a missing reach; '
                           f'per-region files would not be self-contained')
    hy.topology.assert_nested_set_is_valid(metadata)
    metadata = hy.schema.enforce_int32(metadata)
    logging.info(f'global ordering validated: {len(metadata):,} reaches, 0 cross-region edges, '
                 f'nested-set property holds')

    global_root.mkdir(parents=True, exist_ok=True)
    watersheds = watershed_table(metadata)
    # checked before anything is written: the file is skipped when it exists, so a wrong one would
    # be accepted for the rest of the release's life
    covered = int(watersheds[reach_count].sum())
    if covered != len(metadata):
        raise RuntimeError(f'the watershed ranges cover {covered:,} reaches against '
                           f'{len(metadata):,} in the network')
    if not np.array_equal(watersheds[index_start].to_numpy()[1:],
                          watersheds[index_end].to_numpy()[:-1] + 1):
        raise RuntimeError('the watershed ranges are not a partition of the riverIndex space')
    if not watersheds_out.exists():
        hy.parquet.write_parquet(watersheds, watersheds_out)
        logging.info(f'{len(watersheds):,} terminal watersheds, each one contiguous riverIndex '
                     f'range, largest {int(watersheds[reach_count].max()):,} reaches '
                     f'-> {watersheds_out.name}')

    if not (metadata_out.exists() and zarr_out.exists()):
        # the one published table deliberately left in one row group: this is the whole world in
        # one piece, taken as a bulk download, and the per-region files are what serve subsetting
        hy.parquet.write_parquet(metadata, metadata_out, row_group_size=None)
        logging.info(f'wrote {len(metadata):,} rows -> {metadata_out.name}')
        zarr_int_vars = [hy.schema.river_id, hy.schema.river_index, hy.schema.upstream_count,
                         hy.schema.next_river_id, hy.schema.last_river_id]
        zarr_float_vars = [hy.schema.lat_field, hy.schema.lon_field]
        (
            metadata[zarr_int_vars + zarr_float_vars]
            .to_xarray()
            .chunk({'index': 10_000})
            .to_zarr(
                zarr_out, mode='w', zarr_format=3, consolidated=False,
                encoding={
                    **{v: {'dtype': 'int32', 'compressors': BloscCodec(cname='zstd', clevel=5, shuffle='shuffle')}
                       for v in zarr_int_vars},
                    **{v: {'dtype': 'float32', 'compressors': BloscCodec(cname='zstd', clevel=5, shuffle='shuffle')}
                       for v in zarr_float_vars},
                }
            )
        )
        logging.info('wrote metadata.zarr')

    def process(region: str) -> tuple:
        if all(p.exists() for p in region_outputs(region, catchments_present[region])):
            print(f'region {region}: outputs exist, skipped')
            return region, False
        return region, publish_region(region, frames[region], catchments_present[region])

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rebuilt = sum(1 for _, did in pool.map(process, regions) if did)

    print(f'done: {rebuilt} region(s) rebuilt, {time.time() - started:.0f}s. '
          f'Region boundaries come from step 6.')
    dataset_summary(regions, metadata, watersheds)
