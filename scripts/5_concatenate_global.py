#!/usr/bin/env python
"""Assemble the release: global ordering, published group files, global products, leaf tile band."""
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
group_root = hy.paths.group_root
global_root = hy.paths.global_root
logs_root = hy.paths.logs_root

WORKERS = max(1, int(os.environ.get('CONCAT_WORKERS', 8)))

LARGE_GEOMETRY_KINDS = {'streams', 'catchments'}
GEOMETRY_KINDS = LARGE_GEOMETRY_KINDS | {'confluences'}
SUFFIXES = {'metadata': '.parquet'}
REGION_KINDS = ['metadata', 'streams', 'confluences']

LEAF_TOLERANCE = max(hy.basins.zoom_tolerance(hy.basins.LEAF_ZOOMS[1]), 1.0)
LEAF_CHUNK = 20_000


def out_name(kind: str, group_id: int) -> str:
    return f'{kind}_{group_id}{SUFFIXES.get(kind, ".geo.parquet")}'


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


def region_group_runs(meta: pd.DataFrame, region: str) -> pd.DataFrame:
    local = meta[hy.schema.river_index].to_numpy()
    if not np.array_equal(local, np.arange(len(meta), dtype=local.dtype)):
        raise ValueError(f'{region}: the region-local riverIndex is not the row position; '
                         f'rerun step 3')
    groups = meta[hy.schema.group_id].to_numpy()
    edges = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1], True])
    starts, sizes = edges[:-1], np.diff(edges)
    keys = groups[starts]
    if len(keys) != len(np.unique(keys)):
        raise ValueError(f'{region}: a group occupies more than one run of the region ordering; '
                         f'rerun step 3')
    return pd.DataFrame({hy.schema.group_id: keys, 'region': region,
                         'local_start': starts, 'size': sizes})


def region_outputs(region: str, runs: pd.DataFrame, has_catchments: bool) -> list:
    kinds = REGION_KINDS + (['catchments'] if has_catchments else [])
    paths = []
    for run in runs.itertuples():
        group_id = int(getattr(run, hy.schema.group_id))
        paths += [hy.paths.group_dir(group_id) / out_name(kind, group_id) for kind in kinds]
    if has_catchments:
        leaf = region_root / region / f'catchments_tile_{region}.fgb'
        paths += [leaf, lines_path(leaf)]
    return paths


def split_region(region: str, meta: pd.DataFrame, runs: pd.DataFrame) -> bool:
    region_dir = region_root / region
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
    confluences = confluences.sort_values('_pos', kind='stable').reset_index(drop=True)

    catchments_src = region_dir / f'catchments_{region}.geo.parquet'
    catchments = None
    if catchments_src.exists():
        catchments = gpd.read_parquet(catchments_src)
        if not np.array_equal(catchments[hy.schema.river_id].to_numpy(),
                              meta[hy.schema.river_id].to_numpy()):
            raise ValueError(f'{region}: catchments and metadata are not in the same row order; '
                             f'rerun step 4')
        catchments[hy.schema.river_index] = meta[hy.schema.river_index].to_numpy()
        catchments = hy.schema.enforce_int32(catchments)

    for run in runs.itertuples():
        group_id = int(run.groupId) if hasattr(run, 'groupId') else int(getattr(run, hy.schema.group_id))
        start, end = int(run.local_start), int(run.local_start + run.size)
        out_dir = hy.paths.group_dir(group_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        conf_lo, conf_hi = np.searchsorted(confluences['_pos'].to_numpy(), [start, end])
        parts = {
            'metadata': meta.iloc[start:end],
            'streams': streams.iloc[start:end],
            'confluences': confluences.iloc[conf_lo:conf_hi].drop(columns=['_pos']),
        }
        if catchments is not None:
            parts['catchments'] = catchments.iloc[start:end]

        counts = {}
        for kind, part in parts.items():
            part = part.drop(columns=[hy.schema.group_id], errors='ignore')
            if hy.schema.river_index in part.columns and len(part):
                stamped = part[hy.schema.river_index].to_numpy()
                if not (np.diff(stamped) == 1).all():
                    raise ValueError(f'{kind} for group {group_id} is not one unbroken run of '
                                     f'riverIndex')
            out_path = out_dir / out_name(kind, group_id)
            if kind in GEOMETRY_KINDS:
                row_group_size = hy.parquet.GEOMETRY_ROW_GROUP_SIZE \
                    if kind in LARGE_GEOMETRY_KINDS else None
                hy.parquet.write_geoparquet(part, out_path, row_group_size=row_group_size)
            else:
                hy.parquet.write_parquet(part, out_path)
            counts[kind] = len(part)

        logging.info(f'region {region} -> group {group_id}: '
                     + ', '.join(f'{n:,} {k}' for k, n in counts.items()))

    if catchments is not None:
        cut_leaf_band(catchments, region_dir / f'catchments_tile_{region}.fgb')
    print(f'region {region}: {len(runs)} group(s) written'
          + ('' if catchments is None else ' + leaf band'))
    return True


SUMMARY_KINDS = ['metadata', 'streams', 'confluences', 'catchments']


def dataset_summary(runs: pd.DataFrame, metadata: pd.DataFrame) -> None:
    group_ids = [int(g) for g in runs[hy.schema.group_id]]
    counts = dict.fromkeys(SUMMARY_KINDS, 0)
    sizes = dict.fromkeys(SUMMARY_KINDS, 0)
    absent = dict.fromkeys(SUMMARY_KINDS, 0)
    for group_id in group_ids:
        for kind in SUMMARY_KINDS:
            path = hy.paths.group_dir(group_id) / out_name(kind, group_id)
            if not path.exists():
                absent[kind] += 1
                continue
            counts[kind] += pq.read_metadata(path).num_rows
            sizes[kind] += path.stat().st_size

    metadata_out = global_root / 'metadata.parquet'
    zarr_out = global_root / 'metadata.zarr'
    streams = pq.read_metadata(metadata_out).num_rows
    zarr_size = sum(p.stat().st_size for p in zarr_out.rglob('*') if p.is_file())
    published = sum(sizes.values()) + metadata_out.stat().st_size + zarr_size

    outlets = int((metadata[hy.schema.next_river_id] == -1).sum())
    largest = int(metadata[hy.schema.upstream_count].max()) + 1

    rows = [
        ('regions', f'{len(runs["region"].unique()):,}'),
        ('groups', f'{len(group_ids):,}'),
        ('streams', f'{streams:,}'),
        ('  outlets (drainage networks)', f'{outlets:,}'),
        ('  reaches in the largest network', f'{largest:,}'),
    ]
    for kind in SUMMARY_KINDS:
        if absent[kind] == len(group_ids):
            rows.append((f'{kind} in group files', 'not built'))
            continue
        short = f'  ({absent[kind]} group(s) missing)' if absent[kind] else ''
        rows.append((f'{kind} in group files',
                     f'{counts[kind]:,} rows, {hy.console.humanize_bytes(sizes[kind])}{short}'))
    rows += [
        ('metadata.parquet', hy.console.humanize_bytes(metadata_out.stat().st_size)),
        ('metadata.zarr', hy.console.humanize_bytes(zarr_size)),
        ('published total', hy.console.humanize_bytes(published)),
    ]

    notes = []
    if counts['metadata'] != streams and not absent['metadata']:
        notes.append(f'WARNING: group metadata totals {counts["metadata"]:,} rows against '
                     f'{streams:,} in metadata.parquet')
    for kind in ('streams', 'catchments'):
        if not absent[kind] and counts[kind] != streams:
            notes.append(f'WARNING: {kind} totals {counts[kind]:,} rows against {streams:,} '
                         f'reaches')
    notes.append(f'{hy.paths.group_root}')
    hy.console.summary('RELEASE SUMMARY', rows, notes)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if args:
        WORKERS = max(1, int(args[0]))
        INNER_THREADS = max(1, (os.cpu_count() or 8) // WORKERS)
    hy.console.banner('Assemble the release')

    logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=logs_root / 'concatenate_global.log', filemode='w',
                        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    started = time.time()

    metadata_paths = natsorted(region_root.glob('*/metadata_*.parquet'), key=str)
    regions = [p.parent.name for p in metadata_paths]

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        frames = dict(zip(regions, pool.map(pd.read_parquet, metadata_paths)))
    runs = pd.concat([region_group_runs(frames[r], r) for r in regions], ignore_index=True)
    if runs[hy.schema.group_id].duplicated().any():
        split = runs.loc[runs[hy.schema.group_id].duplicated(), hy.schema.group_id].tolist()
        raise RuntimeError(f'group(s) {split[:5]} appear in more than one region')
    runs = runs.sort_values(hy.schema.group_id, ignore_index=True)
    runs['global_start'] = np.concatenate(([0], runs['size'].cumsum().to_numpy()[:-1]))
    logging.info(f'{len(runs)} groups over {len(regions)} regions, '
                 f'{int(runs["size"].sum()):,} reaches; global riverIndex is offset arithmetic')

    metadata_out = global_root / 'metadata.parquet'
    zarr_out = global_root / 'metadata.zarr'
    runs_by_region = {r: g for r, g in runs.groupby('region', sort=False)}
    catchments_present = {r: (region_root / r / f'catchments_{r}.geo.parquet').exists()
                          for r in regions}
    outputs = [metadata_out, zarr_out]
    for region in regions:
        outputs += region_outputs(region, runs_by_region[region], catchments_present[region])
    if all(path.exists() for path in outputs):
        # exit 0, not 1: pipeline.sh runs under `set -e`, so a step with nothing to do must report
        # success or it aborts the whole run. sys.exit(<string>) prints to stderr and exits 1.
        print(f'all {len(outputs)} outputs exist, nothing to do')
        sys.exit(0)

    for row in runs.itertuples():
        frame = frames[row.region]
        block = slice(int(row.local_start), int(row.local_start + row.size))
        frame.loc[frame.index[block], hy.schema.river_index] = np.arange(
            int(row.global_start), int(row.global_start + row.size), dtype=np.int32)

    metadata = pd.concat(
        [frames[row.region].iloc[int(row.local_start):int(row.local_start + row.size)]
         for row in runs.itertuples()], ignore_index=True)
    if not np.array_equal(metadata[hy.schema.river_index].to_numpy(),
                          np.arange(len(metadata), dtype=np.int32)):
        raise RuntimeError('the stamped global riverIndex is not the row position - the offset '
                           'arithmetic and the group table disagree')
    if not metadata[hy.schema.river_id].is_unique:
        raise RuntimeError('riverId is not globally unique across regions')
    group_of = pd.Series(metadata[hy.schema.group_id].to_numpy(),
                         index=metadata[hy.schema.river_id].to_numpy())
    flowing = metadata[metadata[hy.schema.next_river_id] != -1]
    downstream_group = group_of.reindex(flowing[hy.schema.next_river_id].to_numpy()).to_numpy()
    crossing = int((downstream_group != flowing[hy.schema.group_id].to_numpy()).sum())
    if crossing or pd.isna(downstream_group).any():
        raise RuntimeError(f'{crossing:,} reach(es) drain into another group and '
                           f'{int(pd.isna(downstream_group).sum()):,} into a missing reach; '
                           f'per-group files would not be self-contained')
    hy.topology.assert_nested_set_is_valid(metadata)
    metadata = hy.schema.enforce_int32(metadata)
    logging.info(f'global ordering validated: {len(metadata):,} reaches, 0 cross-group edges, '
                 f'nested-set property holds')

    global_root.mkdir(parents=True, exist_ok=True)
    if not (metadata_out.exists() and zarr_out.exists()):
        hy.parquet.write_parquet(metadata, metadata_out)
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
        region_runs = runs_by_region[region]
        if all(p.exists() for p in region_outputs(region, region_runs,
                                                  catchments_present[region])):
            print(f'region {region}: outputs exist, skipped')
            return region, False
        return region, split_region(region, frames[region], region_runs)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rebuilt = sum(1 for _, did in pool.map(process, regions) if did)

    print(f'done: {rebuilt} region(s) rebuilt, {time.time() - started:.0f}s. '
          f'Group boundaries come from step 6.')
    dataset_summary(runs, metadata)
