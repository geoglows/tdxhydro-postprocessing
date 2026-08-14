#!/usr/bin/env python
"""
Assemble the release: the global ordering, the published group files, and the global products,
in one pass over the region files.

This step absorbed three older ones - the global attribute stamp, the group split, and the global
concatenation - because run separately they read and rewrote the same bytes repeatedly: the
ordering step read every region's metadata, ran a 5.5M-node traversal, and rewrote every region's
metadata AND streams just to stamp three columns; the group split read all of it again to write
the published copies; the concatenation read all the metadata a third time. Merged, every region
file is read exactly once, nothing under regions/ is ever rewritten, and every published file is
written exactly once.

**There is no global traversal.** Step 3's region-local ordering is group-major with the same sort
keys the global ordering would use, so a group is one contiguous run of a region's rows in exactly
its final internal order, and the global ordering is nothing but the groups concatenated in
ascending groupId. The globally unique riverIndex is therefore pure arithmetic - the group's
global offset plus the row's position within the group - applied to each table as it streams past
on its way into the group files. The nested-set property that ordering exists to provide is then
*checked* globally (the vectorized validator, O(n) numpy) rather than re-derived, which is a
stronger guarantee at a vanishing fraction of the cost.

The scope rule for riverIndex: region files carry the region-local position step 3 stamped;
everything written here - the group-partitioned files and the group=0 products - carries the
globally unique one. See docs/river-index.md.

    reads   regions/<region>/{metadata,streams,confluences,catchments}_<region>[.geo].parquet
    writes  group=<id>/{metadata,streams,confluences,catchments}_<id>[.geo].parquet
            group=0/metadata.parquet and metadata.zarr, the global network
            regions/<region>/catchments_tile_<region>.fgb (+ .lines.fgb), the leaf tile band

The leaf band is cut here because this is the one place that holds a region's catchments in
memory *and* knows their global riverIndex: the band is the published catchments thinned to what
the leaf zooms resolve (hydrography/basins.py owns the banding), carrying the same id and index
every other product does. Cutting it anywhere else would mean a second read of the largest
geometry in the dataset or a band without its index.

Group boundaries are NOT dissolved here any more: unioning a group's full-resolution catchments
was this step's most expensive geometry, and the frozen level-8 basins already contain the same
outline at band resolution - a basin never crosses a group divide except 24 measured coastal
stragglers - so step 6, which holds the stamped basins, derives the boundaries and the
groups.geo.parquet stack from them at a fraction of the cost.

A region whose group outputs and leaf band all exist is skipped without reading any geometry, but
only after a footer-statistics probe confirms its stamped offsets still match this run's group
table - if the region set changed, every offset after the change is different, and an existence
check alone would ship stale indices.

    RFS_DATA_ROOT=... TDXHYDRO_ROOT=... python 5_concatenate_global.py [workers]
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
group_root = hy.paths.group_root
global_root = hy.paths.global_root
logs_root = hy.paths.logs_root

# Regions processed concurrently. The heavy work in a worker - pyogrio reads, GEOS coverage
# simplification, arrow writes - all releases the GIL, so threads scale, and the ceiling is
# memory: a worker holds one region's streams and catchments at once.
WORKERS = max(1, int(os.environ.get('CONCAT_WORKERS', 8)))

# how the published splits are written - see hydrography/parquet.py. Only the tables whose rows
# carry a whole reach's geometry need the small row groups; confluences are a point per row and
# metadata is light, so both stay on the default.
LARGE_GEOMETRY_KINDS = {'streams', 'catchments'}
GEOMETRY_KINDS = LARGE_GEOMETRY_KINDS | {'confluences'}
SUFFIXES = {'metadata': '.parquet'}

# the leaf band's cut tolerance - a quarter pixel at the finest zoom the band is drawn at - and
# the chunking that keeps a whole-region coverage_simplify inside memory. Chunking is safe because
# the pinned chunk outlines make the seams exact, and cheap only because the rows are in the
# published order, where a contiguous run is a compact clump.
LEAF_TOLERANCE = max(hy.basins.zoom_tolerance(hy.basins.LEAF_ZOOMS[1]), 1.0)
LEAF_CHUNK = 20_000


def out_name(kind: str, group_id: int) -> str:
    return f'{kind}_{group_id}{SUFFIXES.get(kind, ".geo.parquet")}'


def lines_path(path: Path) -> Path:
    return path.with_name(path.name.replace('.fgb', '.lines.fgb'))


# ---------------------------------------------------------------------------
# the leaf tile band (ported from the retired basins step, global index attached)
# ---------------------------------------------------------------------------
def write_band(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """One band as the polygon + boundary-line FlatGeobuf pair tippecanoe reads. Integer columns
    go out as float64 because tippecanoe's fgb reader fails -j comparisons on integers and
    silently drops the features. The aside name keeps the .fgb extension - handed anything else,
    GDAL writes a directory dataset tippecanoe cannot mmap."""
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
    """The published catchments thinned to the leaf band's tolerance, in place - the group files
    are already written from the full-resolution geometry by the time this runs, and the cut copy
    is the last thing the region needs, so mutating saves holding both."""
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


# ---------------------------------------------------------------------------
# phase 1: the group table and the global products, from the metadata alone
# ---------------------------------------------------------------------------
def region_group_runs(meta: pd.DataFrame, region: str) -> pd.DataFrame:
    """One row per group in this region: (group, local start, size). Also where the two
    region-local invariants everything downstream leans on are checked: the stamped riverIndex is
    the row position, and every group is one contiguous run of it."""
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


def stamped_start(path: Path) -> int:
    """The first riverIndex a written group part carries, from the parquet footer statistics -
    the cheap probe that tells a skipped region its offsets are still current."""
    metadata = pq.ParquetFile(path).metadata
    index = metadata.schema.names.index(hy.schema.river_index)
    statistics = metadata.row_group(0).column(index).statistics
    return int(statistics.min) if statistics is not None else -1


# ---------------------------------------------------------------------------
# phase 2: one region, read once, split into its groups, leaf band cut
# ---------------------------------------------------------------------------
def split_region(region: str, meta: pd.DataFrame, runs: pd.DataFrame) -> bool:
    """Write every published file this region contributes.

    ``meta`` arrives already stamped with the global riverIndex. The geometry tables never join
    anything: streams and catchments are asserted to be row-for-row the same reaches as the
    metadata and take their columns positionally; confluences map each junction to its reach's
    position once and sort. Group parts are contiguous row slices after that - no hashing.
    """
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


def region_is_current(region: str, runs: pd.DataFrame, has_catchments: bool) -> bool:
    """True when every file this region contributes exists and its stamped offsets match this
    run's group table - checked from parquet footers, no geometry read."""
    kinds = ['metadata', 'streams', 'confluences'] + (['catchments'] if has_catchments else [])
    leaf = region_root / region / f'catchments_tile_{region}.fgb'
    if has_catchments and not (leaf.exists() and lines_path(leaf).exists()):
        return False
    for run in runs.itertuples():
        group_id = int(getattr(run, hy.schema.group_id))
        for kind in kinds:
            path = hy.paths.group_dir(group_id) / out_name(kind, group_id)
            if not path.exists():
                return False
        probe = hy.paths.group_dir(group_id) / out_name('metadata', group_id)
        if stamped_start(probe) != int(run.global_start):
            logging.info(f'{region}: group {group_id} is stamped from a different group table, '
                         f'rebuilding the region')
            return False
    return True


# ---------------------------------------------------------------------------
# what the release came out to
# ---------------------------------------------------------------------------
SUMMARY_KINDS = ['metadata', 'streams', 'confluences', 'catchments']


def dataset_summary(runs: pd.DataFrame, metadata: pd.DataFrame) -> None:
    """Print the release's headline numbers, counted off the files this step just wrote.

    Every row count comes from a parquet footer - a seek and a few KB, no row group decoded - so
    the whole dataset is counted for the price of opening the files. Counting from the footers
    rather than from the frames still in memory is the point: it is the written bytes that ship,
    and a group total that disagrees with the global one is a real defect this surfaces for free.
    """
    group_ids = [int(g) for g in runs[hy.schema.group_id]]
    counts = dict.fromkeys(SUMMARY_KINDS, 0)
    sizes = dict.fromkeys(SUMMARY_KINDS, 0)
    absent = dict.fromkeys(SUMMARY_KINDS, 0)
    for group_id in group_ids:
        for kind in SUMMARY_KINDS:
            path = hy.paths.group_dir(group_id) / out_name(kind, group_id)
            if not path.exists():
                absent[kind] += 1  # catchments are absent whenever step 4 has not run
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
    # the group parts are the global table, partitioned - anything else means a stale group file
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
    # todo pull this from a file or config? env-overridable so a partial tree can be assembled
    # deliberately (and tested) without editing the script
    n_regions_expected = int(os.environ.get('EXPECT_REGIONS', 50))
    if len(metadata_paths) != n_regions_expected:
        raise RuntimeError(f'Expected {n_regions_expected} region metadata files, '
                           f'found {len(metadata_paths)}')
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

    # stamp the global riverIndex into each region's metadata frame, in place, no traversal
    for row in runs.itertuples():
        frame = frames[row.region]
        block = slice(int(row.local_start), int(row.local_start + row.size))
        frame.loc[frame.index[block], hy.schema.river_index] = np.arange(
            int(row.global_start), int(row.global_start + row.size), dtype=np.int32)

    # the global frame: group slices concatenated in groupId order. riverIndex == row position by
    # construction; everything else about the ordering is CHECKED here, vectorized, rather than
    # re-derived - the nested-set validator proves parent-after-child, the upstreamCount blocks,
    # and seamless watershed tiling in one pass, which also cross-checks step 3's upstreamCount
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

    # the global products, written only when a region file is newer than they are
    global_root.mkdir(parents=True, exist_ok=True)
    metadata_out = global_root / 'metadata.parquet'
    zarr_out = global_root / 'metadata.zarr'
    newest_region = max(p.stat().st_mtime for p in metadata_paths)
    if not (metadata_out.exists() and zarr_out.exists()
            and metadata_out.stat().st_mtime >= newest_region):
        hy.parquet.write_parquet(metadata, metadata_out)
        logging.info(f'wrote {len(metadata):,} rows -> {metadata_out.name}')
        # the projection a client walking the network needs, chunked for range reads
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

    # phase 2: regions in parallel, each read once, skipped without any geometry read when its
    # outputs exist and their stamped offsets still match this run's group table
    runs_by_region = {r: g for r, g in runs.groupby('region', sort=False)}
    catchments_present = {r: (region_root / r / f'catchments_{r}.geo.parquet').exists()
                          for r in regions}

    def process(region: str) -> tuple:
        region_runs = runs_by_region[region]
        if region_is_current(region, region_runs, catchments_present[region]):
            print(f'region {region}: outputs current, skipped')
            return region, False
        return region, split_region(region, frames[region], region_runs)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rebuilt = sum(1 for _, did in pool.map(process, regions) if did)

    print(f'done: {rebuilt} region(s) rebuilt, {time.time() - started:.0f}s. '
          f'Group boundaries come from step 6.')
    dataset_summary(runs, metadata)
