"""
Assign the global ordering: riverIndex, upstreamCount, and a recomputed shreveOrder.

This is the step that decides the row order of every published file, and with it how the dataset
behaves under both of its workloads - marching a Muskingum kernel through long arrays, and asking a
map "what is upstream of here". It has to run globally because riverIndex is a position in a single
ordering spanning all 50 regions; nothing about it can be known while one region is processed alone.

What it does NOT do is redo any of step 2's work. The ordering is a permutation of rows plus three
derived integers, so the expensive simplification (lakes, dissolves, pruning, catchments) is
untouched and does not need to re-run when the ordering changes.

See docs/river-index.md for what the ordering guarantees and why it is built this way; the traversal
itself lives in hydrography/topology.py::nested_set_order.

    RFS_DATA_ROOT=... TDXHYDRO_ROOT=... python 3_global_stream_attributes.py [workers]
"""
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from natsort import natsorted

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

# This step REWRITES the files step 2 wrote, so it has to write them the same way — see
# hydrography/parquet.py. Going through pyarrow's defaults instead would silently collapse them
# back into one row group per file (measured: 10 row groups in, 1 out), undoing the only thing that
# makes the geometry subsettable, and re-encode the geometry column back to WKB.
#
# Only metadata_* and streams_* reach here, and streams_* are the tables the small row groups are
# for. Confluences would not want them — one point per row is ~29 bytes, so a whole file is about a
# megabyte — but this step's globs never match them.

# every output path hangs off the data root - see hydrography/paths.py and $RFS_DATA_ROOT
region_root = hy.paths.region_root
logs_root = hy.paths.logs_root
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# the columns this step derives and hands back to every region file
index_lookup_columns = [hy.schema.river_id] + hy.schema.index_columns + [hy.schema.shreve_order]


def stamp_index_columns(frame_path: Path, index_lookup: pd.DataFrame) -> None:
    """Join the index columns onto a region frame, reorder its rows by riverIndex, and write back."""
    is_geo = frame_path.name.endswith('.geo.parquet')
    frame = (gpd.read_parquet if is_geo else pd.read_parquet)(frame_path)
    published = hy.schema.final_columns_to_keep if is_geo else hy.schema.metadata_columns_to_keep
    columns = hy.schema.insert_index_columns(published)

    # drop rather than skip if the columns are already there: this step has to be re-runnable, and an
    # earlier build's riverIndex is exactly what a rebuild exists to replace
    frame = frame.drop(columns=hy.schema.index_columns, errors='ignore')
    stamped = frame.merge(index_lookup, on=hy.schema.river_id, how='left', suffixes=('_stale', ''))
    if stamped[hy.schema.river_index].isna().any():
        missing = int(stamped[hy.schema.river_index].isna().sum())
        raise ValueError(f'{frame_path.name}: {missing:,} reach(es) got no riverIndex from the '
                         f'global ordering. The region files and the ordering are out of step.')
    missing_columns = [c for c in columns if c not in stamped.columns]
    if missing_columns:
        raise ValueError(f'{frame_path.name}: expected column(s) {missing_columns} are not present')

    # THE point of this step: riverIndex is only useful as a position if it IS the row position
    stamped = stamped.sort_values(hy.schema.river_index)[columns].reset_index(drop=True)
    # a left merge widens the joined columns to int64 (and to float if anything failed to match),
    # so re-assert the dtypes rather than let the write undo what step 2 set
    stamped = hy.schema.enforce_int32(stamped)
    if is_geo:
        hy.parquet.write_geoparquet(stamped, frame_path, index=False)
    else:
        hy.parquet.write_parquet(stamped, frame_path, index=False)
    logging.info(f'stamped {len(stamped):,} rows -> {frame_path.name}')


if __name__ == '__main__':
    # keep console output (basicConfig above) and also tee the run into data/logs/
    logs_root.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(logs_root / 'global_stream_attributes.log', mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(file_handler)
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    MAX_WORKERS = int(args[0]) if args else 4

    metadata_parquets = natsorted(region_root.glob('*/metadata_*.parquet'), key=str)

    # Whether this step has already run is asked of the files it writes, not of a marker beside
    # them. The stamped region files ARE the output - a marker is a second claim about them that can
    # outlive the thing it describes, and this one was also being published into group=0 as if it
    # were a product. Reading a parquet footer is cheap enough to just look.
    if metadata_parquets and all(
        set(hy.schema.index_columns) <= set(pq.read_schema(p).names) for p in metadata_parquets
    ):
        logging.info(f'all {len(metadata_parquets)} region metadata files already carry '
                     f'{hy.schema.index_columns}, so the ordering is stamped. Delete them to '
                     f'recompute it.')
        exit(0)

    logging.info(f'reading {len(metadata_parquets)} region metadata files')
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        frames = list(pool.map(pd.read_parquet, metadata_parquets))  # map() preserves input order
        df = (
            pd.concat(frames, ignore_index=True)
            .drop(columns=hy.schema.index_columns, errors='ignore')  # idempotent -> a rebuild replaces them
            .reset_index(drop=True)
        )
        del frames
        logging.info(f'ordering {len(df):,} reaches in {df[hy.schema.group_id].nunique()} groups')

        # the whole point of the step. everything below is bookkeeping around this one call
        df = hy.topology.nested_set_order(df)
        df[hy.schema.river_index] = np.arange(len(df), dtype=np.int32)

        # Groups are ordered whole, so each has to come out as one unbroken range of riverIndex. That
        # is what lets a group file be treated as a dense array indexed by riverIndex - riverIndexStart,
        # and it is checked here because every per-group consumer downstream assumes it.
        groups = df[hy.schema.group_id].to_numpy()
        runs = int((groups[1:] != groups[:-1]).sum()) + 1
        if runs != df[hy.schema.group_id].nunique():
            raise ValueError(f'groupId occupies {runs:,} runs of riverIndex but there are only '
                             f'{df[hy.schema.group_id].nunique():,} groups, so at least one group is '
                             f'split across the ordering.')

        index_lookup = hy.schema.enforce_int32(df[index_lookup_columns].copy())
        parquets_to_update = natsorted(
            list(region_root.glob('*/metadata*.parquet')) +
            list(region_root.glob('*/streams*.geo.parquet')),
            key=str,
        )
        list(pool.map(lambda p: stamp_index_columns(p, index_lookup), parquets_to_update))
        logging.info(f'stamped {len(parquets_to_update)} region files with the global ordering')
