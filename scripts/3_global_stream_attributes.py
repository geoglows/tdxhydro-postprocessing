import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd
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
global_root = hy.paths.global_root
logs_root = hy.paths.logs_root
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def slot_river_index(frame_path: Path, index_lookup: pd.DataFrame) -> None:
    """Stamp riverIndex onto a frame by riverId, slot it right after the 3 id columns, write back"""
    reader = gpd.read_parquet if frame_path.name.endswith('.geo.parquet') else pd.read_parquet
    frame = reader(frame_path)
    if hy.schema.river_index in frame.columns:
        return
    frame = frame.drop(columns=[hy.schema.river_index], errors='ignore')
    cols = list(frame.columns)
    new_cols = cols[:3] + [hy.schema.river_index] + cols[3:]
    stamped = frame.merge(index_lookup, on=hy.schema.river_id, how='left')[new_cols]
    # a left merge widens the joined column to int64 (and to float if anything failed to match),
    # so re-assert the dtypes rather than let the write undo what step 2 set
    stamped = hy.schema.enforce_int32(stamped)
    if frame_path.name.endswith('.geo.parquet'):
        hy.parquet.write_geoparquet(stamped, frame_path, index=False)
    else:
        hy.parquet.write_parquet(stamped, frame_path, index=False)
    logging.info(f'stamped riverIndex -> {frame_path.name}')


if __name__ == '__main__':
    global_root.mkdir(parents=True, exist_ok=True)
    # keep console output (basicConfig above) and also tee the run into data/logs/
    logs_root.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(logs_root / 'global_stream_attributes.log', mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(file_handler)
    MAX_WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    global_output_sentinel = global_root / 'riverId_riverIndex.parquet'

    if global_output_sentinel.exists():
        logging.info('All riverIndexes presumed to be stamped because the global ID->IDX mapper exists')
        exit(0)

    metadata_parquets = natsorted(region_root.glob('*/metadata_*.parquet'), key=str)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        frames = list(pool.map(pd.read_parquet, metadata_parquets))  # map() preserves input order
        df = (
            pd.concat(frames, ignore_index=True)
            .drop(columns=[hy.schema.river_index], errors='ignore')  # idempotent -> drop if exists riverIndex
            .reset_index(drop=True)
            .reset_index(names=[hy.schema.river_index])
        )
        river_index_lookup = hy.schema.enforce_int32(df[[hy.schema.river_id, hy.schema.river_index]].copy())
        parquets_to_update = natsorted(
            list(region_root.glob('*/metadata*.parquet')) +
            list(region_root.glob('*/streams*.geo.parquet')),
            key=str,
        )
        list(pool.map(lambda p: slot_river_index(p, river_index_lookup), parquets_to_update))
        # cache the lookups last and use it as a sentinel whose existence makes the script skip
        hy.parquet.write_parquet(river_index_lookup, global_output_sentinel, index=False)
