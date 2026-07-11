import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd
from natsort import natsorted

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hydrography as hy

region_root = Path('/Users/rchales/code/untitled folder/tdxhydro-postprocessing/data/regions')
tdx_root = Path('/Users/rchales/code/untitled folder/tdxhydro-postprocessing/data/TDXHydroGeoParquet')
network_data_root = Path('/Users/rchales/code/untitled folder/tdxhydro-postprocessing/network_data')
global_root = Path('/Users/rchales/code/untitled folder/tdxhydro-postprocessing/data/global')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def slot_river_index(frame_path: Path, index_lookup: pd.DataFrame) -> None:
    """Merge riverIndex onto a frame by riverId, slot it right after the 3 id columns, write back.
    Merging by riverId (not row order) makes it correct on any streams variant and preserves the
    frame's geometry/CRS; dropping any existing riverIndex first keeps a re-run from colliding."""
    reader = gpd.read_parquet if frame_path.name.endswith('.geo.parquet') else pd.read_parquet
    frame = reader(frame_path)
    if hy.schema.river_index in frame.columns:
        return
    frame = frame.drop(columns=[hy.schema.river_index], errors='ignore')
    cols = list(frame.columns)
    new_cols = cols[:3] + [hy.schema.river_index] + cols[3:]
    frame.merge(index_lookup, on=hy.schema.river_id, how='left')[new_cols].to_parquet(frame_path, index=False)
    logging.info(f'stamped riverIndex -> {frame_path.name}')


if __name__ == '__main__':
    global_root.mkdir(parents=True, exist_ok=True)
    MAX_WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    metadata_parquets = natsorted(region_root.glob('*/metadata_*.parquet'), key=str)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        frames = list(pool.map(pd.read_parquet, metadata_parquets))  # map() preserves input order
        df = (
            pd.concat(frames, ignore_index=True)
            .drop(columns=[hy.schema.river_index], errors='ignore')  # idempotent -> drop if exists riverIndex
            .reset_index(drop=True)
            .reset_index(names=[hy.schema.river_index])
        )
        river_index_lookup = df[[hy.schema.river_id, hy.schema.river_index]]
        river_index_lookup.to_parquet(global_root / 'riverId_riverIndex.parquet', index=False)
        parquets_to_update = natsorted(
            list(region_root.glob('*/metadata*.parquet')) +
            list(region_root.glob('*/streams*.geo.parquet')),
            key=str,
        )
        list(pool.map(lambda p: slot_river_index(p, river_index_lookup), parquets_to_update))
