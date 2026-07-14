import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
import hydrography as hy

region_root = root / 'data' / 'regions'
global_root = root / 'data' / 'global'
logs_root = root / 'data' / 'logs'

if __name__ == '__main__':
    global_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logs_root / 'concatenate_global.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    # scan every region directory for the simplified-stream and metadata tables step 2 wrote and
    # concatenate whatever is present. this can run before every region is finished and is safe to
    # rerun: it rebuilds the global files from scratch each time rather than appending.
    region_dirs = sorted((d for d in region_root.glob('*') if d.is_dir()), key=str)

    simplified_frames = []
    metadata_frames = []
    region_counts = {}
    for d in region_dirs:
        region = d.name
        metadata_path = d / f'metadata_{region}.parquet'
        simplified_path = d / f'streams_simplified_{region}.geo.parquet'

        if metadata_path.exists():
            mdf = pd.read_parquet(metadata_path)
            metadata_frames.append(mdf)
            region_counts[region] = len(mdf)
        if simplified_path.exists():
            sdf = gpd.read_parquet(simplified_path)
            simplified_frames.append(sdf)
            region_counts.setdefault(region, len(sdf))

    if not metadata_frames and not simplified_frames:
        sys.exit(f'No metadata or simplified-stream files found under {region_root}; run steps 2-3 first')

    # concatenate the metadata (attribute) tables into one global table
    if metadata_frames:
        global_metadata = pd.concat(metadata_frames, ignore_index=True)
        metadata_out = global_root / 'metadata_global.parquet'
        global_metadata.to_parquet(metadata_out)
        logging.info(f'Wrote {len(global_metadata):,} rows to {metadata_out}')
        print(f'Metadata: {len(global_metadata):,} reaches -> {metadata_out}')

    # concatenate the simplified-geometry tables into one global table
    if simplified_frames:
        global_simplified = gpd.GeoDataFrame(
            pd.concat(simplified_frames, ignore_index=True),
            geometry=hy.schema.geometry,
            crs=simplified_frames[0].crs,
        )
        simplified_out = global_root / 'streams_simplified_global.geo.parquet'
        global_simplified.to_parquet(simplified_out)
        logging.info(f'Wrote {len(global_simplified):,} rows to {simplified_out}')
        print(f'Simplified streams: {len(global_simplified):,} reaches -> {simplified_out}')

    # summary of total stream count across the concatenated regions
    total = sum(region_counts.values())
    logging.info(f'Total streams across {len(region_counts)} region(s): {total:,}')
    print(f'\nConcatenated {len(region_counts)} region(s), {total:,} total streams:')
    for region, n in sorted(region_counts.items(), key=lambda kv: int(kv[0])):
        print(f'  region {region}: {n:,} streams')
