import glob
import logging
import os
import sys

import geopandas as gpd
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydrography as hy

region_root = '/Users/rchales/code/untitled folder/tdxhydro-postprocessing/data/regions'
global_root = '/Users/rchales/code/untitled folder/tdxhydro-postprocessing/data/global'

if __name__ == '__main__':
    os.makedirs(global_root, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(global_root, 'concatenate_log.log'),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    # scan every region directory for the simplified-stream and metadata tables step 2 wrote and
    # concatenate whatever is present. this can run before every region is finished and is safe to
    # rerun: it rebuilds the global files from scratch each time rather than appending.
    region_dirs = sorted(d for d in glob.glob(os.path.join(region_root, '*')) if os.path.isdir(d))

    simplified_frames = []
    metadata_frames = []
    region_counts = {}
    for d in region_dirs:
        region = os.path.basename(d)
        metadata_path = os.path.join(d, f'metadata_{region}.parquet')
        simplified_path = os.path.join(d, f'streams_simplified_{region}.geo.parquet')

        if os.path.exists(metadata_path):
            mdf = pd.read_parquet(metadata_path)
            metadata_frames.append(mdf)
            region_counts[region] = len(mdf)
        if os.path.exists(simplified_path):
            sdf = gpd.read_parquet(simplified_path)
            simplified_frames.append(sdf)
            region_counts.setdefault(region, len(sdf))

    if not metadata_frames and not simplified_frames:
        sys.exit(f'No metadata or simplified-stream files found under {region_root}; run steps 2-3 first')

    # concatenate the metadata (attribute) tables into one global table
    if metadata_frames:
        global_metadata = pd.concat(metadata_frames, ignore_index=True)
        metadata_out = os.path.join(global_root, 'metadata_global.parquet')
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
        simplified_out = os.path.join(global_root, 'streams_simplified_global.geo.parquet')
        global_simplified.to_parquet(simplified_out)
        logging.info(f'Wrote {len(global_simplified):,} rows to {simplified_out}')
        print(f'Simplified streams: {len(global_simplified):,} reaches -> {simplified_out}')

    # summary of total stream count across the concatenated regions
    total = sum(region_counts.values())
    logging.info(f'Total streams across {len(region_counts)} region(s): {total:,}')
    print(f'\nConcatenated {len(region_counts)} region(s), {total:,} total streams:')
    for region, n in sorted(region_counts.items(), key=lambda kv: int(kv[0])):
        print(f'  region {region}: {n:,} streams')
