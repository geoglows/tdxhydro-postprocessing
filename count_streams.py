import os
import glob
import pandas as pd
import geopandas as gpd
from multiprocessing import Pool
import numpy as np


def count_streams(gpkg):
    return os.path.basename(gpkg).split('_')[2], gpd.read_file(gpkg, ignore_geometry=True).shape[0]


if __name__ == '__main__':
    streams_paths = sorted(glob.glob('/tdxrapid/*/TDX_streamnet*model.gpkg'))
    with Pool(12) as p:
        results = p.map(count_streams, streams_paths)
    results = np.array(results)
    pd.DataFrame(results, columns=['region', 'count']).to_csv('network_data/stream_counts.csv', index=False)

    merged_df = (
        pd
        .read_csv('network_data/stream_counts.csv')
        .merge(pd.read_csv('network_data/stream_counts_source.csv'),
               how='outer', left_on='region', right_on='region', suffixes=('_calc', '_source'))
        .sort_values(by='region')
    )
    merged_df['percent_removed'] = (merged_df['count_source'] - merged_df['count_calc']) / merged_df['count_source']
    merged_df['percent_removed'] = merged_df['percent_removed'].round(4)
    merged_df.to_csv('network_data/stream_counts_merged.csv', index=False)

