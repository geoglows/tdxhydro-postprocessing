import os
import glob
import pandas as pd
import geopandas as gpd
from multiprocessing import Pool
import numpy as np


def count_streams(gpkg):
    return os.path.basename(gpkg).split('_')[2], gpd.read_file(gpkg, ignore_geometry=True).shape[0]


if __name__ == '__main__':
    with Pool(12) as p:
        results = p.map(count_streams, sorted(glob.glob('/Users/rchales/Data/TDXHydro/TDX_streamnet*')))
    results = np.array(results)
    pd.DataFrame(results, columns=['region', 'count']).to_csv('network_data/stream_counts_source.csv', index=False)
