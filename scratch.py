import json

import geopandas as gpd
import pandas as pd

with open('regions_to_skip.json', 'r') as f:
    regions_to_skip = json.load(f)

df = pd.read_csv('./network_data/stream_counts_merged.csv')
gdf = gpd.read_file('/Users/rchales/Data/TDXHydro/hydrobasins_level2.geojson')

gdf = gdf.loc[~gdf['HYBAS_ID'].isin(regions_to_skip)]
df = df.loc[~df['region'].isin(regions_to_skip)]

df.to_csv('./stream_counts.csv', index=False)
gdf.to_file('./dropped_basins.geojson', driver='GeoJSON')
