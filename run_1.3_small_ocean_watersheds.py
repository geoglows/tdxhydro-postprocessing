import os
import glob

import tqdm
import pandas as pd
import networkx as nx
import geopandas as gpd

from tdxhydrorapid.network import create_directed_graphs


world_seas = r"C:\Users\lrr43\Downloads\World_Seas_IHO_v3\World_Seas_IHO_v3.parquet"
tdx_parquets = glob.glob(r'D:\geoglows_v3\parquets\TDX_streamnet_*_01.parquet')
output_dir = os.path.join(os.path.dirname(__file__), 'tdxhydrorapid', 'network_data')
MAX_AREA = 100_000_000

# Load the shapefiles
seas = gpd.read_parquet(world_seas)
if seas.sindex is None:
    seas.sindex = seas.sindex

small_ocean_watersheds = set()
for tdx_parquet in tqdm.tqdm(tdx_parquets, total=len(tdx_parquets)):
    tdx = gpd.read_parquet(tdx_parquet)
    tdx = tdx.to_crs(seas.crs)
    G = create_directed_graphs(tdx)

    bbox = tdx.total_bounds
    seas_sub = seas.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    # Select rivers that intersect with the seas
    outlets = tdx[tdx['DSLINKNO'] == -1]
    spatial_index = outlets.sindex
    possible_matches_index = spatial_index.intersection(seas_sub.total_bounds)
    possible_matches = outlets.iloc[possible_matches_index]
    ocean_outlets = set(gpd.sjoin(possible_matches, seas_sub)['LINKNO'])

    ocean_outlets = set(tdx.loc[(tdx['LINKNO'].isin(ocean_outlets)) & (tdx['DSContArea'] <= MAX_AREA), 'LINKNO'].values)
    for outlet in ocean_outlets:
        small_ocean_watersheds.update(nx.ancestors(G, outlet))
        small_ocean_watersheds.add(outlet)

pd.DataFrame(small_ocean_watersheds, columns=['drop']).to_csv(os.path.join(output_dir, 'small_ocean_watersheds.csv'), index=False)