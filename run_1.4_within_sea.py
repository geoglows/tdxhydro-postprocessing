import os
import glob

import tqdm
import pandas as pd
import networkx as nx
import geopandas as gpd

from tdxhydrorapid.network import create_directed_graphs


world_seas = r"C:\Users\lrr43\Downloads\World_Seas_IHO_v3\World_Seas_IHO_v3.parquet"
tdx_parquets = glob.glob(r"D:\geoglows_v3\parquets\TDX_streamnet_*_01.parquet")
output_dir = os.path.join(os.path.dirname(__file__), 'tdxhydrorapid', 'network_data')
ID_FIELD = 'LINKNO'
MAX_AREA = 100_000_000

# Load the shapefiles
seas = gpd.read_file(world_seas)
if seas.sindex is None:
    seas.sindex = seas.sindex

def find_sink_from_node(G: nx.DiGraph, start_node: int) -> int:
    """
    Given a directed graph G and a starting node, find the sink node reachable from start_node.
    A sink is a node with no outgoing edges.
    """
    current_node = start_node
    successors = list(G.successors(current_node))
    while successors:
        current_node = successors[0]
        successors = list(G.successors(current_node))

    return current_node

    

within_sea = set()
for tdx_parquet in tqdm.tqdm(tdx_parquets, total=len(tdx_parquets)):
    tdx = gpd.read_parquet(tdx_parquet)
    tdx = tdx.to_crs(seas.crs)
    G = create_directed_graphs(tdx)

    # Select rivers are entirely within the seas
    sea_rivers = set(gpd.sjoin(tdx, seas, predicate='within')[ID_FIELD].values)

    # Sea rivers should all have a terminal node with a dslinkno of -1
    true_sea_rivers = set()
    sea_g = create_directed_graphs(tdx.loc[tdx[ID_FIELD].isin(sea_rivers)])
    for potential_downstream in sea_rivers:
        if tdx.loc[tdx[ID_FIELD] == find_sink_from_node(sea_g, potential_downstream), 'DSLINKNO'].values[0] == -1:
            true_sea_rivers.add(potential_downstream)
    within_sea.update(true_sea_rivers)

pd.DataFrame(list(within_sea), columns=['drop']).to_csv(os.path.join(output_dir, 'within_sea.csv'), index=False)