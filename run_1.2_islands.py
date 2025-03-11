import logging
import sys
import glob
import os
import networkx as nx
import tqdm
from multiprocessing import Pool

import geopandas as gpd
import pandas as pd
from shapely import box


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

islands_parquet = r"C:\Users\lrr43\Downloads\BigIslandsless8000\Big Islands.parquet" # https://garslab.com/?p=234 August 24, 2022
gpq_dir = r"D:\geoglows_v3\parquets"
# gpq_dir = '/Users/ricky/tdxhydro-postprocessing/test/pqs'
save_dir = os.path.join('.', 'tdxhydrorapid', 'network_data')

def create_directed_graphs(df: gpd.GeoDataFrame,
                           id_field='LINKNO',
                           ds_id_field='DSLINKNO', ) -> nx.DiGraph:
    G: nx.DiGraph = nx.from_pandas_edgelist(df[df[ds_id_field] != -1], source=id_field, target=ds_id_field, create_using=nx.DiGraph)
    G.add_nodes_from(df[id_field].values)
    return G

def _helper(gpq: str) -> set:
    islands = gpd.read_parquet(islands_parquet).to_crs('EPSG:4326')
    island_streams_to_drop = set()
    df = gpd.read_parquet(gpq)
    G = create_directed_graphs(df)

    # Watersheds, sorted smallest to largest
    connected_components = sorted(nx.weakly_connected_components(G), key=len)
    for watershed_ids in connected_components:
        # Ignore really large watersheds... probably not a relevant island
        if len(watershed_ids) > 200:
            break

        # Find the intersection with the islands
        bbox = box(*df[df['LINKNO'].isin(watershed_ids)].total_bounds)
        intersected_islands: gpd.GeoSeries = islands[islands.intersects(bbox)]
        if intersected_islands.empty:
            continue

        # Test how many streams intersect the islands.
        # If it is more than 33% of the streams, drop the watershed
        intersected_streams = df[df['LINKNO'].isin(watershed_ids)].intersects(intersected_islands.union_all())
        if intersected_streams.sum() / len(watershed_ids) > 0.33:
            island_streams_to_drop.update(watershed_ids)

    return island_streams_to_drop

if __name__ == "__main__":
    logging.info('Finding Island rivers to drop')
    
    pqs = glob.glob(os.path.join(gpq_dir, '*streamnet*14250*.parquet'))

    # For every 8 GB of total memory, we can process 1 pq in parallel
    with Pool(4) as p, tqdm.tqdm(total=len(pqs)) as pbar:
        island_streams_to_drop = set()
        for island_streams in p.imap_unordered(_helper, pqs):
            island_streams_to_drop.update(island_streams)
            pbar.update(1)

    logging.info("Saving the islands to drop")
    pd.Series(list(island_streams_to_drop), name='island_id').to_csv(os.path.join(save_dir, 'island_table.csv'), index=False)        
    