from multiprocessing import Pool

import geopandas as gpd
import pandas as pd
import json


def merge_streams(upstream_ids: list, network_gdf: gpd.GeoDataFrame):
    return network_gdf[network_gdf["LINKNO"].isin(upstream_ids)].dissolve()


if __name__ == '__main__':
    network_gpkg = '/Users/rchales/Data/NGA_delineation/Caribbean/TDX_streamnet_7020065090_01.shp'
    gdf = gpd.read_file(network_gpkg)

    order2json = "./streamlink_json/carribean_order_2.json"

    with open(order2json) as f:
        order_2_dict = json.load(f)

    allorderjson = "./streamlink_json/carribean_allorders.json"

    with open (allorderjson) as f:
        allorders_dict = json.load(f)

    toporder2 = []
    allorder = []

    for value in list(order_2_dict.values()):
        toporder2.append(value[-1])

    toporder2 = set(toporder2)

    toporder_2s = gdf[gdf["LINKNO"].isin(toporder2)]
    # order_2s = gdf[gdf["strmOrder"] == 2]
    # order_2s.to_file("output_shps/allorder2s.shp")

    # for each stream order 2 merge point/node, merge the corrects together, and create a big list

    with Pool(15) as p:
        merged_features = p.starmap(merge_streams, [(allorders_dict[str(rivid)], gdf) for rivid in toporder2])

    # list all ids that were merged
    all_merged_rivids = [allorders_dict[str(rivid)] for rivid in toporder2]
    # turn a list of lists into a flat list
    all_merged_rivids = [item for sublist in all_merged_rivids for item in sublist]
    # remove duplicates
    all_merged_rivids = set(all_merged_rivids)

    # drop rivids that were merged
    gdf = gdf[~gdf["LINKNO"].isin(all_merged_rivids)]
    # concat the merged features
    gdf = pd.concat([gdf, *merged_features])

    gdf.to_file("merged_carribean.gpkg", driver="GPKG")
