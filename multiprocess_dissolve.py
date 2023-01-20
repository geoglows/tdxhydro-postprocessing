from multiprocessing import Pool

import geopandas as gpd
import pandas as pd
import json


def merge_streams(upstream_ids: list, network_gdf: gpd.GeoDataFrame):
    return network_gdf[network_gdf["LINKNO"].isin(upstream_ids)].dissolve()


network_gpkg = '/Users/rchales/Data/NGA_delineation/Caribbean/TDX_streamnet_7020065090_01.shp'
output_gpkg_name = 'merged_carribean.gpkg'

if __name__ == '__main__':
    # todo: make only 1 input which is the path to the geopackage/shapefile of the stream network
    # todo: do not read json files, generate them in the script
    # todo: make the output file name one of the parameters you can specify at the start of the script
    # todo: make the readme.md file in the root of the repo

    gdf = gpd.read_file(network_gpkg)

    order2json = "./streamlink_json/carribean_order_2.json"

    with open(order2json) as f:
        order_2_dict = json.load(f)

    allorderjson = "./streamlink_json/carribean_allorders.json"

    with open(allorderjson) as f:
        allorders_dict = json.load(f)

    toporder2 = set([value[-1] for value in list(order_2_dict.values())])

    with Pool() as p:
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

    gdf.to_file(output_gpkg_name, driver="GPKG")
