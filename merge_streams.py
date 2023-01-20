import geopandas as gpd
import pandas as pd
import json

gdf = gpd.read_file("Carribean/TDX_streamnet_7020065090_01.shp")

# new_gdf = gdf[gdf["strmOrder"] != 1]
# stream_order1s = list(gdf[gdf["strmOrder"] == 1]['LINKNO'])
# make a copy of all the streams that are not going to get dissolved (drop the ones that will)
# new_gdf = gdf[gdf['LINKNO'].notin(streamorder_1)]

order2json = "output_json/carribean_order_2.json"

f = open(order2json)
order_2_dict = json.load(f)

allorderjson = "output_json/carribean_allorders.json"

f = open(allorderjson)

allorders_dict = json.load(f)

toporder2 = []
allorder = []

for value in list(order_2_dict.values()):
    toporder2.append(value[-1])

toporder2 = list(set(toporder2))
toporder_2s = gdf[gdf["LINKNO"].isin(toporder2)]
order_2s = gdf[gdf["strmOrder"] == 2]
order_2s.to_file("output_shps/allorder2s.shp")
print(toporder2)


# for each stream order 2 merge point/node, merge the corrects together, and create a big list
list_of_merged_streams = []
for id in toporder2:
    upstreamids = allorders_dict[str(id)]
    features_to_dissolve = gdf[gdf["LINKNO"].isin(upstreamids)].dissolve()
    gdf = pd.concat([
        gdf[~gdf["LINKNO"].isin(upstreamids)],
        features_to_dissolve
    ])
print(gdf)

    # run the geopandas merge command

# slight change in syntax will be required
# concat wants 1 arg: [gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, ...]
# concat wants 1 arg: [gpd.GeoDataFrame, [gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, ...] ]
# new_gdf = pd.concat([new_gdf, list_of_merged_streams])
#
gdf.to_file("output_shps/merged_carribean.shp")
