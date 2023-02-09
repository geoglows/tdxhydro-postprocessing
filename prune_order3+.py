import json

import geopandas as gpd

from AdjoinUpdown import NpEncoder


# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NpEncoder, self).default(obj)


def get_downstream_order(x, network_gdf: gpd.GeoDataFrame):
    ord = 0
    if x != -1:
        orders = network_gdf.loc[network_gdf[riv_id_col] == x, order_col].values
        if len(orders) > 0:
            ord = orders[0]
        # print(ord)
    return ord


def merge_catchments(row, network_gdf: gpd.GeoDataFrame, upstream_dict: dict):
    print(row)
    print(row[riv_id_col])
    ids_to_merge = [row[ds_id_col], ] + list(upstream_dict[str(row[riv_id_col])])
    return network_gdf.loc[network_gdf[riv_id_col].isin(ids_to_merge)].dissolve()


def prune_network(network_gdf: gpd.GeoDataFrame, upstream_dict: dict) -> gpd.GeoDataFrame:
    """
    Finds all order ones that feed into a larger order river (without forming an order 2 with another river), joins
    their area down onto the larger river, and drops their river segment from the network.

    Args:
        network_gdf: geodataframe containing the stream network
        upstream_dict: dictionary containing all upstream stream ids for each stream segment

    Returns: pruned network geodataframe

    """
    network_gdf['_next_down_order'] = network_gdf[ds_id_col].apply(get_downstream_order, network_gdf=network_gdf)

    network_gdf.loc[(network_gdf[order_col] == 1) & (network_gdf['_next_down_order'] >= 3)] = \
        network_gdf.loc[(network_gdf[order_col] == 1) & (network_gdf['_next_down_order'] >= 3)] \
        .apply(merge_catchments, network_gdf=network_gdf, upstream_dict=upstream_dict, axis=1)

    network_gdf.drop(network_gdf[(network_gdf[order_col] == 1) & (network_gdf['_next_down_order'] >= 3)].index,
                     inplace=True)
    return network_gdf


def prune_stream(stream_id: int, upstream_list: list, streams_to_merge: list):
    """
    Removes all streams upstream of a given segment whose order is less than (<) the given max_order
    Args:
        stream_id: unique id of the stream to prune
        upstream_list: list produced using AdjointCatchments that contains all upstream segment ids

    Returns: dictionary attaching a list of the pruned ids to the stream_id so that the catchment areas corresponding to
    those regions can be merged down
    """
    upstream_rows = gdf[gdf[ds_id_col] == stream_id]
    print(upstream_rows)
    if 1 in upstream_rows[order_col]:
        upstream_order_1_id = int(upstream_rows[upstream_rows[order_col] == 1][riv_id_col])
        rows_to_delete = gdf[gdf[riv_id_col].isin(upstream_list[str(upstream_order_1_id)])]
        gdf.drop(index=rows_to_delete.index, axis=0)
        streams_to_merge.append({stream_id: rows_to_delete[riv_id_col].to_list()})
        pbar.update(1)
    return streams_to_merge


region = 'japan'
all_order_json = f'output_jsons/{region}_allorders.json'
network_gpkg = f'Japan/TDX_streamnet_4020034510_01.shp'
output_gpkg_name = f'pruned_{region}.gpkg'
output_json_name = f'output_jsons/{region}_streams_to_prune.json'

max_order = 2
riv_id_col = 'LINKNO'
ds_id_col = 'DSLINKNO'
order_col = 'strmOrder'

if __name__ == '__main__':
    with open(all_order_json) as f:
        allorders_dict = json.load(f)

    gdf = gpd.read_file(network_gpkg)

    # order_3up = stream_net[stream_net[order_col] > 2]
    order_3up_ids = gdf[gdf[order_col] > 2][riv_id_col].to_list()

    # streams_to_merge = []
    # print()
    gdf = prune_network(gdf, allorders_dict)

    # rivid = order_3up_ids[100]
    # prune_stream(rivid, allorders_dict[str(rivid)], streams_to_merge)
    # for rivid in order_3up_ids:
    #     streams_to_merge = prune_stream(rivid, allorders_dict[str(rivid)], streams_to_merge)
    # with Pool() as p:
    #     gdf, streams_to_merge = p.starmap(prune_stream, [(rivid, max_order, allorders_dict[str(rivid)], stream_net, streams_to_merge) for rivid in order_3up_ids])

    # print(streams_to_merge)
    # with open(output_json_name, "w") as f:
    #     json.dump(streams_to_merge, f, cls=NpEncoder)

    gdf.to_file(output_gpkg_name, driver="GPKG")
