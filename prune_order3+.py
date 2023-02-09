from multiprocessing import Pool
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

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
pbar = tqdm()

if __name__ == '__main__':
    with open(all_order_json) as f:
        allorders_dict = json.load(f)

    gdf = gpd.read_file(network_gpkg)

    # order_3up = stream_net[stream_net[order_col] > 2]
    order_3up_ids = gdf[gdf[order_col] > 2][riv_id_col].to_list()

    streams_to_merge = []
    pbar = tqdm(total=len(order_3up_ids))
    # rivid = order_3up_ids[100]
    # prune_stream(rivid, allorders_dict[str(rivid)], streams_to_merge)
    for rivid in order_3up_ids:
        streams_to_merge = prune_stream(rivid, allorders_dict[str(rivid)], streams_to_merge)
    # with Pool() as p:
    #     gdf, streams_to_merge = p.starmap(prune_stream, [(rivid, max_order, allorders_dict[str(rivid)], stream_net, streams_to_merge) for rivid in order_3up_ids])

    print(streams_to_merge)
    with open(output_json_name, "w") as f:
        json.dump(streams_to_merge, f, cls=NpEncoder)

    gdf.to_file(output_gpkg_name, driver="GPKG")
