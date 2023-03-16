from multiprocessing import Pool
from collections.abc import Iterable
from itertools import chain

import shapely.geometry as sg
import json
import os
import queue
import geopandas as gpd
import pandas as pd
import numpy as np
import time

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def make_tree_up(df: pd.DataFrame, order: int = 0, stream_id_col: str = "COMID", next_down_id_col: str = "NextDownID", order_col: str = "order_") -> dict:
    """
    Makes a dictionary depicting a tree where each segment id as a key has a tuple containing the ids of its parent segments, or the ones that
    have it as the next down id. Either does this for every id in the tree, or only includes ids of a given stream order
    and their parent of the same stream order, if they have one. This function attempts to use pandas vectorization and
    indexing to improve efficiency on make_tree, but may miss some edge cases, further testing is underway.
    Args:
        df: dataframe to parse the tree from. Must contain:
            - a column with the segment/catchment ID ("HydroID")
            - a column with the IDs for the next down segment ("NextDownID")
            - an order column
        order: number of stream order to limit it to. If zero, will make tree for all segments,
               otherwise only contains ids that match the given stream order.
        stream_id_col: the name of the column that contains the unique ids for the streams
        next_down_id_col: the name of the column that contains the unique id of the next down stream for each row, the
                          one that the stream for that row feeds into.
        order_col: name of the column that contains the stream order
    Returns: dictionary where for each key, a tuple of all values that have that key as their next down id is assigned.
             if order==0, values will be either length 0 or 2, otherwise will be 0 or one as only a maximum of
             one parent will be of the given order.
    """
    if order == 0:
        out = df[[stream_id_col, next_down_id_col]].set_index(next_down_id_col)
        out.drop(-1, inplace=True)
        tree = {}
        for hydroid in df[stream_id_col]:
            if hydroid in out.index:
                rows = out.loc[hydroid][stream_id_col]
                if not (isinstance(rows, np.floating) or isinstance(rows, np.generic)):
                    tree[hydroid] = tuple(rows.tolist())
                else:
                    tree[hydroid] = (rows,)
            else:
                tree[hydroid] = ()
        return tree
    out = df[df[order_col] == order][[stream_id_col, next_down_id_col, order_col]].set_index(next_down_id_col)
    tree = {hydroid: ((int(out.loc[hydroid][stream_id_col]),) if hydroid in out.index else ()) for hydroid in
            df[df[order_col] == order][stream_id_col]}
    return tree

def make_tree_down(df: pd.DataFrame, order: int = 0, stream_id_col: str = "COMID", next_down_id_col: str = "NextDownID", 
                   order_col: str = "order_") -> dict:
    """
    Performs the simpler task of pairing segment ids as keys with their next down ids as values.
    Args:
        df: dataframe to parse the tree from. Must contain:
            - a column with the segment/catchment ID ("HydroID")
            - a column with the IDs for the next down segment ("NextDownID")
            - an order column
        order: number of stream order to limit it to. If zero, will make tree for all segments,
               otherwise only contains ids that match the given stream order.
        stream_id_col: the name of the column that contains the unique ids for the streams
        next_down_id_col: the name of the column that contains the unique id of the next down stream for each row, the
                          one that the stream for that row feeds into.
        order_col: name of the column that contains the stream order
    Returns: dictionary where for each key its next down id from the dataframe is given as a value.
    """
    if order == 0:
        tree = dict(zip(df[stream_id_col], df[next_down_id_col]))
        return tree
    out = df[[stream_id_col, next_down_id_col, order_col]][df[order_col] == order]
    out_2 = out[out[next_down_id_col].isin(out[stream_id_col])]
    out.loc[~out[stream_id_col].isin(out_2[stream_id_col]), next_down_id_col] = -1
    # tree = dict(zip(out.loc[out['NextDownID'] != -1, 'HydroID'], out.loc[out['NextDownID'] != -1, 'NextDownID']))
    tree = dict(zip(out[stream_id_col], out[next_down_id_col]))
    return tree

def trace_tree(tree: dict, search_id: int, cuttoff_n: int = 200) -> list:
    """
    Universal function that traces a tree produced by make_tree_up or make_tree_down from the search id all the way to
    the end of the segment. If the given tree was produced for a given order, it will produce a list with all down or
    upstream segments that share that order. If the tree was produced including all segments, it will get the whole
    upstream network or the whole path to the ocean.
    Args:
        tree: Tree where each key has a tuple containing the ids of each of its parent segments, or an integer of
              a single child, as its value, i.e.:
              {2: (3, 5), 3: (), 4: (): 5: (6, 7), 6: (), 7: ()} for an upstream tree
                or
              {2: -1, 3: 2, 5: 2, 4: -1, 6: 5, 7: 5} for a downstream tree
        search_id: id to search from.
    Returns: list containing all ids that will be upstream of the search_id.
    """
    q = queue.Queue()
    q.put((search_id,))
    upstream = []
    i = 0

    while not q.empty():
        n = q.get()
        if i > cuttoff_n:  # cuts off infinite loops. Number may need to be adjusted if adjoint catchments start to contain more than 200 individual regions
            break
        if isinstance(n, Iterable):
            for s in n:
                if s != -1:
                    upstream.append(s)
                if s in tree:
                    q.put(tree[s])
        else:
            if n != -1:
                upstream.append(n)
            if n in tree:
                q.put(tree[n])
        i += 1
    return upstream

def create_adjoint_dict(network_shp, out_file: str = None, stream_id_col: str = "COMID",
                        next_down_id_col: str = "NextDownID", order_col: str = "order_", trace_up: bool = True,
                        order_filter: int = 0):
    """
    Creates a dictionary where each unique id in a stream network is assigned a list of all ids upstream or downstream
    of that stream, as specified. By default is designed to trace upstream on GEOGloWS Delineation Catchment shapefiles,
    but can be customized for other files with column name parameters, customized to trace down, or filtered by stream
    order. If filtered by stream order, the dictionary will only contain ids of the given stream order, with the
    upstream or downstream ids for the other streams in the chain that share that stream order.
    Args:
        network_shp: path to  .shp file that contains the stream network. This file
                     must contain attributes for a unique id and a next down id, and if filtering by order number is
                     specified, it must also contain a column with stream order values.
        out_file: a path to an output file to write the dictionary as a .json, if desired.
        stream_id_col: the name of the column that contains the unique ids for the stream segments
        next_down_id_col: the name of the column that contains the unique id of the next down stream for each row, the
                          one that the stream for that row feeds into.
        order_col: name of the column that contains the stream order
        trace_up: if true, trace up from each stream, otherwise trace down.
        order_filter: if set to number other than zero, limits values traced to only ids that match streams with that
                      stream order
    Returns:
    """
    # Added to forgo computation on precreated json files, for testing
    if out_file is not None and os.path.exists(out_file):
        with open(out_file, 'r') as f:
            upstream_lists_dict = json.load(f)
        return upstream_lists_dict

    network_df = gpd.read_file(network_shp)
    columns_to_search = [stream_id_col, next_down_id_col]
    if order_filter != 0:
        columns_to_search.append(order_col)
    for col in columns_to_search:
        if col not in network_df.columns:
            print(f"Column {col} not present")
            return {}
    if trace_up:
        tree = make_tree_up(network_df, order_filter, stream_id_col, next_down_id_col, order_col)
    else:
        tree = make_tree_down(network_df, order_filter, stream_id_col, next_down_id_col, order_col)
    if order_filter != 0:
        upstream_lists_dict = {str(hydro_id): trace_tree(tree, hydro_id) for hydro_id in network_df[network_df[order_col] == order_filter][stream_id_col]}
    else:
        upstream_lists_dict = {str(hydro_id): trace_tree(tree, hydro_id) for hydro_id in network_df[stream_id_col]}
    if out_file is not None:
        if not os.path.exists(out_file):
            with open(out_file, "w") as f:
                json.dump(upstream_lists_dict, f, cls=NpEncoder)
        else:
            print("File already created")
            return upstream_lists_dict # Added to return dictionary anyways
    return upstream_lists_dict

def dissolve_network(network_gdf, upstream_ids, streamid) -> gpd.GeoDataFrame:
    """
    Some stream segments have linestrings that aren't in the right order, and so when dissolved they create a MultiLineString object, which we don't want.
    This is ensures that the geometry is a LineString by finding the correct order and concatenating the linestrings into a new LineString
    """
    stuff_to_dissolve = network_gdf[network_gdf[streamid].isin(upstream_ids)]
    if len(upstream_ids) == 2:
        line1 = list(stuff_to_dissolve.iloc[0].geometry.coords)
        line2 = list(stuff_to_dissolve.iloc[1].geometry.coords)
        line1_start = line1[0]
        line2_end = line2[-1]
        if line1_start == line2_end:
            newline = line2 + line1
            line = sg.LineString(newline)
        else: # We assume that the end of line1 is the beginning of line2
            newline = line1 + line2
            line = sg.LineString(newline)
        return gpd.GeoDataFrame(geometry=[line], crs=network_gdf.crs)
    return stuff_to_dissolve.dissolve()

def merge_streams(upstream_ids: list, network_gdf: gpd.GeoDataFrame, model: bool, streamid: str = 'LINKNO', 
                  dsid: str = 'DSLINKNO', length: str = 'Length'):
    """
    Selects the stream segments that are upstream of the given stream segments, and merges them into a single geodataframe.
    """
    order2_stream = network_gdf[network_gdf[streamid] == upstream_ids[0]]
    dwnstrm_id = order2_stream[dsid].values[0]
    DSCONTAREA = order2_stream['DSContArea'].values[0]
    # A little ugly, but we use an if/else to return the index to use based on which of the upstream_ids is longer
    ids_to_use = [upstream_ids[0],upstream_ids[2 if network_gdf[network_gdf[streamid] == upstream_ids[1]][length].values <= network_gdf[network_gdf[streamid] == upstream_ids[2]][length].values else 1]]
    Length = sum(network_gdf[network_gdf[streamid].isin(ids_to_use)][length].values)
    Magnitude = order2_stream['Magnitude'].values[0]
    strmDrop = sum(network_gdf[network_gdf[streamid].isin(ids_to_use)]['strmDrop'].values)
    WSNO = order2_stream['WSNO'].values[0]
    DOUTEND = order2_stream['DOUTEND'].values[0]
    DOUTSTART = network_gdf[network_gdf[streamid] == ids_to_use[1]]["DOUTSTART"].values[0]
    DSNODEID = order2_stream['DSNODEID'].values[0]

    if DSNODEID != -1:
        print(f"    The stream {upstream_ids[0]} has a DSNODEID other than -1...")

    if not model:
        network_gdf = dissolve_network(network_gdf, upstream_ids,streamid)
    else:
        # Get rid of the shortest stream segment that isn't the order 2!!!
        network_gdf = dissolve_network(network_gdf, ids_to_use,streamid)

    network_gdf[streamid] = upstream_ids[0] 
    network_gdf[dsid] = dwnstrm_id 
    network_gdf["USLINKNO1"] = -1
    network_gdf["USLINKNO2"] = -1
    network_gdf["DSNODEID"] = DSNODEID
    network_gdf["strmOrder"] = 2
    network_gdf[length] = Length
    network_gdf["Magnitude"] = Magnitude
    network_gdf["DSContArea"] = DSCONTAREA
    network_gdf["strmDrop"] = strmDrop
    network_gdf["Slope"] = strmDrop / Length
    network_gdf["StraightL"] = -1
    network_gdf["USContArea"] = -1
    network_gdf["WSNO"] = WSNO
    network_gdf["DOUTEND"] = DOUTEND
    network_gdf["DOUTSTART"] = DOUTSTART
    network_gdf["DOUTMID"] = round((DOUTEND + DOUTSTART) / 2, 2)
    network_gdf['MERGEIDS'] = ','.join(str(num) for num in upstream_ids[1:])
    
    return network_gdf

def prune(linkno: int, gdf: gpd.GeoDataFrame, streamid, dsid) -> gpd.GeoDataFrame:
    """
    Prune the order 1's, adding the missing DSContArea and updating the UPSTRMIDS
    """
    linkgdf = gdf[gdf[streamid] == linkno]
    dsconta = linkgdf['DSContArea'].values[0]
    dwnstrm_linkno = linkgdf[dsid].values[0]
    new_feature = gdf[gdf[streamid] == dwnstrm_linkno].copy()
    new_feature['DSContArea'] += dsconta 
    new_feature['UPSTRMIDS'] = str(linkno)
    new_feature['USLINKNO2'] = -1 # It seems that every uslinko2 in this case is the order 1.
    
    return new_feature

def create_order_set(gdf: gpd.GeoDataFrame, order: int = 1, streamid: str = 'LINKNO', dsid: str = 'DSLINKNO') -> set:
    """
    Returns a set of all stream link numbers that are of the specified order and have a downstream magnitude greater than 2 
    (Indicating that this is a small segment joining a larger river, not an order 1 that will me merged with a top order 2)
    """
    order1_gdf = gdf[gdf['strmOrder'] == order]
    order1_list = []
    for linkno in order1_gdf[streamid].values:
        dslinkno = order1_gdf[order1_gdf[streamid] == linkno][dsid].values[0]
        if dslinkno == -1:
            continue
        if gdf[gdf[streamid] == dslinkno]['Magnitude'].values[0] > 2:
            order1_list.append(linkno)
    return set(order1_list)

def merge_basins(upstream_ids: list, basin_gdf: gpd.GeoDataFrame):
    """
    Dissolves basins based on list of upstream river ids, and returns that feature
    """
    return basin_gdf[basin_gdf["streamID"].isin(upstream_ids)].dissolve()

def fix_0_Length(gdf,basin_gdf,streamid,dsid,length) -> gpd.GeoDataFrame:
    """
    Fix streams that have 0 length.
    General Error Cases:
    1) Feature is coastal w/ no upstream or downstream
        -> Delete the stream and its basin
    2) Feature is bridging a 3-river confluence (Has downstream and upstreams)
        -> Artificially create a basin with 0 area, and force a length on the point of 1 meter
    3) Feature is costal w/ upstreams but no downstream
        -> Force a length on the point of 1 meter
    4) Feature doesn't match any previous case
        -> Raise an error for now
    """
    bad_streams = gdf[gdf[length] == 0]
    case2_gdfs = []
    rivids2drop = []

    for rivid in bad_streams[streamid].values:
        river_length = 0.001
        feat = bad_streams[bad_streams[streamid] == rivid]

        # Case 1
        if feat[dsid].values == -1 and feat['USLINKNO1'].values == -1 and feat['USLINKNO2'].values == -1:
            rivids2drop.append(rivid)

        # Case 2
        elif feat[dsid].values != -1 and feat['USLINKNO1'].values != -1 and feat['USLINKNO2'].values != -1:
            gdf.loc[gdf[streamid] == rivid,length] = river_length #####
            coords = feat.iloc[0].geometry.coords[0]
            box = sg.box(coords[0], coords[1], coords[0], coords[1])
            ###box = sg.Point(coords[0], coords[1]).buffer(0.00001,quad_segs=4, cap_style=3)#####
            
            case2_gdfs.append(gpd.GeoDataFrame({'streamID': [rivid]}, geometry=[box], crs=basin_gdf.crs))

        # Case 3
        elif feat[dsid].values == -1 and feat['USLINKNO1'].values != -1 and feat['USLINKNO2'].values != -1:
            gdf.loc[gdf[streamid] == rivid, length] = river_length #####

        # Case 4
        else:
            raise(f"The stream segement {feat[streamid]} has condtitions we've not yet considered")
        
    basin_gdf = pd.concat([basin_gdf] + case2_gdfs)
    gdf = gdf[~gdf[streamid].isin(rivids2drop)]
    basin_gdf = basin_gdf[~basin_gdf['streamID'].isin(rivids2drop)]
        
    return gdf, basin_gdf


def main(network_gpkg: str, basin_gpkg: str, output_gpkg_name: str = None, model: bool = False, 
         streamid='LINKNO', dsid: str = 'DSLINKNO', length: str = 'Length'):
    """"
    Ensure that shapely >= 2.0.1, otherwise you will get access violations

    Dissolves order 1 streams with their downstream order 2 segments (along with their associated catchments).
    Writes the dissolved networks and catchments to new gpkg files.

    Parameters
    ----------
    network_gpkg : string
        Path to delineation network file
    basin_gpkg : string
        Path to basins or catchments file
    output_gpkg_name : string, optional
        Optional output path for new delineation network. If not specified, the name will be the same as the input + _connectivity.gpkg
        If specified, enter a path such as: "C:\\Users\\user\\Desktop\\output"; "_mapping.gpkg", "_model.gpkg", and "_basins.gpkg" will be added
    model : bool, optional
        If true, the only files created will be the model and basins gpkg's. Otherwise, the mapping gpkg will also be made (default False)
    streamid : string, optional
        Field in network file that corresponds to the unique id of each stream segment
    dsid : string, optional
        Field in network file that corresponds to the unique downstream id of each stream segment
    length : string, optional
        Field in network file that corresponds to the length of each stream segment
    """
    start = time.time()

    gdf = gpd.read_file(network_gpkg)
    print(f"--- Finished at {round((time.time() - start) / 60 ,2)} minutes to read {os.path.basename(network_gpkg)} ---")
    basin_gdf = gpd.read_file(basin_gpkg)
    print(f"--- Finished at {round((time.time() - start) / 60 ,2)} minutes to read {os.path.basename(basin_gpkg)} ---")
    if not model:
        mapping_gdf = gdf.copy()
        mapping_gdf.set_crs(epsg=3857, allow_override=True)

    gdf.set_crs(epsg=3857, allow_override=True)
    basin_gdf.set_crs(epsg=3857, allow_override=True)

    gdf['MERGEIDS'] = np.nan
    
    if 0 in gdf[length].values:
        gdf, basin_gdf = fix_0_Length(gdf, basin_gdf, streamid,dsid,length)
        print(f"--- Segments of length 0 found. Finished at {round((time.time() - start) / 60 ,2)} minutes to fix ---")

    out_json_path = os.path.splitext(network_gpkg)[0] + '_orders.json'
    allorders_dict = create_adjoint_dict(network_gpkg,
                    out_json_path,
                    stream_id_col=streamid,
                    next_down_id_col=dsid,
                    order_col="strmOrder")

    order_2_dict = create_adjoint_dict(network_gpkg,
                        stream_id_col=streamid,
                        next_down_id_col=dsid,
                        order_col="strmOrder",
                        order_filter=2)
    
    print(f"--- Finished at {round((time.time() - start) / 60 ,2)} minutes to create/read {os.path.basename(out_json_path)}. Dissolving... ---")

    toporder2 = {value[-1] for value in list(order_2_dict.values())}

    with Pool() as p:
        merged_streams = p.starmap(merge_streams, [(allorders_dict[str(rivid)], gdf, True) for rivid in toporder2])
        merged_basins = p.starmap(merge_basins,[(allorders_dict[str(rivid)], basin_gdf) for rivid in toporder2])
        if not model:
            merged_mapping = p.starmap(merge_streams, [(allorders_dict[str(rivid)],basin_gdf, False) for rivid in toporder2])
        #order1_features = p.starmap(prune, [(rivid, gdf) for rivid in order1_list])

    print(f"--- Finished dissolving at {round((time.time() - start) / 60 ,2)} minutes ---")

    # Get downstream values of order 1s that must be removed 
    #downstreams = set(pd.concat(order1_features)['LINKNO'].values)
    
    # list all ids that were merged, turn a list of lists into a flat list, remove duplicates by converting to a set (saves ~5 sec)
    all_merged_rivids = set(chain.from_iterable([allorders_dict[str(rivid)] for rivid in toporder2])) #| order1_list | downstreams

    # drop rivids that were merged
    gdf = gdf[~gdf[streamid].isin(all_merged_rivids)]
    basin_gdf = basin_gdf[~basin_gdf["streamID"].isin(all_merged_rivids)]
    if not model:
        mapping_gdf = mapping_gdf[~mapping_gdf[streamid].isin(all_merged_rivids)]

    # concat the merged features
    gdf = pd.concat([gdf, *merged_streams])
    basin_gdf = pd.concat([basin_gdf, *merged_basins])
    #gdf = pd.concat([gdf, *order1_features])
    gdf['reach_id'] = gdf[streamid]
    basin_gdf['reach_id'] = basin_gdf['streamID']
    if not model:
        mapping_gdf = pd.concat([mapping_gdf, *merged_mapping])
    print(f"--- Finished at {round((time.time() - start) / 60 ,2)} minutes to drop and merge ---")
    
    gdf.sort_values('strmOrder', inplace=True)

    if output_gpkg_name is None:
        network_out_path = os.path.splitext(network_gpkg)[0] + '_model1.gpkg'
        basin_out_path = os.path.splitext(network_gpkg)[0] + '_basins1.gpkg'
        if not model:
            mapping_path = os.path.splitext(network_gpkg)[0] + '_mapping.gpkg'
    else:
        network_out_path = output_gpkg_name + '_model.gpkg'
        basin_out_path = output_gpkg_name + '_basins.gpkg'
        if not model:
            mapping_path = output_gpkg_name + '_mapping.gpkg'

    # gdf.to_file(network_out_path, driver="GPKG", index=False)
    # print(f"--- Finished at {round((time.time() - start) / 60 ,2)} minutes to save model gpkg ---")
    # basin_gdf.to_file(basin_out_path, driver="GPKG", index=False)
    # print(f"--- Finished at {round((time.time() - start) / 60 ,2)} minutes to save basins gpkg ---")
    # if not model:
    #     mapping_gdf.to_file(mapping_path, driver="GPKG", index=False)
    #     print(f"--- Finished at {round((time.time() - start) / 60 ,2)} minutes to save mapping gpkg ---")
    files = [network_out_path, basin_out_path]
    gdf_list = [gdf,basin_gdf]
    if not model:
        files.append(mapping_path)
        gdf_list.append(mapping_gdf)
    with Pool(processes=len(files)) as p:
        p.starmap(save_geopackage, zip(files, gdf_list, [start]*len(files)))

def save_geopackage(filename, gpd, start):
    gpd.to_file(filename, driver="GPKG", index=False)
    print(f"--- Finished at {round((time.time() - start) / 60 ,2)} minutes to save {os.path.basename(filename)} ---")