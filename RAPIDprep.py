from multiprocessing import Pool
from collections.abc import Iterable
from itertools import chain
from shapely.ops import voronoi_diagram, unary_union
from shapely import Point, MultiPoint
from pyproj import Geod

import shapely.geometry as sg
import json
import os
import queue
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import pyproj
import netCDF4 as NET
################################################################
#   Disolving functions:
################################################################
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def _make_tree_up(df: pd.DataFrame, order: int = 0, stream_id_col: str = "COMID", next_down_id_col: str = "NextDownID", order_col: str = "order_") -> dict:
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

def _make_tree_down(df: pd.DataFrame, order: int = 0, stream_id_col: str = "COMID", next_down_id_col: str = "NextDownID", 
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

def _trace_tree(tree: dict, search_id: int, cuttoff_n: int = 200) -> list:
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

def _create_adjoint_dict(network_shp, out_file: str = None, stream_id_col: str = "COMID",
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
        tree = _make_tree_up(network_df, order_filter, stream_id_col, next_down_id_col, order_col)
    else:
        tree = _make_tree_down(network_df, order_filter, stream_id_col, next_down_id_col, order_col)
    if order_filter != 0:
        upstream_lists_dict = {str(hydro_id): _trace_tree(tree, hydro_id) for hydro_id in network_df[network_df[order_col] == order_filter][stream_id_col]}
    else:
        upstream_lists_dict = {str(hydro_id): _trace_tree(tree, hydro_id) for hydro_id in network_df[stream_id_col]}
    if out_file is not None:
        if not os.path.exists(out_file):
            with open(out_file, "w") as f:
                json.dump(upstream_lists_dict, f, cls=NpEncoder)
        else:
            print("File already created")
            return upstream_lists_dict # Added to return dictionary anyways
    return upstream_lists_dict

def _dissolve_network(network_gdf, upstream_ids, streamid) -> gpd.GeoDataFrame:
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

def _merge_streams(upstream_ids: list, network_gdf: gpd.GeoDataFrame, model: bool, streamid: str = 'LINKNO', 
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
        network_gdf = _dissolve_network(network_gdf, upstream_ids,streamid)
    else:
        # Get rid of the shortest stream segment that isn't the order 2!!!
        network_gdf = _dissolve_network(network_gdf, ids_to_use,streamid)

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

def _prune(linkno: int, gdf: gpd.GeoDataFrame, streamid, dsid) -> gpd.GeoDataFrame:
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

def _create_order_set(gdf: gpd.GeoDataFrame, order: int = 1, streamid: str = 'LINKNO', dsid: str = 'DSLINKNO') -> set:
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

def _merge_basins(upstream_ids: list, basin_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Dissolves basins based on list of upstream river ids, and returns that feature. Ensure that id is the stream order 2 id
    """
    gdf = basin_gdf[basin_gdf["streamID"].isin(upstream_ids)].dissolve()
    gdf['streamID'] = upstream_ids[0]
    return gdf

def _fix_0_Length(gdf: gpd.GeoDataFrame,basin_gdf: gpd.GeoDataFrame,streamid: str, dsid: str, length: str) -> gpd.GeoDataFrame:
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

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Stream network
    basin_gdf : gpd.GeoDataFrame
        Basins network
    streamid : string
        Field in stream network that corresponds to the unique id of each stream segment
    dsid : string
        Field in stream network that corresponds to the unique downstream id of each stream segment
    length : string
        Field in basins network that corresponds to the unique length of each stream segment
    """
    bad_streams = gdf[gdf[length] == 0]
    case2_gdfs = []
    rivids2drop = []

    river_length = 0.001
    for rivid in bad_streams[streamid].values:
        feat = bad_streams[bad_streams[streamid] == rivid]

        # Case 1
        if feat[dsid].values == -1 and feat['USLINKNO1'].values == -1 and feat['USLINKNO2'].values == -1:
            rivids2drop.append(rivid)

        # Case 2
        elif feat[dsid].values != -1 and feat['USLINKNO1'].values != -1 and feat['USLINKNO2'].values != -1:
            gdf.loc[gdf[streamid] == rivid,length] = river_length #####
            coords = feat.iloc[0].geometry.coords[0]
            ###box = sg.box(coords[0], coords[1], coords[0], coords[1])
            box = sg.Point(coords[0], coords[1]).buffer(0.00001,quad_segs=4, cap_style=3)#####
            
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

def _save_geopackage(filename: str, gpd: gpd.GeoDataFrame, start: float, EPSG:int) -> None:
    """
    Function for multiprocessing. Executes the geopandas 'to_file' method on the gpd, saving it at the location filename with the projection of EPSG, and displays a timestamp
    """
    gpd.to_file(filename, driver="GPKG", index=False, crs=EPSG)
    print(f"--- Finished at {round((time.time() - start) / 60 ,2)} minutes to save {os.path.basename(filename)} ---")

def _read_geopackage(filename: str, start: float) -> gpd.GeoDataFrame:
    """
    Function for multiprocessing. Executes the geopandas 'read_file' method on the filename, and displays timestamp
    """
    pkcg =  gpd.read_file(filename)
    print(f"--- Finished at {round((time.time() - start) / 60 ,2)} minutes to read {os.path.basename(filename)} ---")
    return pkcg

def _main_dissolve(network_gpkg: str, basin_gpkg: str, output_gpkg_name: str = None, model: bool = False, 
         streamid='LINKNO', dsid: str = 'DSLINKNO', length: str = 'Length', EPSG=3857) -> None:
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
    EPSG : int, optional
        The projection that will be used to read and write the gpkgs.
    """
    start = time.time()

    with Pool(processes=2) as p:
        gdf, basin_gdf = p.starmap(_read_geopackage, zip([network_gpkg, basin_gpkg], [start]*2))

    #EPSG = pyproj.CRS.from_epsg(EPSG)
    if not model:
        mapping_gdf = gdf.copy()
        mapping_gdf = mapping_gdf.to_crs(EPSG)

    gdf = gdf.to_crs(epsg=EPSG)
    basin_gdf = basin_gdf.to_crs(EPSG)

    gdf['MERGEIDS'] = np.nan
    
    if 0 in gdf[length].values:
        gdf, basin_gdf = _fix_0_Length(gdf, basin_gdf, streamid, dsid, length)
        print(f"--- Segments of length 0 found. Finished at {round((time.time() - start) / 60 ,2)} minutes to fix ---")

    out_json_path = os.path.splitext(network_gpkg)[0] + '_orders.json'
    allorders_dict = _create_adjoint_dict(network_gpkg,
                    out_json_path,
                    stream_id_col=streamid,
                    next_down_id_col=dsid,
                    order_col="strmOrder")

    order_2_dict = _create_adjoint_dict(network_gpkg,
                        stream_id_col=streamid,
                        next_down_id_col=dsid,
                        order_col="strmOrder",
                        order_filter=2)
    
    print(f"--- Finished at {round((time.time() - start) / 60 ,2)} minutes to create/read {os.path.basename(out_json_path)}. Dissolving... ---")

    toporder2 = {value[-1] for value in list(order_2_dict.values())}

    with Pool() as p:
        merged_streams = p.starmap(_merge_streams, [(allorders_dict[str(rivid)], gdf, True) for rivid in toporder2])
        merged_basins = p.starmap(_merge_basins,[(allorders_dict[str(rivid)], basin_gdf) for rivid in toporder2])
        if not model:
            merged_mapping = p.starmap(_merge_streams, [(allorders_dict[str(rivid)],basin_gdf, False) for rivid in toporder2])
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
        network_out_path = os.path.splitext(network_gpkg)[0] + '_model.gpkg'
        basin_out_path = os.path.splitext(network_gpkg)[0] + '_basins.gpkg'
        if not model:
            mapping_path = os.path.splitext(network_gpkg)[0] + '_mapping.gpkg'
    else:
        network_out_path = output_gpkg_name + '_model.gpkg'
        basin_out_path = output_gpkg_name + '_basins.gpkg'
        if not model:
            mapping_path = output_gpkg_name + '_mapping.gpkg'

    files = [network_out_path, basin_out_path]
    gdf_list = [gdf,basin_gdf]
    if not model:
        files.append(mapping_path)
        gdf_list.append(mapping_gdf)

    with Pool(processes=len(files)) as p:
        p.starmap(_save_geopackage, zip(files, gdf_list, [start]*len(files), [EPSG]*len(files)))

################################################################
#   RAPID Preprocessing functions
################################################################
        
def _CreateComidLatLonZ(network,out_dir,id_field,start):
    network.sort_values(id_field, inplace=True)
    gdf = gpd.GeoDataFrame.copy(network)
    gdf['lat'] = network.geometry.apply(lambda geom: geom.xy[1][0])
    gdf['lon'] = network.geometry.apply(lambda geom: geom.xy[0][0])
    data = {id_field: network[id_field].values,
            "lat": gdf['lat'].values,
            "lon": gdf['lon'].values,
            "z": 0}
    
    pd.DataFrame(data).to_csv(os.path.join(out_dir, "comid_lat_lon_z.csv"), index=False, header=True)
    print(f"\tCreated comid_lat_lon_z.csv at {round((time.time() - start) / 60 ,2)} minutes.")

def _CreateRivBasId(network, out_dir,downstream_field,id_field,start):
    network.sort_values([downstream_field, id_field], inplace=True, ascending=[False,False])
    network[id_field].to_csv(os.path.join(out_dir, "riv_bas_id.csv"), index=False, header=False)   

    print(f"\tCreated riv_bas_id.csv at {round((time.time() - start) / 60 ,2)}")

def _CalculateMuskingum(network,out_dir,k,x, id_field,start):
    network.sort_values(id_field, inplace=True)
    network = network.to_crs(epsg=4326) # Calculation of geodesic lengths must occur in epsg 4326
    network["LENGTH_GEO"] = network.geometry.apply(_calculate_geodesic_length)
    network["Musk_kfac"] = network["LENGTH_GEO"] * 3600
    network["Musk_k"] = network["Musk_kfac"] * k
    network["Musk_x"] = x * 0.1
    network.to_file("C:\\Users\\lrr43\Desktop\\Lab\\GEOGLOWSData\\RAPID\\PythonV\\Test.gpkg")

    network["Musk_kfac"].to_csv(os.path.join(out_dir, "kfac.csv"), index=False, header=False)
    network["Musk_k"].to_csv(os.path.join(out_dir, "k.csv"), index=False, header=False)
    network["Musk_x"].to_csv(os.path.join(out_dir, "x.csv"), index=False, header=False)
    print(f"\tCreated muskingum parameters at {round((time.time() - start) / 60 ,2)} minutes.")

def _calculate_geodesic_length(line):
    geod = Geod(ellps='WGS84')
    length = geod.geometry_length(line) / 1000 # To convert to km

    # This is for the outliers that have 0 length
    if length < 0.00000000001:
        length = 0.001
    return length

def _CreateRapidConnect(network, out_dir, id_field, downstream_field,start):
    network.sort_values(id_field, inplace=True)
    list_all = []
    max_count_Upstream = 0

    for hydroid in network[id_field].values:
        # find the HydroID of the upstreams
        list_upstreamID = network.loc[network[downstream_field] == hydroid, id_field].values
        # count the total number of the upstreams
        count_upstream = len(list_upstreamID)
        if count_upstream > max_count_Upstream:
            max_count_Upstream = count_upstream
        nextDownID = network.loc[network[id_field] == hydroid, downstream_field].values[0]

        # append the list of Stream HydroID, NextDownID, Count of Upstream ID, and  HydroID of each Upstream into a larger list
        list_all.append(np.concatenate([np.array([hydroid, nextDownID, count_upstream]), list_upstreamID]))

    with open(os.path.join(out_dir, "rapid_connect.csv"), 'w') as f:
        for row in list_all:
            out = np.concatenate([row, np.array([0 for i in range(max_count_Upstream - row[2])])])
            for i, item in enumerate(out):
                if i + 1 != len(out):
                    f.write(f"{item},")
                else: f.write(f"{item}")
            f.write('\n')

    print(f"\tCreated rapid_connect.csv at {round((time.time() - start) / 60 ,2)} minutes.")

def _CreateWeightTable(out_dir, basins_file, nc_file, basin_id,start):
    print(f"\tBeginning Weight Table Creation at {round((time.time() - start) / 60 ,2)} minutes.")
    basins_gdf = gpd.read_file(basins_file)
    if not basin_id in basins_gdf.columns:
        raise ValueError(f"The id field {basin_id} is not in the basins file in {out_dir}")
    basins_gdf = basins_gdf.to_crs('EPSG:4326')

    # Obtain catchment extent
    extent = basins_gdf.total_bounds

    data_nc = NET.Dataset(nc_file)

    # Obtain geographic coordinates
    variables_list = data_nc.variables.keys()
    lat_var = 'lat'
    if 'latitude' in variables_list:
        lat_var = 'latitude'
    lon_var = 'lon'
    if 'longitude' in variables_list:
        lon_var = 'longitude'
    lon = (data_nc.variables[lon_var][:] + 180) % 360 - 180 # convert [0, 360] to [-180, 180]
    lat = data_nc.variables[lat_var][:]

    data_nc.close()

    # Create Thiessen polygons based on the points within the extent
    print("\tCreating Thiessen polygons")
    buffer = 2 * max(abs(lat[0]-lat[1]),abs(lon[0] - lon[1]))
    # Extract the lat and lon within buffered extent (buffer with 2* interval degree)
    lat0 = lat[(lat >= (extent[1] - buffer)) & (lat <= (extent[3] + buffer))]
    lon0 = lon[(lon >= (extent[0] - buffer)) & (lon <= (extent[2] + buffer))]

    # Create a list of geographic coordinate pairs
    pointGeometryList = [Point(lon, lat) for lat in lat0 for lon in lon0]

    # Create Thiessen polygon based on the point feature
    regions = voronoi_diagram(MultiPoint(pointGeometryList))

    lon_list = []
    lat_list = []

    for point in pointGeometryList:
        lon_list.append(point.x)
        lat_list.append(point.y)

    polygons_gdf = gpd.GeoDataFrame(geometry=[region for region in regions.geoms], crs=4326)
    #polygons_gdf = polygons_gdf.to_crs('EPSG:3857')
    polygons_gdf['POINT_X'] = polygons_gdf.geometry.centroid.x
    polygons_gdf['POINT_Y'] = polygons_gdf.geometry.centroid.y
    #polygons_gdf.to_crs(epsg=4326, inplace=True)
    #polygons_gdf.to_file(os.path.join(out_dir, 'test.gpkg'), driver="GPKG")

    print("Intersecting Thiessen polygons with catchment...")
    intersect = gpd.overlay(basins_gdf, polygons_gdf, how='intersection')

    print("Calculating geodesic areas...")
    intersect['AREA_GEO'] = intersect['geometry'].to_crs('EPSG:3857').area 
    #intersect.to_file(os.path.join(out_dir, 'intersect.gpkg'), driver="GPKG")

    area_arr = pd.DataFrame(data={
        basin_id: intersect[basin_id].values,
        'POINT_X': intersect['POINT_X'].values,
        'POINT_Y': intersect['POINT_Y'].values,
        'AREA_GEO': intersect['AREA_GEO']
    })
    area_arr.sort_values([basin_id, 'AREA_GEO'], inplace=True, ascending=[True,False])
    connectivity_table = pd.read_csv(os.path.join(out_dir,'rapid_connect.csv'), header=None)
    streamID_unique_list = connectivity_table.iloc[:,0].astype(int).unique().tolist()

    #   If point not in array append dummy data for one point of data
    lon_dummy = area_arr['POINT_X'].iloc[0]
    lat_dummy = area_arr['POINT_Y'].iloc[0]
    try:
        index_lon_dummy = int(np.where(lon == lon_dummy)[0])
    except TypeError as _:
        index_lon_dummy = int((np.abs(lon-lon_dummy)).argmin())
        pass

    try:
        index_lat_dummy= int(np.where(lat == lat_dummy)[0])
    except TypeError as _:
        index_lat_dummy = int((np.abs(lat-lat_dummy)).argmin())
        pass

    with open(os.path.join(out_dir, 'weight_table_py2.csv'), 'w') as csvfile:
        csvfile.write(f'{basin_id},area_sqm,lon_index,lat_index,npoints,lon,lat\n')

        for streamID_unique in streamID_unique_list:
                ind_points = np.where(area_arr[basin_id]==streamID_unique)[0]
                num_ind_points = len(ind_points)
    
                if num_ind_points <= 0:
                    # if point not in array, append dummy data for one point of data
                    # streamID, area_sqm, lon_index, lat_index, npoints
                    csvfile.write(f'{streamID_unique},0,{index_lon_dummy},{index_lat_dummy},1,{lon_dummy},{lat_dummy}\n')

                else:
                    for ind_point in ind_points:
                        area_geo_each = float(area_arr['AREA_GEO'].iloc[ind_point])
                        lon_each = area_arr['POINT_X'].iloc[ind_point]
                        lat_each = area_arr['POINT_Y'].iloc[ind_point]
                        
                        try:
                            index_lon_each = int(np.where(lon == lon_each)[0])
                        except:
                            index_lon_each = int((np.abs(lon-lon_each)).argmin())
                            pass

                        try:
                            index_lat_each = int(np.where(lat == lat_each)[0])
                        except:
                            index_lat_each = int((np.abs(lat-lat_each)).argmin())
                        csvfile.write(f'{streamID_unique},{area_geo_each},{index_lon_each},{index_lat_each},{num_ind_points},{lon_each},{lat_each}\n')
    
    
    print(f"\tCreated weight table at {round((time.time() - start) / 60 ,2)} minutes.")

def main(main_dir: str, nc_file: str, id_field: str = 'LINKNO', downstream_field: str = 'DSLINKNO', basin_id: str = 'streamID', 
         k: float = 0.35, x: float = 3.0, overwrite: bool = False):
    """
    Fix stream network, dissolve network and cathcment gpkgs, create RAPID preprocessing files

    Parameters
    ----------
    main_dir : string
        The overarching directory, containing subfolders, which each contain two geopackages, one with 'model' in its name and another with 'basin' in its name
    nc_file : string
        Path to the nc file.
    id_field : string, optional
        Field in network file that corresponds to the unique id of each stream segment
    downstream_field : string, optional
        Field in network file that corresponds to the unique downstream id of each stream segment
    basin_id : string, optional
        Field in basins file that corresponds to the unique id of each catchment

    overwrite : bool, optional
        If set to True, will overwrite existing files. The default is False.

    """
    start = time.time()
    job_list = []
    for file in os.listdir(main_dir):
        d = os.path.join(main_dir, file)
        if os.path.isdir(d):
            job_list.append(d)
    for folder in job_list:
        print(f'In {folder}')
        files = os.listdir(folder)
        basins_file = None
        stream_ntwk_file = None
        for file in files:
            if 'basin' in file:
                basins_file = os.path.join(folder,file)
            if 'model' in file:
                stream_ntwk_file = os.path.join(folder,file)
        if basins_file is not None and stream_ntwk_file is not None:
            out_dir = folder
            network = gpd.read_file(stream_ntwk_file)
            network.set_crs(epsg=3857, allow_override=True)

            # Some checks: 
            if not id_field in network.columns:
                raise ValueError(f"The id field {id_field} is not in the network file in {out_dir}")
            if not downstream_field in network.columns:
                raise ValueError(f"The downstream field {downstream_field} is not in the network file in {out_dir}")
            
            # Run main functions
            _CreateComidLatLonZ(network, out_dir, id_field,start)
            _CreateRivBasId(network, out_dir, downstream_field,id_field,start)
            _CalculateMuskingum(network,out_dir,k,x, id_field,start)
            _CreateRapidConnect(network, out_dir, id_field, downstream_field,start)
            _CreateWeightTable(out_dir, basins_file, nc_file, basin_id,start)
            print('\tFinished successfully!\n')
            return