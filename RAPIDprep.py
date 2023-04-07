import datetime
import glob
import json
import logging
import os
import queue
import warnings
from collections.abc import Iterable
from itertools import chain
from multiprocessing import Pool

import geopandas as gpd
import netCDF4 as nc
import numpy as np
import pandas as pd
import shapely.geometry as sg
from pyproj import Geod
from shapely import Point, MultiPoint
from shapely.ops import voronoi_diagram

hydrobasin_cache = {
    1020000010: 11, 1020011530: 12, 1020018110: 13, 1020021940: 14, 1020027430: 15, 1020034170: 16,
    1020035180: 17, 1020040190: 18, 2020033490: 21, 2020041390: 22, 2020057170: 23, 2020065840: 24,
    2020071190: 25, 2020000010: 26, 2020003440: 27, 2020018240: 28, 2020024230: 29, 3020009320: 31,
    3020024310: 32, 3020000010: 33, 3020003790: 34, 3020005240: 35, 3020008670: 36, 4020034510: 41,
    4020050210: 42, 4020050220: 43, 4020050290: 44, 4020050470: 45, 4020000010: 46, 4020006940: 47,
    4020015090: 48, 4020024190: 49, 5020054880: 51, 5020055870: 52, 5020082270: 53, 5020000010: 54,
    5020015660: 55, 5020037270: 56, 5020049720: 57, 6020017370: 61, 6020021870: 62, 6020029280: 63,
    6020000010: 64, 6020006540: 65, 6020008320: 66, 6020014330: 67, 7020038340: 71, 7020046750: 72,
    7020047840: 73, 7020065090: 74, 7020000010: 75, 7020014250: 76, 7020021430: 77, 7020024600: 78,
    8020022890: 81, 8020032840: 82, 8020044560: 83, 8020000010: 84, 8020008900: 85, 8020010700: 86,
    8020020760: 87, 9020000010: 91
}

# set up logging
logger = logging.getLogger(__name__)


################################################################
#   Dissolving functions:
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


def _make_tree_up(df: pd.DataFrame, order: int = 0, stream_id_col: str = "COMID", next_down_id_col: str = "NextDownID",
                  order_col: str = "order_") -> dict:
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


def _make_tree_down(df: pd.DataFrame, order: int = 0, stream_id_col: str = "COMID",
                    next_down_id_col: str = "NextDownID",
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


def _create_adjoint_dict(network_gdf: gpd.GeoDataFrame, out_file: str = None, stream_id_col: str = "COMID",
                         next_down_id_col: str = "NextDownID", order_col: str = "order_", trace_up: bool = True,
                         order_filter: int = 0) -> dict:
    """
    Creates a dictionary where each unique id in a stream network is assigned a list of all ids upstream or downstream
    of that stream, as specified. By default is designed to trace upstream on GEOGloWS Delineation Catchment shapefiles,
    but can be customized for other files with column name parameters, customized to trace down, or filtered by stream
    order. If filtered by stream order, the dictionary will only contain ids of the given stream order, with the
    upstream or downstream ids for the other streams in the chain that share that stream order.
    Args:
        network_gdf: The stream network. This file
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
    # Added to forgo computation on pre-created json files
    if out_file is not None and os.path.exists(out_file):
        with open(out_file, 'r') as f:
            upstream_lists_dict = json.load(f)
        return upstream_lists_dict

    columns_to_search = [stream_id_col, next_down_id_col]
    if order_filter != 0:
        columns_to_search.append(order_col)
    for col in columns_to_search:
        if col not in network_gdf.columns:
            logger.info(f"Column {col} not present")
            return {}
    if trace_up:
        tree = _make_tree_up(network_gdf, order_filter, stream_id_col, next_down_id_col, order_col)
    else:
        tree = _make_tree_down(network_gdf, order_filter, stream_id_col, next_down_id_col, order_col)
    if order_filter != 0:
        upstream_lists_dict = {str(hydro_id): _trace_tree(tree, hydro_id) for hydro_id in
                               network_gdf[network_gdf[order_col] == order_filter][stream_id_col]}
    else:
        upstream_lists_dict = {str(hydro_id): _trace_tree(tree, hydro_id) for hydro_id in network_gdf[stream_id_col]}
    if out_file is not None:
        if not os.path.exists(out_file):
            with open(out_file, "w") as f:
                json.dump(upstream_lists_dict, f, cls=NpEncoder)
        else:
            logger.info("File already created")
            return upstream_lists_dict  # Added to return dictionary anyways
    return upstream_lists_dict


def _merge_streams(upstream_ids: list, network_gdf: gpd.GeoDataFrame, make_model_version: bool,
                   streamid: str = 'LINKNO', dsid: str = 'DSLINKNO', length_col: str = 'Length') -> gpd.GeoDataFrame:
    """
    Selects the stream segments that are upstream of the given stream segments, and merges them into a single geodataframe.
    """
    if make_model_version:
        # A little ugly, but we use an if/else to return the index to use based on which of the upstream_ids is longer
        upstream_ids = [upstream_ids[0], upstream_ids[
            2 if network_gdf[network_gdf[streamid] == upstream_ids[1]][length_col].values <=
                 network_gdf[network_gdf[streamid] == upstream_ids[2]][length_col].values else 1]]

    dissolved_feature = network_gdf[network_gdf[streamid].isin(upstream_ids)]
    if len(upstream_ids) == 2:
        # This is for when we merge only two upstream streams, and this is where only using GeoPandas dissolve may
        # create MultiLineString objects (because geoms in wrong order?)
        line1 = list(dissolved_feature.iloc[0].geometry.coords)
        line2 = list(dissolved_feature.iloc[1].geometry.coords)
        line1_start = line1[0]
        line2_end = line2[-1]
        if line1_start == line2_end:
            newline = line2 + line1
            line = sg.LineString(newline)
        else:  # We assume that the end of line1 is the beginning of line2
            newline = line1 + line2
            line = sg.LineString(newline)
        dissolved_feature = gpd.GeoDataFrame(geometry=[line], crs=network_gdf.crs)
    else:
        dissolved_feature = dissolved_feature.dissolve()

    order2_stream = network_gdf[network_gdf[streamid] == upstream_ids[0]]
    downstream_id = order2_stream[dsid].values[0]
    dscontarea = order2_stream['DSContArea'].values[0]
    length = sum(network_gdf[network_gdf[streamid].isin(upstream_ids)][length_col].values)
    magnitude = order2_stream['Magnitude'].values[0]
    strm_drop = sum(network_gdf[network_gdf[streamid].isin(upstream_ids)]['strmDrop'].values)
    wsno = order2_stream['WSNO'].values[0]
    doutend = order2_stream['DOUTEND'].values[0]
    doutstart = network_gdf[network_gdf[streamid] == upstream_ids[1]]["DOUTSTART"].values[0]
    dsnodeid = order2_stream['DSNODEID'].values[0]
    upstream_comids = upstream_ids[1:]

    if dsnodeid != -1:
        logging.warning(f"  The stream {upstream_ids[0]} has a DSNODEID other than -1...")

    dissolved_feature[streamid] = upstream_ids[0]
    dissolved_feature[dsid] = downstream_id
    dissolved_feature["USLINKNO1"] = -1
    dissolved_feature["USLINKNO2"] = -1
    dissolved_feature["DSNODEID"] = dsnodeid
    dissolved_feature["strmOrder"] = 2
    dissolved_feature[length_col] = length
    dissolved_feature["Magnitude"] = magnitude
    dissolved_feature["DSContArea"] = dscontarea
    dissolved_feature["strmDrop"] = strm_drop
    dissolved_feature["Slope"] = strm_drop / length
    dissolved_feature["StraightL"] = -1
    dissolved_feature["USContArea"] = -1
    dissolved_feature["WSNO"] = wsno
    dissolved_feature["DOUTEND"] = doutend
    dissolved_feature["DOUTSTART"] = doutstart
    dissolved_feature["DOUTMID"] = round((doutend + doutstart) / 2, 2)
    dissolved_feature['MERGEIDS'] = ','.join(str(num) for num in upstream_comids)

    return dissolved_feature


def _merge_basins(basin_gdf: gpd.GeoDataFrame, upstream_ids: list) -> gpd.GeoDataFrame:
    """
    Dissolves basins based on list of upstream river ids, and returns that feature.
    """
    gdf = basin_gdf[basin_gdf["streamID"].isin(upstream_ids)].dissolve()
    gdf['streamID'] = upstream_ids[0]  # the first ID should be the most downstream ID
    return gdf


def identify_0_length_fixes(gdf: gpd.GeoDataFrame, stream_id_col: str, ds_id_col: str, length_col: str) -> pd.DataFrame:
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
    stream_id_col : string
        Field in stream network that corresponds to the unique id of each stream segment
    ds_id_col : string
        Field in stream network that corresponds to the unique downstream id of each stream segment
    length_col : string
        Field in basins network that corresponds to the unique length of each stream segment
    """
    case1_ids = []
    case2_ids = []
    case2_xs = []
    case2_ys = []
    case3_ids = []
    case4_ids = []

    for rivid in gdf[gdf[length_col] == 0][stream_id_col].values:
        feat = gdf[gdf[stream_id_col] == rivid]

        # Case 1
        if feat[ds_id_col].values == -1 and feat['USLINKNO1'].values == -1 and feat['USLINKNO2'].values == -1:
            case1_ids.append(rivid)

        # Case 2
        elif feat[ds_id_col].values != -1 and feat['USLINKNO1'].values != -1 and feat['USLINKNO2'].values != -1:
            case2_ids.append(rivid)
            case2_xs.append(feat['geometry'].values[0].coords[0][0])
            case2_ys.append(feat['geometry'].values[0].coords[0][1])

        # Case 3
        elif feat[ds_id_col].values == -1 and feat['USLINKNO1'].values != -1 and feat['USLINKNO2'].values != -1:
            case3_ids.append(rivid)

        # Case 4
        else:
            logging.warning(f"The stream segement {feat[stream_id_col]} has condtitions we've not yet considered")
            case4_ids.append(rivid)

    # variable length lists with np.nan to make them the same length
    longest_list = max([len(case1_ids), len(case2_ids), len(case3_ids), len(case4_ids), len(case2_xs), len(case2_ys)])
    case1_ids = case1_ids + [np.nan] * (longest_list - len(case1_ids))
    case2_ids = case2_ids + [np.nan] * (longest_list - len(case2_ids))
    case2_xs = case2_xs + [np.nan] * (longest_list - len(case2_xs))
    case2_ys = case2_ys + [np.nan] * (longest_list - len(case2_ys))
    case3_ids = case3_ids + [np.nan] * (longest_list - len(case3_ids))
    case4_ids = case4_ids + [np.nan] * (longest_list - len(case4_ids))

    return pd.DataFrame({
        'case1': case1_ids,
        'case2': case2_ids,
        'case3': case3_ids,
        'case4': case4_ids,
        'case2_x': case2_xs,
        'case2_y': case2_ys
    })


def apply_0_length_stream_fixes(streams_gdf: gpd.GeoDataFrame, zero_length_df: pd.DataFrame,
                                stream_id_col: str, length_col: str, river_length: float = 0.001) -> gpd.GeoDataFrame:
    """
    Apply fixes to streams that have 0 length.

    Args:
        streams_gdf:
        zero_length_df:
        stream_id_col:
        length_col:
        river_length:

    Returns:

    """
    corrected_streams = streams_gdf.copy()

    # Case 1 - Coastal w/ no upstream or downstream - Delete the stream and its basin
    corrected_streams = corrected_streams[~corrected_streams[stream_id_col].isin(zero_length_df['case1'])]

    # Case 2 - Allow 3-river confluence - Create a basin with small non-zero area, assign small non-zero length
    corrected_streams.loc[corrected_streams[stream_id_col].isin(zero_length_df['case2']), length_col] = river_length

    # Case 3 - Coastal w/ upstreams but no downstream - Assign small non-zero length
    corrected_streams.loc[corrected_streams[stream_id_col].isin(zero_length_df['case3']), length_col] = river_length

    return corrected_streams


def apply_0_length_basin_fixes(basins_gdf: gpd.GeoDataFrame, zero_length_df: pd.DataFrame,
                               stream_id_col: str, buffer_size: float = 1) -> gpd.GeoDataFrame:
    """
    Apply fixes to streams that have 0 length.

    Args:
        basins_gdf:
        zero_length_df:
        stream_id_col:
        buffer_size:

    Returns:

    """
    corrected_basins = basins_gdf.copy()

    # Case 1 - Coastal w/ no upstream or downstream - Delete the stream and its basin
    corrected_basins = corrected_basins[~corrected_basins[stream_id_col].isin(zero_length_df['case1'])]

    # Case 2 - Allow 3-river confluence - Create a basin with small non-zero area, assign small non-zero length
    boxes = (
        zero_length_df[['case2', 'case2_x', 'case2_y']]
        .apply(lambda x: sg.box(
            x.case2_x - buffer_size,
            x.case2_y - buffer_size,
            x.case2_x + buffer_size,
            x.case2_y + buffer_size
        ), axis=1)
    )
    corrected_basins = pd.concat([
        corrected_basins,
        gpd.GeoDataFrame({'geometry': boxes, stream_id_col: zero_length_df['case2']})
    ])

    # Case 3 - Coastal w/ upstreams but no downstream - Assign small non-zero length
    # NO FIXES APPLIED TO BASINS FOR CASE 3 - ALREADY HAVE

    return corrected_basins


def dissolve_streams(streams_gpkg: str, save_dir: str,
                     stream_id_col='LINKNO', ds_id_col: str = 'DSLINKNO', length_col: str = 'Length',
                     mp_dissolve: bool = True, n_processes: int or None = None) -> gpd.GeoDataFrame:
    """"
    Ensure that shapely >= 2.0.1, otherwise you will get access violations

    Dissolves order 1 streams with their downstream order 2 segments (along with their associated catchments).
    Writes the dissolved networks and catchments to new gpkg files.

    Args:
        streams_gpkg (str): Path to delineation network file
        save_dir (str): Path to directory where dissolved network and catchments will be saved
        stream_id_col (str, optional): Field in network file that corresponds to the unique id of each stream segment
        ds_id_col (str, optional): Field in network file that corresponds to the unique downstream id of each stream segment
        length_col (str, optional): Field in network file that corresponds to the length of each stream segment
        mp_dissolve (bool, optional): Whether to use multiprocessing to dissolve streams
        n_processes (int, optional): Number of processes to use for parallel processing

    Returns:
        gpd.GeoDataFrame: Dissolved streams
    """
    logger.info(datetime.datetime.now().strftime("%H:%M:%S"))
    streams_gdf = gpd.read_file(streams_gpkg)
    logger.info(f"Finished reading {streams_gpkg}")
    logger.info(datetime.datetime.now().strftime("%H:%M:%S"))
    logger.info(f'Total features {streams_gdf.shape[0]}')

    streams_gdf['MERGEIDS'] = np.nan

    if 0 in streams_gdf[length_col].values:
        logger.info("Fixing length 0 segments")
        zero_length_fixes_df = identify_0_length_fixes(streams_gdf, stream_id_col, ds_id_col, length_col)
        zero_length_fixes_df.to_csv(os.path.join(save_dir, 'zero_length_fixes.csv'), index=False)
        streams_gdf = apply_0_length_stream_fixes(streams_gdf, zero_length_fixes_df, stream_id_col, length_col)
        zero_length_fixes_df = None

    adjoint_dict = _create_adjoint_dict(streams_gdf, stream_id_col=stream_id_col, next_down_id_col=ds_id_col,
                                        order_col="strmOrder")

    adjoint_order_2_dict = _create_adjoint_dict(streams_gdf, stream_id_col=stream_id_col, next_down_id_col=ds_id_col,
                                                order_col="strmOrder", order_filter=2)

    # list all ids that were merged, turn a list of lists into a flat list, remove duplicates by converting to a set
    top_order_2s = {str(value[-1]) for value in list(adjoint_order_2_dict.values())}
    adjoint_order_2_dict = {key: adjoint_dict[key] for key in top_order_2s}
    all_merged_rivids = set(chain.from_iterable([adjoint_dict[rivid] for rivid in top_order_2s]))

    with open(os.path.join(save_dir, 'adjoint_tree.json'), 'w') as f:
        json.dump(adjoint_dict, f)
        adjoint_dict = None

    with open(os.path.join(save_dir, 'adjoint_dissolves_tree.json'), 'w') as f:
        json.dump(adjoint_order_2_dict, f)

    with Pool(n_processes) as p:
        # Process each chunk of basin_gdf separately
        logger.info("Merging streams (model)")
        merged_streams_model = p.starmap(_merge_streams, [
            (adjoint_order_2_dict[str(rivid)], streams_gdf, True) for rivid in top_order_2s])
        logger.info("Merging streams (mapping)")
        merged_streams_mapping = p.starmap(_merge_streams, [
            (adjoint_order_2_dict[str(rivid)], streams_gdf, False) for rivid in top_order_2s])

    # concat the merged features
    logger.info("Concatenating dissolved features")
    streams_gdf = streams_gdf[~streams_gdf[stream_id_col].isin(all_merged_rivids)]
    mapping_gdf = pd.concat([streams_gdf, pd.concat(merged_streams_mapping)])
    merged_streams_mapping = None
    streams_gdf = pd.concat([streams_gdf, pd.concat(merged_streams_model)])
    merged_streams_model = None

    # Sort streams for csvs
    streams_gdf.sort_values('strmOrder', inplace=True)
    mapping_gdf.sort_values('strmOrder', inplace=True)

    # Save the files
    logger.info('Writing geopackages')
    streams_gdf.to_file(os.path.join(save_dir, os.path.basename(os.path.splitext(streams_gpkg)[0]) + '_model.gpkg'))
    mapping_gdf.to_file(os.path.join(save_dir, os.path.basename(os.path.splitext(streams_gpkg)[0]) + '_vis.gpkg'))

    return streams_gdf


def dissolve_basins(basins_gpkg: str, save_dir: str, mp_dissolve: bool = True,
                    stream_id_col='LINKNO', n_process: int or None = None) -> gpd.GeoDataFrame:
    """"
    Ensure that shapely >= 2.0.1, otherwise you will get access violations

    Dissolves order 1 streams with their downstream order 2 segments (along with their associated catchments).
    Writes the dissolved networks and catchments to new gpkg files.

    Args:
        basins_gpkg (str): Path to delineation network file
        save_dir (str): Path to directory where dissolved network and catchments will be saved
        mp_dissolve (bool, optional): Whether to use multiprocessing to dissolve the network
        stream_id_col (str, optional): Field in network file that corresponds to the unique id of each stream segment
        n_process (int, optional): Number of processes to use for parallel processing

    Returns:
        gpd.GeoDataFrame: Dissolved network
    """
    logger.info(datetime.datetime.now().strftime("%H:%M:%S"))
    basins_gdf = gpd.read_file(basins_gpkg)
    logger.info(f"Finished reading {basins_gpkg}")
    logger.info(datetime.datetime.now().strftime("%H:%M:%S"))
    logger.info(basins_gdf.shape[0])

    with open(os.path.join(save_dir, 'adjoint_dissolves_tree.json'), 'r') as f:
        adjoint_dict = json.load(f)
    all_merged_basins = set(chain.from_iterable([adjoint_dict[rivid] for rivid in adjoint_dict.keys()]))

    # Check for 0 length segments
    zero_length_fixes_df_path = os.path.join(save_dir, 'zero_length_fixes.csv')
    if os.path.exists(zero_length_fixes_df_path):
        zero_length_fixes_df = pd.read_csv(os.path.join(save_dir, 'zero_length_fixes.csv'))
        basins_gdf = apply_0_length_basin_fixes(basins_gdf, zero_length_fixes_df,
                                                stream_id_col=stream_id_col, buffer_size=.001)

    if mp_dissolve:
        with Pool(n_process) as p:
            # Process each chunk of basin_gdf separately
            logger.info("Merging basins")
            merged_basins = p.starmap(_merge_basins,
                                      [(basins_gdf, adjoint_dict[rivid]) for rivid in adjoint_dict.keys()])

        logger.info("Concatenating dissolved features")
        # drop the basins that were merged
        basins_gdf = basins_gdf[~basins_gdf[stream_id_col].isin(all_merged_basins)]
        merged_basins = pd.concat(merged_basins)
        basins_gdf = pd.concat([basins_gdf, merged_basins])
    else:
        for rivid in adjoint_dict.keys():
            merged_basins = _merge_basins(basins_gdf, adjoint_dict[rivid])
            basins_gdf = pd.concat([
                basins_gdf[~basins_gdf[stream_id_col].isin(adjoint_dict[rivid])],
                merged_basins
            ])
    merged_basins = None

    # Save the files
    logger.info('Writing modeled basins geopackage')
    basins_gdf.to_file(os.path.join(save_dir, os.path.basename(os.path.splitext(basins_gpkg)[0]) + '_model.gpkg'))

    return basins_gdf


################################################################
#   RAPID Preprocessing functions
################################################################
def create_comid_lat_lon_z(streams_gdf: gpd.GeoDataFrame, out_dir: str, id_field: str) -> None:
    """
    Assumes that geometry of the network are shapely LineStrings.
    If there are MultiLineStrings than something has gone wrong in the dissolving step.

    Args:
        streams_gdf (gpd.GeoDataFrame): GeoDataFrame of the streams
        out_dir (str): Path to directory where comid_lat_lon_z.csv will be saved
        id_field (str): Field in streams_gdf that corresponds to the unique id of each stream segment

    Returns:
        None
    """
    logger.info("Creating comid_lat_lon_z.csv")
    # todo sort the output df not gdf
    # todo apply lambda to gdf only once
    temp_network = streams_gdf.sort_values(id_field)
    lats = temp_network.geometry.apply(lambda geom: geom.xy[1][0]).values
    lons = temp_network.geometry.apply(lambda geom: geom.xy[0][0]).values

    pd.DataFrame({
        id_field: temp_network[id_field].values,
        "lat": lats,
        "lon": lons,
        "z": 0
    }).to_csv(os.path.join(out_dir, "comid_lat_lon_z.csv"), index=False, header=True)
    return


def create_riv_bas_id(streams_gdf: gpd.GeoDataFrame, out_dir: str, downstream_field: str, id_field: str) -> None:
    """
    Creates riv_bas_id.csv. Network is sorted to match the outputs of the ArcGIS tool this was designed from, 
    and it is likely that the second element in the list for ascending may be True without impacting RAPID

    Args:
        streams_gdf (gpd.GeoDataFrame):
        out_dir (str):
        downstream_field (str):
        id_field (str):

    Returns:
        None
    """
    logger.info("Creating riv_bas_id.csv")

    # todo sort the output df not gdf
    temp_network = streams_gdf.sort_values([downstream_field, id_field], ascending=[False, False])
    temp_network[id_field].to_csv(os.path.join(out_dir, "riv_bas_id.csv"), index=False, header=False)
    return


def calculate_muskingum(streams_gdf: gpd.GeoDataFrame, out_dir: str, k: float, x: float) -> None:
    """
    Calculates muskingum parameters by using pyproj's Geod.geometry_length. Note that the network must be in EPSG 4326
    """
    logger.info("Creating muskingum parameters")

    # todo split into two functions, one for calculating the parameters and one for writing them to csv
    streams_gdf["LENGTH_GEO"] = streams_gdf.geometry.apply(_calculate_geodesic_length)
    streams_gdf["Musk_kfac"] = streams_gdf["LENGTH_GEO"] * 3600
    streams_gdf["Musk_k"] = streams_gdf["Musk_kfac"] * k
    streams_gdf["Musk_x"] = x * 0.1

    streams_gdf["Musk_kfac"].to_csv(os.path.join(out_dir, "kfac.csv"), index=False, header=False)
    streams_gdf["Musk_k"].to_csv(os.path.join(out_dir, "k.csv"), index=False, header=False)
    streams_gdf["Musk_x"].to_csv(os.path.join(out_dir, "x.csv"), index=False, header=False)
    return


def _calculate_geodesic_length(line) -> float:
    """
    Input is shapely geometry, should be all shapely LineString objects
    """
    geod = Geod(ellps='WGS84')
    length = geod.geometry_length(line) / 1000  # To convert to km

    # This is for the outliers that have 0 length
    if length < 0.00000000001:
        length = 0.001
    return length


def create_rapid_connect(network: gpd.GeoDataFrame, out_dir: str, id_field: str, downstream_field: str) -> None:
    """
    Creates rapid_connect.csv

    todo document the columns of the csv
    rapid_connect is a csv file that contains the following columns:
    HydroID: the HydroID of the stream
    NextDownID: the HydroID of the next downstream stream
    CountUpstreamID: the number of upstream streams
    UpstreamID: the HydroID of the upstream streams

    Args:
        network:
        out_dir:
        id_field:
        downstream_field:

    Returns:

    """
    logger.info("Creating rapid_connect.csv")

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

        row_dict = {'HydroID': hydroid, 'NextDownID': nextDownID, 'CountUpstreamID': count_upstream}
        for i in range(count_upstream):
            row_dict[f'UpstreamID{i + 1}'] = list_upstreamID[i]
        list_all.append(row_dict)

    # Fill in NaN values for any missing upstream IDs
    for i in range(max_count_Upstream):
        col_name = f'UpstreamID{i + 1}'
        for row in list_all:
            if col_name not in row:
                row[col_name] = 0

    pd.DataFrame(list_all).to_csv(os.path.join(out_dir, 'rapid_connect.csv'), index=False, header=None)
    return


def create_weight_table(out_dir: str, basins_gdf: gpd.GeoDataFrame, nc_file: str, basin_id: str = 'streamID') -> None:
    """
    Creates weight table for a given netcdf file named after the netcdf file

    todo rewrite this function to be more memory and speed efficient
    todo document the columns of the csv
    the columns of the weight table are

    Args:
        out_dir:
        basins_gdf:
        nc_file:
        basin_id:

    Returns:

    """
    out_name = 'weight_' + os.path.basename(os.path.splitext(nc_file)[0]) + '.csv'
    logger.info(f"Creating weight table: {os.path.basename(out_name)}")

    # Obtain catchment extent
    extent = basins_gdf.total_bounds

    data_nc = nc.Dataset(nc_file)

    # Obtain geographic coordinates
    variables_list = data_nc.variables.keys()
    lat_var = 'lat'
    if 'latitude' in variables_list:
        lat_var = 'latitude'
    lon_var = 'lon'
    if 'longitude' in variables_list:
        lon_var = 'longitude'
    lon = (data_nc.variables[lon_var][:] + 180) % 360 - 180  # convert [0, 360] to [-180, 180]
    lat = data_nc.variables[lat_var][:]

    data_nc.close()

    # Create a buffer
    buffer = 2 * max(abs(lat[0] - lat[1]), abs(lon[0] - lon[1]))

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

    # Supress the warning about finding the centroid of a geographic crs 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        polygons_gdf = gpd.GeoDataFrame(geometry=[region for region in regions.geoms], crs=4326)
        polygons_gdf['POINT_X'] = polygons_gdf.geometry.centroid.x
        polygons_gdf['POINT_Y'] = polygons_gdf.geometry.centroid.y

    intersect = gpd.overlay(basins_gdf, polygons_gdf, how='intersection')

    intersect['AREA_GEO'] = intersect['geometry'].to_crs({'proj': 'cea'}).area

    area_arr = pd.DataFrame(data={
        basin_id: intersect[basin_id].values,
        'POINT_X': intersect['POINT_X'].values,
        'POINT_Y': intersect['POINT_Y'].values,
        'AREA_GEO': intersect['AREA_GEO']
    })
    area_arr.sort_values([basin_id, 'AREA_GEO'], inplace=True, ascending=[True, False])
    connectivity_table = pd.read_csv(os.path.join(out_dir, 'rapid_connect.csv'), header=None)
    streamID_unique_list = connectivity_table.iloc[:, 0].astype(int).unique().tolist()

    #   If point not in array append dummy data for one point of data
    lon_dummy = area_arr['POINT_X'].iloc[0]
    lat_dummy = area_arr['POINT_Y'].iloc[0]
    try:
        index_lon_dummy = int(np.where(lon == lon_dummy)[0])
    except TypeError as _:
        index_lon_dummy = int((np.abs(lon - lon_dummy)).argmin())
        pass

    try:
        index_lat_dummy = int(np.where(lat == lat_dummy)[0])
    except TypeError as _:
        index_lat_dummy = int((np.abs(lat - lat_dummy)).argmin())
        pass

    df = pd.DataFrame(columns=[f'{basin_id}', "area_sqm", "lon_index", "lat_index", "npoints", "lon", "lat"])

    for streamID_unique in streamID_unique_list:
        ind_points = np.where(area_arr[basin_id] == streamID_unique)[0]
        num_ind_points = len(ind_points)

        if num_ind_points <= 0:
            # if point not in array, append dummy data for one point of data
            # streamID, area_sqm, lon_index, lat_index, npoints
            df.loc[len(df)] = [streamID_unique, 0, index_lon_dummy, index_lat_dummy, 1, lon_dummy, lat_dummy]

        else:
            for ind_point in ind_points:
                area_geo_each = float(area_arr['AREA_GEO'].iloc[ind_point])
                lon_each = area_arr['POINT_X'].iloc[ind_point]
                lat_each = area_arr['POINT_Y'].iloc[ind_point]

                try:
                    index_lon_each = int(np.where(lon == lon_each)[0])
                except:
                    index_lon_each = int((np.abs(lon - lon_each)).argmin())

                try:
                    index_lat_each = int(np.where(lat == lat_each)[0])
                except:
                    index_lat_each = int((np.abs(lat - lat_each)).argmin())

                df.loc[len(df)] = [streamID_unique, area_geo_each, index_lon_each, index_lat_each, num_ind_points,
                                   lon_each, lat_each]

    df.to_csv(os.path.join(out_dir, out_name), index=False)
    logger.info(f"Finished weight table: {os.path.basename(out_name)}")
    return


################################################################
#   Master function
################################################################
def preprocess_for_rapid(stream_file: str, basins_file: str, nc_files: list, save_dir: str,
                         id_field: str = 'LINKNO', ds_field: str = 'DSLINKNO', length_field: str = 'Length',
                         k: float = 0.35, x: float = 3,
                         n_processes: int or None = 1, mp_streams: bool = True, mp_basins: bool = True) -> None:
    """
    Master function for preprocessing stream delineations and catchments and creating RAPID inputs.

    Args:
        stream_file (str): Path to stream network file
        basins_file (str): Path to the basins/catchments file
        nc_files (list): List of paths to sample netCDF files for making weight tables
        save_dir (str): Path to the output directory
        id_field (str): Field in network file that corresponds to the unique id of each stream segment. Defaults to 'LINKNO'.
        ds_field (str): Field in network file that corresponds to the unique downstream id of each stream segment. Defaults to 'DSLINKNO'.
        length_field (str): Field in network file that corresponds to the length of each stream segment. Defaults to 'Length'.
        k (float): K parameter to use when generating Muskingum parameters. Defaults to 0.35.
        x (float): X parameter to use when generating Muskingum parameters. Defaults to 3.
        n_processes (int): Number of processes to use for multiprocessing. If None, will use all available cores. Defaults to 1.
        mp_streams (bool): Whether to use multiprocessing for stream processing. Defaults to True.
        mp_basins (bool): Whether to use multiprocessing for basin processing. Defaults to True.

    Returns:
        None
    """
    logger.info('Dissolving streams')
    # Dissolve streams and basins
    streams_gdf = dissolve_streams(stream_file, save_dir=save_dir,
                                   stream_id_col=id_field, ds_id_col=ds_field, length_col=length_field,
                                   mp_dissolve=mp_streams, n_processes=n_processes * 2)

    # Create rapid preprocessing files
    logger.info('Creating RAPID files')
    create_comid_lat_lon_z(streams_gdf, save_dir, id_field)
    create_riv_bas_id(streams_gdf, save_dir, ds_field, id_field)
    calculate_muskingum(streams_gdf, save_dir, k, x)
    create_rapid_connect(streams_gdf, save_dir, id_field, ds_field)

    # clear memory for large network processing
    logger.info('Clearing stream GDF from memory')
    streams_gdf = None

    # dissolve basins
    logger.info('Dissolving basins')
    basins_gdf = dissolve_basins(basins_file, mp_dissolve=mp_basins,
                                 save_dir=save_dir, stream_id_col="streamID", n_process=n_processes)

    # Create weight table
    logger.info('Creating weight tables')
    # for nc_file in nc_files:
    #     create_weight_table(save_dir, basins_gdf, nc_file)
    with Pool(min(n_processes, len(nc_files))) as p:
        p.starmap(create_weight_table, [(save_dir, basins_gdf, nc_file) for nc_file in nc_files])

    # clear memory for large basin processing
    logger.info('Clearing basin GDF from memory')
    basins_gdf = None
    return


def validate_rapid_directory(directory: str):
    """
    Validate that the directory contains the necessary files for RAPID.

    Args:
        directory (str): Path to the directory to validate
    """
    required_rapid_files = [
        'rapid_connect.csv',
        'riv_bas_id.csv',
        'comid_lat_lon_z.csv',
        'k.csv',
        'kfac.csv',
        'x.csv',
    ]
    expected_network_files = [
        'adjoint_tree.json',
        'adjoint_dissolves_tree.json',
        'zero_length_fixes.csv',
    ]
    expected_geopackages = [
        'TDX_streamnet_*_model.gpkg',
        'TDX_streamnet_*_vis.gpkg',
        'TDX_streamreach_basins_*_model.gpkg'
    ]
    # Look for RAPID files
    missing_rapid_files = [f for f in required_rapid_files if not os.path.isfile(os.path.join(directory, f))]

    # look for weight tables
    weight_tables = glob.glob(os.path.join(directory, 'weight_*.csv'))

    # look for dissolved support files
    missing_network_files = [f for f in expected_network_files if not os.path.isfile(os.path.join(directory, f))]

    # look for geopackages
    missing_geopackages = [f for f in expected_geopackages if len(glob.glob(os.path.join(directory, f))) == 0]

    # summarize findings
    logger.info(f'Validating directory: {directory}')
    if all([
        len(missing_rapid_files) == 0,
        len(weight_tables) > 0,
        len(missing_network_files) == 0,
        len(missing_geopackages) == 0
    ]):
        logger.info('All expected files found in this directory')
        logger.info(f'Found {len(weight_tables)} weight tables')
    else:
        if len(missing_rapid_files) != 0:
            logger.info('Missing RAPID files:')
            for file in missing_rapid_files:
                logger.info(file)

        if len(weight_tables) == 0:
            logger.info('No weight tables found')
        else:
            logger.info(f'Found {len(weight_tables)} weight tables')

        if len(missing_network_files) != 0:
            logger.info('Missing network files:')
            for file in missing_network_files:
                logger.info(file)

        if len(missing_geopackages) != 0:
            logger.info('Missing geopackages:')
            for file in missing_geopackages:
                logger.info(file)

    logger.info('')
    return

