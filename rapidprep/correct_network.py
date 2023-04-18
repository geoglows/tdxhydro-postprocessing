import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as sg

__all__ = [
    'merge_headwater_streams',
    'merge_basins',
    'identify_0_length',
    'apply_0_length_basin_fixes',
    'apply_0_length_stream_fixes',
    'correct_0_length_basins'
]

logger = logging.getLogger(__name__)


def merge_headwater_streams(upstream_ids: list, network_gdf: gpd.GeoDataFrame, make_model_version: bool,
                            streamid: str = 'LINKNO', dsid: str = 'DSLINKNO',
                            length_col: str = 'Length') -> gpd.GeoDataFrame:
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


def merge_basins(basin_gdf: gpd.GeoDataFrame, upstream_ids: list) -> gpd.GeoDataFrame:
    """
    Dissolves basins based on list of upstream river ids, and returns that feature.
    """
    gdf = basin_gdf[basin_gdf["streamID"].isin(upstream_ids)].dissolve()
    gdf['streamID'] = upstream_ids[0]  # the first ID should be the most downstream ID
    return gdf


def identify_0_length(gdf: gpd.GeoDataFrame, stream_id_col: str, ds_id_col: str, length_col: str) -> pd.DataFrame:
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
            logging.warning(f"The stream segment {feat[stream_id_col]} has conditions we've not yet considered")
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
        .dropna(axis=0, how='any')
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


def correct_0_length_basins(basins_gpkg: gpd.GeoDataFrame, save_dir: str, stream_id_col: str, buffer_size: float = 1):
    """
    Apply fixes to streams that have 0 length.

    Args:
        basins_gpkg: Basins to correct
        save_dir: Directory to save the corrected basins to
        stream_id_col:
        buffer_size:

    Returns:

    """
    logger.info('Revising basins with 0 length streams')
    zero_length_df = pd.read_csv(os.path.join(save_dir, 'mod_zero_length_streams.csv'))

    basin_gdf = gpd.read_file(basins_gpkg)

    # Case 1 - Coastal w/ no upstream or downstream - Delete the stream and its basin
    basin_gdf = basin_gdf[~basin_gdf[stream_id_col].isin(zero_length_df['case1'])]

    # Case 2 - Allow 3-river confluence - Create a basin with small non-zero area, assign small non-zero length
    boxes = (
        zero_length_df[['case2', 'case2_x', 'case2_y']]
        .dropna(axis=0, how='any')
        .apply(lambda x: sg.box(
            x.case2_x - buffer_size,
            x.case2_y - buffer_size,
            x.case2_x + buffer_size,
            x.case2_y + buffer_size
        ), axis=1)
    )
    basin_gdf = pd.concat([
        basin_gdf,
        gpd.GeoDataFrame({'geometry': boxes, stream_id_col: zero_length_df['case2']})
    ])

    # Case 3 - Coastal w/ upstreams but no downstream - Assign small non-zero length
    # NO FIXES APPLIED TO BASINS FOR CASE 3 - ALREADY HAVE

    basin_gdf.to_file(os.path.join(save_dir, os.path.basename(os.path.splitext(basins_gpkg)[0]) + '_corrected.gpkg'))
    return
