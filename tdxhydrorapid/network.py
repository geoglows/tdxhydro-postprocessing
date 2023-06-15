import logging
import os

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely.geometry as sg

__all__ = [
    'sort_topologically',
    'create_directed_graphs',
    'find_headwater_streams_to_dissolve',
    'merge_headwater_streams',
    'identify_0_length',
    'correct_0_length_streams',
    'correct_0_length_basins',
    'make_vpu_streams',
    'make_vpu_basins',
]

logger = logging.getLogger(__name__)


def sort_topologically(digraph_from_headwaters: nx.DiGraph) -> np.array:
    return np.array(list(nx.topological_sort(digraph_from_headwaters))).astype(int)


def create_directed_graphs(df: pd.DataFrame, id_field='LINKNO', ds_id_field='DSLINKNO'):
    G = nx.DiGraph()

    for node in df[id_field].values:
        G.add_node(node)
    for i, row in df.iterrows():
        if row[ds_id_field] != -1:
            G.add_edge(row[id_field], row[ds_id_field])
    return G


def find_headwater_streams_to_dissolve(sdf: pd.DataFrame or gpd.GeoDataFrame) -> pd.DataFrame:
    # todo parameterize the column names
    us_cols = sorted([c for c in sdf.columns if c.startswith('USLINKNO')])
    o1 = sdf[sdf['strmOrder'] == 1]['LINKNO'].values.flatten()
    o2 = sdf[sdf['strmOrder'] == 2]

    # select rows where 2+ of the 2+ upstreams are 1st order (ie this is the first 2nd order in the chain)
    # o2 = o2[us_cols].isin(o1).sum(axis=1) >= 2
    # alternatively the only upstreams must be 1st orders, all else are -1
    o2 = o2[o2[us_cols].isin(o1).sum(axis=1) + (o2[us_cols] == -1).sum(axis=1) == len(us_cols)]
    o2 = o2[['LINKNO', ] + us_cols]
    return o2


def merge_headwater_streams(upstream_ids: list, network_gdf: gpd.GeoDataFrame, make_model_version: bool,
                            streamid: str = 'LINKNO', dsid: str = 'DSLINKNO',
                            length_col: str = 'Length') -> gpd.GeoDataFrame:
    """
    Selects stream segments upstream of the given segments and merges them into a single geodataframe.
    """
    # todo old, reconfigure and delete
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

        # Case 3
        elif feat[ds_id_col].values == -1 and feat['USLINKNO1'].values != -1 and feat['USLINKNO2'].values != -1:
            case3_ids.append(rivid)

        # Case 4
        else:
            logging.warning(f"The stream segment {feat[stream_id_col]} has conditions we've not yet considered")
            case4_ids.append(rivid)

    # variable length lists with np.nan to make them the same length
    longest_list = max([len(case1_ids), len(case2_ids), len(case3_ids), len(case4_ids), ])
    case1_ids = case1_ids + [np.nan] * (longest_list - len(case1_ids))
    case2_ids = case2_ids + [np.nan] * (longest_list - len(case2_ids))
    case3_ids = case3_ids + [np.nan] * (longest_list - len(case3_ids))
    case4_ids = case4_ids + [np.nan] * (longest_list - len(case4_ids))

    return pd.DataFrame({
        'case1': case1_ids,
        'case2': case2_ids,
        'case3': case3_ids,
        'case4': case4_ids,
    })


def correct_0_length_streams(sgdf: gpd.GeoDataFrame, zero_length_df: pd.DataFrame,
                             id_field: str) -> gpd.GeoDataFrame:
    """
    Apply fixes to streams that have 0 length.

    Args:
        sgdf:
        zero_length_df:
        id_field:

    Returns:

    """
    # Case 1 - Coastal w/ no upstream or downstream - Delete the stream and its basin
    c1 = zero_length_df['case1'].dropna().astype(int).values
    sgdf = sgdf[~sgdf[id_field].isin(c1)]

    # Case 3 - Coastal w/ upstreams but no downstream - Assign small non-zero length
    # Apply before case 2 to handle some edges cases where zero length basins drain into other zero length basins
    c3_us_ids = sgdf[sgdf[id_field].isin(zero_length_df['case3'].dropna().values)][
        ['USLINKNO1', 'USLINKNO2']].values.flatten()
    sgdf.loc[sgdf[id_field].isin(c3_us_ids), 'DSLINKNO'] = -1
    sgdf = sgdf[~sgdf['LINKNO'].isin(zero_length_df['case3'].dropna().values)]

    # Case 2 - Allow 3-river confluence - Delete the temporary basin and modify the connectivity properties
    # Sort by DSLINKNO to handle some edges cases where zero length basins drain into other zero length basins
    c2 = sgdf[sgdf['LINKNO'].isin(zero_length_df['case2'].dropna().astype(int).values)]
    c2 = c2.sort_values(by=['DSLINKNO'], ascending=True)
    c2 = c2['LINKNO'].values
    for river_id in c2:
        ids_to_apply = sgdf.loc[sgdf[id_field] == river_id, ['USLINKNO1', 'USLINKNO2', 'DSLINKNO']]
        sgdf.loc[
            sgdf[id_field].isin(ids_to_apply[['USLINKNO1', 'USLINKNO2']].values.flatten()), 'DSLINKNO'] = \
            ids_to_apply['DSLINKNO'].values[0]
    # Remove the rows corresponding to the rivers to be deleted
    sgdf = sgdf[~sgdf['LINKNO'].isin(c2)]

    return sgdf


def correct_0_length_basins(basins_gpq: str, save_dir: str, stream_id_col: str) -> gpd.GeoDataFrame:
    """
    Apply fixes to streams that have 0 length.

    Args:
        basins_gpq: Basins to correct
        save_dir: Directory to save the corrected basins to
        stream_id_col:

    Returns:

    """
    basin_gdf = gpd.read_parquet(basins_gpq)

    zero_fix_csv_path = os.path.join(save_dir, 'mod_basin_zero_centroid.csv')
    if os.path.exists(zero_fix_csv_path):
        box_radius_degrees = 0.015
        basin_zero_centroid = pd.read_csv(zero_fix_csv_path)
        centroid_x = basin_zero_centroid['centroid_x'].values[0]
        centroid_y = basin_zero_centroid['centroid_y'].values[0]
        link_zero_box = gpd.GeoDataFrame({
            'geometry': [sg.box(
                centroid_x - box_radius_degrees,
                centroid_y - box_radius_degrees,
                centroid_x + box_radius_degrees,
                centroid_y + box_radius_degrees
            )],
            stream_id_col: [0, ]
        }, crs=basin_gdf.crs)
        basin_gdf = pd.concat([basin_gdf, link_zero_box])

    zero_length_csv_path = os.path.join(save_dir, 'mod_zero_length_streams.csv')
    if os.path.exists(zero_length_csv_path):
        logger.info('\tRevising basins with 0 length streams')
        zero_length_df = pd.read_csv(zero_length_csv_path)
        # Case 1 - Coastal w/ no upstream or downstream - Delete the stream and its basin
        logger.info('\tHandling Case 1 0 Length Streams - delete basins')
        basin_gdf = basin_gdf[~basin_gdf[stream_id_col].isin(zero_length_df['case1'])]
        # Case 2 - Allow 3-river confluence - basin does not exist (try to delete just in case)
        logger.info('\tHandling Case 2 0 Length Streams - delete basins')
        basin_gdf = basin_gdf[~basin_gdf[stream_id_col].isin(zero_length_df['case2'])]
        # Case 3 - Coastal w/ upstreams but no downstream - basin exists so delete it
        logger.info('\tHandling Case 3 0 Length Streams - delete basins')
        basin_gdf = basin_gdf[~basin_gdf[stream_id_col].isin(zero_length_df['case3'])]

    small_tree_csv_path = os.path.join(save_dir, 'mod_drop_small_trees.csv')
    if os.path.exists(small_tree_csv_path):
        logger.info('\tDeleting small trees')
        small_tree_df = pd.read_csv(small_tree_csv_path)
        basin_gdf = basin_gdf[~basin_gdf[stream_id_col].isin(small_tree_df.values.flatten())]

    basin_gdf = basin_gdf.reset_index(drop=True)
    return basin_gdf


def make_vpu_streams(final_inputs_directory: str, inputs_directory: str, gpq: str, id_field: str = 'LINKNO') -> None:
    # todo
    vpu = 123
    save_path = os.path.join(save_dir, f'geoglows2_streams_vpu_{vpu}.gpkg')
    logging.info(f'Making GeoPackage:{save_path}')
    gdf = gpd.read_parquet(streams_gpq)

    streams_to_drop = pd.read_csv(os.path.join(save_dir, 'mod_drop_small_trees.csv'))
    gdf = gdf[~gdf[id_field].isin(streams_to_drop.values.flatten())]

    streams_to_dissolve = pd.read_csv(os.path.join(save_dir, 'mod_dissolve_headwater.csv'))
    gdf = dissolve_headwater_table(gdf,
                                   streams_to_dissolve,
                                   geometry_diss=lambda x: x.unary_union)

    gpd.GeoDataFrame(gdf).to_file(save_path, driver='GPKG')  # groupby returns DF not GDF
    return


def make_vpu_basins(final_inputs_directory: str,
                    gpq_dir: str,
                    id_field: str = 'LINKNO',
                    basin_id_field: str = 'streamID', ):
    master_table = pd.read_parquet(os.path.join(final_inputs_directory, 'master_table.parquet'))
    for tdxnumber in sorted(master_table['TDXHydroNumber'].unique()):
        print(tdxnumber)
        gpq = os.path.join(gpq_dir, f'TDX_streamreach_basins_{tdxnumber}_01.parquet')
        output_file = os.path.join(final_inputs_directory, 'gis', f'vpu_basins_{tdxnumber}.geoparquet')
        if os.path.exists(output_file):
            continue

        heads = pd.read_csv(f'/Volumes/EB406_T7_2/TDXHydroRapid_V11/{tdxnumber}/mod_dissolve_headwater.csv')
        idvpudf = (
            master_table
            .loc[master_table['TDXHydroNumber'] == tdxnumber, [id_field, 'VPUCode']]
            .reset_index(drop=True)
        )
        for vpucode in sorted(idvpudf['VPUCode'].unique()):
            matching_rows = heads[heads[id_field].isin(idvpudf[idvpudf['VPUCode'] == vpucode][id_field])]
            all_ids = (
                set(matching_rows[id_field].values)
                .union(matching_rows.iloc[:, 1:].values.flatten())
            )
            idvpudf = pd.concat([
                pd.DataFrame({'LINKNO': list(all_ids), 'VPUCode': vpucode}),
                idvpudf
            ])
        idvpudf = idvpudf.drop_duplicates(subset=['LINKNO', 'VPUCode'])
        (
            gpd
            .read_parquet(gpq)
            .merge(
                idvpudf,
                left_on=basin_id_field,
                right_on=id_field,
                how='inner'
            )
            .drop(columns=[basin_id_field, id_field, ])
            .dissolve(by='VPUCode')
            .reset_index()
            .to_parquet(output_file)
        )
    return
