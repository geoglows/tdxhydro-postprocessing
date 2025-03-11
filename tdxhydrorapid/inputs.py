import glob
import json
import logging
import os
import types
from typing import Union

import geopandas as gpd
import dask.dataframe as dd
import dask_geopandas as dgpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, MultiLineString

try:
    import numba
    ENGINE = 'numba'
except ImportError:
    ENGINE = None

from .network import correct_0_length_streams
from .network import create_directed_graphs
from .network import find_branches_to_prune
from .network import find_headwater_branches_to_dissolve
from .network import identify_0_length
from .network import sort_topologically
from .network import estimate_num_partition

# set up logging
logger = logging.getLogger(__name__)

__all__ = [
    'river_route_master_files',
    'dissolve_branches',
    'prune_branches',
    'routing_configs',
    'concat_tdxregions',
    'vpu_files_from_masters',
    'create_directed_graphs'
    'create_nexus_points',
    'nexus_file_from_masters',
]

def river_route_master_files(streams_gpq: str,
                       save_dir: str,
                       id_field: str = 'LINKNO',
                       ds_id_field: str = 'DSLINKNO',
                       length_field: str = 'Length',
                       default_velocity_factor: float = None,
                       default_x: float = .25,
                       drop_small_watersheds: bool = True,
                       dissolve_headwaters: bool = True,
                       prune_branches_from_main_stems: bool = True,
                       merge_short_streams: bool = True,
                       cache_geometry: bool = True,
                       dissolve_lakes: bool = True,
                       drop_islands: bool = True,
                       drop_ocean_watersheds: bool = True,
                       drop_within_sea: bool = True,
                       drop_low_flow: bool = False,
                       min_drainage_area_m2: float = 200_000_000,
                       min_headwater_stream_order: int = 3,
                       min_velocity_factor: float = 0.25,
                       min_k_value: int = 900,
                       lake_min_k: int = 3600, ) -> None:
    """
    Create river route master files from a stream network

    Saves the following files to the save_dir:
        - rr_inputs_master.parquet
        - {region_num}_dissolved_network.geoparquet (if cache_geometry is True)
        - mod_zero_length_streams.csv (if any 0 length streams are found)
        - mod_basin_zero_centroid.csv (if any basins have an ID of 0 and geometry is not available)
        - mod_drop_small_streams.csv (if drop_small_watersheds is True)
        - mod_dissolved_headwaters.csv (if dissolve_headwaters is True)
        - mod_pruned_branches.csv (if prune_branches_from_main_stems is True)
        - mod_dissovle_lakes.json (if dissolve_lakes is True)


    Args:
        min_k_value: int, target minimum k value to keep a stream segment
        min_velocity_factor: float, minimum velocity factor used to calculate k
        merge_short_streams: bool, merge short streams that are not connected to a confluence
        streams_gpq: str, path to the streams geoparquet
        save_dir: str, path to the directory to save the master files
        id_field: str, field name for the link id
        ds_id_field: str, field name for the downstream link id
        length_field: str, field name for the length of the stream segment
        default_velocity_factor: float, default velocity factor (k) for Muskingum routing
        default_x: float, default attenuation factor (x) for Muskingum routing

        drop_small_watersheds: bool, drop small watersheds
        dissolve_headwaters: bool, dissolve headwater branches
        drop_islands: bool, drop islands
        drop_ocean_watersheds: bool, drop ocean watersheds
        drop_within_sea: bool, drop streams within the sea
        drop_low_flow: bool, drop low flow streams
        prune_branches_from_main_stems: bool, prune branches from main stems
        cache_geometry: bool, save the dissolved geometry as a geoparquet
        dissolve_lakes: bool, clean up lake geometries
        min_drainage_area_m2: float, minimum drainage area in m2 to keep a watershed
        min_headwater_stream_order: int, minimum stream order to keep a headwater branch

    Returns:
        None
    """
    sgdf = gpd.read_parquet(streams_gpq).set_crs('epsg:4326')
    logging.info(f'\tNumber of streams: {sgdf.shape[0]}')

    if not sgdf[sgdf[length_field] <= 0.01].empty:
        logger.info('\tRemoving 0 length segments')
        zero_length_fixes_df = identify_0_length(sgdf, id_field, ds_id_field, length_field)
        zero_length_fixes_df.to_csv(os.path.join(save_dir, 'mod_zero_length_streams.csv'), index=False)
        sgdf = correct_0_length_streams(sgdf, zero_length_fixes_df, id_field)

    # Fix basins with ID of 0
    if 0 in sgdf[id_field].values:
        logger.info('\tFixing basins with ID of 0')
        pd.DataFrame({
            id_field: [0, ],
            'centroid_x': sgdf[sgdf[id_field] == 0].centroid.x.values[0],
            'centroid_y': sgdf[sgdf[id_field] == 0].centroid.y.values[0]
        }).to_csv(os.path.join(save_dir, 'mod_basin_zero_centroid.csv'), index=False)
    
    logger.info('\tCreating Directed Graph')
    G = create_directed_graphs(sgdf, id_field, ds_id_field=ds_id_field) 
    sorted_order = sort_topologically(G)
    sgdf = (
        sgdf
        .set_index(pd.Index(sgdf[id_field]))
        .reindex(sorted_order)
        .reset_index(drop=True)
        .dropna(axis=0, subset=[id_field])
        .astype(sgdf.dtypes.to_dict())
        .set_index(pd.Index(range(1, len(sgdf) + 1)).rename('TopologicalOrder'))
        .reset_index()
        .dropna()
        .astype(sgdf.dtypes.to_dict())
    )

    # Drop trees with small total length/area
    if drop_small_watersheds:
        logger.info('\tFinding and removing small trees')
        small_tree_outlet_ids = sgdf.loc[np.logical_and(
            sgdf[ds_id_field] == -1,
            sgdf['DSContArea'] < min_drainage_area_m2
        ), id_field].values
        small_tree_segments = [nx.ancestors(G, x) for x in small_tree_outlet_ids]
        small_tree_segments = set().union(*small_tree_segments).union(small_tree_outlet_ids)
        (
            pd
            .DataFrame(small_tree_segments, columns=['drop'])
            .to_csv(os.path.join(save_dir, 'mod_drop_small_trees.csv'), index=False)
        )
        sgdf = sgdf.loc[~sgdf[id_field].isin(small_tree_segments)]
        G = create_directed_graphs(sgdf, id_field, ds_id_field=ds_id_field) 

    if drop_islands:
        logging.info('\tFinding and removing islands')
        island_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'network_data', 'island_table.csv'))
        island_ids = set(island_df['island_id'])
        
        sgdf = sgdf[~sgdf[id_field].isin(island_ids)]
        pd.DataFrame({'drop': list(island_ids)}).to_csv(
            os.path.join(save_dir, 'mod_drop_islands.csv'), index=False
        )
        
        G = create_directed_graphs(sgdf, id_field, ds_id_field)

    if drop_low_flow:
        logger.info('\tFinding and removing low flow streams')
        low_flow_rivers = pd.read_csv(os.path.join(os.path.dirname(__file__), 'network_data', 'rivids_lt_5_cms.csv')) 
        low_flow_rivers = set(sgdf[sgdf[id_field].isin(low_flow_rivers['LINKNO'].values)][id_field])

        streams_to_delete = set()
        low_flow_dict: dict[int, list] = {} # Outlets to upstreams
        for river in low_flow_rivers:
            downstream = list(G.successors(river))
            if downstream and downstream[0] in low_flow_rivers:
                    continue # Skip this river until we find an "outlet"
            if downstream:
                downstream = downstream[0]
            else: 
                downstream = -1  
                
            ancestors: set = nx.ancestors(G, river) | {river, }

            if downstream in low_flow_dict:
                low_flow_dict[downstream].extend(ancestors)
            else:
                low_flow_dict[downstream] = list(ancestors)
            streams_to_delete.update(ancestors)

        sgdf = sgdf[~sgdf[id_field].isin(streams_to_delete)]

        with open(os.path.join(save_dir, 'mod_drop_low_flow.json'), 'w') as f:
            json.dump(low_flow_dict, f)

        G = create_directed_graphs(sgdf, id_field, ds_id_field)

    if drop_ocean_watersheds:
        logger.info('\tFinding and removing ocean watersheds')
        small_watersheds_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'network_data', 'small_ocean_watersheds.csv'))
        sgdf = sgdf[~sgdf[id_field].isin(small_watersheds_df['drop'].values)]

        small_watersheds_df[small_watersheds_df['drop'].isin(sgdf[id_field].values)].to_csv(
            os.path.join(save_dir, 'mod_drop_ocean_watersheds.csv'), index=False
        )

        G = create_directed_graphs(sgdf, id_field, ds_id_field)

    if drop_within_sea:
        # From a previous step, we know all ocean streams have downstreams already dealt with, so only need to focus on upstreams
        logger.info('\tFinding and removing streams within the sea')
        within_sea_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'network_data', 'within_sea.csv'))
        sea_rivers = set(within_sea_df['drop'])
        
        # Get only sea rivers that are in sgdf
        sea_rivers = sea_rivers.intersection(sgdf[id_field].values)
        more_sea_rivers = set()

        new_sea_outlets = set()
        for downstream in sea_rivers:
            upstreams = set(G.predecessors(downstream))
            # After all the edits, this might be classified now as a headwater
            # And so we can dissolve it
            if len(upstreams) == 0:
                more_sea_rivers.add(downstream)
            new_sea_outlets.update(upstreams)

        sea_rivers.update(more_sea_rivers)
            
        sgdf.loc[sgdf[id_field].isin(new_sea_outlets), ds_id_field] = -1
        sgdf = sgdf[~sgdf[id_field].isin(sea_rivers)]
        pd.DataFrame({'drop': list(sea_rivers)}).to_csv(
            os.path.join(save_dir, 'mod_drop_within_sea.csv'), index=False
        )

        G = create_directed_graphs(sgdf, id_field, ds_id_field)

    if dissolve_lakes:
        logger.info('\tDissolving lakes')
        lake_csv = os.path.join(os.path.dirname(__file__), 'network_data', 'lake_table.csv')
        if not os.path.exists(lake_csv):
            raise FileNotFoundError('Lake CSV not found')
        
        lake_df = pd.read_csv(lake_csv).astype(int)
        stream_ids = set(sgdf[id_field])

        dissolve_lake_dict: dict[int, dict[str, list]] = {}
        to_remove = set()
        global_inlets = set()
        lake_groups = lake_df.groupby('outlet')

        outlet_geoms_dict = {}
    
        for outlet, group in lake_groups:
            if outlet not in stream_ids:
                continue
            inlets = set(group['inlet'])

            # Compute ancestors and update to_remove
            inside_streams: set = nx.ancestors(G, outlet)
            inlets_to_use = set()
            for inlet in inlets:
                if inlet not in stream_ids:
                    continue

                inlet_ancestors = ancestors_safe(G, inlet)
                # Because of all the edits, this might be classified now as a headwater
                # And so we can dissolve it
                if len(inlet_ancestors) == 0 or len([x for x in inlet_ancestors if G.in_degree(x) == 0]) == len(inlet_ancestors):
                    inside_streams.add(inlet)
                    inside_streams.update(inlet_ancestors)
                    continue

                inlets_to_use.add(inlet)
                inlet_ancestors.add(inlet)
                inside_streams -= inlet_ancestors

            # Update sgdf for previous inlets
            sgdf.loc[sgdf[id_field].isin(inlets_to_use), ds_id_field] = outlet
            global_inlets.update(inlets_to_use)

            dissolve_lake_dict[outlet] = {
                'inlets': list(inlets_to_use),
                'inside': list(inside_streams)
            }
            to_remove.update(inside_streams)

            # By saving this for later, we avoid long wait times in dissolve headwater function
            outlet_geoms_dict[outlet] = sgdf.loc[sgdf[id_field].isin(inside_streams | {outlet,}), 'geometry'].union_all('coverage')

        
        sgdf = sgdf[~sgdf[id_field].isin(to_remove)]

        with open(os.path.join(save_dir, 'mod_dissolve_lakes.json'), 'w') as f:
            json.dump(dissolve_lake_dict, f)

        G = create_directed_graphs(sgdf, id_field, ds_id_field) # Need to recreate the graph after removing streams

    if dissolve_headwaters:
        logger.info('\tFinding headwater streams to dissolve')
        headwater_dissolve_dfs = []
        for strmorder in range(2, min_headwater_stream_order + 1):
            branches = find_headwater_branches_to_dissolve(sgdf, G, strmorder)
            if strmorder == 2:
                sgdf = dissolve_branches(sgdf, branches, geometry_diss=cache_geometry, k_agg_func=_k_agg_order_2)
            elif strmorder == 3:
                sgdf = dissolve_branches(sgdf, branches, geometry_diss=cache_geometry, k_agg_func=_k_agg_order_3)
            headwater_dissolve_dfs.append(branches)
        (
            pd
            .concat(headwater_dissolve_dfs)
            .fillna(-1)
            .astype(int)
            .to_csv(os.path.join(save_dir, 'mod_dissolve_headwater.csv'), index=False)
        )
        headwater_dissolve_dfs = []

        G = create_directed_graphs(sgdf, id_field, ds_id_field=ds_id_field)

    if prune_branches_from_main_stems:
        logger.info('\tFinding branches to prune')
        streams_to_prune = find_branches_to_prune(sgdf, G)
        streams_to_prune.to_csv(os.path.join(save_dir, 'mod_prune_streams.csv'), index=False)
        sgdf = prune_branches(sgdf, streams_to_prune)
        G = create_directed_graphs(sgdf, id_field, ds_id_field=ds_id_field)

    # length is in m, divide by estimated m/s to get k in seconds
    logger.info('\tCalculating Muskingum k and x')
    sgdf['velocity_factor'] = np.exp(0.16842 * np.log(sgdf['DSContArea']) - 4.68).round(3) \
        if default_velocity_factor is None else default_velocity_factor
    sgdf['velocity_factor'] = sgdf['velocity_factor'].clip(lower=min_velocity_factor)
    sgdf['musk_k'] = sgdf['LengthGeodesicMeters'] / sgdf['velocity_factor']
    sgdf['musk_k'] = sgdf['musk_k'].round(0).astype(int)
    sgdf['musk_k'] = sgdf['musk_k'].clip(lower=0, upper=100_000)
    sgdf["musk_x"] = default_x

    if merge_short_streams:
        logging.info('\tFinding small k value streams to merge')
        # find short rivers that have an upstream or downstream link without crossing a confluence point
        short_streams_to_merge = {}
        for river in sgdf.loc[sgdf['musk_k'] < min_k_value, id_field].values:
            # this river is a confluence of 2 upstreams if it has more than 1 upstream
            upstreams = list(G.predecessors(river))
            downstream = list(G.successors(river))
            downstream = downstream[0] if downstream else -1
            downstream_upstreams = [river, ] if downstream == -1 else list(G.predecessors(downstream))
            # if there is a confluence upstream and downstream then it cannot be fixed
            if len(upstreams) != 1 and len(downstream_upstreams) != 1:
                continue
            stream_merge_options = [
                upstreams[0] if len(upstreams) == 1 else -1,
                downstream if len(downstream_upstreams) == 1 else -1
            ]
            if stream_merge_options[0] == stream_merge_options[1]:
                continue
            stream_to_merge_with = (
                sgdf.loc[sgdf[id_field].isin(stream_merge_options)]
                .sort_values(by='musk_k', ascending=True).iloc[0][id_field]
            )
            if stream_to_merge_with not in upstreams:
                stream_to_merge_with, river = river, stream_to_merge_with
            short_streams_to_merge[river] = {'MergeWith': stream_to_merge_with}

        # write the short stream merges to a csv
        if len(short_streams_to_merge):
            short_streams_df = (
                pd.DataFrame.from_dict(short_streams_to_merge)
                .T.reset_index().rename(columns={'index': id_field})
                .astype(int)
            )
            short_streams_df.to_csv(os.path.join(save_dir, 'mod_merge_short_streams.csv'), index=False)
            sgdf = dissolve_short_streams(sgdf, short_streams_df)

            if dissolve_lakes:
                for linkno, mergewith in short_streams_df.itertuples(index=False):
                    if mergewith in dissolve_lake_dict:
                        # Change the index to the linkno of the lake
                        data = dissolve_lake_dict[mergewith]
                        del dissolve_lake_dict[mergewith]
                        dissolve_lake_dict[linkno] = data
                        outlet_geoms_dict[linkno] = outlet_geoms_dict[mergewith]

                with open(os.path.join(save_dir, 'mod_dissolve_lakes.json'), 'w') as f:
                    json.dump(dissolve_lake_dict, f)

        G = create_directed_graphs(sgdf, id_field, ds_id_field=ds_id_field)

    if dissolve_lakes:
        logger.info('\tUpdating Lengths for lake outlets')
        stream_ids = set(sgdf[id_field])
        lake_ids = lake_df['outlet'].unique()
        # We are gonna udpate the k value for lakes
        for outlet, group in lake_groups:
            if outlet not in stream_ids:
                continue

            # change this to find length from outlet to biggest drainage area
            inlets = set(group['inlet'])

            # Find the inlet with the biggest upstream drainage area
            biggest_inlet_id = sgdf.sort_values('DSContArea', ascending=False).iloc[0]['LINKNO']

            # Find the streamline distance from the biggest inlet to the outlet
            max_distance = 0
            downstream = list(G.successors(biggest_inlet_id))
            while downstream:
                downstream = downstream[0]
                if downstream == outlet:
                    break
                max_distance += sgdf.loc[sgdf[id_field] == downstream, 'LengthGeodesicMeters'].values[0]
                downstream = list(G.successors(downstream))

            sgdf.loc[sgdf[id_field] == outlet, 'LengthGeodesicMeters'] += max_distance

        # set the x value to 0.01 for lakes (max attenuation while avoiding possible errors with 0.0)
        sgdf.loc[sgdf['LINKNO'].isin(lake_ids), 'musk_x'] = 0.01

        # Do this again to update the k value
        sgdf['musk_k'] = sgdf['LengthGeodesicMeters'] / sgdf['velocity_factor']
        sgdf['musk_k'] = sgdf['musk_k'].round(0).astype(int)
        sgdf['musk_k'] = sgdf['musk_k'].clip(lower=0, upper=100_000)

        # Also update outlet geometries from the dissolve lakes step
        for outlet in outlet_geoms_dict:
            sgdf.loc[sgdf[id_field] == outlet, 'geometry'] = outlet_geoms_dict[outlet]

    # Do a final pass, filtering out any streams left by themselves after all edits
    drop_dissolve = []
    weakly_connected = sorted(nx.weakly_connected_components(G), key=len)
    for component in weakly_connected:
        if len(component) > 1:
            break

        # Check if the component is a lake inlet
        if dissolve_lakes:
            if list(component)[0] in global_inlets:
                drop_dissolve.append([list(component)[0], sgdf.loc[sgdf[id_field] == list(component)[0], ds_id_field].values[0]])
                continue
            
        drop_dissolve.append([list(component)[0], np.nan])

    if drop_dissolve:
        lonely_df = pd.DataFrame(list(drop_dissolve), columns=['drop', 'dissolve'])
        lonely_df = lonely_df.sort_values('dissolve')
        lonely_df.to_csv(os.path.join(save_dir, 'mod_drop_lonely_streams.csv'), index=False)

        to_delete = set()
        for drop, _dissolve in lonely_df.itertuples(index=False):
            if np.isnan(_dissolve):
                to_delete.add(drop)
            else:
                sgdf.loc[sgdf[id_field] == drop, id_field] = _dissolve
                
        sgdf: gpd.GeoDataFrame = sgdf[~sgdf[id_field].isin(to_delete)]
        sgdf = dissolve(sgdf, True)


        G = create_directed_graphs(sgdf, id_field, ds_id_field=ds_id_field)

    logger.info('\tLabeling watersheds by terminal node')
    for term_node in sgdf[sgdf[ds_id_field] == -1][id_field].values:
        sgdf.loc[sgdf[id_field].isin(list(nx.ancestors(G, term_node)) + [term_node, ]), 'TerminalLink'] = term_node
    try:
        sgdf['TerminalLink'] = sgdf['TerminalLink'].astype(int)
    except pd.errors.IntCastingNaNError:
        print(sgdf[sgdf['TerminalLink'].isna()])
        raise

    # ensure the order is still the topological order
    sgdf = sgdf.sort_values('TopologicalOrder', ascending=True).reset_index(drop=True)

    logger.info(f'Final Stream Count: {sgdf.shape[0]}')

    if cache_geometry:
        logger.info('\tWriting altered geometry to geopackage')
        region_number = sgdf['TDXHydroRegion'].values[0]
        gpd.GeoDataFrame(sgdf)[['LINKNO', 'geometry']].to_parquet(
            os.path.join(save_dir, f"{region_number}_altered_network.geoparquet"))

    logger.info('\tWriting river route master parquet')
    sgdf.drop(columns=['geometry', ]).to_parquet(os.path.join(save_dir, "rr_inputs_master.parquet"))
    return

def create_nexus_points(save_dir: str,
                        min_strm_order: int,
                        id_field: str = 'LINKNO', ) -> None:
    geometry_files = glob.glob(os.path.join(save_dir, '*_altered_network.geoparquet'))
    if not geometry_files:
        return
    
    logger.info('\tCreating Nexus Points')
    df = pd.read_parquet(os.path.join(save_dir, 'rr_inputs_master.parquet'))
    lake_table = pd.read_csv(os.path.join(os.path.dirname(__file__), 'network_data', 'lake_table.csv'))
    gdf = gpd.read_parquet(geometry_files[0])
    gdf = gdf.merge(df[[id_field, 'DSLINKNO', 'strmOrder']], on=id_field, how='left')
    G = create_directed_graphs(gdf, id_field)

    # Search for lake .json
    lake_json = os.path.join(save_dir, 'mod_dissolve_lakes.json')
    if os.path.exists(lake_json):
        with open(lake_json, 'r') as f:
            dissolve_lake_dict = json.load(f)
        ids_to_ignore = set(dissolve_lake_dict.keys())
    else:
        ids_to_ignore = set()
    ids_to_ignore.update(set(lake_table['outlet']))

    nexus_list = []

    # We will ignore headwaters
    ids_to_consider = set(gdf[gdf['strmOrder'] > min_strm_order][id_field]) - ids_to_ignore
    geom_dict = gdf.set_index(id_field)['geometry'].to_dict()
    strm_order_dict = gdf.set_index(id_field)['strmOrder'].to_dict()
    for node in ids_to_consider:
        # Get upstreams
        upstreams = list(G.predecessors(node))

        # Only use streams with 3+ connections
        if len(upstreams) < 2:
            continue

        # Don't use lake streams
        if len(upstreams) > 2:
            continue
        
        # Get first point of the downstream stream - note that if you change the order in which geometries are dissolved (merging small k streams), this will need to be updated
        downstream_geom = geom_dict[node]
        if isinstance(downstream_geom, MultiLineString):
            downstream_geom = downstream_geom.geoms[-1] # Last index is most upstream
        downstream_coords = downstream_geom.coords[-1]
        lat, lon = downstream_coords[1], downstream_coords[0]

        # Get strahler order
        strahler_order = strm_order_dict[node]
        nexus_list.append([lat, lon, node, strahler_order, ",".join(map(str, upstreams)), Point(lon, lat)])

    if nexus_list:
        nexus_file = os.path.join(save_dir, 'nexus_points.gpkg')
        gpd.GeoDataFrame(nexus_list, columns=['Lat', 'Lon', 'DSLINKNO', 'DSStrahlerOrder', 'USLINKNOs', 'geometry'], crs=gdf.crs).to_file(nexus_file, index=False)
    else:
        logger.info('No Nexus Points Found')


def ancestors_safe(G: nx.DiGraph, node: int) -> set:
    try:
        return nx.ancestors(G, node)
    except nx.NetworkXError:
        return set()
    
if ENGINE == 'numba':
    def uscont_helper(values: np.ndarray, index):
        return values[:-1].sum() if len(values) > 1 else values[0]
else:
    def uscont_helper(values: pd.Series):
        return values.iloc[:-1].sum() if len(values) > 1 else values.iloc[0]

def dissolve_branches(sgdf: gpd.GeoDataFrame,
                      head_to_dissolve: pd.DataFrame,
                      geometry_diss: bool = False,
                      k_agg_func: Union[types.FunctionType, str] = 'last', ) -> pd.DataFrame:
    """
    Use pandas groupby to "dissolve" streams in the table by combining rows and handle metadata correctly

    Args:
        sgdf: streams geodataframe with all the metadata columns from the source files
        head_to_dissolve: dataframe with the values of streams to be dissolved and the ID they are dissolved to
        geometry_diss: a boolean to determine if the geometry should be dissolved
        k_agg_func: a string or function to use in groupby to handle combining the k_agg column

    Returns:
        a copy of the streams geodataframe with rows dissolved
    """
    logger.info('\tDissolving headwater streams in inputs master')
    dissolve_map = {stream: streams[0] for streams in head_to_dissolve.values for stream in streams}
    sgdf['LINKNO'] = sgdf['LINKNO'].map(lambda x: dissolve_map.get(x, x))

    # This is the fastest way to groupby and dissolve
    return dissolve(sgdf, geometry_diss=geometry_diss)

def dissolve(sgdf: gpd.GeoDataFrame,
             geometry_diss: bool = False,):
    
    groups = sgdf.groupby('LINKNO')
    
    # This is the fastest way to groupby and dissolve
    if 'musk_k' in sgdf.columns:
        return (gpd.GeoDataFrame({'LINKNO': groups['LINKNO'].first(),
                                'DSLINKNO': groups['DSLINKNO'].last(),
                                'strmOrder': groups['strmOrder'].last(),
                                'Magnitude': groups['Magnitude'].last(),
                                'DSContArea': groups['DSContArea'].last(),
                                'USContArea': groups['USContArea'].agg(uscont_helper, engine=ENGINE),
                                'LengthGeodesicMeters': groups['LengthGeodesicMeters'].last(),
                                'TDXHydroRegion': groups['TDXHydroRegion'].last(),
                                'TopologicalOrder': groups['TopologicalOrder'].last(),
                                'musk_k': groups['musk_k'].last(),
                                'musk_x': groups['musk_x'].last(),
                                'geometry': sgdf.dissolve(by='LINKNO').geometry if geometry_diss else groups['geometry'].last()}) # Default values are fastest
                .reset_index(drop=True)
                .sort_values('TopologicalOrder')
                )
    else:
        return (gpd.GeoDataFrame({'LINKNO': groups['LINKNO'].first(),
                                'DSLINKNO': groups['DSLINKNO'].last(),
                                'strmOrder': groups['strmOrder'].last(),
                                'Magnitude': groups['Magnitude'].last(),
                                'DSContArea': groups['DSContArea'].last(),
                                'USContArea': groups['USContArea'].agg(uscont_helper, engine=ENGINE),
                                'LengthGeodesicMeters': groups['LengthGeodesicMeters'].last(),
                                'TDXHydroRegion': groups['TDXHydroRegion'].last(),
                                'TopologicalOrder': groups['TopologicalOrder'].last(),
                                'geometry': sgdf.dissolve(by='LINKNO').geometry if geometry_diss else groups['geometry'].last()}) # Default values are fastest
                .reset_index(drop=True)
                .sort_values('TopologicalOrder')
                )

def prune_branches(sdf: pd.DataFrame, streams_to_prune: pd.DataFrame) -> pd.DataFrame:
    return sdf[~sdf['LINKNO'].isin(streams_to_prune.iloc[:, 1].values.flatten())]


def dissolve_short_streams(sgdf: gpd.GeoDataFrame, short_streams: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Dissolve short streams that are not connected to a confluence

    Args:
        sgdf: stream network geodataframe

    Returns:
        geodataframe
    """
    logger.info('\tDissolving short streams')
    sgdf = sgdf.reset_index(drop=True) # Allows us to use index-based lookup, faster
    rows_to_drop = set()

    for idx, (river_id_to_keep, river_id_to_drop) in short_streams.iterrows():
        row_to_keep: gpd.GeoSeries = sgdf['LINKNO'] == river_id_to_keep
        row_to_drop: gpd.GeoSeries = sgdf['LINKNO'] == river_id_to_drop

        # if both ids are not in the dataframe then skip. Some rivers will get merged together first
        if not row_to_keep.any() or not row_to_drop.any():
            continue

        keep_row: gpd.GeoSeries = sgdf.loc[row_to_keep].squeeze()  # Extract the row as a series
        drop_row: gpd.GeoSeries = sgdf.loc[row_to_drop].squeeze()
        index = keep_row.name

        combined_geometry = keep_row['geometry'].union(drop_row['geometry'])

        sgdf.at[index, 'geometry'] = combined_geometry
        sgdf.at[index, 'USContArea'] = min(keep_row['USContArea'], drop_row['USContArea'])
        sgdf.at[index, 'DSContArea'] = max(keep_row['DSContArea'], drop_row['DSContArea'])
        sgdf.at[index, 'LengthGeodesicMeters'] = keep_row['LengthGeodesicMeters'] + drop_row['LengthGeodesicMeters']
 
        rows_to_drop.add(river_id_to_drop)
        sgdf.loc[sgdf['DSLINKNO'] == river_id_to_drop, 'DSLINKNO'] = river_id_to_keep

    sgdf = sgdf[~sgdf['LINKNO'].isin(rows_to_drop)]
    return gpd.GeoDataFrame(sgdf.reset_index(drop=True))


def routing_configs(sdf: pd.DataFrame,
                     save_dir: str,
                     id_field: str = 'LINKNO',
                     ds_id_field: str = 'DSLINKNO', ) -> None:
    """
    Create river route input files from a stream network dataframe

    Produces the following files:
        - connectivity.parquet
        - routing_parameters.parquet

    Args:
        sdf: stream network dataframe
        save_dir: directory to save the regions outputs
        id_field: the field in the dataframe that contains the unique ID for each stream
        ds_id_field: the field in the dataframe that contains the unique ID for the downstream stream

    Returns:
        None

    """
    logger.info('\tWriting Connectivity Parquet')
    (
        sdf
        [[id_field, ds_id_field]]
        .rename(columns={id_field: 'river_id', ds_id_field: 'ds_river_id'})
        .to_parquet(os.path.join(save_dir, 'connectivity.parquet'))
    )

    logger.info('\tWriting Routing Parameters')
    (
        sdf
        [[id_field, 'musk_x', 'musk_k']]
        .rename(columns={id_field:'river_id', 'musk_x':'x', 'musk_k':'k'})
        .to_parquet(os.path.join(save_dir, 'routing_parameters.parquet'))
    )
    
    return

def concat_tdxregions(tdxinputs_dir: str, vpu_assignment_table: str, master_table_path: str, original_streams: list[str] = None) -> None:
    """
    If the VPU code is not in the master table, this function will attempt to find the VPU code for each terminal node
    by looking at the VPU code of the downstream terminal node (if og_pqs is not None). If the downstream terminal node does not have a VPU code,
    the function will look at the downstream terminal node of the downstream terminal node. This will continue until a VPU
    code is found or the terminal node is not in the VPU table. If the terminal node is not in the VPU table, the terminal
    node will be written to a csv file and the function will raise an error.
    """
    mdf = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(tdxinputs_dir, '*', 'rr_inputs*.parquet'))])
    vpu_df = pd.read_csv(vpu_assignment_table)
    mdf = mdf.merge(vpu_df, on='TerminalLink', how='left')

    if not mdf[mdf['VPUCode'].isna()].empty:
        if original_streams is not None:
            gdf = dd.read_parquet(original_streams, columns=['LINKNO', 'DSLINKNO']).compute()
            G = create_directed_graphs(gdf, 'LINKNO', ds_id_field='DSLINKNO')

            mapping = {terminal_id: vpu for terminal_id, vpu in vpu_df.values}
            mdf['VPUCode'] = mdf['TerminalLink'].map(mapping).fillna(mdf['VPUCode'])

            mapping = {}
            vpu_dict = vpu_df.set_index('TerminalLink')['VPUCode'].to_dict() # Allows for quick lookup
            for terminal_id in mdf[mdf['VPUCode'].isna()]['TerminalLink'].values:
                downstreams = list(G.successors(terminal_id))
                while downstreams:
                    downstream = downstreams.pop()
                    if downstream in vpu_dict:
                        mapping[terminal_id] = vpu_dict[downstream]
                        break
                    downstreams = list(G.successors(downstream)) 
                else:
                    raise RuntimeError(f"Could not find VPU for terminal node {terminal_id}")
                
            mdf['VPUCode'] = mdf['TerminalLink'].map(mapping).fillna(mdf['VPUCode'])
        else:
            mdf[mdf['VPUCode'].isna()].to_csv(os.path.join(os.path.dirname(master_table_path), 'missing_vpu_label.csv'))
            raise RuntimeError('Some terminal nodes are not in the VPU table and must be fixed before continuing.')
    mdf['VPUCode'] = mdf['VPUCode'].astype(int)

    mdf[[
        'LINKNO',
        'DSLINKNO',
        'strmOrder',
        'USContArea',
        'DSContArea',
        'TDXHydroRegion',
        'VPUCode',
        'TopologicalOrder',
        'LengthGeodesicMeters',
        'TerminalLink',
        'musk_k',
        'musk_x',
    ]].to_parquet(master_table_path)
    return

def vpu_files_from_masters(vpu_df: pd.DataFrame,
                           vpu_dir: str,
                           tdxinputs_directory: str,
                           make_gpkg: bool,
                           gpkg_dir: str, ) -> None:
    tdx_region = vpu_df['TDXHydroRegion'].values[0]
    vpu = vpu_df['VPUCode'].values[0]

    # make the routing configs
    routing_configs(vpu_df, vpu_dir)

    # subset the weight tables
    logging.info('\tSubsetting weight tables')
    weight_tables = glob.glob(os.path.join(tdxinputs_directory, tdx_region, f'weight*.parquet'))
    weight_tables = [x for x in weight_tables if '_full.parquet' not in x]
    for weight_table in weight_tables:
        a = pd.read_parquet(weight_table)
        a = a[a.iloc[:, 0].astype(int).isin(vpu_df['LINKNO'].values)]
        a.to_parquet(os.path.join(vpu_dir, os.path.basename(weight_table)), index=False)

    if not make_gpkg:
        return
    logging.info('\tMaking gpkg')
    altered_network = os.path.join(tdxinputs_directory, tdx_region, f'{tdx_region}_altered_network.geoparquet')
    vpu_network = os.path.join(gpkg_dir, f'streams_{vpu}.gpkg')
    if os.path.exists(altered_network):
        (
            gpd.read_parquet(altered_network)
            .merge(vpu_df, on='LINKNO', how='inner')        
            .set_crs('epsg:4326')
            .to_crs('epsg:3857')
            .to_file(vpu_network, driver='GPKG')
        )  

    return

def nexus_file_from_masters(vpu_boundaries: gpd.GeoDataFrame, 
                            vpu: int,
                            gpkg_dir: str,
                            nexus_region_file: str, ) -> None:
    # Create an enevlope of the current VPU
    vpu_boundary = vpu_boundaries[vpu_boundaries['VPU'] == str(vpu)]
    nexus_file = os.path.join(gpkg_dir, f'nexus_{vpu}.gpkg')
    nexus_df = gpd.read_file(nexus_region_file)

    # Filter the nexus points to the VPU boundary
    vpu_bounds = vpu_boundary.total_bounds
    nexus_df = nexus_df.cx[vpu_bounds[0]:vpu_bounds[2], vpu_bounds[1]:vpu_bounds[3]]
    nexus_df: dgpd.GeoDataFrame = dgpd.from_geopandas(nexus_df, npartitions=estimate_num_partition(nexus_df))
    nexus_df = nexus_df[nexus_df.within(vpu_boundary.geometry.values[0])].compute()

    if nexus_df.empty:
        return
    
    logging.info('\tMaking nexus points')
    (
        nexus_df
        .to_crs('epsg:3857')
        .to_file(nexus_file, driver='GPKG')
    )


def make_vpu_gpkg(df: gpd.GeoDataFrame, vpu_network: str) -> None:
    (
        df            
        .set_crs('epsg:4326')
        .to_crs('epsg:3857')
        .to_file(vpu_network, driver='GPKG')
    )  


def _k_agg_order_3(x: pd.Series) -> np.ndarray:
    return x.mean() * 3.5 if len(x) > 1 else x.iloc[0]


def _k_agg_order_2(x: pd.Series) -> np.ndarray:
    return x.iloc[-1] + x.iloc[:-1].max() if len(x) > 1 else x.iloc[0]


def _geom_diss(x: Union[pd.Series, gpd.GeoSeries]):
    return gpd.GeoSeries(x).unary_union


def _get_tdxhydro_header_number(region_number: int) -> int:
    with open(os.path.join(os.path.dirname(__file__), 'network_data', 'tdx_header_numbers.json')) as f:
        header_numbers = json.load(f)
    return int(header_numbers[str(region_number)])
