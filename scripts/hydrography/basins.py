from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely.geometry as sg

id_field = 'riverId'
next_id_field = 'nextRiverId'


def correct_basins(basins_gpq: str,
                   save_dir: str,
                   stream_id_col: str, ) -> gpd.GeoDataFrame:
    """
    Apply fixes to streams that have 0 length.

    Args:
        basins_gpq: Basins to correct
        save_dir: Directory to save the corrected basins to
        stream_id_col:

    Returns:

    """
    save_dir = Path(save_dir)
    basin_gdf = gpd.read_parquet(basins_gpq)
    basin_gdf = basin_gdf.set_index(stream_id_col)

    zero_fix_csv_path = save_dir / 'mod_basin_zero_centroid.csv'
    if zero_fix_csv_path.exists():
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
        }, crs=basin_gdf.crs).set_index(stream_id_col)
        basin_gdf = pd.concat([basin_gdf, link_zero_box])

    zero_length_csv_path = save_dir / 'mod_zero_length_streams.csv'
    if zero_length_csv_path.exists():
        log.info('\tRevising basins with 0 length streams')
        zero_length_json = pd.read_csv(zero_length_csv_path)
        # Case 1 - Coastal w/ no upstream or downstream - Delete the stream and its basin
        log.info('\tHandling Case 1 0 Length Streams - delete basins')
        basin_gdf = basin_gdf[~basin_gdf.index.isin(zero_length_json['case1'])]
        # Case 2 - Allow 3-river confluence - basin does not exist (try to delete just in case)
        log.info('\tHandling Case 2 0 Length Streams - delete basins')
        basin_gdf = basin_gdf[~basin_gdf.index.isin(zero_length_json['case2'])]
        # Case 3 - Coastal w/ upstreams but no downstream - basin exists so delete it
        log.info('\tHandling Case 3 0 Length Streams - delete basins')
        basin_gdf = basin_gdf[~basin_gdf.index.isin(zero_length_json['case3'])]

    small_tree_csv_path = save_dir / 'mod_drop_small_trees.csv'
    if small_tree_csv_path.exists():
        log.info('\tDeleting small trees')
        small_tree_df = pd.read_csv(small_tree_csv_path)
        basin_gdf = basin_gdf[~basin_gdf.index.isin(small_tree_df.values.flatten())]

    within_sea_streams_path = save_dir / 'mod_drop_within_sea.csv'
    if within_sea_streams_path.exists():
        log.info('\tDeleting basins within the sea')
        within_sea_streams_df = pd.read_csv(within_sea_streams_path)
        basin_gdf = basin_gdf[~basin_gdf.index.isin(within_sea_streams_df.values.flatten())]

    drop_ocean_watersheds_path = save_dir / 'mod_drop_ocean_watersheds.csv'
    if drop_ocean_watersheds_path.exists():
        log.info('\tDeleting small ocean watersheds')
        drop_ocean_watersheds_df = pd.read_csv(drop_ocean_watersheds_path)
        basin_gdf = basin_gdf[~basin_gdf.index.isin(drop_ocean_watersheds_df.values.flatten())]

    drop_lonely_streams_path = save_dir / 'mod_drop_lonely_streams.csv'
    if drop_lonely_streams_path.exists():
        log.info('\tDeleting lonely streams')
        drop_lonely_streams_df = pd.read_csv(drop_lonely_streams_path)
        basin_gdf = basin_gdf[~basin_gdf.index.isin(drop_lonely_streams_df['drop'].values.flatten())]

    drop_islands_path = save_dir / 'mod_drop_islands.csv'
    if drop_islands_path.exists():
        log.info('\tDeleting islands')
        drop_islands_df = pd.read_csv(drop_islands_path)
        basin_gdf = basin_gdf[~basin_gdf.index.isin(drop_islands_df.values.flatten())]

    basin_gdf = basin_gdf.reset_index()
    return basin_gdf
