import json
import logging
import os
import warnings
from itertools import chain
from multiprocessing import Pool

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.ops
import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    'make_weight_table',
    'apply_mods_to_wt'
]


def make_weight_table(lsm_sample: str,
                      out_dir: str,
                      basins_gdf: gpd.GeoDataFrame,
                      basin_id_field: str = 'TDXHydroLinkNo') -> None:
    out_name = os.path.join(out_dir, 'weight_' + os.path.basename(os.path.splitext(lsm_sample)[0]) + '_full.csv')
    if os.path.exists(os.path.join(out_dir, out_name)):
        logger.info(f'Weight table already exists: {os.path.basename(out_name)}')
        return
    logger.info(f'Creating weight table: {os.path.basename(out_name)}')

    # Extract xs and ys dimensions from the ds
    lsm_ds = xr.open_dataset(lsm_sample)
    x_var = [v for v in lsm_ds.variables if v in ('lon', 'longitude',)][0]
    y_var = [v for v in lsm_ds.variables if v in ('lat', 'latitude',)][0]
    xs = lsm_ds[x_var].values
    ys = lsm_ds[y_var].values
    lsm_ds.close()

    # get the resolution of the ds
    resolution = np.abs(xs[1] - xs[0])

    # correct irregular x coordinates
    xs[xs > 180] = xs[xs > 180] - 360

    all_xs = xs.copy()
    all_ys = ys.copy()

    # buffer the min/max in case any basins are close to the edges
    x_min, y_min, x_max, y_max = basins_gdf.total_bounds
    x_min = x_min - resolution
    x_max = x_max + resolution
    y_min = y_min - resolution
    y_max = y_max + resolution

    # find the indexes of the bounding box
    x_min_idx = np.argmin(np.abs(xs - x_min))
    x_max_idx = np.argmin(np.abs(xs - x_max))
    y_min_idx = np.argmin(np.abs(ys - y_min))
    y_max_idx = np.argmin(np.abs(ys - y_max))

    if x_min_idx > x_max_idx:
        xs = np.concatenate((xs[x_min_idx:], xs[:x_max_idx + 1]))
    else:
        xs = xs[x_min_idx:x_max_idx + 1]
    y_min_idx, y_max_idx = min(y_min_idx, y_max_idx), max(y_min_idx, y_max_idx)
    ys = ys[y_min_idx:y_max_idx + 1]

    # create thiessen polygons around the 2d array centers and convert to a geodataframe
    x_grid, y_grid = np.meshgrid(xs, ys)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()

    # Create Thiessen polygon based on the point feature
    # the order of polygons in the voronoi diagram is **guaranteed not** the same as the order of the input points
    logging.info('\tCreating Thiessen polygons')
    regions = shapely.ops.voronoi_diagram(
        shapely.geometry.MultiPoint(
            [shapely.geometry.Point(x, y) for x, y in zip(x_grid, y_grid)]
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.info('\tadding metadata to voronoi polygons gdf')
        # create a geodataframe from the voronoi polygons
        tg_gdf = gpd.GeoDataFrame(geometry=[region for region in regions.geoms], crs=4326)
        tg_gdf['lon'] = tg_gdf.geometry.apply(lambda x: x.centroid.x).astype(float)
        tg_gdf['lat'] = tg_gdf.geometry.apply(lambda y: y.centroid.y).astype(float)
        tg_gdf['lon_index'] = tg_gdf['lon'].apply(lambda x: np.argmin(np.abs(all_xs - x)))
        tg_gdf['lat_index'] = tg_gdf['lat'].apply(lambda y: np.argmin(np.abs(all_ys - y)))

    intersections = gpd.overlay(tg_gdf, basins_gdf, how='intersection')
    intersections['area_sqm'] = intersections.geometry.to_crs({'proj': 'cea'}).area

    intersections.loc[intersections[basin_id_field].isna(), basin_id_field] = 0

    logger.info('\tcalculating number of points')
    intersections['npoints'] = intersections.groupby(basin_id_field)[basin_id_field].transform('count')

    logger.info('\twriting weight table csv')
    (
        intersections[[basin_id_field, 'area_sqm', 'lon_index', 'lat_index', 'npoints', 'lon', 'lat']]
        .sort_values([basin_id_field, 'area_sqm'])
        .to_csv(out_name, index=False)
    )
    return


def _merge_weight_table_rows(wt: pd.DataFrame, new_key: str, values_to_merge: list):
    new_row = wt.loc[wt['streamID'].isin(values_to_merge)]
    new_row = new_row.groupby(['lon_index', 'lat_index', 'lon', 'lat']).sum().reset_index()
    new_row['streamID'] = int(new_key)
    new_row['npoints'] = new_row.shape[0]
    return new_row


def apply_mods_to_wt(wt_path: str, save_dir: str, n_processes: int = None):
    """

    Args:
        wt_path:
        save_dir:
        n_processes:

    Returns:

    """
    wt = pd.read_csv(wt_path)

    # drop the small drainage trees
    small_tree_ids = pd.read_csv(os.path.join(save_dir, 'mod_drop_small_trees.csv')).values.flatten()
    wt = wt[~wt['streamID'].isin(small_tree_ids)]

    # read rows that are dissolved headwater streams
    with open(os.path.join(save_dir, 'mod_dissolve_headwaters.json'), 'r') as f:
        diss_headwaters = json.load(f)

    # read rows that are pruned shoots
    with open(os.path.join(save_dir, 'mod_prune_shoots.json'), 'r') as f:
        pruned_shoots = json.load(f)

    all_headwaters = set(chain.from_iterable([diss_headwaters[rivid][1:] for rivid in diss_headwaters.keys()]))
    all_pruned = set(chain.from_iterable([pruned_shoots[rivid][1:] for rivid in pruned_shoots.keys()]))

    # headwater_rows = [_merge_weight_table_rows(wt, key, values) for key, values in diss_headwaters.items()]
    # pruned_rows = [_merge_weight_table_rows(wt, key, values) for key, values in pruned_shoots.items()]

    # redo the list comprehension to use a multiprocessing pool
    with Pool(n_processes) as pool:
        headwater_rows = pool.starmap(_merge_weight_table_rows,
                                      [(wt, key, values) for key, values in diss_headwaters.items()])
        pruned_rows = pool.starmap(_merge_weight_table_rows,
                                   [(wt, key, values) for key, values in pruned_shoots.items()])

    wt = wt[~wt['streamID'].isin(all_headwaters)]
    wt = wt[~wt['streamID'].isin(all_pruned)]
    wt = pd.concat([wt, *headwater_rows, *pruned_rows])

    wt.to_csv(wt_path.replace('_full.csv', '.csv'), index=False)
    return
