import glob
import json
import logging
import os
from itertools import chain

import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box
from multiprocessing import Pool

logger = logging.getLogger(__name__)


def _intersect_reproject_area(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame):
    return (
        gpd
        .GeoSeries(a, crs='EPSG:4326')
        .to_crs({'proj': 'cea'})
        .intersection(
            gpd.GeoSeries(b, crs='EPSG:4326')
            .to_crs({'proj': 'cea'})
        )
        .area
        .values[0]
    )


def make_weight_table(lsm_sample: str, out_dir: str, basins_gdf: gpd.GeoDataFrame, n_workers: int = 1):
    out_name = os.path.join(out_dir, 'weight_' + os.path.basename(os.path.splitext(lsm_sample)[0]) + '_full.csv')
    if os.path.exists(os.path.join(out_dir, out_name)):
        logger.info(f"Weight table already exists: {os.path.basename(out_name)}")
        return
    logger.info(f"Creating weight table: {os.path.basename(out_name)}")

    # Extract xs and ys dimensions from the dataset
    lsm_ds = xr.open_dataset(lsm_sample)
    x_var = [v for v in lsm_ds.variables if v in ('lon', 'longitude',)][0]
    y_var = [v for v in lsm_ds.variables if v in ('lat', 'latitude',)][0]
    xs = lsm_ds[x_var].values
    ys = lsm_ds[y_var].values
    lsm_ds.close()

    # correct irregular x coordinates
    xs[xs > 180] = xs[xs > 180] - 360

    # create an array of the indices for x and y
    x_idxs = np.arange(len(xs))
    y_idxs = np.arange(len(ys))

    x_min, y_min, x_max, y_max = basins_gdf.total_bounds

    x_min_idx = np.argmin(np.abs(xs - x_min))
    x_max_idx = np.argmin(np.abs(xs - x_max))
    y_min_idx = np.argmin(np.abs(ys - y_min))
    y_max_idx = np.argmin(np.abs(ys - y_max))

    if x_min_idx > x_max_idx:
        xs = np.concatenate((xs[x_min_idx:], xs[:x_max_idx + 1]))
        x_idxs = np.concatenate((x_idxs[x_min_idx:], x_idxs[:x_max_idx + 1]))
    else:
        xs = xs[x_min_idx:x_max_idx + 1]
        x_idxs = x_idxs[x_min_idx:x_max_idx + 1]
    y_min_idx, y_max_idx = min(y_min_idx, y_max_idx), max(y_min_idx, y_max_idx)
    ys = ys[y_min_idx:y_max_idx + 1]
    y_idxs = y_idxs[y_min_idx:y_max_idx + 1]

    resolution = np.abs(xs[1] - xs[0])

    # create thiessen polygons around the 2d array centers and convert to a geodataframe
    x_grid, y_grid = np.meshgrid(xs, ys)
    x_idx, y_idx = np.meshgrid(x_idxs, y_idxs)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    x_idx = x_idx.flatten()
    y_idx = y_idx.flatten()
    x_left = x_grid - resolution / 2
    x_right = x_grid + resolution / 2
    y_bottom = y_grid - resolution / 2
    y_top = y_grid + resolution / 2

    def _point_to_box(row: pd.Series) -> box:
        return box(row.x_left, row.y_bottom, row.x_right, row.y_top)

    tg_gdf = dd.from_pandas(
        pd.DataFrame(
            np.transpose(np.array([x_left, y_bottom, x_right, y_top, x_idx, y_idx, x_grid, y_grid])),
            columns=['x_left', 'y_bottom', 'x_right', 'y_top', 'lon_index', 'lat_index', 'lon', 'lat']
        ),
        npartitions=n_workers
    )
    tg_gdf['geometry'] = tg_gdf.apply(lambda row: _point_to_box(row), axis=1, meta=('geometry', 'object'))
    tg_gdf = tg_gdf.compute()

    # drop the columns used for determining the bounding boxes
    tg_gdf = tg_gdf[['lon_index', 'lat_index', 'lon', 'lat', 'geometry']]
    tg_gdf = gpd.GeoDataFrame(
        tg_gdf,
        geometry='geometry',
        crs={'proj': 'latlong', 'ellps': 'WGS84', 'datum': 'WGS84', 'no_defs': True}
    ).to_crs(epsg=4326)

    # Spatial join the two dataframes using the 'intersects' predicate
    logger.info('performing spatial join')
    intersections = gpd.sjoin(tg_gdf, basins_gdf, predicate='intersects')

    logger.info('merging dataframes')
    intersections = gpd.GeoDataFrame(
        intersections
        .merge(basins_gdf[['geometry', ]], left_on=['index_right'], right_index=True, suffixes=('_tp', '_sb'))
    )

    logger.info('calculating area of intersections')
    with Pool(n_workers) as p:
        intersections['area_sqm'] = p.starmap(
            _intersect_reproject_area,
            zip(intersections['geometry_tp'], intersections['geometry_sb'])
        )

    logger.info('calculating number of points')
    intersections['npoints'] = intersections.groupby('streamID')['streamID'].transform('count')

    logger.info('writing weight table csv')
    (
        intersections[['streamID', 'area_sqm', 'lon_index', 'lat_index', 'npoints', 'lon', 'lat']]
        .sort_values(['streamID', 'area_sqm'])
        .to_csv(out_name, index=False)
    )
    return


def _merge_weight_table_rows(wt: pd.DataFrame, new_key: str, values_to_merge: list):
    new_row = wt.loc[wt['streamID'].isin(values_to_merge)]
    new_row = new_row.groupby(['lon_index', 'lat_index', 'lon', 'lat']).sum().reset_index()
    new_row['streamID'] = int(new_key)
    new_row['npoints'] = new_row.shape[0]
    return new_row


def apply_modifications(wt_path: str, save_dir: str, n_processes: int = None):
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
