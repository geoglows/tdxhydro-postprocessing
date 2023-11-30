import logging
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
import shapely.ops
import xarray as xr

logger = logging.getLogger(__name__)

__all__ = [
    'make_thiessen_grid_from_netcdf_sample',
    'make_weight_table_from_thiessen_grid',
    'make_weight_table_from_netcdf',
    'apply_weight_table_simplifications',
]


def make_thiessen_grid_from_netcdf_sample(lsm_sample: str,
                                          out_dir: str, ) -> None:
    # Extract xs and ys dimensions from the ds
    lsm_ds = xr.open_dataset(lsm_sample)
    x_var = [v for v in lsm_ds.variables if v in ('lon', 'longitude',)][0]
    y_var = [v for v in lsm_ds.variables if v in ('lat', 'latitude',)][0]
    xs = lsm_ds[x_var].values
    ys = lsm_ds[y_var].values
    lsm_ds.close()

    # correct irregular x coordinates
    xs[xs > 180] = xs[xs > 180] - 360

    all_xs = xs.copy()
    all_ys = ys.copy()

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

    # save the thiessen grid to disc
    logging.info('\tSaving Thiessen grid to disc')
    tg_gdf.to_parquet(os.path.join(out_dir, os.path.basename(lsm_sample).replace('.nc', '_thiessen_grid.parquet')))
    return


def make_weight_table_from_thiessen_grid(tg_parquet: str,
                                         out_dir: str,
                                         basins_gdf: gpd.GeoDataFrame,
                                         basin_id_field: str = 'TDXHydroLinkNo') -> None:
    out_name = os.path.join(out_dir,
                            'weight_' + os.path.basename(tg_parquet).replace('_thiessen_grid.parquet', '_full.csv'))
    if os.path.exists(os.path.join(out_dir, out_name)):
        logger.info(f'Weight table already exists: {os.path.basename(out_name)}')
        return
    logger.info(f'Creating weight table: {os.path.basename(out_name)}')

    # load the thiessen grid
    logger.info('\tloading thiessen grid')
    tg_gdf = gpd.read_parquet(tg_parquet)

    # filter the thiessen grid to only include points within the basins bounding box
    logger.info('\tfiltering thiessen grid by bounding box')
    basins_bbox = basins_gdf.total_bounds
    tg_gdf = tg_gdf.cx[basins_bbox[0]:basins_bbox[2], basins_bbox[1]:basins_bbox[3]]

    logger.info('\tcalculating intersections and areas')
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


def make_weight_table_from_netcdf(lsm_sample: str,
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


def apply_weight_table_simplifications(save_dir: str,
                                       weight_table_in_path: str,
                                       weight_table_out_path: str,
                                       basin_id_field: str = 'streamID') -> None:
    logging.info(f'Processing {weight_table_in_path}')

    wt = pd.read_csv(weight_table_in_path)

    headwater_dissolve_path = os.path.join(save_dir, 'mod_dissolve_headwater.csv')
    if os.path.exists(headwater_dissolve_path):
        o2_to_dissolve = (
            pd
            .read_csv(headwater_dissolve_path)
            .fillna(-1)
            .astype(int)
        )
        for streams_to_merge in o2_to_dissolve.values:
            wt.loc[wt[basin_id_field].isin(streams_to_merge), basin_id_field] = streams_to_merge[0]

    streams_to_prune_path = os.path.join(save_dir, 'mod_prune_streams.csv')
    if os.path.exists(streams_to_prune_path):
        ids_to_prune = (
            pd
            .read_csv(streams_to_prune_path)
            .astype(int)
            .set_index('LINKTODROP')
        )
        wt[basin_id_field] = wt[basin_id_field].replace(ids_to_prune['LINKNO'])

    drop_streams_path = os.path.join(save_dir, 'mod_drop_small_trees.csv')
    if os.path.exists(drop_streams_path):
        ids_to_drop = (
            pd
            .read_csv(drop_streams_path)
            .astype(int)
        )
        wt = wt[~wt[basin_id_field].isin(ids_to_drop.values.flatten())]

    # group by matching values in columns except for area_sqm and sum the areas in grouped rows
    wt = wt.groupby(wt.columns.drop('area_sqm').tolist()).sum().reset_index()
    wt = wt.sort_values([basin_id_field, 'area_sqm'], ascending=[True, False])
    wt['npoints'] = wt.groupby(basin_id_field)[basin_id_field].transform('count')

    wt.to_csv(weight_table_out_path, index=False)
    return
