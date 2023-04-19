import glob
import json
import logging
import os
from itertools import chain
from multiprocessing import Pool

import geopandas as gpd
import pandas as pd
from pyproj import Geod

# set up logging
logger = logging.getLogger(__name__)


def _calculate_geodesic_length(line) -> float:
    """
    Input is shapely geometry, should be all shapely LineString objects
    """
    length = Geod(ellps='WGS84').geometry_length(line) / 1000  # To convert to km

    # This is for the outliers that have 0 length
    if length < 0.0000001:
        length = 0.001
    return length


def _make_rapid_connect_row(stream_id, streams_gdf):
    id_field = 'LINKNO'
    ds_id_field = 'DSLINKNO'
    upstreams = streams_gdf.loc[streams_gdf[ds_id_field] == stream_id, id_field].values
    return {
        'HydroID': stream_id,
        'NextDownID': streams_gdf.loc[streams_gdf[id_field] == stream_id, ds_id_field].values[0],
        'CountUpstreamID': len(upstreams),
        **{f'UpstreamID{i + 1}': upstreams[i] for i in range(len(upstreams))}
    }


def _combine_routing_rows(streams_df: pd.DataFrame, id_to_preserve: str, ids_to_merge: list):
    target_row = streams_df.loc[streams_df['LINKNO'] == int(id_to_preserve)]
    if target_row.empty:
        print(id_to_preserve)
        return None
    # todo only sum muskingum parameters if they are part of the branches being kept
    musk_k, musk_kfac = streams_df.loc[streams_df['LINKNO'].isin(ids_to_merge[1:]), ['musk_k', 'musk_kfac']].sum()
    musk_x = target_row['musk_x'].values[0]
    return pd.DataFrame({
        'LINKNO': int(id_to_preserve),
        'DSLINKNO': int(target_row['DSLINKNO'].values[0]),
        'strmOrder': int(target_row['strmOrder'].values[0]),
        'musk_k': float(musk_k),
        'musk_kfac': float(musk_kfac),
        'musk_x': float(musk_x),
        'lat': float(target_row['lat'].values[0]),
        'lon': float(target_row['lon'].values[0]),
        'z': int(target_row['z'].values[0]),
    }, index=[0])


def prepare_rapid_inputs(streams_gpkg: str,
                         save_dir: str,
                         id_field: str = 'LINKNO',
                         ds_field: str = 'DSLINKNO',
                         order_field: str = 'strmOrder',
                         default_k: float = 0.35,
                         default_x: float = 3,
                         n_workers: int or None = 1) -> None:
    # Create rapid preprocessing files
    logger.info('Creating RAPID files')
    streams_gdf = gpd.read_file(streams_gpkg)
    streams_gdf = streams_gdf.sort_values([order_field, id_field], ascending=[True, True])

    logger.info('Reading stream modification files')
    with open(os.path.join(save_dir, 'mod_dissolve_headwaters.json'), 'r') as f:
        diss_headwaters = json.load(f)
        all_merged_headwater = set(chain.from_iterable([diss_headwaters[rivid] for rivid in diss_headwaters.keys()]))
    with open(os.path.join(save_dir, 'mod_prune_shoots.json'), 'r') as f:
        pruned_shoots = json.load(f)
        pruned_shoots = set([ids[-1] for _, ids in pruned_shoots.items()])
    small_trees = pd.read_csv(os.path.join(save_dir, 'mod_drop_small_trees.csv')).values.flatten()

    logger.info('Dropping small trees')
    streams_gdf = streams_gdf.loc[~streams_gdf[id_field].isin(small_trees)]
    logger.info('Dropping pruned shoots')
    streams_gdf = streams_gdf.loc[~streams_gdf[id_field].isin(pruned_shoots)]

    with Pool(n_workers) as p:
        logger.info('Calculating lengths')
        streams_gdf["LENGTH_GEO"] = p.map(_calculate_geodesic_length, streams_gdf.geometry.values)

        streams_gdf['lat'] = streams_gdf.geometry.apply(lambda geom: geom.xy[1][0]).values
        streams_gdf['lon'] = streams_gdf.geometry.apply(lambda geom: geom.xy[0][0]).values

        logger.info('Applying unit conversions to k and x')
        streams_gdf["musk_kfac"] = streams_gdf["LENGTH_GEO"] * 3600
        streams_gdf["musk_k"] = streams_gdf["musk_kfac"] * default_k
        streams_gdf["musk_x"] = default_x * 0.1
        streams_gdf['z'] = 0

        streams_gdf = streams_gdf[[id_field, ds_field, order_field, 'musk_k', 'musk_kfac', 'musk_x', 'lat', 'lon', 'z']]

        # Apply corrections based on the stream modification files
        logger.info('Merging head water stream segment rows')
        corrected_headwater_rows = p.starmap(
            _combine_routing_rows,
            [[streams_gdf, id_to_keep, ids_to_merge] for id_to_keep, ids_to_merge in diss_headwaters.items()]
        )
        logger.info('Applying corrected rows to gdf')
        streams_gdf = pd.concat([
            streams_gdf.loc[~streams_gdf[id_field].isin(all_merged_headwater)],
            *corrected_headwater_rows
        ])

        logger.info('Calculating RAPID connect file')
        rapid_connect = p.starmap(_make_rapid_connect_row, [[x, streams_gdf] for x in streams_gdf[id_field].values])

    logger.info('Writing RAPID csvs')
    (
        pd
        .DataFrame(rapid_connect)
        .fillna(0)
        .astype(int)
        .to_csv(os.path.join(save_dir, 'rapid_connect.csv'), index=False, header=None)
    )
    streams_gdf[id_field].to_csv(os.path.join(save_dir, "riv_bas_id.csv"), index=False, header=False)
    streams_gdf["musk_kfac"].to_csv(os.path.join(save_dir, "kfac.csv"), index=False, header=False)
    streams_gdf["musk_k"].to_csv(os.path.join(save_dir, "k.csv"), index=False, header=False)
    streams_gdf["musk_x"].to_csv(os.path.join(save_dir, "x.csv"), index=False, header=False)
    streams_gdf[[id_field, 'lat', 'lon', 'z']].to_csv(os.path.join(save_dir, "comid_lat_lon_z.csv"), index=False)

    return
