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
    geod = Geod(ellps='WGS84')
    length = geod.geometry_length(line) / 1000  # To convert to km

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
    target_row = streams_df.loc[streams_df['LINKNO'] == id_to_preserve]
    musk_k, musk_kfac = streams_df.loc[streams_df['LINKNO'].isin(ids_to_merge), ['musk_k', 'musk_kfac']].sum()
    musk_x = streams_df
    return pd.DataFrame({
        'LINKNO': id_to_preserve,
        'DSLINKNO': target_row['DSLINKNO'].values[0],
        'strmOrder': target_row['strmOrder'].values[0],
        'musk_k': musk_k,
        'musk_kfac': musk_kfac,
        'musk_x': musk_x,
        'lat': target_row['lat'].values[0],
        'lon': target_row['lon'].values[0],
        'z': target_row['z'].values[0],
    })


def prepare_rapid_inputs(save_dir: str,
                         id_field: str = 'LINKNO',
                         ds_field: str = 'DSLINKNO',
                         order_field: str = 'strmOrder',
                         default_k: float = 0.35,
                         default_x: float = 3,
                         n_workers: int or None = 1) -> None:
    # Create rapid preprocessing files
    logger.info('Creating RAPID files')
    streams_gdf = gpd.read_file(glob.glob(os.path.join(save_dir, 'TDX_streamnet*_model.gpkg'))[0])
    streams_gdf = streams_gdf.sort_values([order_field, id_field], ascending=[True, True])

    logger.info('Calculating lengths')
    streams_gdf["LENGTH_GEO"] = streams_gdf.geometry.apply(_calculate_geodesic_length)
    streams_gdf['lat'] = streams_gdf.geometry.apply(lambda geom: geom.xy[1][0]).values
    streams_gdf['lon'] = streams_gdf.geometry.apply(lambda geom: geom.xy[0][0]).values

    logger.info('Applying unit conversions to k and x')
    streams_gdf["musk_kfac"] = streams_gdf["LENGTH_GEO"] * 3600
    streams_gdf["musk_k"] = streams_gdf["musk_kfac"] * default_k
    streams_gdf["musk_x"] = default_x * 0.1
    streams_gdf['z'] = 0

    streams_gdf = streams_gdf[[id_field, ds_field, order_field, 'musk_k', 'musk_kfac', 'musk_x', 'lat', 'lon', 'z']]

    # Apply corrections based on the stream modification files
    logger.info('Applying stream modifications')
    with open(os.path.join(save_dir, 'mod_dissolve_headwaters.json'), 'r') as f:
        dissolved_headwaters = json.load(f)
    all_merged_headwater = set(
        chain.from_iterable([dissolved_headwaters[rivid] for rivid in dissolved_headwaters.keys()]))

    corrected_headwater_rows = [_combine_routing_rows(streams_gdf, id_to_preserve, ids_to_merge) for
                                id_to_preserve, ids_to_merge in dissolved_headwaters.items()]
    streams_gdf = pd.concat([
        streams_gdf.loc[~streams_gdf[id_field].isin(all_merged_headwater)],
        *corrected_headwater_rows
    ])

    with open(os.path.join(save_dir, 'mod_prune_shoots.json'), 'r') as f:
        pruned_shoots = json.load(f)
    pruned_shoots = set([ids[-1] for _, ids in pruned_shoots.items()])
    streams_gdf = streams_gdf.loc[~streams_gdf[id_field].isin(pruned_shoots)]

    with Pool(n_workers) as p:
        rapid_connect = p.starmap(_make_rapid_connect_row, [[x, streams_gdf] for x in streams_gdf[id_field].values])

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

    return
