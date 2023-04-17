import glob
import logging
import os
from multiprocessing import Pool

import geopandas as gpd
import pandas as pd
from pyproj import Geod

# set up logging
logger = logging.getLogger(__name__)


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


def create_riv_bas_id(streams_gdf: gpd.GeoDataFrame, out_dir: str, id_field: str) -> None:
    """
    Creates riv_bas_id.csv. Network is sorted to match the outputs of the ArcGIS tool this was designed from,
    and it is likely that the second element in the list for ascending may be True without impacting RAPID

    Args:
        streams_gdf (gpd.GeoDataFrame): GeoDataFrame of the streams
        out_dir (str): Path to directory where riv_bas_id.csv will be saved
        id_field (str): Field in streams_gdf that corresponds to the unique id of each stream segment

    Returns:
        None
    """
    logger.info("Creating riv_bas_id.csv")
    streams_gdf[id_field].to_csv(os.path.join(out_dir, "riv_bas_id.csv"), index=False, header=False)
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


def create_rapid_connect(streams_gdf: gpd.GeoDataFrame,
                         save_dir: str, id_field: str,
                         n_workers: int or None = 1) -> None:
    """
    Creates rapid_connect.csv

    rapid_connect is a csv file with no header or index column that contains the following columns:
        HydroID: the HydroID of the stream
        NextDownID: the HydroID of the next downstream stream
        CountUpstreamID: the number of upstream streams
        UpstreamID1: the HydroID of the first upstream segment, if it exits
        UpstreamID2: the HydroID of the second upstream segment, if it exits

    Args:
        streams_gdf: GeoDataFrame of the streams
        save_dir: Path to directory where rapid_connect.csv will be saved
        id_field: Field in streams_gdf that corresponds to the unique id of each stream segment
        n_workers: Number of workers to use for multiprocessing. If None, then all available workers will be used.

    Returns:

    """
    logger.info("Creating rapid_connect.csv")

    with Pool(n_workers) as p:
        rapid_connect = p.starmap(_make_rapid_connect_row, [[x, streams_gdf] for x in streams_gdf[id_field].values])

    rapid_connect = (
        pd
        .DataFrame(rapid_connect)
        .fillna(0)
        .astype(int)
    )
    rapid_connect.to_csv(os.path.join(save_dir, 'rapid_connect.csv'), index=False, header=None)
    return


def prepare_rapid_inputs(save_dir: str,
                         id_field: str = 'LINKNO',
                         order_field: str = 'strmOrder',
                         default_k: float = 0.35, default_x: float = 3,
                         n_workers: int or None = 1) -> None:
    # Create rapid preprocessing files
    logger.info('Creating RAPID files')
    streams_gdf = gpd.read_file(glob.glob(os.path.join(save_dir, 'TDX_streamnet*_model.gpkg'))[0])
    streams_gdf = streams_gdf.sort_values([order_field, id_field], ascending=[True, True])
    create_comid_lat_lon_z(streams_gdf, save_dir, id_field=id_field)
    create_riv_bas_id(streams_gdf, save_dir, id_field=id_field)
    calculate_muskingum(streams_gdf, save_dir, k=default_k, x=default_x)
    create_rapid_connect(streams_gdf, save_dir, id_field=id_field, n_workers=n_workers)
    return
