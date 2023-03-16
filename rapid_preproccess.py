import geopandas as gpd
import pandas as pd
import os
import numpy as np
import netCDF4 as NET

from shapely.ops import voronoi_diagram, unary_union
from shapely import Point, MultiPoint
from pyproj import Geod
from time import time

def main(main_dir: str, nc_file: str, id_field: str = 'LINKNO', downstream_field: str = 'DSLINKNO', basin_id: str = 'streamID', 
         k: float = 0.35, x: float = 3.0, overwrite: bool = False):
    """
    
    Parameters
    ----------
    main_dir : string
        The overarching directory, containing subfolders, which each contain two geopackages, one with 'model' in its name and another with 'basin' in its name
    nc_file : string
        Path to the nc file.
    id_field : string, optional
        Field in network file that corresponds to the unique id of each stream segment
    downstream_field : string, optional
        Field in network file that corresponds to the unique downstream id of each stream segment
    basin_id : string, optional
        Field in basins file that corresponds to the unique id of each catchment

    overwrite : bool, optional
        If set to True, will overwrite existing files. The default is False.

    """
    start = time()
    job_list = []
    for file in os.listdir(main_dir):
        d = os.path.join(main_dir, file)
        if os.path.isdir(d):
            job_list.append(d)
    for folder in job_list:
        print(f'In {folder}')
        files = os.listdir(folder)
        basins_file = None
        stream_ntwk_file = None
        for file in files:
            if 'basin' in file:
                basins_file = os.path.join(folder,file)
            if 'model' in file:
                stream_ntwk_file = os.path.join(folder,file)
        if basins_file is not None and stream_ntwk_file is not None:
            out_dir = folder
            network = gpd.read_file(stream_ntwk_file)
            network.set_crs(epsg=3857, allow_override=True)

            # Some checks: 
            if not id_field in network.columns:
                raise ValueError(f"The id field {id_field} is not in the network file in {out_dir}")
            if not downstream_field in network.columns:
                raise ValueError(f"The downstream field {downstream_field} is not in the network file in {out_dir}")
            
            # Run main functions
            _CreateComidLatLonZ(network, out_dir, id_field,start)
            _CreateRivBasId(network, out_dir, downstream_field,id_field,start)
            _CalculateMuskingum(network,out_dir,k,x, id_field,start)
            _CreateRapidConnect(network, out_dir, id_field, downstream_field,start)
            _CreateWeightTable(out_dir, basins_file, nc_file, basin_id,start)
            print('\tFinished successfully!\n')
            return

        
def _CreateComidLatLonZ(network,out_dir,id_field,start):
    network.sort_values(id_field, inplace=True)
    gdf = gpd.GeoDataFrame.copy(network)
    gdf['lat'] = network.geometry.apply(lambda geom: geom.xy[1][0])
    gdf['lon'] = network.geometry.apply(lambda geom: geom.xy[0][0])
    data = {id_field: network[id_field].values,
            "lat": gdf['lat'].values,
            "lon": gdf['lon'].values,
            "z": 0}
    
    pd.DataFrame(data).to_csv(os.path.join(out_dir, "comid_lat_lon_z.csv"), index=False, header=True)
    print(f"\tCreated comid_lat_lon_z.csv at {round((time() - start) / 60 ,2)}")

def _CreateRivBasId(network, out_dir,downstream_field,id_field,start):
    network.sort_values([downstream_field, id_field], inplace=True, ascending=[False,False])
    network[id_field].to_csv(os.path.join(out_dir, "riv_bas_id.csv"), index=False, header=False)   

    print(f"\tCreated riv_bas_id.csv at {round((time() - start) / 60 ,2)}")

def _CalculateMuskingum(network,out_dir,k,x, id_field,start):
    network.sort_values(id_field, inplace=True)
    network["LENGTH_GEO"] = network.geometry.apply(_calculate_geodesic_length)
    network["Musk_kfac"] = network["LENGTH_GEO"] * 3600
    network["Musk_k"] = network["Musk_kfac"] * k
    network["Musk_x"] = x * 0.1
    network.to_file("C:\\Users\\lrr43\Desktop\\Lab\\GEOGLOWSData\\RAPID\\PythonV\\Test.gpkg")

    network["Musk_kfac"].to_csv(os.path.join(out_dir, "kfac.csv"), index=False, header=False)
    network["Musk_k"].to_csv(os.path.join(out_dir, "k.csv"), index=False, header=False)
    network["Musk_x"].to_csv(os.path.join(out_dir, "x.csv"), index=False, header=False)
    print(f"\tCreated muskingum parameters at {round((time() - start) / 60 ,2)}")

def _calculate_geodesic_length(line,start):
    geod = Geod(ellps='WGS84')
    length = geod.geometry_length(line) / 1000 # To convert to km

    # This is for the outliers that have 0 length
    if length < 0.00000000001:
        length = 0.001
    return length

def _CreateRapidConnect(network, out_dir, id_field, downstream_field,start):
    network.sort_values(id_field, inplace=True)
    list_all = []
    max_count_Upstream = 0

    for hydroid in network[id_field].values:
        # find the HydroID of the upstreams
        list_upstreamID = network.loc[network[downstream_field] == hydroid, id_field].values
        # count the total number of the upstreams
        count_upstream = len(list_upstreamID)
        if count_upstream > max_count_Upstream:
            max_count_Upstream = count_upstream
        nextDownID = network.loc[network[id_field] == hydroid, downstream_field].values[0]

        # append the list of Stream HydroID, NextDownID, Count of Upstream ID, and  HydroID of each Upstream into a larger list
        list_all.append(np.concatenate([np.array([hydroid, nextDownID, count_upstream]), list_upstreamID]))

    with open(os.path.join(out_dir, "rapid_connect.csv"), 'w') as f:
        for row in list_all:
            out = np.concatenate([row, np.array([0 for i in range(max_count_Upstream - row[2])])])
            for item in out:
                f.write(f"{item},")
            f.write('\n')

    print(f"\tCreated rapid_connect.csv at {round((time() - start) / 60 ,2)}")

def _CreateWeightTable(out_dir, basins_file, nc_file, basin_id,start):
    print(f"\tBeginning Weight Table Creation at {round((time() - start) / 60 ,2)}")
    basins_gdf = gpd.read_file(basins_file)
    if not basin_id in basins_gdf.columns:
        raise ValueError(f"The id field {basin_id} is not in the basins file in {out_dir}")
    basins_gdf.set_crs(epsg=3857, allow_override=True)

    # Obtain catchment extent
    extent = basins_gdf.total_bounds

    data_nc = NET.Dataset(nc_file)

    # Obtain geographic coordinates
    variables_list = data_nc.variables.keys()
    lat_var = 'lat'
    if 'latitude' in variables_list:
        lat_var = 'latitude'
    lon_var = 'lon'
    if 'longitude' in variables_list:
        lon_var = 'longitude'
    lon = (data_nc.variables[lon_var][:] + 180) % 360 - 180 # convert [0, 360] to [-180, 180]
    lat = data_nc.variables[lat_var][:]

    data_nc.close()

    # Create Thiessen polygons based on the points within the extent
    print("\tCreating Thiessen polygons")
    buffer = 2 * max(abs(lat[0]-lat[1]),abs(lon[0] - lon[1]))
    # Extract the lat and lon within buffered extent (buffer with 2* interval degree)
    lat0 = lat[(lat >= (extent[1] - buffer)) & (lat <= (extent[3] + buffer))]
    lon0 = lon[(lon >= (extent[0] - buffer)) & (lon <= (extent[2] + buffer))]

    # Create a list of geographic coordinate pairs
    pointGeometryList = [Point(lon, lat) for lat in lat0 for lon in lon0]

    # Create Thiessen polygon based on the point feature
    regions = voronoi_diagram(MultiPoint(pointGeometryList))

    lon_list = []
    lat_list = []

    for point in pointGeometryList:
        lon_list.append(point.x)
        lat_list.append(point.y)

    polygons_gdf = gpd.GeoDataFrame(geometry=[region for region in regions.geoms], crs='EPSG:3857')
    polygons_gdf['POINT_X'] = polygons_gdf.geometry.centroid.x
    polygons_gdf['POINT_Y'] = polygons_gdf.geometry.centroid.y
    #polygons_gdf.to_crs(epsg=4326, inplace=True)
    #polygons_gdf.to_file(os.path.join(out_dir, 'test.gpkg'), driver="GPKG")

    print("Intersecting Thiessen polygons with catchment...")
    intersect = gpd.overlay(basins_gdf, polygons_gdf, how='intersection')

    print("Calculating geodesic areas...")
    intersect['AREA_GEO'] = intersect['geometry'].to_crs({'proj':'cea'}).area 
    #intersect.to_file(os.path.join(out_dir, 'intersect.gpkg'), driver="GPKG")

    area_arr = pd.DataFrame(data={
        basin_id: intersect[basin_id].values,
        'POINT_X': intersect['POINT_X'].values,
        'POINT_Y': intersect['POINT_Y'].values,
        'AREA_GEO': intersect['AREA_GEO']
    })
    area_arr.sort_values([basin_id, 'AREA_GEO'], inplace=True, ascending=[True,False])
    connectivity_table = pd.read_csv(os.path.join(out_dir,'rapid_connect.csv'), header=None)
    streamID_unique_list = connectivity_table.iloc[:,0].astype(int).unique().tolist()

    #   If point not in array append dummy data for one point of data
    lon_dummy = area_arr['POINT_X'].iloc[0]
    lat_dummy = area_arr['POINT_Y'].iloc[0]
    try:
        index_lon_dummy = int(np.where(lon == lon_dummy)[0])
    except TypeError as _:
        index_lon_dummy = int((np.abs(lon-lon_dummy)).argmin())
        pass

    try:
        index_lat_dummy= int(np.where(lat == lat_dummy)[0])
    except TypeError as _:
        index_lat_dummy = int((np.abs(lat-lat_dummy)).argmin())
        pass

    with open(os.path.join(out_dir, 'weight_table_py2.csv'), 'w') as csvfile:
        csvfile.write(f'{basin_id},area_sqm,lon_index,lat_index,npoints,lon,lat\n')

        for streamID_unique in streamID_unique_list:
                ind_points = np.where(area_arr[basin_id]==streamID_unique)[0]
                num_ind_points = len(ind_points)
    
                if num_ind_points <= 0:
                    # if point not in array, append dummy data for one point of data
                    # streamID, area_sqm, lon_index, lat_index, npoints
                    csvfile.write(f'{streamID_unique},0,{index_lon_dummy},{index_lat_dummy},1,{lon_dummy},{lat_dummy}\n')

                else:
                    for ind_point in ind_points:
                        area_geo_each = float(area_arr['AREA_GEO'].iloc[ind_point])
                        lon_each = area_arr['POINT_X'].iloc[ind_point]
                        lat_each = area_arr['POINT_Y'].iloc[ind_point]
                        
                        try:
                            index_lon_each = int(np.where(lon == lon_each)[0])
                        except:
                            index_lon_each = int((np.abs(lon-lon_each)).argmin())
                            pass

                        try:
                            index_lat_each = int(np.where(lat == lat_each)[0])
                        except:
                            index_lat_each = int((np.abs(lat-lat_each)).argmin())
                        csvfile.write(f'{streamID_unique},{area_geo_each},{index_lon_each},{index_lat_each},{num_ind_points},{lon_each},{lat_each}\n')
    
    
    print(f"\tCreated rapid_connect.csv at {round((time() - start) / 60 ,2)}")
