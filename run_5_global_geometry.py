# import dask_geopandas as dgpd
from typing import Union

import geopandas as gpd
import pandas as pd
import glob
import tqdm

from shapely.geometry import MultiLineString, LineString

def simplify_lakes(geom: Union[MultiLineString, LineString],) -> gpd.GeoSeries:
    """
    The normal simplify will not modify the dissolved lake MultiLineStrings
    This function will handle both cases
    """
    if isinstance(geom, MultiLineString) and max({len(g.coords) for g in geom.geoms}) < 3:
        # Ignore headwaters and other merged streams
        all_lines = []
        new_lines = []
        start = True
        geoms = geom.geoms 
        length = len(geoms) 

        for i, line in enumerate(geoms):
            coords = line.coords
            if start:
                new_lines.extend([coords[0], coords[-1]])
                start = False
                continue

            # If current line connects to previous line
            if coords[0] == new_lines[-1]:
                # If current line connects to next line
                new_lines.extend([coords[0], coords[-1]])
                
                if i >= length - 1 or  coords[-1] != geoms[i+1].coords[0]:
                    # This is the end of the line
                    start = True
                    all_lines.append(LineString(new_lines))
                    new_lines = []
            else:
                # This is a new start
                start = False
                all_lines.append(LineString(new_lines))
                new_lines = [coords[0], coords[-1]]

        return MultiLineString(all_lines).simplify(.001, preserve_topology=False)

    return geom.simplify(.001, preserve_topology=False)

files = glob.glob(r"D:\geoglows_v3\geoglows_v3\hydrography-tdxhydro-sources\alterations\*\*.geoparquet")   
gdfs = []
for file in tqdm.tqdm(files):
    gdf = gpd.read_parquet(file, columns=['LINKNO', 'geometry'],)
    gdf['geometry'] = gdf['geometry'].apply(simplify_lakes)
    gdfs.append(gdf)

gdf: gpd.GeoDataFrame = pd.concat(gdfs, ignore_index=True)
gdf.to_file(r'D:\geoglows_v3\geoglows_v3\hydrography_global\global_streams_simplified.gpkg', driver='GPKG')
# files = glob.glob(r"D:\geoglows_v3\geoglows_v3\hydrography-tdxhydro-sources\alterations\*\*.geoparquet")
# dgdf: dgpd.GeoDataFrame = dgpd.read_parquet(files, filesystem="arrow", columns=['LINKNO', 'geometry'])
# dgdf['geometry'] = dgdf['geometry'].simplify(.001, preserve_topology=False)
# dgdf.to_file(r'D:\geoglows_v3\geoglows_v3\hydrography_global\global_geometry.gpkg', driver='GPKG')
