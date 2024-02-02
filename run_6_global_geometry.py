import geopandas as gpd
import pandas as pd
import glob
import os
import natsort


gpqs = natsort.natsorted(glob.glob('/Volumes/T9Hales4TB/geoglows2/tdxhydro-inputs/*/*altered_network.geoparquet'))
gdf = pd.concat([gpd.read_parquet(gpq) for gpq in gpqs])
gdf = gpd.GeoDataFrame(gdf, crs='epsg:4326', geometry='geometry')
gdf['geometry'] = gdf['geometry'].simplify(0.001, preserve_topology=False)
gdf[['LINKNO', 'geometry']].to_file('/Volumes/T9Hales4TB/geoglows2/global_streams_simplified.gpkg', driver='GPKG')
