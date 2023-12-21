import geopandas as gpd
import pandas as pd
import glob
import os
import natsort


gpq = natsort.natsorted(glob.glob('/Volumes/T9Hales4TB/geoglows2/tdxhydro-inputs/*/*geoparquet'))
gdf = pd.concat([gpd.read_parquet(g).simplify() for g in gpq])
print(gpq)
