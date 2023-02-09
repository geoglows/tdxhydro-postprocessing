import os
import json
from AdjoinUpdown import create_adjoint_dict
import pandas as pd
import geopandas as gpd

region = 'japan'
networkshp = "Japan/TDX_streamnet_4020034510_01.shp"

outfile = f"output_jsons/{region}_allorders.json"
if not os.path.exists(outfile):
    all_orders_dict = create_adjoint_dict(networkshp, outfile, stream_id_col="LINKNO", next_down_id_col="DSLINKNO",
                                       order_col="strmOrder")
else:
    f = open(outfile)
    all_orders_dict = json.load(f)
