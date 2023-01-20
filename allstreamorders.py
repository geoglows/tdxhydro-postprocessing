import os
import json
from AdjoinUpdown import create_adjoint_dict
import pandas as pd
import geopandas as gpd

networkdir = "/Users/taylormiskin/PycharmProjects/StreamDataBase/Carribean"

outfile = "/Users/taylormiskin/PycharmProjects/StreamDataBase/output_json/carribean_allorders.json"
if not os.path.exists(outfile):
    all_orders_dict = create_adjoint_dict(networkdir, outfile, stream_id_col="LINKNO", next_down_id_col="DSLINKNO",
                                       order_col="strmOrder")
else:
    f = open(outfile)
    all_orders_dict = json.load(f)
