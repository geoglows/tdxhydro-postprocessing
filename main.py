import os
import json
from AdjoinUpdown import create_adjoint_dict
import pandas as pd
import geopandas as gpd

networkdir = "/Users/taylormiskin/PycharmProjects/StreamDataBase/Carribean"

outfile = "output_json/carribean_order_2.json"
if not os.path.exists(outfile):
    order_2_dict = create_adjoint_dict(networkdir, outfile, stream_id_col="LINKNO", next_down_id_col="DSLINKNO",
                                       order_col="strmOrder", order_filter=2)
else:
    f = open(outfile)
    order_2_dict = json.load(f)


toporder2 = []
for value in list(order_2_dict.values()):
    toporder2.append(value[-1])

toporder2 = set(toporder2)
print(toporder2)

