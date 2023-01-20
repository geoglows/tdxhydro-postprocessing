from AdjointCatchments.AdjoinUpdown import create_adjoint_dict

network_gpkg = '/Users/rchales/Data/NGA_delineation/Caribbean/TDX_streamnet_7020065090_01.shp'

create_adjoint_dict(network_gpkg,
                    "./streamlink_json/carribean_allorders.json",
                    stream_id_col="LINKNO",
                    next_down_id_col="DSLINKNO",
                    order_col="strmOrder", )

create_adjoint_dict(network_gpkg,
                    "./streamlink_json/carribean_order_2.json",
                    stream_id_col="LINKNO",
                    next_down_id_col="DSLINKNO",
                    order_col="strmOrder",
                    order_filter=2)
