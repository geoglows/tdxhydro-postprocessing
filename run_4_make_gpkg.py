import tdxhydrorapid as rp
import os

final_gis_dir = '/Volumes/EB406_T7_2/geoglows2/gis'
final_inputs_dir = '/Volumes/EB406_T7_2/geoglows2/inputs'
tdxrapid_dir = '/Volumes/EB406_T7_2/geoglows2/tdxhydro-inputs'

os.makedirs(final_gis_dir, exist_ok=True)

rp.network.make_final_streams(final_inputs_dir, final_gis_dir, tdxrapid_dir)
