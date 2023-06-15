import tdxhydrorapid as rp

# ---------- STEP 3 ----------
save_dir = '/Volumes/EB406_T7_2/GEOGLOWS2'
inputs_directory = '/Volumes/EB406_T7_2/TDXHydroRapid_V11'
final_inputs_directory = '/Volumes/EB406_T7_2/GEOGLOWS2'
gpq_dir = '/Volumes/EB406_T7_2/TDXHydroGeoParquet'
id_field = 'LINKNO'
basin_id_field = 'streamID'

# todo
# rp.inputs.make_vpu_streams()
rp.network.make_vpu_basins(final_inputs_directory, gpq_dir, id_field, basin_id_field)
