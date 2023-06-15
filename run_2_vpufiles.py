import tdxhydrorapid as rp

save_dir = '/Volumes/EB406_T7_2/GEOGLOWS2'
vpu_fixes_csv = 'vpu_fixes_2.csv'

inputs_directory = '/Volumes/EB406_T7_2/TDXHydroRapid_V11'
final_inputs_directory = '/Volumes/EB406_T7_2/GEOGLOWS2/rapid_files'

# ---------- STEP 2 ----------
rp.inputs.fix_vpus(inputs_directory, final_inputs_directory, vpu_fixes_csv)
rp.inputs.rapid_csvs_final(final_inputs_directory, inputs_directory)
