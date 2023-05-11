import rapidprep as rp
import glob
import os

outputs_path = '/Users/rchales/Data/TDXHydroRapid_V6'
outputs_path = '/Volumes/EB406_T7_2/tdxrapid'

for directory in sorted(glob.glob(os.path.join(outputs_path, '*'))):
    if os.path.isdir(directory):
        rp.count_rivers_in_generated_files(directory)
