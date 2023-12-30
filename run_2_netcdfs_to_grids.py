import glob
import logging
import sys

import tdxhydrorapid as rp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

ncs_to_process = glob.glob('./era5_sample_grids/*.nc')

print('Converting Sample Runoff Grids to Thiessen Geoparquet')

for nc in ncs_to_process:
    rp.weights.make_thiessen_grid_from_netcdf_sample(nc, '/Volumes/T9Hales4TB/RunoffSampleGrids')
