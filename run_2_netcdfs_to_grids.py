import glob

import tdxhydrorapid as rp

ncs_to_process = glob.glob('./era5_sample_grids/ifs*.nc')

for nc in ncs_to_process:
    rp.weights.make_thiessen_grid_from_netcdf_sample(nc, './era5_thiessen_grid_parquets')
