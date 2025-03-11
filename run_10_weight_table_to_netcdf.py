import os
import glob

import tqdm
import xarray as xr
import pandas as pd

weight_table_parquets = sorted(glob.glob(r"D:\geoglows_v3\geoglows_v3\routing_configs\*\weight_xinit=-179.5_yinit=-89.75_dx=0.25_dy=0.25.parquet")) # era5
weight_table_parquets = sorted(glob.glob(r"D:\geoglows_v3\geoglows_v3\routing_configs\*\weight_xinit=-179.625_yinit=-59.625_dx=0.25_dy=0.25.parquet")) # GLDAS
weight_table_parquets = sorted(glob.glob(r"D:\geoglows_v3\geoglows_v3\routing_configs\*\weight_xinit=-179.859375_yinit=-89.876375435468_dx=0.0703125_dy=0.070298766768.parquet")) # ifs

original_grid = r"C:\Users\lrr43\tdxhydro-postprocessing\era5_sample_grids\era5_721x1440.nc"
original_grid = r"D:\geoglows_v3\tdx-postprocessing\era5_sample_grids\GLDAS_600x1440.nc"
original_grid = r"D:\geoglows_v3\tdx-postprocessing\era5_sample_grids\ifs_48r1_2560x5120.nc"

dataset_shortname = 'era5'
dataset_shortname = "GLDAS"
dataset_shortname = "IFS_48r1"

if dataset_shortname == 'era5':
    metadata = {"grid_dataset_name": dataset_shortname,
                "grid_spacing_units": "degrees",
                "grid_crs": "WGS84",
                "grid_x_variable": "longitude",
                "grid_y_variable": "latitude",
                "grid_doi":"https://doi.org/10.24381/cds.adbb2d47"}
elif dataset_shortname == "IFS_48r1":
    metadata = {"grid_dataset_name": dataset_shortname,
                "grid_spacing_units": "degrees",
                "grid_crs": "WGS84",
                "grid_x_variable": "lon",
                "grid_y_variable": "lat",
                "grid_doi":"https://confluence.ecmwf.int/display/FCST/Implementation+of+IFS+Cycle+48r1"}
elif dataset_shortname == "GLDAS":
    metadata = {"grid_dataset_name": dataset_shortname,
                "grid_spacing_units": "degrees",
                "grid_crs": "epsg:4326",
                "grid_x_variable": "lon",
                "grid_y_variable": "lat",
                "grid_doi":"https://doi.org/10.5067/E7TYRXPJKWOQ"}
else:
    raise ValueError("Unknown dataset shortname provided.")

with xr.open_dataset(original_grid) as grid_ds:
    # Get x_min, y_min, and resolution
    metadata['x_min'] = grid_ds[metadata["grid_x_variable"]].min().values
    metadata['y_min'] = grid_ds[metadata["grid_y_variable"]].min().values
    metadata['resolution'] = grid_ds[metadata["grid_x_variable"]].diff(dim=metadata["grid_x_variable"]).mean().values# Assuming square grid cells

# master_df = pd.read_parquet(master_table_parquet)
for weight_table_parquet in tqdm.tqdm(weight_table_parquets):
    vpu = os.path.basename(os.path.dirname(weight_table_parquet))
    out_filename = os.path.join(os.path.dirname(weight_table_parquet), f"gridweights_{dataset_shortname}_{vpu}.nc")

    # Read in the parquet file
    wt = pd.read_parquet(weight_table_parquet)

    # Compute proportions
    total_area_per_linkno = wt.groupby(by='LINKNO')['area_sqm'].sum()
    wt['area_sqm_total'] = wt['LINKNO'].map(total_area_per_linkno)
    wt['proportion'] = wt['area_sqm'] / wt['area_sqm_total']

    # there are multiple rows in df with the same river_id column value but we need to sort them all by the sorted_river_ids
    sorted_river_ids = pd.read_parquet(os.path.join(os.path.dirname(weight_table_parquet), "routing_parameters.parquet")).river_id.sort_values().values
    wt: pd.DataFrame = wt.set_index('LINKNO').loc[sorted_river_ids].reset_index()
    wt = wt.drop(columns=['area_sqm'])
    
    # Rename
    wt = wt.rename(columns={'LINKNO': 'river_id', "lon_index": "x_index", "lat_index": "y_index", "lon": "x", "lat": "y"})

    # Now create the netcdf
    ds = wt.to_xarray()
    ds.attrs = metadata
    ds.to_netcdf(out_filename)


    