#!/bin/zsh
python run_1_gpkg_to_geoparquet.py
python run_1_netcdfs_to_grids.py
python run_2_tdxhydro_rapid_masters.py
python run_3_make_vpu_files.py