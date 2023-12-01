#!/bin/zsh
python run_1_gpkg_to_geoparquet.py
python run_2_netcdfs_to_grids.py
python run_3_tdxhydro_rapid_masters.py
python run_4_make_vpu_files.py