# rapid_preprocess
Louis R. Rosas, Dr Riley Hales, Josh Ogden

Inspired by previous work including
- RAPIDpy https://github.com/BYU-Hydroinformatics/RAPIDpy
- ArcGIS RAPID Toolbox https://github.com/Esri/python-toolbox-for-rapid
- RAPID https://rapid-hub.org/

## Overview
Provides the master function 'PreprocessforRAPID', which:

Reads in provided stream networks and catchments

Sorts the modeling network by Strahler stream order, ascending

Fixes stream segments of 0 length for different cases as follows:

1. Feature is coastal w/ no upstream or downstream
    - Delete the stream and its basin
2. Feature is bridging a 3-river confluence (Has downstream and upstreams)
    - Artificially create a basin with 0 area, and force a length on the point of 1 meter
3. Feature is costal w/ upstreams but no downstream
    - Force a length on the point of 1 meter
4. Feature doesn't match any previous case
    - Raise an error for now

Creates three new networks:

1. A visulation network, which has the top order 1 streams dissolved with their downstream order 2 segment (for
   smaller file sizes)
2. A modeling network, which is similar to 1) but only preserves the geometry of the order 2 segment and the longest
   order 1 segment
3. A modified basins network. Any streams that were merged will have their corresponding catchements also dissolved

Calculates the muskingum parameters for the stream network and adds this information to the modeling network

Creates the following six files:

- comid_lat_lon_z.csv
- riv_bas_id.csv
- k.csv
- kfac.csv
- x.csv
- rapid_connect.csv

Creates weight tables for each of the given input ERA netCDF datasets

All out puts are saved to given directory
