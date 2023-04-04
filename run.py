from RAPIDprep import PreprocessForRAPID
import glob
import os
import logging
import sys
import datetime


# Configure logging settings
logging.basicConfig(filename='log.log',
                    encoding='utf-8',
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


if __name__ == '__main__':
    rapid_outputs = '/tdxprocessed/rapidio'
    gpkg_outputs = '/tdxprocessed/gpkg'

    sample_grids = glob.glob('./era5_sample_grids/*.nc')

    # tdx_header_number = sys.argv[1]
    for streams_gpkg, basins_gpkg in zip(
            sorted(glob.glob(f'/tdxhydro/TDX_streamnet*.gpkg')),
            sorted(glob.glob(f'/tdxhydro/TDX_streamreach_basins*.gpkg'))
    ):
        region_number = os.path.basename(streams_gpkg).split('_')[2]

        out_dir = os.path.join(rapid_outputs, region_number)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(os.path.join(gpkg_outputs, region_number)):
            os.makedirs(os.path.join(gpkg_outputs, region_number))

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(streams_gpkg)
        print(basins_gpkg)
        print(region_number)
        print(out_dir)
        print(os.path.join(gpkg_outputs, region_number))

        PreprocessForRAPID(streams_gpkg, basins_gpkg, sample_grids, out_dir)
