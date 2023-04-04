from RAPIDprep import PreprocessForRAPID
import glob
import os
import logging


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

    tdx_header_number = 10
    for streams_gpkg, basins_gpkg in zip(
            sorted(glob.glob(f'/tdxhydro/tdxhydro_streamnet*.gpkg')),
            sorted(glob.glob(f'/tdxhydro/tdxhydro_streamreach_basins*.gpkg'))
    ):
        region_number = os.path.basename(streams_gpkg).split('_')[2]

        out_dir = os.path.join(rapid_outputs, region_number)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        PreprocessForRAPID(streams_gpkg, basins_gpkg, sample_grids, out_dir)

        # move the gpkg files to a separate directory
        if not os.path.exists(os.path.join(gpkg_outputs, region_number)):
            os.makedirs(os.path.join(gpkg_outputs, region_number))
        for f in glob.glob(os.path.join(out_dir, '*.gpkg')):
            os.rename(f, os.path.join(gpkg_outputs, region_number, os.path.basename(f)))
