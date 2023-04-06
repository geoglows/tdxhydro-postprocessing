from RAPIDprep import preprocess_for_rapid
import glob
import os
import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='preprocess.log',
    filemode='w'
)

if __name__ == '__main__':
    rapid_outputs = '/tdxprocessed/rapidio'

    sample_grids = glob.glob('./era5_sample_grids/*.nc')

    numbers_to_skip = [

    ]
    for streams_gpkg, basins_gpkg in zip(
            sorted(glob.glob(f'/tdxhydro/TDX_streamnet*.gpkg')),
            sorted(glob.glob(f'/tdxhydro/TDX_streamreach_basins*.gpkg'))
    ):
        region_number = os.path.basename(streams_gpkg).split('_')[2]
        if region_number in numbers_to_skip:
            continue

        out_dir = os.path.join(rapid_outputs, region_number)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        logging.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logging.info(streams_gpkg)
        logging.info(basins_gpkg)
        logging.info(region_number)
        logging.info(out_dir)

        preprocess_for_rapid(
            streams_gpkg,
            basins_gpkg,
            sample_grids,
            out_dir,
            n_processes=10
        )
