import datetime
import glob
import logging
import os

import pandas as pd

from RAPIDprep import preprocess_for_rapid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='preprocess.log',
    filemode='w'
)

if __name__ == '__main__':
    outputs_path = '/tdxprocessed'

    sample_grids = glob.glob('./era5_sample_grids/*.nc')

    region_sizes_df = pd.read_csv('network_data/stream_counts.csv').astype(int)
    regions_to_skip = [
    ]
    for streams_gpkg, basins_gpkg in zip(
            sorted(glob.glob(f'/tdxhydro/TDX_streamnet*.gpkg')),
            sorted(glob.glob(f'/tdxhydro/TDX_streamreach_basins*.gpkg'))
    ):
        # Identify the region being processed
        region_number = int(os.path.basename(streams_gpkg).split('_')[2])
        n_streams = region_sizes_df.loc[region_sizes_df['region'] == region_number, 'count'].values[0]
        if region_number in regions_to_skip:
            continue
        # create the output folder
        out_dir = os.path.join(outputs_path, f'{region_number}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # log a bunch of stuff
        logging.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logging.info(streams_gpkg)
        logging.info(basins_gpkg)
        logging.info(region_number)
        logging.info(out_dir)

        # scale the number of processes based on the number of streams to process
        n_processes = 12
        if n_streams >= 100_000:
            n_processes = 12
        if n_streams >= 200_000:
            n_processes = 6
        if n_streams >= 300_000:
            n_processes = 4
        if n_streams >= 500_000:
            n_processes = 2

        preprocess_for_rapid(
            streams_gpkg,
            basins_gpkg,
            sample_grids,
            out_dir,
            n_processes=n_processes,
        )

        logging.info('Done')
        logging.info('')

    logging.info('All Regions Processed')
    logging.info('Normal Termination')
    logging.info(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
