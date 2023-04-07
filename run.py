import datetime
import glob
import logging
import os
import shutil

import pandas as pd

from RAPIDprep import preprocess_for_rapid

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='preprocess.log',
    filemode='w'
)

outputs_path = '/tdxrapid'
MP_STREAMS = True
MP_BASINS = True
N_PROCESSES = 4

if __name__ == '__main__':
    sample_grids = glob.glob('./era5_sample_grids/*.nc')

    region_sizes_df = pd.read_csv('network_data/stream_counts_source.csv').astype(int)
    regions_to_skip = [int(os.path.basename(d)) for d in glob.glob(os.path.join(outputs_path, '*'))]
    regions_to_skip = regions_to_skip + []
    logging.info(regions_to_skip)

    for streams_gpkg, basins_gpkg in zip(
            sorted(glob.glob(f'/tdxhydro/TDX_streamnet*.gpkg')),
            sorted(glob.glob(f'/tdxhydro/TDX_streamreach_basins*.gpkg'))
    ):
        # Identify the region being processed
        region_number = int(os.path.basename(streams_gpkg).split('_')[2])
        n_streams = region_sizes_df.loc[region_sizes_df['region'] == region_number, 'count'].values[0]
        if region_number in regions_to_skip:
            logging.info(f'Skipping region {region_number} - directory already exists')
            continue

        # Cap the number of streams to process
        # if n_streams > 300_000:
        #     logging.info(f'Skipping region {region_number} - too many streams')
        #     continue

        # scale the number of processes based on the number of streams to process
        # if n_streams >= 600_000:
        #     n_processes = 2
        # elif n_streams >= 400_000:
        #     n_processes = 4
        # elif n_streams >= 250_000:
        #     n_processes = 6

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
        logging.info(f'Number of processes {N_PROCESSES}')
        logging.info(f'Use multiprocessing for streams: {MP_STREAMS}')
        logging.info(f'Use multiprocessing for basins: {MP_BASINS}')

        try:
            preprocess_for_rapid(
                streams_gpkg,
                basins_gpkg,
                sample_grids,
                out_dir,
                n_processes=N_PROCESSES,
                mp_streams=MP_STREAMS,
                mp_basins=MP_BASINS
            )
        except Exception as e:
            logging.info('-----ERROR')
            logging.info(e)
            shutil.rmtree(out_dir)

        logging.info('Done')
        logging.info('')

    logging.info('All Regions Processed')
    logging.info('Normal Termination')
    logging.info(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
