import glob
import json
import logging
import os
import traceback

import pandas as pd

from RAPIDprep import (
    correct_streams,
    dissolve_basins,
    prepare_rapid_inputs,
    make_weight_table,
    is_valid_rapid_dir,
    REQUIRED_RAPID_FILES
)

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
MIN_N_PROCESSES = 6
id_field = 'LINKNO'
ds_field = 'DSLINKNO'
length_field = 'Length'

if __name__ == '__main__':
    sample_grids = glob.glob('./era5_sample_grids/*.nc')
    region_sizes_df = pd.read_csv('network_data/stream_counts_source.csv').astype(int)

    with open('network_data/regions_to_skip.json', 'r') as f:
        regions_to_skip = json.load(f)
    completed_regions = [d for d in sorted(glob.glob(os.path.join(outputs_path, '*'))) if is_valid_rapid_dir(d)]
    completed_regions = [int(os.path.basename(d)) for d in completed_regions]

    logging.info(f'Base Number of processes {MIN_N_PROCESSES}')
    logging.info(f'Use multiprocessing for streams: {MP_STREAMS}')
    logging.info(f'Use multiprocessing for basins: {MP_BASINS}')
    logging.info(f'Skipping regions: {regions_to_skip}')

    for streams_gpkg, basins_gpkg in zip(
            sorted(glob.glob(f'/tdxhydro/TDX_streamnet*.gpkg')),
            sorted(glob.glob(f'/tdxhydro/TDX_streamreach_basins*.gpkg'))
    ):
        # Identify the region being processed
        region_number = int(os.path.basename(streams_gpkg).split('_')[2])
        if region_number in regions_to_skip or region_number in completed_regions:
            logging.info(f'Skipping region {region_number} - Valid directory already exists\n')
            continue

        n_streams = region_sizes_df.loc[region_sizes_df['region'] == region_number, 'count'].values[0]
        if n_streams > 500_000:
            n_processes = MIN_N_PROCESSES
        else:
            n_processes = 2 * MIN_N_PROCESSES

        # create the output folder
        save_dir = os.path.join(outputs_path, f'{region_number}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # log a bunch of stuff
        logging.info('')
        logging.info(region_number)
        logging.info(streams_gpkg)
        logging.info(basins_gpkg)
        logging.info(save_dir)

        try:
            # Streams and rapid inputs
            if not glob.glob(os.path.join(save_dir, 'TDX_streamnet*.gpkg')):
                correct_streams(streams_gpkg, save_dir=save_dir, stream_id_col=id_field, ds_id_field=ds_field,
                                length_col=length_field, mp_dissolve=MP_STREAMS, n_processes=n_processes)
            if not all([os.path.exists(os.path.join(save_dir, f)) for f in REQUIRED_RAPID_FILES]):
                prepare_rapid_inputs(save_dir=save_dir,
                                     id_field=id_field,
                                     n_workers=min(n_processes, 18))

            # Basins and weight tables
            # if not glob.glob(os.path.join(save_dir, 'TDX_streamreach_basins*.gpkg')):
            #     dissolve_basins(basins_gpkg, mp_dissolve=MP_BASINS,
            #                     save_dir=save_dir, stream_id_col="streamID", n_process=n_processes)
            # if len(list(glob.glob(os.path.join(save_dir, 'weight_*.csv')))) < 3:
            #     for sample_grid in sample_grids:
            #         make_weight_table(sample_grid, save_dir, n_workers=n_processes)

        except Exception as e:
            logging.info('\n----- ERROR -----\n')
            logging.info(e)
            print(traceback.format_exc())
            continue

        logging.info('Done')

    logging.info('All Regions Processed')
    logging.info('Normal Termination')
