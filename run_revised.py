import glob
import json
import logging
import os
import traceback

import pandas as pd
import geopandas as gpd

import rapidprep as rp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='preprocess.log',
    filemode='w'
)

outputs_path = '/Users/rchales/Data/TDXOutputsDescriptors'
MP_STREAMS = True
MP_BASINS = True
N_PROCESSES = os.cpu_count()
id_field = 'LINKNO'
ds_field = 'DSLINKNO'
order_field = 'strmOrder'
length_field = 'Length'

if __name__ == '__main__':
    sample_grids = glob.glob('./era5_sample_grids/*.nc')
    region_sizes_df = pd.read_csv('network_data/stream_counts_source.csv').astype(int)

    with open('network_data/regions_to_skip.json', 'r') as f:
        regions_to_skip = json.load(f)
    completed_regions = [d for d in sorted(glob.glob(os.path.join(outputs_path, '*'))) if rp.is_valid_result(d)]
    completed_regions = [int(os.path.basename(d)) for d in completed_regions]

    logging.info(f'Base Number of processes {N_PROCESSES}')
    logging.info(f'Use multiprocessing for streams: {MP_STREAMS}')
    logging.info(f'Use multiprocessing for basins: {MP_BASINS}')
    logging.info(f'Skipping regions: {regions_to_skip}')
    logging.info(f'Completed regions: {completed_regions}')

    for streams_gpkg, basins_gpkg in zip(
            sorted(glob.glob(f'/Users/rchales/Data/TDXHydro/TDX_streamnet*.gpkg')),
            sorted(glob.glob(f'/Users/rchales/Data/TDXHydro/TDX_streamreach_basins*.gpkg'))
    ):
        # Identify the region being processed
        region_number = int(os.path.basename(streams_gpkg).split('_')[2])

        if region_number in regions_to_skip:
            logging.info(f'Skipping region {region_number} - In regions_to_skip\n')
            continue
        if region_number in completed_regions:
            logging.info(f'Skipping region {region_number} - Valid directory already exists\n')
            continue

        n_streams = region_sizes_df.loc[region_sizes_df['region'] == region_number, 'count'].values[0]

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
        logging.info(f'Streams: {n_streams}')

        try:
            # determine if the preliminary stream analysis has been completed
            if not all([os.path.exists(os.path.join(save_dir, f)) for f in rp.REQUIRED_MODIFICATION_FILES]):
                streams_gdf = gpd.read_file(streams_gpkg)
                rp.analyze.streams(streams_gdf,
                                   save_dir=save_dir,
                                   id_field=id_field,
                                   ds_field=ds_field,
                                   order_field=order_field,
                                   len_field=length_field)

            # correct the raw basins files based on the stream checks
            if not glob.glob(os.path.join(save_dir, 'TDX_streamreach_basins*.gpkg')):
                basins_gdf = rp.correct_network.correct_0_length_basins(basins_gpkg, save_dir=save_dir,
                                                                        stream_id_col="streamID", buffer_size=.001)

            # make the raw weight tables
            if len(list(glob.glob(os.path.join(save_dir, 'weight_*_full.csv')))) < 3:
                for sample_grid in sample_grids:
                    rp.weights.make_weight_table(sample_grid,
                                                 save_dir,
                                                 basin_gdf_path=basins_gpkg,
                                                 n_workers=N_PROCESSES)

            # modify the weight table based on the modification files
            for wt in glob.glob(os.path.join(save_dir, 'weight_*_full.csv')):
                rp.weights.apply_modifications(wt, save_dir)

            if not all([os.path.exists(os.path.join(save_dir, f)) for f in rp.REQUIRED_RAPID_FILES]):
                rp.inputs.prepare_rapid_inputs(save_dir,
                                               id_field=id_field,
                                               ds_field=ds_field,
                                               order_field=order_field,
                                               n_workers=N_PROCESSES)

            # todo dissolve the streams
            if not glob.glob(os.path.join(save_dir, 'TDX_streamnet*.gpkg')):
                continue

        except Exception as e:
            logging.info('\n----- ERROR -----\n')
            logging.info(e)
            print(traceback.format_exc())
            continue

        logging.info('Done')
        logging.info('All Regions Processed')
        logging.info('Normal Termination')
