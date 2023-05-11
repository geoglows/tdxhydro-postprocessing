import glob
import json
import logging
import os
import shutil
import traceback
import warnings

import geopandas as gpd
import pandas as pd

import rapidprep as rp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='preprocess.log',
    filemode='w'
)

N_PROCESSES = os.cpu_count()
inputs_path = '/tdxhydro'
outputs_path = '/tdxrapid/input'

gis_iterable = zip(
    sorted(glob.glob(os.path.join(inputs_path, 'TDX_streamnet_*.gpkg')), reverse=True),
    sorted(glob.glob(os.path.join(inputs_path, 'TDX_streamreach_basins_*.gpkg')), reverse=True),
)
CORRECT_TAUDEM_ERRORS = True
SLIM_NETWORK = False
MAKE_GPKG = False
id_field = 'LINKNO'
basin_id_field = 'streamID'
ds_field = 'DSLINKNO'
order_field = 'strmOrder'
length_field = 'Length'

# gis_iterable = list(
#     zip(
#         sorted(glob.glob('/Users/rchales/Data/geoglows_delineation/drainlines_shapefile/j*-drainageline/*.shp')),
#         sorted(glob.glob('/Users/rchales/Data/geoglows_delineation/catchment_shapefile/j*/*.shp')),
#     ),
# )
# CORRECT_TAUDEM_ERRORS = False
# outputs_path = '/Users/rchales/Data/GEOGLOWS_1_RAPID_REPEAT'
# id_field = 'COMID'
# # basin_id_field = 'DrainLnID'
# basin_id_field = 'COMID'
# ds_field = 'NextDownID'
# order_field = 'order_'
# length_field = 'Length'

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    sample_grids = glob.glob('./era5_sample_grids/era5*.nc')
    region_sizes_df = pd.read_csv('network_data/stream_counts_source.csv').astype(int)

    with open('network_data/regions_to_skip.json', 'r') as f:
        regions_to_skip = json.load(f)
    completed_regions = [d for d in sorted(glob.glob(os.path.join(outputs_path, '*'))) if rp.has_base_files(d)]
    completed_regions = [int(os.path.basename(d)) for d in completed_regions]

    logging.info(f'Base Number of processes {N_PROCESSES}')
    logging.info(f'Skipping regions: {regions_to_skip}')
    logging.info(f'Completed regions: {completed_regions}')

    for streams_gpkg, basins_gpkg in gis_iterable:
        # Identify the region being processed
        region_number = os.path.basename(streams_gpkg)
        region_number = region_number.split('_')[2]
        region_number = int(region_number)

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
        logging.info(save_dir)
        logging.info(streams_gpkg)
        logging.info(basins_gpkg)
        logging.info(f'Streams: {n_streams}')

        try:
            # make the master rapid input files
            if not all([os.path.exists(os.path.join(save_dir, f)) for f in rp.RAPID_MASTER_FILES]):
                rp.inputs.rapid_master_files(streams_gpkg,
                                             save_dir=save_dir,
                                             id_field=id_field,
                                             ds_id_field=ds_field,
                                             length_field=length_field,
                                             n_workers=N_PROCESSES)

            # look for streams to trim
            if SLIM_NETWORK and not all([os.path.exists(os.path.join(save_dir, f)) for f in rp.MODIFICATION_FILES]):
                rp.slim_net.find_streams_to_slim(save_dir, id_field=id_field, order_field=order_field)

            # make the rapid input files
            if not all([os.path.exists(os.path.join(save_dir, f)) for f in rp.RAPID_FILES]):
                streams_df = pd.read_parquet(os.path.join(save_dir, 'rapid_inputs_master.parquet'))
                rapcon_df = pd.read_parquet(os.path.join(save_dir, 'rapid_connect_master.parquet')).astype(int)

                # apply the slimming modifications
                if SLIM_NETWORK:
                    streams_df = rp.slim_net.slim_streams_df(save_dir,
                                                             streams_df,
                                                             id_field=id_field,
                                                             n_processes=N_PROCESSES)

                rp.inputs.rapid_input_csvs(save_dir,
                                           id_field=id_field,
                                           ds_id_field=ds_field, )

            # make the master weight tables
            if len(list(glob.glob(os.path.join(save_dir, 'weight_*_full.csv')))) < len(sample_grids):
                if CORRECT_TAUDEM_ERRORS:
                    # edit the basins in memory - not cached to save time
                    basins_gdf = rp.correct_network.correct_0_length_basins(
                        basins_gpkg,
                        save_dir=save_dir,
                        stream_id_col=basin_id_field,
                        buffer_size=.1
                    )
                else:
                    basins_gdf = gpd.read_file(basins_gpkg)

                # reproject the basins to epsg 4326 if needed
                if basins_gdf.crs != 'epsg:4326':
                    basins_gdf = basins_gdf.to_crs('epsg:4326')

                for sample_grid in sorted(sample_grids):
                    rp.weights.make_weight_table(
                        sample_grid,
                        save_dir,
                        basins_gdf=basins_gdf,
                        n_workers=N_PROCESSES,
                        basin_id_field=basin_id_field
                    )
                basins_gdf = None

            # todo slim the weight tables if needed
            for wt in sorted(glob.glob(os.path.join(save_dir, 'weight_*_full.csv'))):
                if not os.path.exists(wt.replace('_full', '')):
                    shutil.copy(wt, wt.replace('_full', ''))

            # check that number streams in rapid inputs matches the number of streams in the weight tables
            if not rp.count_rivers_in_generated_files(save_dir):
                logging.info(f'Number of streams in rapid inputs does not match number of streams in weight tables')
                continue

            # todo dissolve the streams
            if MAKE_GPKG and not glob.glob(os.path.join(save_dir, 'TDX_streamnet*.gpkg')):
                continue

        except Exception as e:
            logging.info('\n----- ERROR -----\n')
            logging.info(e)
            print(traceback.format_exc())
            continue

    logging.info('All Regions Processed')
