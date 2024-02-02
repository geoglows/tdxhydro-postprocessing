import glob
import logging
import os
import sys
import traceback
import warnings

import pandas as pd

import tdxhydrorapid as rp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

inputs_path = '/Volumes/T9Hales4TB/TDXHydroGeoParquet'
outputs_path = '/Volumes/T9Hales4TB/geoglows2/tdxhydro-inputs'
sample_grids = glob.glob('/Volumes/T9Hales4TB/RunoffSampleGrids/*.parquet')
net_df = pd.read_excel('./tdxhydrorapid/network_data/processing_options.xlsx')
region_select = '*'

id_field = 'LINKNO'
ds_field = 'DSLINKNO'
order_field = 'strmOrder'
length_field = 'LengthGeodesicMeters'

MAKE_RAPID_INPUTS = False
MAKE_WEIGHT_TABLES = True
CACHE_GEOMETRY = True
VELOCITY_FACTOR = None
MIN_VELOCITY_FACTOR = 0.25
MIN_K_VALUE = 900  # 15 minutes in seconds
LAKE_K_VALUE = 3600  # 1 hour in seconds

warnings.filterwarnings("ignore")

gis_iterable = zip(
    sorted(glob.glob(os.path.join(inputs_path, f'TDX_streamnet_{region_select}.parquet')), reverse=False),
    sorted(glob.glob(os.path.join(inputs_path, f'TDX_streamreach_basins_{region_select}.parquet')), reverse=False),
)

for streams_gpq, basins_gpq in gis_iterable:
    region_num = os.path.basename(streams_gpq)
    region_num = region_num.split('_')[2]
    region_num = int(region_num)
    logging.info(region_num)

    if not bool(net_df.loc[net_df['region_number'] == region_num, 'process'].values[0]):
        logging.warning(f'Skipping region {region_num}')
        continue

    save_dir = os.path.join(outputs_path, f'{region_num}')
    os.makedirs(save_dir, exist_ok=True)

    # get configs for each region
    CORRECT_TAUDEM_ERRORS = net_df.loc[net_df['region_number'] == region_num, 'fix_taudem_errors'].values[0]
    DROP_SMALL_WATERSHEDS = net_df.loc[net_df['region_number'] == region_num, 'drop_small_watersheds'].values[0]
    DISSOLVE_HEADWATERS = net_df.loc[net_df['region_number'] == region_num, 'dissolve_headwaters'].values[0]
    PRUNE_MAIN_STEMS = net_df.loc[net_df['region_number'] == region_num, 'prune_main_stems'].values[0]
    MERGE_SHORT_STREAMS = net_df.loc[net_df['region_number'] == region_num, 'merge_short_streams'].values[0]
    MIN_DRAINAGE_AREA_M2 = net_df.loc[net_df['region_number'] == region_num, 'min_area_km2'].values[0] * 1e6
    MIN_HEADWATER_STREAM_ORDER = net_df.loc[net_df['region_number'] == region_num, 'min_stream_order'].values[0]

    # cast configs as correct data types
    CORRECT_TAUDEM_ERRORS = bool(CORRECT_TAUDEM_ERRORS)
    DROP_SMALL_WATERSHEDS = bool(DROP_SMALL_WATERSHEDS)
    DISSOLVE_HEADWATERS = bool(DISSOLVE_HEADWATERS)
    PRUNE_MAIN_STEMS = bool(PRUNE_MAIN_STEMS)
    MERGE_SHORT_STREAMS = bool(MERGE_SHORT_STREAMS)
    MIN_DRAINAGE_AREA_M2 = float(MIN_DRAINAGE_AREA_M2)
    MIN_HEADWATER_STREAM_ORDER = int(MIN_HEADWATER_STREAM_ORDER)

    try:
        # make the master rapid input files
        if not os.path.exists(os.path.join(save_dir, 'rapid_inputs_master.parquet')) or \
                (CACHE_GEOMETRY and not len(list(glob.glob(os.path.join(save_dir, '*.geoparquet'))))):
            rp.inputs.rapid_master_files(streams_gpq,
                                         save_dir=save_dir, id_field=id_field, ds_id_field=ds_field,
                                         length_field=length_field,
                                         default_velocity_factor=VELOCITY_FACTOR,
                                         drop_small_watersheds=DROP_SMALL_WATERSHEDS,
                                         dissolve_headwaters=DISSOLVE_HEADWATERS,
                                         prune_branches_from_main_stems=PRUNE_MAIN_STEMS,
                                         merge_short_streams=MERGE_SHORT_STREAMS,
                                         cache_geometry=CACHE_GEOMETRY,
                                         min_drainage_area_m2=MIN_DRAINAGE_AREA_M2,
                                         min_headwater_stream_order=MIN_HEADWATER_STREAM_ORDER,
                                         min_velocity_factor=MIN_VELOCITY_FACTOR,
                                         min_k_value=MIN_K_VALUE,
                                         lake_min_k=LAKE_K_VALUE, )

        # make the rapid input files
        if MAKE_RAPID_INPUTS and not all([os.path.exists(os.path.join(save_dir, f)) for f in rp.RAPID_FILES]):
            rp.inputs.rapid_input_csvs(pd.read_parquet(os.path.join(save_dir, 'rapid_inputs_master.parquet')),
                                       save_dir,
                                       id_field=id_field,
                                       ds_id_field=ds_field, )

        # break for weight tables
        if not MAKE_WEIGHT_TABLES:
            continue

        # make the master weight tables
        basins_gdf = None
        expect_tables = [f'weight_{os.path.basename(f)}' for f in sample_grids]
        expect_tables = [f.replace('_thiessen_grid.parquet', '_full.csv') for f in expect_tables]
        expect_tables = [os.path.join(save_dir, f) for f in expect_tables]
        if not all([os.path.exists(os.path.join(save_dir, f)) for f in expect_tables]):
            logging.info('Reading basins')
            basins_gdf = rp.network.correct_0_length_basins(basins_gpq,
                                                            save_dir=save_dir,
                                                            stream_id_col=id_field)

            # reproject the basins to epsg 4326 if needed
            if basins_gdf.crs != 'epsg:4326':
                basins_gdf = basins_gdf.to_crs('epsg:4326')

            for sample_grid in sorted(sample_grids):
                rp.weights.make_weight_table_from_thiessen_grid(sample_grid,
                                                                save_dir,
                                                                basins_gdf=basins_gdf,
                                                                id_field=id_field)

        for weight_table in glob.glob(os.path.join(save_dir, 'weight*full.csv')):
            out_path = weight_table.replace('_full.csv', '.csv')

            if os.path.exists(out_path):
                logging.info(f'Weight table already exists: {os.path.basename(out_path)}')
                continue

            rp.weights.apply_weight_table_simplifications(save_dir, weight_table, out_path, id_field)

        # check that number streams in rapid inputs matches the number of streams in the weight tables
        rp.tdxhydro_corrections_consistent(save_dir)

    except Exception as e:
        logging.info('\n----- ERROR -----\n')
        logging.info(e)
        logging.error(traceback.format_exc())
        continue

logging.info('All TDX Hydro Regions Processed')
