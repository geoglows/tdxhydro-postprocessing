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

inputs_path = '/Users/rchales/Data/TDXHydroGeoParquet'
outputs_path = '/Volumes/EB406_T7_2/geoglows2/tdxhydro-inputs'
regions_to_select = '*'

gis_iterable = zip(
    sorted(glob.glob(os.path.join(inputs_path, f'TDX_streamnet_{regions_to_select}.parquet')), reverse=False),
    sorted(glob.glob(os.path.join(inputs_path, f'TDX_streamreach_basins_{regions_to_select}.parquet')), reverse=False),
)

id_field = 'LINKNO'
basin_id_field = 'streamID'
ds_field = 'DSLINKNO'
order_field = 'strmOrder'
length_field = 'Length'

MAKE_RAPID_INPUTS = True
MAKE_WEIGHT_TABLES = True
CACHE_GEOMETRY = True
VELOCITY_FACTOR = None  # 0.4 otherwise

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    sample_grids = glob.glob('./era5_thiessen_grid_parquets/*.parquet')
    net_df = pd.read_excel('./tdxhydrorapid/network_data/processing_options.xlsx')

    for streams_gpq, basins_gpq in gis_iterable:
        # Identify the region being processed
        region_num = os.path.basename(streams_gpq)
        region_num = region_num.split('_')[2]
        region_num = int(region_num)

        save_dir = os.path.join(outputs_path, f'{region_num}')
        os.makedirs(save_dir, exist_ok=True)

        if not net_df.loc[net_df['region_number'] == region_num, 'process'].values[0]:
            logging.warning(f'Skipping region {region_num}')
            continue

        # get configs for each region
        CORRECT_TAUDEM_ERRORS = net_df.loc[net_df['region_number'] == region_num, 'fix_taudem_errors'].values[0]
        DROP_SMALL_WATERSHEDS = net_df.loc[net_df['region_number'] == region_num, 'drop_small_watersheds'].values[0]
        DISSOLVE_HEADWATERS = net_df.loc[net_df['region_number'] == region_num, 'dissolve_headwaters'].values[0]
        PRUNE_MAIN_STEMS = net_df.loc[net_df['region_number'] == region_num, 'prune_main_stems'].values[0]
        MIN_DRAINAGE_AREA_M2 = net_df.loc[net_df['region_number'] == region_num, 'min_area_km2'].values[0] * 1e6
        MIN_HEADWATER_STREAM_ORDER = net_df.loc[net_df['region_number'] == region_num, 'min_stream_order'].values[0]

        # log a bunch of stuff
        logging.info('')
        logging.info(region_num)
        logging.info(save_dir)
        logging.info(streams_gpq)
        logging.info(basins_gpq)

        try:
            # make the master rapid input files
            if not os.path.exists(os.path.join(save_dir, 'rapid_inputs_master.parquet')) or \
                    (CACHE_GEOMETRY and not len(list(glob.glob(os.path.join(save_dir, '*.geoparquet'))))):
                rp.inputs.rapid_master_files(streams_gpq, save_dir=save_dir, id_field=id_field, ds_id_field=ds_field,
                                             length_field=length_field, default_velocity_factor=VELOCITY_FACTOR,
                                             drop_small_watersheds=DROP_SMALL_WATERSHEDS,
                                             dissolve_headwaters=DISSOLVE_HEADWATERS,
                                             prune_branches_from_main_stems=PRUNE_MAIN_STEMS,
                                             cache_geometry=CACHE_GEOMETRY, min_drainage_area_m2=MIN_DRAINAGE_AREA_M2,
                                             min_headwater_stream_order=MIN_HEADWATER_STREAM_ORDER)

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
            expected_tables = [os.path.basename(f) for f in sample_grids]
            expected_tables = [f'weight_{f.replace("_thiessen_grid.parquet", "_full.csv")}' for f in expected_tables]
            expected_tables = [os.path.join(save_dir, f) for f in expected_tables]
            if not all([os.path.exists(os.path.join(save_dir, f)) for f in expected_tables]):
                logging.info('Reading basins')
                basins_gdf = rp.network.correct_0_length_basins(basins_gpq,
                                                                save_dir=save_dir,
                                                                stream_id_col=basin_id_field)

                # reproject the basins to epsg 4326 if needed
                if basins_gdf.crs != 'epsg:4326':
                    basins_gdf = basins_gdf.to_crs('epsg:4326')

                for sample_grid in sorted(sample_grids):
                    rp.weights.make_weight_table_from_thiessen_grid(sample_grid,
                                                                    save_dir,
                                                                    basins_gdf=basins_gdf,
                                                                    basin_id_field=basin_id_field)

            for weight_table in glob.glob(os.path.join(save_dir, 'weight*full.csv')):
                out_path = weight_table.replace('_full.csv', '.csv')

                if os.path.exists(out_path):
                    logging.info(f'Weight table already exists: {os.path.basename(out_path)}')
                    continue

                rp.weights.apply_weight_table_simplifications(save_dir, weight_table, out_path, basin_id_field)

                if basins_gdf is None:
                    basins_gdf = pd.read_parquet(basins_gpq, columns=[basin_id_field, 'TDXHydroLinkNo'])

                # swap the linkno for the globally unique number
                # todo could place this somewhere earlier when the table is first created
                (
                    pd.read_csv(out_path)
                    .merge(basins_gdf[[basin_id_field, 'TDXHydroLinkNo']], on=basin_id_field, how='left')
                    .drop(columns=[basin_id_field, ])
                    [['TDXHydroLinkNo', 'area_sqm', 'lon_index', 'lat_index', 'npoints', 'lon', 'lat']]
                    .to_csv(out_path, index=False)
                )

            basins_gdf = None

            # check that number streams in rapid inputs matches the number of streams in the weight tables
            rp.tdxhydro_corrections_consistent(save_dir)

        except Exception as e:
            logging.info('\n----- ERROR -----\n')
            logging.info(e)
            logging.error(traceback.format_exc())
            continue

    logging.info('All TDX Hydro Regions Processed')
