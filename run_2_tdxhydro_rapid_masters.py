import glob
import json
import logging
import os
import traceback
import warnings

import pandas as pd

import tdxhydrorapid as rp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename='log.log',
    filemode='w'
)

inputs_path = '/Volumes/EB406_T7_2/TDXHydroGeoParquet'
outputs_path = '/Volumes/EB406_T7_2/TEST_10/inputs'
regions_to_select = '*7020065090*'

gis_iterable = zip(
    sorted(glob.glob(os.path.join(inputs_path, f'TDX_streamnet_{regions_to_select}.parquet')), reverse=False),
    sorted(glob.glob(os.path.join(inputs_path, f'TDX_streamreach_basins_{regions_to_select}.parquet')), reverse=False),
)

id_field = 'LINKNO'
basin_id_field = 'streamID'
ds_field = 'DSLINKNO'
order_field = 'strmOrder'
length_field = 'Length'

CORRECT_TAUDEM_ERRORS = True
MAKE_RAPID_INPUTS = True
MAKE_WEIGHT_TABLES = True
CACHE_GEOMETRY = False

DROP_SMALL_WATERSHEDS = True
DISSOLVE_HEADWATERS = False
PRUNE_BRANCHES_FROM_MAIN_STEMS = False
MIN_DRAINAGE_AREA_M2 = 200_000_000
MIN_HEADWATER_STREAM_ORDER = 2

K_VALUE = 0.5

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    sample_grids = glob.glob('./era5_thiessen_grid_parquets/era5*.parquet')
    with open('tdxhydrorapid/network_data/regions_to_skip.json', 'r') as f:
        regions_to_skip = json.load(f)

    logging.info(f'Skipping regions: {regions_to_skip}')

    for streams_gpq, basins_gpq in gis_iterable:
        # Identify the region being processed
        region_number = os.path.basename(streams_gpq)
        region_number = region_number.split('_')[2]
        region_number = int(region_number)

        save_dir = os.path.join(outputs_path, f'{region_number}')
        os.makedirs(save_dir, exist_ok=True)

        if region_number in regions_to_skip:
            logging.warning(f'Skipping region {region_number} - In regions_to_skip')
            continue

        # log a bunch of stuff
        logging.info('')
        logging.info(region_number)
        logging.info(save_dir)
        logging.info(streams_gpq)
        logging.info(basins_gpq)

        try:
            # make the master rapid input files
            if not os.path.exists(os.path.join(save_dir, 'rapid_inputs_master.parquet')) or \
                    (CACHE_GEOMETRY and not len(list(glob.glob(os.path.join(save_dir, '*.geoparquet'))))):
                rp.inputs.rapid_master_files(streams_gpq,
                                             save_dir=save_dir,
                                             id_field=id_field,
                                             ds_id_field=ds_field,
                                             length_field=length_field,
                                             cache_geometry=CACHE_GEOMETRY,
                                             default_k=K_VALUE,
                                             drop_small_watersheds=DROP_SMALL_WATERSHEDS,
                                             min_drainage_area_m2=MIN_DRAINAGE_AREA_M2,
                                             dissolve_headwaters=DISSOLVE_HEADWATERS,
                                             min_headwater_stream_order=MIN_HEADWATER_STREAM_ORDER,
                                             prune_branches_from_main_stems=PRUNE_BRANCHES_FROM_MAIN_STEMS, )

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
