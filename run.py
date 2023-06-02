import glob
import json
import logging
import os
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

inputs_path = '/Volumes/EB406_T7_2/TDXHydroGeoParquet'
outputs_path = '/Volumes/EB406_T7_2/TDXHydroRapid_V10/'

gis_iterable = zip(
    sorted(glob.glob(os.path.join(inputs_path, 'TDX_streamnet_7020065090*.parquet')), reverse=False),
    sorted(glob.glob(os.path.join(inputs_path, 'TDX_streamreach_basins_7020065090*.parquet')), reverse=False),
)
CALCULATE_CONNECTIVITY = False
CORRECT_TAUDEM_ERRORS = True
SLIM_NETWORK = True
MAKE_WEIGHT_TABLES = True
MAKE_GPKG = False
id_field = 'LINKNO'
basin_id_field = 'streamID'
ds_field = 'DSLINKNO'
order_field = 'strmOrder'
length_field = 'Length'

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    sample_grids = glob.glob('./era5_sample_grids/era*.nc')
    with open('network_data/regions_to_skip.json', 'r') as f:
        regions_to_skip = json.load(f)
    completed_regions = [d for d in sorted(glob.glob(os.path.join(outputs_path, '*'))) if rp.has_base_files(d)]
    completed_regions = [int(os.path.basename(d)) for d in completed_regions]

    logging.info(f'Skipping regions: {regions_to_skip}')
    logging.info(f'Completed regions: {completed_regions}')

    for streams_gpq, basins_gpq in gis_iterable:
        # Identify the region being processed
        region_number = os.path.basename(streams_gpq)
        region_number = region_number.split('_')[2]
        region_number = int(region_number)

        if region_number in regions_to_skip:
            logging.info(f'Skipping region {region_number} - In regions_to_skip\n')
            continue
        if region_number in completed_regions:
            logging.info(f'Skipping region {region_number} - Valid directory already exists\n')
            continue

        # create the output folder
        save_dir = os.path.join(outputs_path, f'{region_number}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # log a bunch of stuff
        logging.info('')
        logging.info(region_number)
        logging.info(save_dir)
        logging.info(streams_gpq)
        logging.info(basins_gpq)

        try:
            # make the master rapid input files
            if not rp.has_rapid_master_files(save_dir):
                rp.inputs.rapid_master_files(streams_gpq,
                                             save_dir=save_dir,
                                             id_field=id_field,
                                             ds_id_field=ds_field,
                                             length_field=length_field)

            if not os.path.exists(os.path.join(save_dir, 'rapid_inputs_slim.parquet')):
                rp.inputs.dissolve_headwater_table(save_dir)

            # make the rapid input files
            if not all([os.path.exists(os.path.join(save_dir, f)) for f in rp.RAPID_FILES]):
                rp.inputs.rapid_input_csvs(save_dir,
                                           id_field=id_field,
                                           ds_id_field=ds_field, )

            # break for weight tables
            if not MAKE_WEIGHT_TABLES:
                continue

            # make the master weight tables
            if len(list(glob.glob(os.path.join(save_dir, 'weight_*full.csv')))) < len(sample_grids):
                logging.info('Reading basins')
                if CORRECT_TAUDEM_ERRORS:
                    # edit the basins in memory - not cached to save time
                    basins_gdf = rp.correct_network.correct_0_length_basins(basins_gpq,
                                                                            save_dir=save_dir,
                                                                            stream_id_col=basin_id_field)
                else:
                    basins_gdf = gpd.read_file(basins_gpq)

                # reproject the basins to epsg 4326 if needed
                if basins_gdf.crs != 'epsg:4326':
                    basins_gdf = basins_gdf.to_crs('epsg:4326')

                for sample_grid in sorted(sample_grids):
                    rp.weights.make_weight_table(
                        sample_grid,
                        save_dir,
                        basins_gdf=basins_gdf,
                        basin_id_field=basin_id_field
                    )
                basins_gdf = None

            for weight_table in glob.glob(os.path.join(save_dir, 'weight*full.csv')):
                out_path = weight_table.replace('_full.csv', '.csv')
                if os.path.exists(out_path):
                    continue

                logging.info(f'Merging rows in weight table {weight_table}')

                # mod the weight tables
                wt = pd.read_csv(weight_table)
                if SLIM_NETWORK:
                    o2_to_dissolve = rp.correct_network.find_headwater_streams_to_dissolve(
                        pd.read_parquet(os.path.join(save_dir, 'rapid_inputs_master.parquet'))
                    )
                    # consolidate the weight table rows
                    for streams_to_merge in o2_to_dissolve.values:
                        wt.loc[wt[basin_id_field].isin(streams_to_merge), basin_id_field] = streams_to_merge[0]

                    # group the weight table by matching columns except for area_sqm then sum by that column
                    wt['npoints'] = wt.groupby(basin_id_field)[basin_id_field].transform('count')
                    wt = wt.groupby(wt.columns.drop('area_sqm').tolist()).sum().reset_index()
                    wt = wt.sort_values([basin_id_field, 'area_sqm'], ascending=[True, False])
                    wt['npoints'] = wt.groupby(basin_id_field)[basin_id_field].transform('count')
                    wt = wt[[basin_id_field, 'area_sqm', 'lon_index', 'lat_index', 'npoints', 'lon', 'lat']]

                wt.to_csv(out_path, index=False)

            # check that number streams in rapid inputs matches the number of streams in the weight tables
            if not rp.count_rivers_in_generated_files(save_dir):
                logging.error(f'Number of streams in rapid inputs does not match number of streams in weight tables')
                continue

            if MAKE_GPKG:  # and not glob.glob(os.path.join(save_dir, 'TDX_streamnet*.gpkg')):
                logging.info(f'Making GeoPackage: {MAKE_GPKG}')
                gdf = gpd.read_parquet(streams_gpq)
                streams_to_dissolve = rp.correct_network.find_headwater_streams_to_dissolve(
                    pd.read_parquet(os.path.join(save_dir, 'rapid_inputs_master.parquet'))
                )
                streams_to_drop = pd.read_csv(os.path.join(save_dir, 'mod_drop_small_trees.csv'))
                gdf = gdf[~gdf[id_field].isin(streams_to_drop.values.flatten())]
                for streams_to_merge in streams_to_dissolve.values:
                    gdf = pd.concat([
                        gdf[~gdf[id_field].isin(streams_to_merge)],
                        gdf[gdf[id_field].isin(streams_to_merge)].dissolve()
                    ])
                gdf.to_file(os.path.join(save_dir, f'TDX_streamnet_{region_number}.gpkg'), driver='GPKG')
                continue

        except Exception as e:
            logging.info('\n----- ERROR -----\n')
            logging.info(e)
            print(traceback.format_exc())
            continue

    logging.info('All Regions Processed')
