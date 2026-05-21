import json
import logging
import os
import sys
from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd
from natsort import natsorted

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydrography
import hydrography.schema as schema

modifications_root = '/Users/rchales/code/untitled folder/tdxhydro-postprocessing/data/modifications'
tdx_root = '/Users/rchales/code/untitled folder/tdxhydro-postprocessing/data/TDXHydroGeoParquet'
network_data_root = '/Users/rchales/code/untitled folder/tdxhydro-postprocessing/network_data'

if __name__ == '__main__':
    # find the ID of the region to process
    if len(sys.argv) != 2:
        sys.exit('usage: 2_simplify_streams.py <region_number>')
    region_number = int(sys.argv[1])
    # region_number = 1020000010  # Example region number

    # prepare directories and logging
    outputs_dir = os.path.join(modifications_root, f'{region_number}')
    os.makedirs(os.path.join(outputs_dir, 'mods'), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(outputs_dir, 'mods', 'log.log'),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    gdf = gpd.read_parquet(os.path.join(tdx_root, f'TDX_streamnet_{region_number}_01.parquet'))
    logging.info(f'Initial shape: {gdf.shape}')

    # add attributes
    gdf[schema.area] = gdf[schema.tdx_ds_area_field] - gdf[schema.tdx_us_area_field]
    # the header number should be applied to the LINKNO and DSLINKNO fields
    with open(os.path.join(network_data_root, 'tdxhydro_splits/tdx_header_numbers.json')) as f:
        header_numbers_lookup = json.load(f)
    spacer = 10_000_000 * header_numbers_lookup[str(region_number)]
    gdf[schema.river_id] = (gdf[schema.tdx_link_field] + spacer).astype(int)
    # everywhere except the DSLINKNO = -1 should have the unique spacer added to it
    gdf[schema.next_river_id] = -1
    gdf.loc[gdf[schema.tdx_ds_link_field] != -1, schema.next_river_id] = gdf[schema.tdx_ds_link_field] + spacer
    gdf = gdf.drop(columns=[schema.tdx_link_field, schema.tdx_ds_link_field, ])
    gdf.rename(columns=schema.rename_map, inplace=True)

    # prepare the topology attributes
    gdf = hydrography.topology.compute_topology(gdf)

    # remove watersheds with outlets in the defined lists of areas to ignore
    drop_lists = natsorted(glob(os.path.join(network_data_root, 'dropped_watersheds/*.csv')))
    for drop_list in drop_lists:
        drop_ids = pd.read_csv(drop_list).values.flatten()
        gdf = gdf[~gdf[schema.last_river_id].isin(drop_ids)]
        logging.info(f'After dropping {drop_list}, shape is {gdf.shape}')

    # drop all small watersheds less than 250 km^2
    to_drop = (
        gdf
        [np.logical_and(gdf[schema.tdx_ds_area_field] < 250_000_000, gdf[schema.next_river_id] == -1)]
        [schema.last_river_id]
        .tolist()
    )
    gdf = gdf[~gdf[schema.last_river_id].isin(to_drop)]

    # assign vpu groups based on outletRiverId
    vpu_df = pd.read_csv(os.path.join(network_data_root, 'vpu_table.csv'))
    vpu_map = vpu_df.set_index(schema.last_river_id)[schema.vpu_id].to_dict()
    gdf[schema.vpu_id] = gdf[schema.last_river_id].map(vpu_map)
    missing_vpu = gdf.loc[gdf[schema.vpu_id].isna(), schema.last_river_id].unique()
    if len(missing_vpu):
        logging.warning(f'{len(missing_vpu)} terminal node(s) missing from vpu_table.csv, e.g. {missing_vpu[:10]}')
        gdf[schema.vpu_id] = gdf[schema.vpu_id].fillna(-1).astype(int)

    # Step 2 - lake and reservoir treatments
    lake_edits = hydrography.lakes.find_lake_edits(gdf)
    gdf = hydrography.lakes.apply_lake_edits(gdf, lake_edits)
    logging.info(f'After applying lake edits, shape is {gdf.shape}')
    hydrography.topology.assert_topology_is_valid(gdf)

    # remove 0 length rivers
    zero_lengths = hydrography.streams.find_zero_length(gdf)
    gdf = hydrography.streams.remove_zero_length(gdf, zero_length_json=zero_lengths)
    logging.info(f'After correcting zero length rivers, shape is {gdf.shape}')
    hydrography.topology.assert_topology_is_valid(gdf)

    # dissolve headwater streams with min order of 2
    header_mergers = hydrography.streams.find_headwater_mergers(gdf, min_order=2)
    gdf = hydrography.streams.merge_headwaters(gdf, header_mergers=header_mergers)
    logging.info(f'After dissolving headwaters, shape is {gdf.shape}')
    hydrography.topology.assert_topology_is_valid(gdf)

    # prune branches with min order of 2
    branches_to_prune = hydrography.streams.find_branches_to_prune(gdf)
    gdf = hydrography.streams.prune_branches(gdf, branches_to_prune=branches_to_prune)
    logging.info(f'After pruning branches, shape is {gdf.shape}')
    hydrography.topology.assert_topology_is_valid(gdf)

    # consolidate shorter streams into their neighbors where possible
    consolidations = hydrography.streams.find_short_streams(gdf, min_length=2000)
    gdf = hydrography.streams.consolidate_short_streams(gdf, consolidations=consolidations)
    logging.info(f'After consolidating short streams, shape is {gdf.shape}')
    hydrography.topology.assert_topology_is_valid(gdf)

    # reset the topological numbering after all modifications are done
    gdf = gdf.sort_values(schema.topo_sort).reset_index(drop=True)
    gdf[schema.topo_sort] = np.arange(len(gdf), dtype=np.int32)

    with open(os.path.join(outputs_dir, 'mods', 'lake_edits.json'), 'w') as f:
        json.dump(lake_edits, f)
    with open(os.path.join(outputs_dir, 'mods', 'zero_length_streams.json'), 'w') as f:
        json.dump(zero_lengths, f)
    with open(os.path.join(outputs_dir, 'mods', 'headwater_dissolves.json'), 'w') as f:
        json.dump(header_mergers, f)
    with open(os.path.join(outputs_dir, 'mods', 'branches_to_prune.json'), 'w') as f:
        json.dump(branches_to_prune, f)
    with open(os.path.join(outputs_dir, 'mods', 'short_consolidations.json'), 'w') as f:
        json.dump(consolidations, f)

    # length is in m, divide by estimated m/s to get k in seconds
    gdf[schema.static_velocity_factor] = np.exp(0.16842 * np.log(gdf[schema.tdx_ds_area_field]) - 4.68).round(3)
    gdf[schema.static_musk_k] = gdf[schema.length] / gdf[schema.static_velocity_factor]
    gdf[schema.static_musk_k] = gdf[schema.static_musk_k].round(0).astype(int)
    gdf[schema.static_musk_x] = 0.20
    # for seconds in (3600, 1800, 900):
    #     gdf[schema.case2_field(seconds)] = gdf[schema.static_musk_k] * 2 * (1 - gdf[schema.static_musk_x]) < seconds

    gdf = gdf[schema.final_columns_to_keep]
    gdf.to_parquet(f'../../data/modifications/{region_number}/streams_{region_number}.parquet')
