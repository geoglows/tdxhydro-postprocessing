import json
import logging
import os
import sys
from glob import glob

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from natsort import natsorted
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydrography as hy

region_root = '/Users/rchales/code/untitled folder/tdxhydro-postprocessing/data/regions'
tdx_root = '/Users/rchales/code/untitled folder/tdxhydro-postprocessing/data/TDXHydroGeoParquet'
network_data_root = '/Users/rchales/code/untitled folder/tdxhydro-postprocessing/network_data'

if __name__ == '__main__':
    # find the ID of the region to process
    if len(sys.argv) != 2:
        sys.exit('usage: 2_simplify_streams.py <region>')
    region = int(sys.argv[1])
    # region = 1020000010  # Example region number

    # final outputs to check for existence before computing
    final_geoparquet_output = os.path.join(region_root, f'{region}', f'streams_{region}.geo.parquet')
    simple_streams_output = os.path.join(region_root, f'{region}', f'streams_simplified_{region}.geo.parquet')
    mapping_streams_output = os.path.join(region_root, f'{region}', f'streams_mapping_{region}.geo.parquet')
    final_metadata_output = os.path.join(region_root, f'{region}', f'metadata_{region}.parquet')
    confluences_output = os.path.join(region_root, f'{region}', f'confluences_{region}.geo.parquet')
    outputs = [final_geoparquet_output, simple_streams_output, mapping_streams_output,
               final_metadata_output, confluences_output]
    if all(os.path.exists(output) for output in outputs):
        print(f'All final outputs for region {region} already exist, skipping')
        sys.exit(0)

    # prepare directories and logging
    outputs_dir = os.path.join(region_root, f'{region}')
    os.makedirs(os.path.join(outputs_dir, 'mods'), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(outputs_dir, 'log.log'),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    gdf = gpd.read_parquet(os.path.join(tdx_root, f'TDX_streamnet_{region}_01.parquet'))
    logging.info(f'Initial shape: {gdf.shape}')

    # add unique river ids and attributes
    gdf[hy.schema.area] = gdf[hy.schema.tdx_ds_area_field] - gdf[hy.schema.tdx_us_area_field]
    with open(os.path.join(network_data_root, 'tdxhydro_splits/tdx_header_numbers.json')) as f:
        header_numbers_lookup = json.load(f)
    spacer = 10_000_000 * header_numbers_lookup[str(region)]
    gdf[hy.schema.river_id] = (gdf[hy.schema.tdx_link_field] + spacer).astype(int)
    gdf[hy.schema.next_river_id] = -1
    gdf.loc[gdf[hy.schema.tdx_ds_link_field] != -1, hy.schema.next_river_id] = gdf[hy.schema.tdx_ds_link_field] + spacer
    gdf = gdf.drop(columns=[hy.schema.tdx_link_field, hy.schema.tdx_ds_link_field, ])
    gdf.rename(columns=hy.schema.rename_map, inplace=True)

    # prepare the topology attributes
    gdf = hy.topology.compute_topology(gdf)

    # remove watersheds with outlets in the defined lists of areas to ignore
    drop_lists = natsorted(glob(os.path.join(network_data_root, 'dropped_watersheds/*.csv')))
    for drop_list in drop_lists:
        drop_ids = pd.read_csv(drop_list).values.flatten()
        gdf = gdf[~gdf[hy.schema.last_river_id].isin(drop_ids)]
        logging.info(f'After dropping {drop_list}, shape is {gdf.shape}')

    # remove watersheds less than 250 km^2
    to_drop = (
        gdf
        [np.logical_and(gdf[hy.schema.tdx_ds_area_field] < 250_000_000, gdf[hy.schema.next_river_id] == -1)]
        [hy.schema.last_river_id]
        .tolist()
    )
    gdf = gdf[~gdf[hy.schema.last_river_id].isin(to_drop)]

    # assign vpu groups based on outletRiverId
    vpu_df = pd.read_csv(os.path.join(network_data_root, 'vpu_table.csv'))
    vpu_map = vpu_df.set_index(hy.schema.last_river_id)[hy.schema.vpu_id].to_dict()
    gdf[hy.schema.vpu_id] = gdf[hy.schema.last_river_id].map(vpu_map)
    missing_vpu = gdf.loc[gdf[hy.schema.vpu_id].isna(), hy.schema.last_river_id].unique()
    if len(missing_vpu):
        # write the invalid copy to file so that it can be debugged and fixed, then rerun
        gdf.to_parquet(os.path.join(outputs_dir, 'mods', 'missing_vpu_debug.parquet'))
        raise RuntimeError(f'{len(missing_vpu)} reaches have no vpuId; e.g. {missing_vpu[:10]}')

    # modify lake and reservoirs
    lake_edits = hy.lakes.find_lake_edits(gdf, min_inlet_area=100_000_000)
    gdf = hy.lakes.apply_lake_edits(gdf, lake_edits)
    logging.info(f'After applying lake edits, shape is {gdf.shape}')
    hy.topology.assert_topology_is_valid(gdf)

    # remove zero length rivers
    zero_lengths = hy.streams.find_zero_length(gdf)
    gdf = hy.streams.remove_zero_length(gdf, zero_length_json=zero_lengths)
    logging.info(f'After correcting zero length rivers, shape is {gdf.shape}')
    hy.topology.assert_topology_is_valid(gdf)

    # consolidate order-1 reaches orphaned by zero length outlet removal into their
    # neighbor. uses the STALE outletRiverId to recover the sibling set, so it must
    # run before recompute_outlets overwrites those ids
    coastal_orphans = hy.streams.find_orphaned_coastal_outlets(gdf)
    gdf = hy.streams.prune_branches(gdf, branches_to_prune=coastal_orphans)
    logging.info(f'After consolidating orphaned coastal order-1s, shape is {gdf.shape}')

    # outletRiverId is stale after zero length removal where the zero length was the outlet
    gdf = hy.topology.recompute_outlets(gdf)
    hy.topology.assert_topology_is_valid(gdf)

    # dissolve headwater streams with min order of 2
    header_mergers = hy.streams.find_headwater_mergers(gdf, min_order=2)
    gdf = hy.streams.merge_headwaters(gdf, header_mergers=header_mergers)
    logging.info(f'After dissolving headwaters, shape is {gdf.shape}')
    hy.topology.assert_topology_is_valid(gdf)

    # prune branches with min order of 2
    branches_to_prune = hy.streams.find_branches_to_prune(gdf)
    gdf = hy.streams.prune_branches(gdf, branches_to_prune=branches_to_prune)
    logging.info(f'After pruning branches, shape is {gdf.shape}')
    hy.topology.assert_topology_is_valid(gdf)

    # consolidate shorter streams into their neighbors where possible targeting minimum 2km length
    consolidations = hy.streams.find_short_streams(gdf, min_length=2000)
    gdf = hy.streams.consolidate_short_streams(gdf, consolidations=consolidations)
    logging.info(f'After consolidating short streams, shape is {gdf.shape}')
    hy.topology.assert_topology_is_valid(gdf)

    # reset the topological numbering after all modifications are done
    gdf = gdf.sort_values(hy.schema.topo_sort).reset_index(drop=True)
    gdf[hy.schema.topo_sort] = np.arange(len(gdf), dtype=np.int32)

    with open(os.path.join(outputs_dir, 'mods', 'lake_edits.json'), 'w') as f:
        json.dump(lake_edits, f)
    with open(os.path.join(outputs_dir, 'mods', 'zero_length_streams.json'), 'w') as f:
        json.dump(zero_lengths, f)
    with open(os.path.join(outputs_dir, 'mods', 'coastal_orphans.json'), 'w') as f:
        json.dump(coastal_orphans, f)
    with open(os.path.join(outputs_dir, 'mods', 'headwater_dissolves.json'), 'w') as f:
        json.dump(header_mergers, f)
    with open(os.path.join(outputs_dir, 'mods', 'branches_to_prune.json'), 'w') as f:
        json.dump(branches_to_prune, f)
    with open(os.path.join(outputs_dir, 'mods', 'short_consolidations.json'), 'w') as f:
        json.dump(consolidations, f)

    # length is in m, divide by estimated m/s to get k in seconds
    gdf[hy.schema.static_velocity_factor] = (
            np.exp(0.10 * np.log(gdf[hy.schema.tdx_ds_area_field]) - 4.68).round(3) + 0.1
    )
    gdf[hy.schema.static_musk_k] = gdf[hy.schema.length] / gdf[hy.schema.static_velocity_factor]
    gdf[hy.schema.static_musk_k] = gdf[hy.schema.static_musk_k].round(0).astype(int)
    gdf[hy.schema.static_musk_x] = 0.20

    gdf = gdf[hy.schema.final_columns_to_keep]

    logging.info('Writing final outputs')
    gdf.to_parquet(final_geoparquet_output)
    logging.info(f'Final streams written to {final_geoparquet_output}')
    gdf.drop(columns=hy.schema.geometry).to_parquet(final_metadata_output)
    logging.info(f'Metadata written to {final_metadata_output}')
    gdf.set_geometry(gdf.simplify(tolerance=10)).to_parquet(simple_streams_output)
    logging.info(f'Simplified streams written to {simple_streams_output}')
    # full-resolution streams in web mercator with coordinates rounded to whole metres; the
    # source for the map tiles built in stream_revisions.sh
    hy.pmtiling.to_mapping_geometry(gdf).to_parquet(mapping_streams_output)
    logging.info(f'Mapping streams written to {mapping_streams_output}')

    confluences = (
        gdf
        .groupby(hy.schema.next_river_id)[hy.schema.river_id]
        .agg(list)
        .reset_index()
        .rename(columns={hy.schema.next_river_id: hy.schema.river_id, hy.schema.river_id: 'upstream_ids'})
    )
    reach_start_points = (
        gdf
        .set_index(hy.schema.river_id).geometry
        .apply(lambda g: Point(shapely.get_coordinates(g)[0]))
        .to_dict()
    )
    confluences[hy.schema.geometry] = confluences[hy.schema.river_id].map(reach_start_points)
    confluences['upstream_ids'] = confluences['upstream_ids'].apply(lambda x: ','.join(map(str, x)))
    confluences = gpd.GeoDataFrame(confluences, geometry=hy.schema.geometry, crs=gdf.crs)
    confluences.to_parquet(confluences_output)
    logging.info(f'Confluences written to {confluences_output}')
