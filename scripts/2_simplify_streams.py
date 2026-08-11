import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from natsort import natsorted

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

# 16 bits puts the Hilbert grid at 65,536 cells across the globe, ~600 m at the equator — finer than
# any reach's outlet point needs in order to be distinguished from its neighbour's.
HILBERT_BITS = 16
SIMPLIFY_TOLERANCE_METERS = 10.0

# every output path hangs off the data root - see hydrography/paths.py and $RFS_DATA_ROOT
region_root = hy.paths.region_root
tdx_root = hy.paths.tdx_root
network_data_root = hy.paths.network_data_root
logs_root = hy.paths.logs_root

if __name__ == '__main__':
    # find the ID of the region to process
    if len(sys.argv) != 2:
        sys.exit('usage: 2_simplify_streams.py <region>')
    region = int(sys.argv[1])
    # region = 1020000010  # Example region number

    # final outputs to check for existence before computing
    final_geoparquet_output = region_root / f'{region}' / f'streams_{region}.geo.parquet'
    final_metadata_output = region_root / f'{region}' / f'metadata_{region}.parquet'
    confluences_output = region_root / f'{region}' / f'confluences_{region}.geo.parquet'
    outputs = [final_geoparquet_output, final_metadata_output, confluences_output]
    if all(output.exists() for output in outputs):
        print(f'All final outputs for region {region} already exist, skipping')
        sys.exit(0)

    # prepare directories and logging
    outputs_dir = region_root / f'{region}'
    (outputs_dir / 'mods').mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logs_root / f'simplify_streams_{region}.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    gdf = gpd.read_parquet(tdx_root / f'TDX_streamnet_{region}_01.parquet')
    logging.info(f'Initial shape: {gdf.shape}')

    # the standardized parquets already carry the outlet point; derive it for older files
    # while the geometry is still single-part lines and coordinate 0 is unambiguous
    if not {hy.schema.lon_field, hy.schema.lat_field}.issubset(gdf.columns):
        gdf = hy.streams.add_outlet_coordinates(gdf)

    # add unique river ids and attributes
    gdf[hy.schema.area] = gdf[hy.schema.tdx_ds_area_field] - gdf[hy.schema.tdx_us_area_field]
    with open(network_data_root / 'tdxhydro_splits' / 'tdx_header_numbers.json') as f:
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
    drop_lists = natsorted((network_data_root / 'dropped_watersheds').glob('*.csv'), key=str)
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

    # assign groups based on outletRiverId
    groups_df = pd.read_csv(network_data_root / 'groupIds_table.csv')
    group_id_map = groups_df.set_index(hy.schema.last_river_id)[hy.schema.group_id].to_dict()
    gdf[hy.schema.group_id] = gdf[hy.schema.last_river_id].map(group_id_map)
    missing_group = gdf.loc[gdf[hy.schema.group_id].isna(), hy.schema.last_river_id].unique()
    if len(missing_group):
        gdf.to_parquet(outputs_dir / 'mods' / 'missing_group_debug.parquet')
        raise RuntimeError(f'{len(missing_group)} reaches have no groupId; e.g. {missing_group[:10]}')

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
    # OLD: union the order-2 line with its order-1 upstream tributaries into one geometry
    # gdf = hy.streams.merge_headwaters(gdf, header_mergers=header_mergers)
    # NEW: keep only the order-2 geometry; order-1 tributaries are not mapped
    gdf = hy.streams.merge_headwaters_order2_geom(gdf, header_mergers=header_mergers)
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

    # Region-local nested-set ordering. riverIndex is NOT assigned here - it is a position in a
    # single global ordering and cannot be known while one region is processed alone, so step 3
    # redoes this traversal across all 50 regions at once. What this call is for is leaving the
    # region files in a sensible topological order and stamping upstreamCount and the recomputed
    # shreveOrder, which are region-local quantities (no reach drains across a region boundary).
    gdf = hy.topology.nested_set_order(gdf, bits=HILBERT_BITS)
    logging.info(f'Ordered {len(gdf):,} reaches upstream-to-downstream')

    with open(outputs_dir / 'mods' / 'lake_edits.json', 'w') as f:
        json.dump(lake_edits, f)
    with open(outputs_dir / 'mods' / 'zero_length_streams.json', 'w') as f:
        json.dump(zero_lengths, f)
    with open(outputs_dir / 'mods' / 'coastal_orphans.json', 'w') as f:
        json.dump(coastal_orphans, f)
    with open(outputs_dir / 'mods' / 'headwater_dissolves.json', 'w') as f:
        json.dump(header_mergers, f)
    with open(outputs_dir / 'mods' / 'branches_to_prune.json', 'w') as f:
        json.dump(branches_to_prune, f)
    with open(outputs_dir / 'mods' / 'short_consolidations.json', 'w') as f:
        json.dump(consolidations, f)

    # length is in m, divide by estimated m/s to get k in seconds
    gdf[hy.schema.static_velocity_factor] = (
            np.exp(0.10 * np.log(gdf[hy.schema.tdx_ds_area_field]) - 4.68).round(3) + 0.1
    )
    gdf[hy.schema.static_musk_k] = gdf[hy.schema.length] / gdf[hy.schema.static_velocity_factor]
    gdf[hy.schema.static_musk_k] = gdf[hy.schema.static_musk_k].round(0).astype(int)
    gdf[hy.schema.static_musk_x] = 0.20

    # Ids and indices go out as int32 (see schema.enforce_int32). Cast here, once, before the
    # column selection, so every output below inherits it rather than each write repeating it.
    gdf = hy.schema.enforce_int32(gdf)

    # lat/lon are metadata only - the geometry already carries them in the streams outputs
    gdf = gdf[hy.schema.final_columns_to_keep + [hy.schema.lat_field, hy.schema.lon_field]]
    streams = gdf[hy.schema.final_columns_to_keep]

    logging.info('Writing final outputs')
    streams = hy.projection.to_web_mercator(streams)
    before = int(shapely.get_num_coordinates(streams[hy.schema.geometry].values).sum())
    streams[hy.schema.geometry] = shapely.simplify(
        streams[hy.schema.geometry].values, SIMPLIFY_TOLERANCE_METERS, preserve_topology=True)
    after = int(shapely.get_num_coordinates(streams[hy.schema.geometry].values).sum())
    logging.info(f'Simplified stream geometry at {SIMPLIFY_TOLERANCE_METERS:g} m: '
                 f'{before:,} -> {after:,} vertices ({100 * after / before:.1f}%)')
    hy.parquet.write_geoparquet(streams, final_geoparquet_output)
    logging.info(f'Final streams written to {final_geoparquet_output}')
    hy.parquet.write_parquet(gdf[hy.schema.metadata_columns_to_keep], final_metadata_output)
    logging.info(f'Metadata written to {final_metadata_output}')

    # exclude the -1 group: those reaches leave the network, they do not meet at a junction
    confluences = (
        gdf
        [gdf[hy.schema.next_river_id] != -1]
        .groupby(hy.schema.next_river_id)[hy.schema.river_id]
        .agg(list)
        .reset_index()
        .rename(columns={hy.schema.next_river_id: hy.schema.river_id, hy.schema.river_id: 'upstream_ids'})
    )
    # the junction sits at the outlet of the upstream reaches, not at the outlet of the reach
    # they flow into. where a lake edit repointed an inlet, the inlet's own outlet point is
    # still the most defensible location for it
    outlet_points = dict(zip(
        gdf[hy.schema.river_id],
        gpd.points_from_xy(gdf[hy.schema.lon_field], gdf[hy.schema.lat_field]),
    ))
    confluences[hy.schema.geometry] = confluences['upstream_ids'].apply(lambda ids: outlet_points[ids[0]])
    confluences['upstream_ids'] = confluences['upstream_ids'].apply(lambda x: ','.join(map(str, x)))
    confluences = gpd.GeoDataFrame(confluences, geometry=hy.schema.geometry, crs=gdf.crs)
    confluences = hy.projection.to_web_mercator(confluences)
    # the groupby above rebuilds riverId as int64, so cast it back
    confluences = hy.schema.enforce_int32(confluences)
    hy.parquet.write_geoparquet(confluences, confluences_output, row_group_size=None)
    logging.info(f'Confluences written to {confluences_output}')
