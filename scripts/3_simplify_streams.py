import json
import logging
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from natsort import natsorted

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

HILBERT_BITS = 16

region_root = hy.paths.region_root
tdx_root = hy.paths.tdx_root
network_data_root = hy.paths.network_data_root
logs_root = hy.paths.logs_root

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: 3_simplify_streams.py <region>')
    region = int(sys.argv[1])

    final_geoparquet_output = region_root / f'{region}' / f'streams_{region}.geo.parquet'
    final_metadata_output = region_root / f'{region}' / f'metadata_{region}.parquet'
    confluences_output = region_root / f'{region}' / f'confluences_{region}.geo.parquet'
    outputs = [final_geoparquet_output, final_metadata_output, confluences_output]
    if all(output.exists() for output in outputs):
        print(f'region {region}: streams exist, skipping')
        sys.exit(0)

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

    gdf[hy.schema.area] = gdf[hy.schema.tdx_ds_area_field] - gdf[hy.schema.tdx_us_area_field]
    id_column = hy.schema.tdx_link_no_field if hy.schema.tdx_link_no_field in gdf.columns \
        else hy.schema.tdx_link_field
    gdf[hy.schema.river_id] = gdf[id_column].astype(int)
    to_global = pd.Series(gdf[hy.schema.river_id].to_numpy(),
                          index=gdf[hy.schema.tdx_link_field].to_numpy())
    ds_local = gdf[hy.schema.tdx_ds_link_field].to_numpy()
    ds_global = to_global.reindex(ds_local).to_numpy()
    dangling = (ds_local != -1) & pd.isna(ds_global)
    if dangling.any():
        raise RuntimeError(f'{int(dangling.sum()):,} reach(es) flow into an id that is not in the '
                           f'file, e.g. DSLINKNO {ds_local[dangling][:5].tolist()}')
    gdf[hy.schema.next_river_id] = np.where(ds_local == -1, -1,
                                            np.nan_to_num(ds_global, nan=-1)).astype(int)
    gdf = (
        gdf
        .drop(columns=[hy.schema.tdx_link_field, hy.schema.tdx_ds_link_field], errors='ignore')
        .drop(columns=[hy.schema.tdx_link_no_field], errors='ignore')
        .rename(columns=hy.schema.rename_map)
    )

    gdf = hy.topology.compute_topology(gdf)

    drop_lists = natsorted((network_data_root / 'dropped_watersheds').glob('*.csv'), key=str)
    for drop_list in drop_lists:
        drop_ids = pd.read_csv(drop_list).values.flatten()
        gdf = gdf[~gdf[hy.schema.last_river_id].isin(drop_ids)]
        logging.info(f'After dropping {drop_list}, shape is {gdf.shape}')

    to_drop = (
        gdf
        [np.logical_and(gdf[hy.schema.tdx_ds_area_field] < 250_000_000, gdf[hy.schema.next_river_id] == -1)]
        [hy.schema.last_river_id]
        .tolist()
    )
    gdf = gdf[~gdf[hy.schema.last_river_id].isin(to_drop)]

    groups_df = pd.read_csv(network_data_root / 'groupIds_table.csv')
    group_id_map = groups_df.set_index(hy.schema.last_river_id)[hy.schema.group_id].to_dict()
    gdf[hy.schema.group_id] = gdf[hy.schema.last_river_id].map(group_id_map)
    missing_group = gdf.loc[gdf[hy.schema.group_id].isna(), hy.schema.last_river_id].unique()
    if len(missing_group):
        gdf.to_parquet(outputs_dir / 'mods' / 'missing_group_debug.parquet')
        raise RuntimeError(f'{len(missing_group)} reaches have no groupId; e.g. {missing_group[:10]}')

    protected = hy.lakes.lake_outlets(gdf)
    logging.info(f'{len(protected):,} lake outlets are protected from simplification')

    lake_edits = hy.lakes.find_lake_edits(gdf, min_inlet_area=100_000_000)
    gdf = hy.lakes.apply_lake_edits(gdf, lake_edits)
    logging.info(f'After applying lake edits, shape is {gdf.shape}')
    hy.topology.assert_topology_is_valid(gdf)

    zero_lengths = hy.streams.find_zero_length(gdf)
    gdf = hy.streams.remove_zero_length(gdf, zero_length_json=zero_lengths)
    logging.info(f'After correcting zero length rivers, shape is {gdf.shape}')
    hy.topology.assert_topology_is_valid(gdf)

    coastal_orphans = hy.streams.find_orphaned_coastal_outlets(gdf, protected=protected)
    gdf = hy.streams.prune_branches(gdf, branches_to_prune=coastal_orphans)
    logging.info(f'After consolidating orphaned coastal order-1s, shape is {gdf.shape}')

    gdf = hy.topology.recompute_outlets(gdf)
    hy.topology.assert_topology_is_valid(gdf)

    header_mergers = hy.streams.find_headwater_mergers(gdf, min_order=2, protected=protected)
    gdf = hy.streams.merge_headwaters_order2_geom(gdf, header_mergers=header_mergers)
    logging.info(f'After dissolving headwaters, shape is {gdf.shape}')
    hy.topology.assert_topology_is_valid(gdf)

    branches_to_prune = hy.streams.find_branches_to_prune(gdf, protected=protected)
    gdf = hy.streams.prune_branches(gdf, branches_to_prune=branches_to_prune)
    logging.info(f'After pruning branches, shape is {gdf.shape}')
    hy.topology.assert_topology_is_valid(gdf)

    consolidations = hy.streams.find_short_streams(gdf, min_length=2000, protected=protected)
    gdf = hy.streams.consolidate_short_streams(gdf, consolidations=consolidations)
    logging.info(f'After consolidating short streams, shape is {gdf.shape}')
    hy.topology.assert_topology_is_valid(gdf)

    gdf = hy.topology.nested_set_order(gdf, bits=HILBERT_BITS)
    gdf[hy.schema.river_index] = np.arange(len(gdf), dtype=np.int32)
    logging.info(f'Ordered {len(gdf):,} reaches upstream-to-downstream, region-local riverIndex '
                 f'stamped')

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

    gdf[hy.schema.static_velocity_factor] = (
            np.exp(0.10 * np.log(gdf[hy.schema.tdx_ds_area_field]) - 4.68).round(3) + 0.1
    )
    gdf[hy.schema.static_musk_k] = gdf[hy.schema.length] / gdf[hy.schema.static_velocity_factor]
    gdf[hy.schema.static_musk_k] = gdf[hy.schema.static_musk_k].round(0).astype(int)
    gdf[hy.schema.static_musk_x] = 0.20

    gdf = hy.schema.enforce_int32(gdf)

    gdf = gdf[hy.schema.final_columns_to_keep + [hy.schema.lat_field, hy.schema.lon_field]]
    streams = gdf[hy.schema.final_columns_to_keep]

    logging.info('Writing final outputs')
    streams = hy.projection.to_web_mercator(streams)
    vertices = int(shapely.get_num_coordinates(streams[hy.schema.geometry].values).sum())
    logging.info(f'Stream geometry kept at source resolution: {vertices:,} vertices')
    hy.parquet.write_geoparquet(streams, final_geoparquet_output)
    logging.info(f'Final streams written to {final_geoparquet_output}')
    hy.parquet.write_parquet(gdf[hy.schema.metadata_columns_to_keep], final_metadata_output)
    logging.info(f'Metadata written to {final_metadata_output}')

    confluences = (
        gdf
        [gdf[hy.schema.next_river_id] != -1]
        .groupby(hy.schema.next_river_id)[hy.schema.river_id]
        .agg(list)
        .reset_index()
        .rename(columns={hy.schema.next_river_id: hy.schema.river_id, hy.schema.river_id: 'upstream_ids'})
    )
    outlet_points = dict(zip(
        gdf[hy.schema.river_id],
        gpd.points_from_xy(gdf[hy.schema.lon_field], gdf[hy.schema.lat_field]),
    ))
    confluences[hy.schema.geometry] = confluences['upstream_ids'].apply(lambda ids: outlet_points[ids[0]])
    confluences['upstream_ids'] = confluences['upstream_ids'].apply(lambda x: ','.join(map(str, x)))
    confluences = gpd.GeoDataFrame(confluences, geometry=hy.schema.geometry, crs=gdf.crs)
    confluences = hy.projection.to_web_mercator(confluences)
    confluences = hy.schema.enforce_int32(confluences)
    hy.parquet.write_geoparquet(confluences, confluences_output, row_group_size=None)
    logging.info(f'Confluences written to {confluences_output}')
