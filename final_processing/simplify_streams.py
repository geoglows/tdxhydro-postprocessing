import json
import logging
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from natsort import natsorted

import hydrography as hy

HILBERT_BITS = 16

# every file this step writes into the region's published mods/ directory
MODS_FILES = ('lake_edits.json', 'zero_length_streams.json', 'coastal_orphans.json',
              'headwater_dissolves.json', 'branches_to_prune.json',
              'short_consolidations.json')

region_root = hy.paths.region_root
tdx_root = hy.paths.tdx_root
network_data_root = hy.paths.network_data_root
logs_root = hy.paths.logs_root

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: simplify_streams.py <region>')
    region = int(sys.argv[1])

    final_geoparquet_output = region_root / f'{region}' / f'streams_{region}.geo.parquet'
    final_metadata_output = region_root / f'{region}' / f'metadata_{region}.parquet'
    confluences_output = region_root / f'{region}' / f'confluences_{region}.geo.parquet'
    # the edits are published, not scratch, and nothing else can regenerate them, so they count
    # as outputs of this step: a region whose mods are gone is rebuilt rather than skipped
    mods_dir = hy.paths.mods_dir(region)
    koppen_path = hy.paths.koppen_part(region)
    country_path = hy.paths.country_part(region)
    outputs = [final_geoparquet_output, final_metadata_output, confluences_output]
    outputs += [mods_dir / name for name in MODS_FILES]
    if all(output.exists() for output in outputs):
        print(f'region {region}: streams exist, skipping')
        sys.exit(0)

    if not koppen_path.exists():
        sys.exit(f'region {region}: {koppen_path} not found - run raw_processing/koppen_classifications.py first')
    if not country_path.exists():
        sys.exit(f'region {region}: {country_path} not found - run raw_processing/country_classifications.py first')

    (region_root / f'{region}').mkdir(parents=True, exist_ok=True)
    mods_dir.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logs_root / f'simplify_streams_{region}.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )
    gdf = gpd.read_parquet(tdx_root / f'TDX_streamnet_{region}_01.parquet')
    logging.info(f'Initial shape: {gdf.shape}')

    gdf[hy.schema.area] = hy.streams.local_area(gdf[hy.schema.tdx_link_field].to_numpy(),
                                                gdf[hy.schema.tdx_ds_link_field].to_numpy(),
                                                gdf[hy.schema.tdx_us_area_field].to_numpy(),
                                                gdf[hy.schema.tdx_ds_area_field].to_numpy())
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

    # Before the topology, not after: repointing a terminal arm at the rest of its own lake
    # changes what accumulates where, which tree a reach belongs to, and whether the sub-250 km²
    # terminal drop below takes it. All of those read the topology this line builds.
    # read from the raw network: once applied, the arms are no longer terminal and cannot be found again
    lake_links = hy.lakes.outlet_links(gdf)
    if lake_links:
        gdf = hy.lakes.apply_outlet_links(gdf, lake_links)
        logging.info(f'{len(lake_links)} lake arm(s) linked across their own water surface')

    gdf = hy.topology.compute_topology(gdf)

    # The drop lists are partitioned by region, one directory each, because an outletRiverId can
    # only ever name a watershed in one of them: the raw id is LINKNO + header * 1e7 and each TDX
    # header belongs to exactly one HydroBASINS level-2 region. A region reading the whole
    # directory was therefore testing 204k ids to use a few thousand, and, worse, gave no way to
    # see what had been decided for a given region without filtering every file by id prefix.
    # A region with nothing dropped has no directory, which is why this does not require one.
    drop_dir = network_data_root / 'dropped_watersheds' / f'{region}'
    drop_lists = natsorted(drop_dir.glob('*.csv'), key=str)
    logging.info(f'{len(drop_lists)} drop list(s) for region {region}: '
                 f'{[p.stem for p in drop_lists]}')
    # Each id may be listed once across the region's lists, so deleting it from the one list that
    # has it is enough to bring the watershed back. A second copy would silently keep it dropped.
    drop_tables = {p: pd.read_csv(p)['drop'].to_numpy() for p in drop_lists}
    listed = pd.Series(np.concatenate([np.empty(0, dtype=np.int64), *drop_tables.values()]))
    if listed.duplicated().any():
        repeated = listed[listed.duplicated()].unique()
        raise RuntimeError(f'{len(repeated):,} id(s) are listed more than once in {drop_dir}, '
                           f'e.g. {repeated[:5].tolist()}')
    for drop_list, drop_ids in drop_tables.items():
        gdf = gdf[~gdf[hy.schema.last_river_id].isin(drop_ids)]
        logging.info(f'After dropping {drop_list.stem}, shape is {gdf.shape}')

    to_drop = (
        gdf
        [np.logical_and(gdf[hy.schema.tdx_ds_area_field] < 250_000_000, gdf[hy.schema.next_river_id] == -1)]
        [hy.schema.last_river_id]
        .tolist()
    )
    gdf = gdf[~gdf[hy.schema.last_river_id].isin(to_drop)]

    protected = hy.lakes.lake_outlets(gdf, lake_links)
    logging.info(f'{len(protected):,} lake outlets are protected from simplification')

    lake_edits = hy.lakes.find_lake_edits(gdf, lake_links, min_inlet_area=100_000_000)
    gdf = hy.lakes.apply_lake_edits(gdf, lake_edits)
    logging.info(f'After applying lake edits, shape is {gdf.shape}')
    hy.topology.assert_topology_is_valid(gdf)

    zero_lengths = hy.streams.find_zero_length(gdf)
    gdf = hy.streams.remove_zero_length(gdf, zero_length_json=zero_lengths)
    logging.info(f'After correcting zero length rivers, shape is {gdf.shape}')
    hy.topology.assert_topology_is_valid(gdf)

    coastal_orphans = hy.streams.find_orphaned_coastal_outlets(gdf, protected=protected)
    # dropped, not folded: their area is not added to any other reach and their catchments are
    # not published, so the land they drained leaves the dataset like a dropped watershed's
    gdf = hy.streams.prune_branches(gdf, branches_to_prune=coastal_orphans, fold_area=False)
    logging.info(f'After dropping orphaned coastal order-1s, shape is {gdf.shape}')

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

    # published provenance: the record of what this step did to the source TDX-Hydro, written
    # once, in the place it is published from. create_catchments.py reads it back from there.
    for name, edit in (('lake_edits.json', lake_edits),
                       ('zero_length_streams.json', zero_lengths),
                       ('coastal_orphans.json', coastal_orphans),
                       ('headwater_dissolves.json', header_mergers),
                       ('branches_to_prune.json', branches_to_prune),
                       ('short_consolidations.json', consolidations)):
        with open(mods_dir / name, 'w') as f:
            json.dump(edit, f)
    logging.info(f'Edits written to {mods_dir}')

    # Koppen-Geiger class of each reach's own catchment. raw_processing/koppen_classifications.py measured the
    # land area per class of every raw catchment; a published reach's catchment is the union of
    # the raw catchments create_catchments.py will dissolve into it, grouped exactly as create_catchments.py groups them, so
    # their areas are summed per keeper and the class with the most land wins. A merged reach is
    # thereby labelled by all of its land, not only by the raw reach whose id it kept.
    area_columns = list(hy.koppen.AREA_COLUMNS)
    koppen = pd.read_parquet(koppen_path, columns=[hy.schema.tdx_link_no_field] + area_columns)
    keeper, deleted = hy.edits.catchment_keepers(koppen[hy.schema.tdx_link_no_field].to_numpy(np.int64),
                                                 mods_dir)
    keep = ~np.isin(keeper, list(deleted))
    reach_areas = koppen.loc[keep, area_columns].groupby(keeper[keep]).sum()
    unlabelled = ~gdf[hy.schema.river_id].isin(reach_areas.index)
    if unlabelled.any():
        raise RuntimeError(f'{int(unlabelled.sum()):,} reach(es) have no catchment in {koppen_path.name}, '
                           f'e.g. {gdf.loc[unlabelled, hy.schema.river_id].tolist()[:5]}; '
                           f'rerun raw_processing/koppen_classifications.py for region {region}')
    reach_areas = reach_areas.reindex(gdf[hy.schema.river_id].to_numpy())
    gdf[hy.schema.koppen] = hy.koppen.dominant(reach_areas.to_numpy(np.float64))
    logging.info(f'Koppen classes ({hy.paths.koppen_period()}) summed from {koppen_path.name}, '
                 f'{int(gdf[hy.schema.koppen].isna().sum()):,} reach(es) with no land class')

    # The country of each reach's own catchment, by the same grouping: the raw catchments' land per
    # country (raw_processing/country_classifications.py) summed per keeper, the most land winning.
    # A reach with no land in any country (an offshore sliver) gets null.
    country = pd.read_parquet(country_path, columns=[hy.schema.tdx_link_no_field, hy.countries.NAME, 'km2'])
    keeper, deleted = hy.edits.catchment_keepers(country[hy.schema.tdx_link_no_field].to_numpy(np.int64),
                                                 mods_dir)
    keep = ~np.isin(keeper, list(deleted))
    winner = hy.countries.dominant(keeper[keep], country.loc[keep, hy.countries.NAME].to_numpy(),
                                   country.loc[keep, 'km2'].to_numpy())
    gdf[hy.schema.country] = pd.array(winner.reindex(gdf[hy.schema.river_id].to_numpy()).to_numpy(),
                                      dtype='string')
    logging.info(f'Countries summed from {country_path.name}, '
                 f'{int(gdf[hy.schema.country].isna().sum()):,} reach(es) with no country')

    gdf = hy.schema.enforce_int32(gdf)

    gdf = gdf[hy.schema.final_columns_to_keep + [hy.schema.lat_field, hy.schema.lon_field]]
    streams = gdf[hy.schema.final_columns_to_keep]

    logging.info('Writing final outputs')
    streams = hy.projection.to_web_mercator(streams)
    vertices = int(shapely.get_num_coordinates(streams[hy.schema.geometry].values).sum())
    logging.info(f'Stream geometry kept at source resolution: {vertices:,} vertices')
    hy.parquet.write_geoparquet(streams, final_geoparquet_output)
    logging.info(f'Final streams written to {final_geoparquet_output}')
    # scratch, not published: concatenate_global.py reads this whole, so it takes the one-row-group default
    hy.parquet.write_parquet(gdf[hy.schema.metadata_columns_to_keep], final_metadata_output,
                             row_group_size=None)
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
