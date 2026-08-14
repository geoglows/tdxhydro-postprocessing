#!/usr/bin/env python
"""
Duplicate the frozen global basins into the published dataset, stamped with this release's ids.

The basins themselves - codes, polygons, raw attributes - were generated once by 2_global_basins.py
and never change. What changes per release is the network they sit over: reaches are dropped and
merged, and riverIndex is a position in an ordering that only exists once step 4 has run. This
step is the bridge: it copies the per-level basin files and adds, for each basin, the release
reach its pour point survives as - ``riverId`` and ``riverIndex`` - so a lookup that starts from a
basin lands on a valid row of the published network, and a lookup that starts from a reach can
climb to its basins.

The mapping is the same one the revisions themselves record: a pour point that survives keeps its
id (revised ids are raw ids); one that was folded into a keeper follows the chain in the region's
``mods/`` journal to the surviving reach. A basin whose pour point has no release representation -
its watershed was dropped, or its whole region is not in this release - is left out of the
published copy and counted, because a row that cannot be looked up is not a lookup table.

    reads   $TDXHYDRO_ROOT/global_basins/basins_level{2..8}.geo.parquet, the frozen product
            hydrography/group=0/metadata.parquet, this release's ids and indices
            regions/<region>/mods/*.json, the edit journals
    writes  hydrography/group=0/basins_level{2..8}.geo.parquet
            pmtiles/basin_bands/basin_level{3..8}.fgb (+ .lines.fgb), the tiling feed
            pmtiles/basin_bands/basin_level2.lines.fgb, region boundaries, lines only
            group=<id>/boundary_<id>.geo.parquet and group=0/groups.geo.parquet, the group
            outlines, dissolved from the stamped level-8 basins (see below)

The fgb pairs are the basin half of tile_catchments.sh's input (step 5 cuts the leaf half): the
stamped, release-filtered basins with their attributes embedded, written where the geometry is
already in memory. The polygons go out largest-first so a pinprick enclave is written after - and
drawn above - the solid basin whose filled hole it sits in; the tiling passes
--preserve-input-order to keep that. Level 2 contributes boundaries only: the region divides join
the catchment_lines layer at every zoom, but region polygons are not a drawable band.

Run after 5_concatenate_global.py. Rerunning is cheap and idempotent; each level rewrites only
when its inputs are newer than the published copy, and the band feed only when the published copy
is newer than it.
"""
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

LEVELS = [2, 3, 4, 5, 6, 7, 8]

# what rides in the tiles: the same attribute set the old catchment tiles carried, so the map
# style and its -j filters keep working unchanged
BAND_COLUMNS = [hy.schema.river_id, hy.schema.river_index, 'basinId', 'level', 'pfafCode',
                'riverCount', hy.schema.area, 'strahlerOrder']

band_dir = hy.paths.pmtiles_root / 'basin_bands'


def band_paths(level: int) -> tuple:
    """(polygon fgb, lines fgb) for a level; level 2 has no polygon band."""
    poly = band_dir / f'basin_level{level}.fgb'
    lines = band_dir / f'basin_level{level}.lines.fgb'
    return (None if level == 2 else poly), lines


def write_band_feed(basins: gpd.GeoDataFrame, level: int) -> None:
    """One level's tiling feed. Same writer conventions as step 5's leaf band: integer columns as
    float64 for tippecanoe's -j reader, no spatial index, written aside and renamed."""
    band_dir.mkdir(parents=True, exist_ok=True)
    frame = basins.sort_values(hy.schema.area, ascending=False, ignore_index=True)
    frame = frame[BAND_COLUMNS + [hy.schema.geometry]].copy()
    for column in frame.columns:
        if pd.api.types.is_integer_dtype(frame[column]):
            frame[column] = frame[column].astype('float64')
    options = dict(driver='FlatGeobuf', promote_to_multi=True, SPATIAL_INDEX='NO')
    poly, lines = band_paths(level)
    # the aside name must keep the .fgb extension: handed anything else, GDAL's FlatGeobuf driver
    # treats the path as a directory dataset and buries the real file one level down - pyogrio
    # reads that back transparently, tippecanoe cannot mmap it
    if poly is not None:
        partial = poly.with_name(poly.name.replace('.fgb', '.partial.fgb'))
        pyogrio.write_dataframe(frame, partial, geometry_type='MultiPolygon', **options)
        partial.replace(poly)
    frame = frame.set_geometry(frame.geometry.boundary)
    partial = lines.with_name(lines.name.replace('.fgb', '.partial.fgb'))
    pyogrio.write_dataframe(frame, partial, geometry_type='MultiLineString', **options)
    partial.replace(lines)

MERGE_MOD_FILES = ('coastal_orphans.json', 'headwater_dissolves.json',
                   'branches_to_prune.json', 'short_consolidations.json')


def member_to_keeper_map(region_root: Path) -> dict:
    """Every merged-away raw reach mapped to its final surviving keeper, replayed from each
    processed region's edit journal (lake interiors fold into their lake outlet; the merge edits
    are already shaped {keeper: [members]}). Keeper chains are followed to their end."""
    direct = {}
    for mods in sorted(region_root.glob('*/mods')):
        with open(mods / 'lake_edits.json') as f:
            for outlet_id, edit in json.load(f).items():
                for deleted in edit.get('delete', []):
                    direct[int(deleted)] = int(outlet_id)
        for name in MERGE_MOD_FILES:
            with open(mods / name) as f:
                for keeper, members in json.load(f).items():
                    for member in members:
                        direct[int(member)] = int(keeper)
    terminal = {}
    for start in direct:
        node, path = start, []
        while node in direct and node not in terminal:
            path.append(node)
            node = direct[node]
        end = terminal.get(node, node)
        for reach in path:
            terminal[reach] = end
    return terminal


if __name__ == '__main__':
    hy.console.banner('Publish basins and group boundaries')
    metadata_path = hy.paths.global_root / 'metadata.parquet'
    if not metadata_path.exists():
        sys.exit(f'{metadata_path} not found - run 5_concatenate_global.py first')
    sources = {lv: hy.paths.global_basins_root / f'basins_level{lv}.geo.parquet' for lv in LEVELS}
    missing = [p.name for p in [*sources.values(), hy.paths.global_basins_root / "pfaf_codes.parquet"] if not p.exists()]
    if missing:
        sys.exit(f'frozen basins missing ({", ".join(missing)}) - run 2_global_basins.py first')

    hy.paths.logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=hy.paths.logs_root / 'publish_basins.log', filemode='w',
                        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    started = time.time()
    metadata = pd.read_parquet(metadata_path, columns=[hy.schema.river_id, hy.schema.river_index,
                                                       hy.schema.group_id])
    index_of = pd.Series(metadata[hy.schema.river_index].to_numpy(),
                         index=metadata[hy.schema.river_id].to_numpy())
    keeper_of = member_to_keeper_map(hy.paths.region_root)
    logging.info(f'{len(index_of):,} release reaches, {len(keeper_of):,} merged raw reaches mapped')

    for level in LEVELS:
        out = hy.paths.global_root / f'basins_level{level}.geo.parquet'
        published = None
        if not (out.exists() and out.stat().st_mtime >= max(sources[level].stat().st_mtime,
                                                            metadata_path.stat().st_mtime)):
            basins = gpd.read_parquet(sources[level])
            pour = basins[hy.schema.tdx_link_no_field].to_numpy().astype(np.int64)
            # a surviving pour point is its own release reach; a merged one follows its keeper
            release = np.where(pd.Series(pour).isin(index_of.index).to_numpy(), pour,
                               pd.Series(pour).map(keeper_of).fillna(-1).to_numpy().astype(np.int64))
            release[(release != -1) & ~pd.Series(release).isin(index_of.index).to_numpy()] = -1

            kept = release != -1
            published = basins.loc[kept].copy()
            published.insert(0, hy.schema.river_id, release[kept])
            published.insert(1, hy.schema.river_index,
                             index_of.reindex(release[kept]).to_numpy().astype(np.int64))
            published = hy.schema.enforce_int32(published)
            duplicates = int(published[hy.schema.river_id].duplicated().sum())

            partial = out.with_name(f'{out.name}.partial')
            hy.parquet.write_geoparquet(published, partial)
            partial.replace(out)
            logging.info(f'level {level}: {len(published):,} of {len(basins):,} basins published, '
                         f'{len(basins) - len(published):,} have no release reach, {duplicates:,} '
                         f'share one, {out.stat().st_size / 1e6:.0f} MB')
            print(f'level {level}: {len(published):,} of {len(basins):,} basins '
                  f'({len(basins) - len(published):,} outside this release) -> {out.name}')

        # the tiling feed follows the published copy: rebuilt when it is missing or older
        poly, lines = band_paths(level)
        stale = any(p is not None and (not p.exists() or p.stat().st_mtime < out.stat().st_mtime)
                    for p in (poly, lines))
        if stale:
            if published is None:
                published = gpd.read_parquet(out)
            write_band_feed(published, level)
            names = ' + '.join(p.name for p in (poly, lines) if p is not None)
            logging.info(f'level {level}: band feed -> {names}')
            print(f'level {level}: band feed -> {names}')
    # ------------------------------------------------------------------
    # Group boundaries, dissolved from the stamped level-8 basins.
    #
    # A group is a set of whole terminal watersheds and a level-8 basin never crosses a watershed
    # divide, so a group's outline is the union of its basins' polygons - a few hundred solid,
    # band-simplified shapes instead of the tens of thousands of full-resolution catchments the
    # old dissolve unioned (measured 2.5x faster on the worst group, with far fewer vertices).
    # The exception is the handful of coastal basins whose Hilbert-bundled watersheds straddle a
    # group border (24 of 380,745 measured): those are excluded wholly and BOTH their sides are
    # patched exactly from the published group catchments, so no minority area lands in the wrong
    # group. The same patch covers reaches whose basin has no published polygon at all.
    # ------------------------------------------------------------------
    boundaries_out = hy.paths.global_root / 'groups.geo.parquet'
    level8_out = hy.paths.global_root / 'basins_level8.geo.parquet'
    freshest = max(level8_out.stat().st_mtime, metadata_path.stat().st_mtime)
    if boundaries_out.exists() and boundaries_out.stat().st_mtime >= freshest:
        print('group boundaries newer than their inputs, skipped')
    else:
        t0 = time.time()
        basins8 = gpd.read_parquet(level8_out)
        # a code names a basin only within its region, so the basin key is region + code - keyed
        # on the bare code, identical prefixes from different regions collide into false straddlers
        codes = pd.read_parquet(hy.paths.global_basins_root / 'pfaf_codes.parquet',
                                columns=[hy.schema.tdx_link_no_field, hy.schema.tdx_region_field,
                                         'pfafCode'])
        basin_of = pd.Series((codes[hy.schema.tdx_region_field] + codes['pfafCode']).to_numpy(),
                             index=codes[hy.schema.tdx_link_no_field].to_numpy())
        del codes
        reach = pd.DataFrame({
            'basin': basin_of.reindex(metadata[hy.schema.river_id].to_numpy()).to_numpy(),
            'group': metadata[hy.schema.group_id].to_numpy(),
            hy.schema.river_id: metadata[hy.schema.river_id].to_numpy(),
        })
        if pd.isna(reach['basin']).any():
            raise RuntimeError('a release reach has no frozen code; rerun 2_global_basins.py')

        published_basins = set((basins8[hy.schema.tdx_region_field] + basins8['pfafCode']).tolist())
        groups_per_basin = reach.groupby('basin')['group'].nunique()
        straddlers = set(groups_per_basin[groups_per_basin > 1].index.tolist())
        clean = reach['basin'].isin(published_basins) & ~reach['basin'].isin(straddlers)
        patch_reaches = reach[~clean]
        basin_group = reach[clean].groupby('basin')['group'].first()
        logging.info(f'boundaries: {len(straddlers)} straddler basin(s); '
                     f'{len(patch_reaches):,} reach(es) patched from catchments')

        basins8['_basin'] = basins8[hy.schema.tdx_region_field] + basins8['pfafCode']
        basins8['_group'] = basins8['_basin'].map(basin_group)
        # a basin whose own reaches were all merged away carries a code no release reach has, so
        # the reach contingency cannot place it - but its drainage rides in its pour's keeper, and
        # the stamped riverId IS that keeper, so it belongs to the keeper's group (measured: 1,176
        # of 4,980 published basins in the heaviest-merged region, 5.8% of its area)
        keeper_group = pd.Series(metadata[hy.schema.group_id].to_numpy(),
                                 index=metadata[hy.schema.river_id].to_numpy())
        basins8['_group'] = basins8['_group'].fillna(
            basins8[hy.schema.river_id].map(keeper_group))
        by_group = {int(g): part.geometry.to_numpy()
                    for g, part in basins8.dropna(subset=['_group']).groupby('_group')}
        patches_by_group = {int(g): part[hy.schema.river_id].to_numpy()
                            for g, part in patch_reaches.groupby('group')}

        def outline_of(group_id: int):
            pieces = list(by_group.get(group_id, ()))
            wanted = patches_by_group.get(group_id)
            if wanted is not None and len(wanted):
                catchments = gpd.read_parquet(
                    hy.paths.group_dir(group_id) / f'catchments_{group_id}.geo.parquet',
                    columns=[hy.schema.river_id, hy.schema.geometry],
                    filters=[(hy.schema.river_id, 'in', wanted.tolist())])
                pieces.extend(catchments.geometry.to_numpy())
            if not pieces:
                return group_id, None
            outline = shapely.make_valid(shapely.union_all(np.array(pieces, dtype=object)))
            # solid, on the 1 m lattice, valid - the same finishing the catchment dissolve had
            shells = shapely.polygons(shapely.get_exterior_ring(shapely.get_parts(outline)))
            outline = shapely.union_all(shells)
            snapped = hy.projection.snap_to_grid(outline)
            if not shapely.is_valid(snapped):
                snapped = shapely.set_precision(outline, hy.projection.precision_meters)
            fixed, _ = hy.geometry.repair(np.array([snapped], dtype=object))
            return group_id, fixed[0]

        group_ids = sorted(int(g) for g in reach['group'].unique())
        with ThreadPoolExecutor(max_workers=8) as pool:
            outlines = dict(pool.map(lambda g: outline_of(g), group_ids))
        empty = [g for g, o in outlines.items() if o is None or o.is_empty]
        if empty:
            raise RuntimeError(f'group(s) {empty[:5]} dissolved to nothing')

        for group_id in group_ids:
            boundary = gpd.GeoDataFrame(geometry=[outlines[group_id]], crs=basins8.crs)
            hy.parquet.write_geoparquet(
                boundary, hy.paths.group_dir(group_id) / f'boundary_{group_id}.geo.parquet',
                row_group_size=None)
        stacked = gpd.GeoDataFrame(
            {hy.schema.group_id: np.array(group_ids, dtype='int32')},
            geometry=[outlines[g] for g in group_ids], crs=basins8.crs)
        # one row group per row: a row is a whole continent's divide, so the usual 500 would put
        # the entire world in a single fetch
        hy.parquet.write_geoparquet(stacked, boundaries_out, row_group_size=1)
        logging.info(f'{len(stacked)} group boundaries from the level-8 basins in '
                     f'{time.time() - t0:.0f}s -> {boundaries_out.name}')
        print(f'{len(stacked)} group boundaries from the level-8 basins '
              f'({len(straddlers)} straddler(s) patched exactly), {time.time() - t0:.0f}s')

    print(f'done, {time.time() - started:.0f}s')
