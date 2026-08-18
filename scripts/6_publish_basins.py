#!/usr/bin/env python
"""Stamp the frozen global basins with this release's ids and dissolve the group outlines."""
import json
import logging
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

LEVELS = [2, 3, 4, 5, 6, 7, 8]

BAND_COLUMNS = [hy.schema.river_id, hy.schema.river_index, 'basinId', 'level', 'pfafCode',
                'riverCount', hy.schema.area, 'strahlerOrder']

band_dir = hy.paths.pmtiles_root / 'basin_bands'


def band_paths(level: int) -> tuple:
    poly = band_dir / f'basin_level{level}.fgb'
    lines = band_dir / f'basin_level{level}.lines.fgb'
    return (None if level == 2 else poly), lines


def write_band_feed(basins: gpd.GeoDataFrame, level: int) -> None:
    band_dir.mkdir(parents=True, exist_ok=True)
    frame = basins.sort_values(hy.schema.area, ascending=False, ignore_index=True)
    frame = frame[BAND_COLUMNS + [hy.schema.geometry]].copy()
    for column in frame.columns:
        if pd.api.types.is_integer_dtype(frame[column]):
            frame[column] = frame[column].astype('float64')
    options = dict(driver='FlatGeobuf', promote_to_multi=True, SPATIAL_INDEX='NO')
    poly, lines = band_paths(level)
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

    published_of = {lv: hy.paths.global_root / f'basins_level{lv}.geo.parquet' for lv in LEVELS}
    feeds_of = {lv: [p for p in band_paths(lv) if p is not None] for lv in LEVELS}
    boundaries_out = hy.paths.global_root / 'groups.geo.parquet'
    outputs = [*published_of.values(), *[p for feeds in feeds_of.values() for p in feeds],
               boundaries_out]
    if all(path.exists() for path in outputs):
        # exit 0, not 1: pipeline.sh runs under `set -e`, so a step with nothing to do must report
        # success or it aborts the whole run. sys.exit(<string>) prints to stderr and exits 1.
        print('all outputs exist, nothing to do')
        sys.exit(0)

    metadata_path = hy.paths.global_root / 'metadata.parquet'
    if not metadata_path.exists():
        sys.exit(f'{metadata_path} not found - run 5_concatenate_global.py first')
    sources = {lv: hy.paths.global_basins_root / f'basins_level{lv}.geo.parquet' for lv in LEVELS}
    missing = [p.name for p in sources.values() if not p.exists()]
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
        out = published_of[level]
        published = None
        if not out.exists():
            basins = gpd.read_parquet(sources[level])
            pour = basins[hy.schema.tdx_link_no_field].to_numpy().astype(np.int64)
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

        if not all(p.exists() for p in feeds_of[level]):
            if published is None:
                published = gpd.read_parquet(out)
            write_band_feed(published, level)
            names = ' + '.join(p.name for p in feeds_of[level])
            logging.info(f'level {level}: band feed -> {names}')
            print(f'level {level}: band feed -> {names}')

    group_ids = sorted(int(g) for g in metadata[hy.schema.group_id].unique())
    catchment_paths = {g: hy.paths.group_dir(g) / f'catchments_{g}.geo.parquet' for g in group_ids}
    absent = [p for p in catchment_paths.values() if not p.exists()]
    if absent:
        sys.exit(f'{len(absent)} group catchment file(s) missing, e.g. {absent[0]} - '
                 f'run 5_concatenate_global.py first')
    if boundaries_out.exists():
        print(f'{boundaries_out.name} exists, skipped')
    else:
        t0 = time.time()

        def outline_of(group_id: int):
            geometries = gpd.read_parquet(catchment_paths[group_id],
                                          columns=[hy.schema.geometry]).geometry.to_numpy()
            geometries, _ = hy.geometry.repair(geometries)
            outline = hy.geometry.union_coverage(geometries)
            if outline is None:
                return group_id, None, len(geometries)
            snapped = hy.projection.snap_to_grid(outline)
            if not shapely.is_valid(snapped):
                snapped = shapely.set_precision(outline, hy.projection.precision_meters)
            fixed, _ = hy.geometry.repair(np.array([snapped], dtype=object), np.array([outline],
                                                                                     dtype=object))
            return group_id, fixed[0], len(geometries)

        outlines = {}
        for group_id in group_ids:
            started_group = time.time()
            group_id, outline, count = outline_of(group_id)
            outlines[group_id] = outline
            logging.info(f'group {group_id}: {count:,} catchments dissolved in '
                         f'{time.time() - started_group:.1f}s')
        empty = [g for g, o in outlines.items() if o is None or o.is_empty]
        if empty:
            raise RuntimeError(f'group(s) {empty[:5]} dissolved to nothing')

        dissolved = [outlines[g] for g in group_ids]
        tree = shapely.STRtree(dissolved)
        filled, closed, trimmed = [], 0, 0
        for index, group_id in enumerate(group_ids):
            geometry, shut, around = hy.geometry.fill_holes(dissolved[index], tree, skip=index)
            filled.append(geometry)
            closed += shut
            trimmed += around
            if shut or around:
                logging.info(f'group {group_id}: {shut:,} hole(s) closed, '
                             f'{around:,} closed around an occupant')
        repaired, lost = hy.geometry.repair(np.array(filled, dtype=object),
                                            np.array(dissolved, dtype=object))
        if lost:
            logging.warning(f'{lost} filled outline(s) kept their unfilled geometry')
        outlines = dict(zip(group_ids, repaired))
        logging.info(f'{closed:,} hole(s) closed and {trimmed:,} closed around an occupant, '
                     f'across {len(group_ids)} outlines')
        print(f'{closed + trimmed:,} holes closed in the group outlines '
              f'({trimmed:,} around something standing in them)')

        ordered = [outlines[g] for g in group_ids]
        tree = shapely.STRtree(ordered)
        overlaps = []
        for left, right in zip(*tree.query(ordered, predicate='intersects')):
            if left >= right:
                continue
            area = shapely.area(shapely.intersection(ordered[left], ordered[right]))
            if area > 0:
                overlaps.append((area, group_ids[left], group_ids[right]))
        if overlaps:
            overlaps.sort(reverse=True)
            for area, left, right in overlaps[:20]:
                logging.warning(f'groups {left} and {right} overlap by {area / 1e6:.3f} km2')
            print(f'WARNING: {len(overlaps)} overlapping group pair(s), '
                  f'{sum(a for a, _, _ in overlaps) / 1e6:.3f} km2 total, worst '
                  f'{overlaps[0][1]}/{overlaps[0][2]} at {overlaps[0][0] / 1e6:.3f} km2')
        else:
            logging.info(f'no overlap between any of the {len(group_ids)} group outlines')
            print(f'{len(group_ids)} group outlines, none overlapping')

        crs = f'EPSG:{hy.projection.web_mercator_epsg}'
        for group_id in group_ids:
            boundary = gpd.GeoDataFrame(geometry=[outlines[group_id]], crs=crs)
            hy.parquet.write_geoparquet(
                boundary, hy.paths.group_dir(group_id) / f'boundary_{group_id}.geo.parquet',
                row_group_size=None)
        stacked = gpd.GeoDataFrame(
            {hy.schema.group_id: np.array(group_ids, dtype='int32')},
            geometry=[outlines[g] for g in group_ids], crs=crs)
        hy.parquet.write_geoparquet(stacked, boundaries_out, row_group_size=1)
        logging.info(f'{len(stacked)} group boundaries dissolved from the published catchments '
                     f'in {time.time() - t0:.0f}s -> {boundaries_out.name}')
        print(f'{len(stacked)} group boundaries from the published catchments, '
              f'{time.time() - t0:.0f}s')

    print(f'done, {time.time() - started:.0f}s')
