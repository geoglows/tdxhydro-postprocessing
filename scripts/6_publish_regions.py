#!/usr/bin/env python
"""Dissolve each region's published catchments into the outline of the level-2 region."""
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

# Regions dissolve independently and every heavy call in ``outline_of`` is a GEOS call that
# releases the GIL - the coverage union and the repair inside it are each a single call on one
# core, however many threads the elementwise helpers get - so the wall clock across regions is a
# scheduling question and nothing else. It ran this way until the dissolve moved from the
# band-simplified level-8 basins to the full-resolution catchments, where the pool was dropped
# along with everything else that version did. Sequential, the 107 partitions this replaced took
# 21.4 min; there are 47 regions now and the same catchments go through them.
default_jobs = 8


if __name__ == '__main__':
    hy.console.banner('Publish region boundaries')

    boundaries_out = hy.paths.global_root / 'regions.geo.parquet'
    if boundaries_out.exists():
        # exit 0, not 1: pipeline.sh runs under `set -e`, so a step with nothing to do must report
        # success or it aborts the whole run. sys.exit(<string>) prints to stderr and exits 1.
        print(f'{boundaries_out.name} exists, nothing to do')
        sys.exit(0)

    metadata_path = hy.paths.global_root / 'metadata.parquet'
    if not metadata_path.exists():
        sys.exit(f'{metadata_path} not found - run 5_concatenate_global.py first')

    hy.paths.logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=hy.paths.logs_root / 'publish_regions.log', filemode='w',
                        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    jobs = int(sys.argv[1]) if len(sys.argv) > 1 \
        else int(os.environ.get('BOUNDARY_JOBS', default_jobs))
    started = time.time()
    metadata = pd.read_parquet(metadata_path, columns=[hy.schema.tdx_region_field])
    regions = sorted(metadata[hy.schema.tdx_region_field].unique(), key=int)
    catchment_paths = {r: hy.paths.publish_dir(r) / f'catchments_{r}.geo.parquet' for r in regions}
    absent = [p for p in catchment_paths.values() if not p.exists()]
    if absent:
        sys.exit(f'{len(absent)} region catchment file(s) missing, e.g. {absent[0]} - '
                 f'run 5_concatenate_global.py first')

    def outline_of(region: str):
        started_region = time.time()
        geometries = gpd.read_parquet(catchment_paths[region],
                                      columns=[hy.schema.geometry]).geometry.to_numpy()
        geometries, _ = hy.geometry.repair(geometries)
        outline = hy.geometry.union_coverage(geometries)
        if outline is None:
            return region, None, len(geometries), time.time() - started_region
        # snap and repair, the same pair every other layer here is written with.
        # ``set_precision`` used to stand between them as the escalation for a ring that
        # rounding self-intersected, and it is the wrong tool twice over: ``repair`` already
        # fixes that ring, and ``set_precision`` refuses an invalid input outright - it threw
        # a side location conflict on one partition whose dissolve came back pinched before
        # anything here had rounded it.
        snapped = hy.projection.snap_to_grid(outline)
        fixed, _ = hy.geometry.repair(np.array([snapped], dtype=object), np.array([outline],
                                                                                 dtype=object))
        return region, fixed[0], len(geometries), time.time() - started_region

    outlines = {}
    print(f'dissolving {len(regions)} regions, {jobs} at a time')
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        # map, so the log reads in region order however the threads finish
        for region, outline, count, seconds in pool.map(outline_of, regions):
            outlines[region] = outline
            logging.info(f'region {region}: {count:,} catchments dissolved in {seconds:.1f}s')
    logging.info(f'{len(regions)} regions dissolved in {time.time() - started:.0f}s '
                 f'on {jobs} threads')
    print(f'{len(regions)} regions dissolved, {time.time() - started:.0f}s')
    empty = [r for r, o in outlines.items() if o is None or o.is_empty]
    if empty:
        raise RuntimeError(f'region(s) {empty[:5]} dissolved to nothing')

    dissolved = [outlines[r] for r in regions]
    tree = shapely.STRtree(dissolved)
    filled, closed, trimmed = [], 0, 0
    for index, region in enumerate(regions):
        geometry, shut, around = hy.geometry.fill_holes(dissolved[index], tree, skip=index)
        filled.append(geometry)
        closed += shut
        trimmed += around
        if shut or around:
            logging.info(f'region {region}: {shut:,} hole(s) closed, '
                         f'{around:,} closed around an occupant')
    repaired, lost = hy.geometry.repair(np.array(filled, dtype=object),
                                        np.array(dissolved, dtype=object))
    if lost:
        logging.warning(f'{lost} filled outline(s) kept their unfilled geometry')
    outlines = dict(zip(regions, repaired))
    logging.info(f'{closed:,} hole(s) closed and {trimmed:,} closed around an occupant, '
                 f'across {len(regions)} outlines')
    print(f'{closed + trimmed:,} holes closed in the region outlines '
          f'({trimmed:,} around something standing in them)')

    ordered = [outlines[r] for r in regions]
    tree = shapely.STRtree(ordered)
    overlaps = []
    for left, right in zip(*tree.query(ordered, predicate='intersects')):
        if left >= right:
            continue
        area = shapely.area(shapely.intersection(ordered[left], ordered[right]))
        if area > 0:
            overlaps.append((area, regions[left], regions[right]))
    if overlaps:
        overlaps.sort(reverse=True)
        for area, left, right in overlaps[:20]:
            logging.warning(f'regions {left} and {right} overlap by {area / 1e6:.3f} km2')
        print(f'WARNING: {len(overlaps)} overlapping region pair(s), '
              f'{sum(a for a, _, _ in overlaps) / 1e6:.3f} km2 total, worst '
              f'{overlaps[0][1]}/{overlaps[0][2]} at {overlaps[0][0] / 1e6:.3f} km2')
    else:
        logging.info(f'no overlap between any of the {len(regions)} region outlines')
        print(f'{len(regions)} region outlines, none overlapping')

    crs = f'EPSG:{hy.projection.web_mercator_epsg}'
    for region in regions:
        boundary = gpd.GeoDataFrame(geometry=[outlines[region]], crs=crs)
        hy.parquet.write_geoparquet(
            boundary, hy.paths.publish_dir(region) / f'boundary_{region}.geo.parquet',
            row_group_size=None)
    stacked = gpd.GeoDataFrame(
        {hy.schema.tdx_region_field: list(regions)},
        geometry=[outlines[r] for r in regions], crs=crs)
    hy.parquet.write_geoparquet(stacked, boundaries_out, row_group_size=1)
    logging.info(f'{len(stacked)} region boundaries dissolved from the published catchments '
                 f'in {time.time() - started:.0f}s -> {boundaries_out.name}')
    print(f'{len(stacked)} region boundaries from the published catchments, '
          f'{time.time() - started:.0f}s')
