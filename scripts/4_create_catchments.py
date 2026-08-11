"""
Build one region's leaf catchment polygons: one per surviving reach, in the published row order,
simplified as a coverage.

Two things happen here, and they are one step because the second wants the geometry the first is
already holding:

1. **The leaf catchments.** The source basins are one polygon per original TDX reach; step 2
   dissolved reaches into each other, so the surviving reach's catchment is the union of the basins
   that were folded into it. Which basins those are is read back out of the json journal step 2
   wrote, replayed in the same order.

2. **Simplification.** The source basins are polygonised DEM cells, so a boundary is a run of
   ~3.4 m stair treads: 1,903 vertices per polygon on average, ~10.4 billion across the network,
   12 GB on disk. None of that is information -- the source DEM is 1/9 arcsec -- and no renderer
   can carry it. 100 m of coverage simplification keeps ~8% of the vertices for an area error under
   0.0001%, and stays sub-pixel until z11.

Two properties of the geometry govern how the middle step is done, both measured in the design note:

**Coverage simplification, not per-polygon.** Neighbouring catchments share a boundary.
Douglas-Peucker run on each polygon independently simplifies the shared stretch twice, from two
different ring start points, and leaves a sliver in the gap between the two answers. GEOS
CoverageSimplify simplifies each shared edge once and hands the same line to both neighbours, so
the coverage stays exact. It keeps ~2x more vertices than DP and that is the price of not shredding
the layer -- and it is what lets a consumer dissolve these polygons cheaply, because unioning an
exact coverage is edge cancellation (``coverage_union_all``) rather than a general overlay.

**The coverage boundary is held fixed, and the chunks are spatially coherent.**
``simplify_boundary=False`` leaves the outer edge of the input untouched, which is what lets this
run per region -- a region outline is a drainage divide shared vertex-for-vertex with the next
region's catchments -- and, for the same reason, what lets a region be simplified in chunks at all.
A pinned outline is only cheap if the chunk is a compact blob, so the catchments are put in the
published riverIndex order *before* they are chunked: Hilbert-ordered watersheds, DFS post-order
within each, so a contiguous run is a compact clump. Chunking the same region in riverId order
keeps 78.8% of the vertices where this keeps 19.7%.

    RFS_DATA_ROOT=... TDXHYDRO_ROOT=... python 4_create_catchments.py <region> [--force]
"""
import json
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

# every output path hangs off the data root - see hydrography/paths.py and $RFS_DATA_ROOT
region_root = hy.paths.region_root
tdx_root = hy.paths.tdx_root
logs_root = hy.paths.logs_root

dissolve_threads = os.cpu_count() or 8
TOLERANCE_METERS = 100.0
CHUNK_SIZE = 20_000


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def build_basin_edits(mods_dir: Path) -> tuple[dict, set]:
    """
    Replay, in the same order as 2_simplify_streams.py, the id-level edits that the
    stream simplification recorded in its json side-files, expressed for basins.

    Returns:
        redirect: {original_river_id: river_id it was merged into}. A basin's surviving
                  catchment is found by following redirect to a fixed point.
        deleted:  river_ids whose reach (and basin) were removed outright.
    """
    redirect: dict = {}
    deleted: set = set()

    # 1. lakes: every interior reach (including the inlet->outlet geometry path) collapses
    #    into the lake outlet, so its local catchment area belongs to the outlet's catchment.
    #    inlets and the outlet survive and keep their own basins.
    for outlet, edit in _load_json(mods_dir / 'lake_edits.json').items():
        outlet = int(outlet)
        for d in edit.get('delete', []):
            redirect[int(d)] = outlet

    # 2. zero-length streams: their basins are deleted in every case (1, 2, 3)
    zero_lengths = _load_json(mods_dir / 'zero_length_streams.json')
    for case in ('case1', 'case2', 'case3'):
        deleted.update(int(i) for i in zero_lengths.get(case, {}).get('ids', []))

    # 3. headwater dissolves, 4. branch pruning, 5. short consolidations:
    #    each {keeper: [members]} group folds every non-keeper member's basin into the keeper
    for fname in ('headwater_dissolves.json', 'branches_to_prune.json', 'short_consolidations.json'):
        for keeper, members in _load_json(mods_dir / fname).items():
            keeper = int(keeper)
            for member in members:
                redirect[int(member)] = keeper

    return redirect, deleted


def resolve_keeper(rid: int, redirect: dict) -> int:
    """Follow the redirect chain to the reach that ultimately absorbed rid."""
    seen = set()
    while rid in redirect and rid not in seen:
        seen.add(rid)
        rid = redirect[rid]
    return rid


def dissolve_threaded(gdf: gpd.GeoDataFrame, by: str, workers: int, groups_per_task: int = 300) -> gpd.GeoDataFrame:
    """
    Union the geometries within each `by` group, one row out per group. Equivalent to
    gdf.dissolve(by=by) for a geometry-only frame, but runs the per-group GEOS unions
    across worker threads (shapely.union_all drops the GIL during the union).
    """
    grouped = gdf.groupby(by)[hy.schema.geometry].apply(lambda s: s.values)
    keys = grouped.index.to_numpy()
    arrays = grouped.to_list()

    def union_chunk(start: int) -> list:
        return [shapely.union_all(a) for a in arrays[start:start + groups_per_task]]

    geometries = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(union_chunk, range(0, len(arrays), groups_per_task)):
            geometries.extend(part)
    return gpd.GeoDataFrame({by: keys}, geometry=geometries, crs=gdf.crs)


def simplify_coverage(geometries, chunk_size: int = CHUNK_SIZE) -> tuple[int, int]:
    """Simplify an ordered array of coverage polygons in place, chunk by chunk.

    In place so the raw geometry is released as it goes: one region's source geometry is ~78 bytes
    per vertex once GEOS has it, or roughly 16 GB for a large region, and holding the simplified
    copy alongside all of it would be the peak of this script rather than the dissolve.

    Returns the vertex count before and after.
    """
    before = after = 0
    for start in range(0, len(geometries), chunk_size):
        block = slice(start, start + chunk_size)
        raw = geometries[block]
        before += int(shapely.get_num_coordinates(raw).sum())
        # simplify_boundary=False is what keeps this chunk's outline identical to its neighbours'
        clean = shapely.coverage_simplify(raw, TOLERANCE_METERS, simplify_boundary=False)
        empty = int(shapely.is_empty(clean).sum())
        if empty:
            raise RuntimeError(f'chunk starting at row {start:,} simplified {empty} catchment(s) '
                               f'out of existence')
        after += int(shapely.get_num_coordinates(clean).sum())
        geometries[block] = clean
    return before, after


def build_leaf_catchments(region_number: int, order: pd.DataFrame) -> gpd.GeoDataFrame:
    """The source basins dissolved onto the surviving reaches, in ``order``'s row order, simplified.

    ``order`` is the region's metadata: its row order IS the published order, and this is the only
    place the catchments are put into it. Everything downstream -- the group split, the basin
    dissolve below, a client reading the nth row of two products -- depends on that.
    """
    outputs_dir = region_root / f'{region_number}'
    mods_dir = outputs_dir / 'mods'

    # load the original basins. TDXHydroLinkNo is the basin id with the region spacer applied,
    # so it is the same numbering as the stream riverId.
    basins_src = tdx_root / f'TDX_streamreach_basins_{region_number}_01.parquet'
    basins = gpd.read_parquet(basins_src)
    basins = (
        basins[[hy.schema.tdx_link_no_field, hy.schema.geometry]]
        .rename(columns={hy.schema.tdx_link_no_field: hy.schema.river_id})
    )
    basins[hy.schema.river_id] = basins[hy.schema.river_id].astype(int)
    logging.info(f'Read {len(basins):,} source basins from {basins_src}')

    # replay step 2's edits to map each original basin to the reach that absorbed it
    redirect, deleted = build_basin_edits(mods_dir)
    keeper_map = {rid: resolve_keeper(rid, redirect) for rid in basins[hy.schema.river_id].unique()}
    basins[hy.schema.river_id] = basins[hy.schema.river_id].map(keeper_map)

    # drop basins whose reach was removed outright (the zero-length cases)
    before = len(basins)
    basins = basins[~basins[hy.schema.river_id].isin(deleted)]
    logging.info(f'Dropped {before - len(basins):,} basins for deleted (zero-length) reaches')

    # dissolve each keeper's absorbed basins into one catchment polygon. only groups with more
    # than one member need a union; singletons (untouched reaches) are passed through for speed.
    counts = basins[hy.schema.river_id].value_counts()
    multi_keepers = set(counts[counts > 1].index)
    singles = basins[~basins[hy.schema.river_id].isin(multi_keepers)]
    multi = basins[basins[hy.schema.river_id].isin(multi_keepers)]
    merged = dissolve_threaded(multi, hy.schema.river_id, dissolve_threads) if len(multi) else multi
    catchments = gpd.GeoDataFrame(
        pd.concat([singles, merged], ignore_index=True),
        geometry=hy.schema.geometry,
        crs=basins.crs,
    )
    logging.info(f'Dissolved into {len(catchments):,} catchments '
                 f'({len(multi_keepers):,} merged, {len(singles):,} unchanged) '
                 f'using {dissolve_threads} threads')
    del basins, singles, multi, merged

    duplicated = catchments[hy.schema.river_id][catchments[hy.schema.river_id].duplicated()].tolist()
    if duplicated:
        raise RuntimeError(f'{len(duplicated)} catchment id(s) are duplicated, e.g. {duplicated[:10]}')

    # every published geometry is web mercator snapped to a 1 m grid - see projection.py. it has to
    # happen before the simplification, whose tolerance is in mercator metres.
    catchments = hy.projection.to_web_mercator(catchments)

    # Reindex onto the published row order. This is also the filter: the metadata holds exactly the
    # reaches that survived step 2, so reaches dropped by the whole-watershed and <250 km^2 rules
    # (which are not in the json journal) fall out here, and anything the journal says survives but
    # has no catchment shows up as a null rather than as a silently missing row.
    river_ids = order[hy.schema.river_id].to_numpy()
    catchments = catchments.set_index(hy.schema.river_id).reindex(river_ids)
    missing = catchments[hy.schema.geometry].isna()
    if missing.any():
        absent = catchments.index[missing].tolist()
        raise RuntimeError(f'{len(absent):,} reach(es) have no catchment, e.g. {absent[:5]}')
    catchments = catchments.reset_index()

    started = time.time()
    before, after = simplify_coverage(catchments[hy.schema.geometry].values)
    logging.info(f'simplified at {TOLERANCE_METERS:g} m in chunks of {CHUNK_SIZE:,}: '
                 f'{before:,} -> {after:,} vertices ({100 * after / before:.2f}%), '
                 f'{time.time() - started:.0f}s')

    # riverIndex rides along so a leaf catchment carries the same id *and* index a reach does, which
    # is what lets one selector address the streams and every catchment layer alike
    catchments.insert(1, hy.schema.river_index, order[hy.schema.river_index].to_numpy())
    return hy.schema.enforce_int32(catchments)


if __name__ == '__main__':
    # find the ID of the region to process
    force = '--force' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--force']
    if len(args) != 1:
        sys.exit('usage: 4_create_catchments.py <region_number> [--force]')
    region_number = int(args[0])
    # region_number = 1020000010  # Example region number

    outputs_dir = region_root / f'{region_number}'
    mods_dir = outputs_dir / 'mods'

    catchments_output = outputs_dir / f'catchments_{region_number}.geo.parquet'
    if not force and catchments_output.exists():
        print(f'Catchments output {catchments_output} already exists, skipping region {region_number}')
        sys.exit(0)

    # prepare directories and logging
    mods_dir.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logs_root / f'create_catchments_{region_number}.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    # The metadata gives this step its two inputs at once: its row order is the published order the
    # catchments have to come out in, and its membership is exactly which reaches survived step 2
    # (the whole-watershed and <250 km^2 drops are not in the json journal).
    metadata = pd.read_parquet(
        outputs_dir / f'metadata_{region_number}.parquet',
        columns=[hy.schema.river_id, hy.schema.river_index],
    )
    logging.info(f'{len(metadata):,} reaches in the published order')

    catchments = build_leaf_catchments(region_number, metadata)
    hy.parquet.write_geoparquet(catchments, catchments_output)
    logging.info(f'Catchments written to {catchments_output}')
    print(f'region {region_number}: {len(catchments):,} catchments -> {catchments_output}')
