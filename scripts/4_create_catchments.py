import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

# Must match the other steps; catchments are geometry, so they get the geometry row groups too
WRITE_OPTS = {'compression': 'zstd', 'compression_level': 3, 'row_group_size': 500}

# every output path hangs off the data root - see hydrography/paths.py and $RFS_DATA_ROOT
region_root = hy.paths.region_root
tdx_root = hy.paths.tdx_root
logs_root = hy.paths.logs_root

# the dissolve (a GEOS union per keeper group) is ~96% of the runtime. shapely's union_all
# releases the GIL during the GEOS work, so unioning groups in worker threads parallelizes the
# heavy part with no inter-process serialization (~7x faster than geopandas .dissolve here).
dissolve_threads = os.cpu_count() or 8


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


if __name__ == '__main__':
    # find the ID of the region to process
    if len(sys.argv) != 2:
        sys.exit('usage: 4_create_catchments.py <region_number>')
    region_number = int(sys.argv[1])
    # region_number = 1020000010  # Example region number

    outputs_dir = region_root / f'{region_number}'
    mods_dir = outputs_dir / 'mods'

    # if the output already exists then skip
    catchments_output = outputs_dir / f'catchments_{region_number}.geo.parquet'
    if catchments_output.exists():
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

    # the simplified stream network defines which reaches (and therefore catchments) survive.
    # the whole-watershed and <250 km^2 drops from step 2 are not in the json side-files, but are
    # captured here by keeping only catchments whose final id is still in the streams output.
    streams_src = outputs_dir / f'streams_{region_number}.geo.parquet'
    surviving_ids = set(pd.read_parquet(streams_src, columns=[hy.schema.river_id])[hy.schema.river_id].astype(int))
    logging.info(f'{len(surviving_ids):,} reaches survive in {streams_src}')

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
                 f'({len(multi_keepers):,} merged, {len(singles):,} unchanged) using {dissolve_threads} threads')

    # keep only catchments for reaches that survived step 2 (whole-watershed / small-area / group drops)
    before = len(catchments)
    catchments = catchments[catchments[hy.schema.river_id].isin(surviving_ids)]
    logging.info(f'Dropped {before - len(catchments):,} catchments not in the simplified stream network')

    # every surviving reach should have exactly one catchment
    missing = surviving_ids - set(catchments[hy.schema.river_id])
    if missing:
        logging.warning(f'{len(missing)} surviving reach(es) have no catchment, e.g. {sorted(missing)[:10]}')
    duplicated = catchments[hy.schema.river_id][catchments[hy.schema.river_id].duplicated()].tolist()
    if duplicated:
        raise RuntimeError(f'{len(duplicated)} catchment id(s) are duplicated, e.g. {duplicated[:10]}')

    catchments = catchments.sort_values(hy.schema.river_id).reset_index(drop=True)
    catchments.to_parquet(catchments_output, **WRITE_OPTS)

    logging.info(f'Catchments written to {catchments_output}')
    print(f'region {region_number}: {len(catchments):,} catchments -> {catchments_output}')
