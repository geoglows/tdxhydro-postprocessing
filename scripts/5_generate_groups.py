import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
import hydrography as hy

region_root = root / 'data' / 'regions'
group_root = root / 'data' / 'groups'
logs_root = root / 'data' / 'logs'

if __name__ == '__main__':
    # find the ID of the region to process
    if len(sys.argv) != 2:
        sys.exit('usage: 5_generate_groups.py <region>')
    region_number = int(sys.argv[1])
    # region = 1020000010  # Example region number

    outputs_dir = region_root / f'{region_number}'
    streams_src = outputs_dir / f'streams_{region_number}.geo.parquet'
    confluences_src = outputs_dir / f'confluences_{region_number}.geo.parquet'
    mapping_src = outputs_dir / f'streams_mapping_{region_number}.geo.parquet'
    catchments_src = outputs_dir / f'catchments_{region_number}.geo.parquet'

    # fast path: if every per-group output already exists, skip before reading any geometry.
    # the datasets that get split are fixed by which region inputs are present, and the group
    # ids come from just the groupId column of the streams file (a cheap, geometry-free read).
    kinds = ['streams', 'confluences']
    if mapping_src.exists():
        kinds.append('streams_mapping')
    if catchments_src.exists():
        kinds.append('catchments')
    if streams_src.exists() and hy.schema.group_id in pq.read_schema(streams_src).names:
        existing_groups = pd.read_parquet(streams_src, columns=[hy.schema.group_id])[hy.schema.group_id]
        existing_groups = sorted(existing_groups.dropna().astype(int).unique().tolist())
        expected_outputs = [
            group_root / str(g) / f'{kind}_{g}.geo.parquet'
            for g in existing_groups for kind in kinds
        ]
        if expected_outputs and all(p.exists() for p in expected_outputs):
            print(f'All {len(expected_outputs)} group outputs for region {region_number} already exist, skipping')
            sys.exit(0)

    # prepare directories and logging
    logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logs_root / f'generate_groups_{region_number}.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    streams_gdf = gpd.read_parquet(streams_src)
    confluences_gdf = gpd.read_parquet(confluences_src)
    logging.info(f'Read {len(streams_gdf):,} reaches and {len(confluences_gdf):,} confluences')

    # use the groupId assigned previously in step 2
    if hy.schema.group_id not in streams_gdf.columns:
        sys.exit(f'{streams_src} has no {hy.schema.group_id} column; rerun step 2 to assign groupIds')
    missing = streams_gdf[hy.schema.group_id].isna()
    if missing.any():
        unmatched = streams_gdf.loc[missing, hy.schema.last_river_id].unique()
        raise ValueError(
            f'{int(missing.sum())} reach(es) in region {region_number} have no {hy.schema.group_id}; '
            f'e.g. outlets {unmatched[:10].tolist()}'
        )
    streams_gdf[hy.schema.group_id] = streams_gdf[hy.schema.group_id].astype(int)

    group_by_river = streams_gdf.set_index(hy.schema.river_id)[hy.schema.group_id]
    confluences_gdf[hy.schema.group_id] = confluences_gdf[hy.schema.river_id].map(group_by_river)
    dropped = int(confluences_gdf[hy.schema.group_id].isna().sum())
    confluences_gdf = confluences_gdf.dropna(subset=[hy.schema.group_id])
    confluences_gdf[hy.schema.group_id] = confluences_gdf[hy.schema.group_id].astype(int)
    if dropped:
        logging.info(f'Dropped {dropped} confluence row(s) with no groupId (e.g. the -1 outlet aggregate)')

    # streams and confluences are always split; simplified streams and catchments are split too if they exist
    datasets = {
        'streams': streams_gdf,
        'confluences': confluences_gdf,
    }

    if mapping_src.exists():
        simplified_gdf = gpd.read_parquet(mapping_src)
        simplified_gdf[hy.schema.group_id] = simplified_gdf[hy.schema.group_id].astype(int)
        datasets['streams_mapping'] = simplified_gdf
        logging.info(f'Read {len(simplified_gdf):,} simplified reaches from {mapping_src}')
    else:
        logging.info(f'No simplified streams at {mapping_src}; skipping that split')

    # catchments carry no groupId; each catchment's id is its reach id, so it inherits the reach's groupId
    if catchments_src.exists():
        catchments_gdf = gpd.read_parquet(catchments_src)
        catchments_gdf[hy.schema.group_id] = catchments_gdf[hy.schema.river_id].map(group_by_river)
        dropped = int(catchments_gdf[hy.schema.group_id].isna().sum())
        catchments_gdf = catchments_gdf.dropna(subset=[hy.schema.group_id])
        catchments_gdf[hy.schema.group_id] = catchments_gdf[hy.schema.group_id].astype(int)
        datasets['catchments'] = catchments_gdf
        logging.info(f'Read {len(catchments_gdf):,} catchments from {catchments_src}')
        if dropped:
            logging.info(f'Dropped {dropped} catchment(s) whose reach has no groupId')
    else:
        logging.info(f'No catchments at {catchments_src}; skipping that split')

    # split every dataset into one parquet each per group
    group_ids = sorted(streams_gdf[hy.schema.group_id].unique().tolist())
    group_root.mkdir(parents=True, exist_ok=True)
    datasets_by_group = {
        kind: dict(tuple(gdf.groupby(hy.schema.group_id)))
        for kind, gdf in datasets.items()
    }
    for group_id in group_ids:
        out_dir = group_root / str(group_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        counts = {}
        for kind, gdf in datasets.items():
            # a group with no rows for a dataset (e.g. no confluences) still gets an empty file for consistency
            part = datasets_by_group[kind].get(group_id, gdf.iloc[0:0])
            out_path = out_dir / f'{kind}_{group_id}.geo.parquet'
            part.drop(columns=[hy.schema.group_id]).to_parquet(out_path)
            counts[kind] = len(part)

        msg = (f'region {region_number} -> group {group_id}: '
               + ', '.join(f'{n:,} {kind}' for kind, n in counts.items())
               + f' -> {out_dir}')
        logging.info(msg)
        print(msg)
