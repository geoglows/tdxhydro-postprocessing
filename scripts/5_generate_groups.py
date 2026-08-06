import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

# Must match 2_simplify_streams.py and 3_global_stream_attributes.py: these are splits of the files
# those steps wrote, and a different codec here would re-encode them on the way through.
COMPRESSION = 'zstd'
COMPRESSION_LEVEL = 3
WRITE_OPTS = {'compression': COMPRESSION, 'compression_level': COMPRESSION_LEVEL}
# Must match 2_simplify_streams.py. These splits are what clients actually download, so the row
# group size that makes them subsettable matters more here than anywhere else in the pipeline.
GEOMETRY_ROW_GROUP_SIZE = 500
# Only the tables whose rows carry a whole reach's geometry, ~5.6 KB each, need to be written in
# small row groups: a row group is the smallest thing a reader can fetch, so a large one forces a
# client wanting a few hundred reaches to decompress hundreds of MB. Confluences are a single point
# per row, ~29 bytes — a whole file is about a megabyte, so there is nothing to subset out of and
# small groups would only add footer. Metadata is likewise light and stays on the default.
LARGE_GEOMETRY_KINDS = {'streams', 'streams_mapping', 'catchments'}

# Everything split here carries geometry except metadata, the attribute table.
SUFFIXES = {'metadata': '.parquet'}


def out_name(kind: str, group_id: int) -> str:
    return f'{kind}_{group_id}{SUFFIXES.get(kind, ".geo.parquet")}'


# every output path hangs off the data root - see hydrography/paths.py and $RFS_DATA_ROOT
region_root = hy.paths.region_root
group_root = hy.paths.group_root
logs_root = hy.paths.logs_root

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
    metadata_src = outputs_dir / f'metadata_{region_number}.parquet'

    # fast path: if every per-group output already exists, skip before reading any geometry.
    # the datasets that get split are fixed by which region inputs are present, and the group
    # ids come from just the groupId column of the streams file (a cheap, geometry-free read).
    kinds = ['streams', 'confluences']
    if metadata_src.exists():
        kinds.append('metadata')
    if mapping_src.exists():
        kinds.append('streams_mapping')
    if catchments_src.exists():
        kinds.append('catchments')
    if streams_src.exists() and hy.schema.group_id in pq.read_schema(streams_src).names:
        existing_groups = pd.read_parquet(streams_src, columns=[hy.schema.group_id])[hy.schema.group_id]
        existing_groups = sorted(existing_groups.dropna().astype(int).unique().tolist())
        expected_outputs = [
            hy.paths.group_dir(g) / out_name(kind, g)
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
    streams_gdf[hy.schema.group_id] = streams_gdf[hy.schema.group_id].astype('int32')

    group_by_river = streams_gdf.set_index(hy.schema.river_id)[hy.schema.group_id]
    confluences_gdf[hy.schema.group_id] = confluences_gdf[hy.schema.river_id].map(group_by_river)
    dropped = int(confluences_gdf[hy.schema.group_id].isna().sum())
    confluences_gdf = confluences_gdf.dropna(subset=[hy.schema.group_id])
    confluences_gdf[hy.schema.group_id] = confluences_gdf[hy.schema.group_id].astype('int32')
    if dropped:
        logging.info(f'Dropped {dropped} confluence row(s) with no groupId (e.g. the -1 outlet aggregate)')

    # streams and confluences are always split; simplified streams and catchments are split too if they exist
    datasets = {
        'streams': streams_gdf,
        'confluences': confluences_gdf,
    }

    # metadata is the one non-geo table, so pandas rather than geopandas reads it. It is every
    # attribute of every reach without the geometry, and parquet is columnar, so a consumer that
    # only wants to walk the network reads riverId/nextRiverId out of it and pays for those column
    # chunks alone — measured at 3.2 MB of a 6.8 MB region file. That is why there is no separate
    # connectivity table here: it would be a strict column subset of this one, same rows in the
    # same order, costing the same bytes to read.
    if metadata_src.exists():
        metadata_df = pd.read_parquet(metadata_src)
        metadata_df[hy.schema.group_id] = metadata_df[hy.schema.group_id].astype('int32')
        datasets['metadata'] = metadata_df
        logging.info(f'Read {len(metadata_df):,} metadata rows from {metadata_src}')
    else:
        logging.info(f'No metadata at {metadata_src}; skipping that split')

    if mapping_src.exists():
        simplified_gdf = gpd.read_parquet(mapping_src)
        simplified_gdf[hy.schema.group_id] = simplified_gdf[hy.schema.group_id].astype('int32')
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
        catchments_gdf[hy.schema.group_id] = catchments_gdf[hy.schema.group_id].astype('int32')
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
        out_dir = hy.paths.group_dir(group_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        counts = {}
        for kind, gdf in datasets.items():
            # a group with no rows for a dataset (e.g. no confluences) still gets an empty file for consistency
            part = datasets_by_group[kind].get(group_id, gdf.iloc[0:0])
            out_path = out_dir / out_name(kind, group_id)
            opts = dict(WRITE_OPTS)
            if kind in LARGE_GEOMETRY_KINDS:
                opts['row_group_size'] = GEOMETRY_ROW_GROUP_SIZE
            part.drop(columns=[hy.schema.group_id]).to_parquet(out_path, **opts)
            counts[kind] = len(part)

        msg = (f'region {region_number} -> group {group_id}: '
               + ', '.join(f'{n:,} {kind}' for kind, n in counts.items())
               + f' -> {out_dir}')
        logging.info(msg)
        print(msg)
