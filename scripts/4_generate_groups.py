import logging
import os
import sys

import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydrography as hy

region_root = '/Users/rchales/code/untitled folder/tdxhydro-postprocessing/data/regions'
vpu_root = '/Users/rchales/code/untitled folder/tdxhydro-postprocessing/data/vpu'

if __name__ == '__main__':
    # find the ID of the region to process
    if len(sys.argv) != 2:
        sys.exit('usage: 4_generate_groups.py <region>')
    region_number = int(sys.argv[1])
    # region = 1020000010  # Example region number

    # prepare directories and logging
    outputs_dir = os.path.join(region_root, f'{region_number}')
    os.makedirs(os.path.join(outputs_dir, 'mods'), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(outputs_dir, 'mods', 'groups_log.log'),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    streams_src = os.path.join(outputs_dir, f'streams_{region_number}.geo.parquet')
    confluences_src = os.path.join(outputs_dir, f'confluences_{region_number}.geo.parquet')
    streams_gdf = gpd.read_parquet(streams_src)
    confluences_gdf = gpd.read_parquet(confluences_src)
    logging.info(f'Read {len(streams_gdf):,} reaches and {len(confluences_gdf):,} confluences')

    # use the vpuId assigned previously in step 2
    if hy.schema.vpu_id not in streams_gdf.columns:
        sys.exit(f'{streams_src} has no {hy.schema.vpu_id} column; rerun step 2 to assign vpu groups')
    missing = streams_gdf[hy.schema.vpu_id].isna()
    if missing.any():
        unmatched = streams_gdf.loc[missing, hy.schema.last_river_id].unique()
        raise ValueError(
            f'{int(missing.sum())} reach(es) in region {region_number} have no {hy.schema.vpu_id}; '
            f'e.g. outlets {unmatched[:10].tolist()}'
        )
    streams_gdf[hy.schema.vpu_id] = streams_gdf[hy.schema.vpu_id].astype(int)

    # confluences carry no vpuId; a confluence belongs to the vpu of its downstream reach (riverId),
    # which shares the junction. the -1 outlet aggregate row has no downstream reach, so it maps to no
    # vpu and is dropped from the per-vpu files.
    vpu_by_river = streams_gdf.set_index(hy.schema.river_id)[hy.schema.vpu_id]
    confluences_gdf[hy.schema.vpu_id] = confluences_gdf[hy.schema.river_id].map(vpu_by_river)
    dropped = int(confluences_gdf[hy.schema.vpu_id].isna().sum())
    confluences_gdf = confluences_gdf.dropna(subset=[hy.schema.vpu_id])
    confluences_gdf[hy.schema.vpu_id] = confluences_gdf[hy.schema.vpu_id].astype(int)
    if dropped:
        logging.info(f'Dropped {dropped} confluence row(s) with no vpu (e.g. the -1 outlet aggregate)')

    # streams and confluences are always split; simplified streams and catchments are split too
    # when steps 2 and 3 produced them. each dataset must carry a vpuId column to group on.
    datasets = {
        'streams': streams_gdf,
        'confluences': confluences_gdf,
    }

    simplified_src = os.path.join(outputs_dir, f'streams_simplified_{region_number}.geo.parquet')
    if os.path.exists(simplified_src):
        simplified_gdf = gpd.read_parquet(simplified_src)
        simplified_gdf[hy.schema.vpu_id] = simplified_gdf[hy.schema.vpu_id].astype(int)
        datasets['streams_simplified'] = simplified_gdf
        logging.info(f'Read {len(simplified_gdf):,} simplified reaches from {simplified_src}')
    else:
        logging.info(f'No simplified streams at {simplified_src}; skipping that split')

    # catchments carry no vpuId; each catchment's id is its reach id, so it inherits the reach's vpu
    catchments_src = os.path.join(outputs_dir, f'catchments_{region_number}.geo.parquet')
    if os.path.exists(catchments_src):
        catchments_gdf = gpd.read_parquet(catchments_src)
        catchments_gdf[hy.schema.vpu_id] = catchments_gdf[hy.schema.river_id].map(vpu_by_river)
        dropped = int(catchments_gdf[hy.schema.vpu_id].isna().sum())
        catchments_gdf = catchments_gdf.dropna(subset=[hy.schema.vpu_id])
        catchments_gdf[hy.schema.vpu_id] = catchments_gdf[hy.schema.vpu_id].astype(int)
        datasets['catchments'] = catchments_gdf
        logging.info(f'Read {len(catchments_gdf):,} catchments from {catchments_src}')
        if dropped:
            logging.info(f'Dropped {dropped} catchment(s) whose reach has no vpu')
    else:
        logging.info(f'No catchments at {catchments_src}; skipping that split')

    # if every per-vpu output already exists then skip
    vpu_ids = sorted(streams_gdf[hy.schema.vpu_id].unique().tolist())
    expected_outputs = [
        os.path.join(vpu_root, str(v), f'{kind}_{v}.geo.parquet')
        for v in vpu_ids for kind in datasets
    ]
    if all(os.path.exists(p) for p in expected_outputs):
        print(f'All {len(expected_outputs)} vpu outputs for region {region_number} already exist, skipping')
        sys.exit(0)

    # split every dataset into one parquet each per vpu
    os.makedirs(vpu_root, exist_ok=True)
    datasets_by_vpu = {
        kind: dict(tuple(gdf.groupby(hy.schema.vpu_id)))
        for kind, gdf in datasets.items()
    }
    for vpu_id in vpu_ids:
        out_dir = os.path.join(vpu_root, str(vpu_id))
        os.makedirs(out_dir, exist_ok=True)

        counts = {}
        for kind, gdf in datasets.items():
            # a vpu with no rows for a dataset (e.g. no confluences) still gets an empty file for consistency
            part = datasets_by_vpu[kind].get(vpu_id, gdf.iloc[0:0])
            out_path = os.path.join(out_dir, f'{kind}_{vpu_id}.geo.parquet')
            part.drop(columns=[hy.schema.vpu_id]).to_parquet(out_path)
            counts[kind] = len(part)

        msg = (f'region {region_number} -> vpu {vpu_id}: '
               + ', '.join(f'{n:,} {kind}' for kind, n in counts.items())
               + f' -> {out_dir}')
        logging.info(msg)
        print(msg)
