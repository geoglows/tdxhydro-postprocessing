"""
Split the per-region simplified stream parquets into per-VPU parquets.

Each region's ``streams_<region>.parquet`` carries an ``outletRiverId`` for every
reach; ``network_data/vpu_table.csv`` maps each outlet to a ``vpuId``. Every VPU
lives entirely within one region (its outlets all share that region's id prefix),
so a region can be split independently and no VPU receives reaches from two
regions.

Run for one region (parallel-friendly, mirrors 2_simplify_streams.py):
    python 4_generate_groups.py <region_number>
Run for every region found under data/modifications/:
    python 4_generate_groups.py
"""
import os
import sys
from glob import glob

import geopandas as gpd
import pandas as pd
from natsort import natsorted

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hydrography.schema as schema

modifications_root = '../../data/modifications'
vpu_root = '../../data/vpu'
vpu_table_path = '../../network_data/vpu_table.csv'


def assign_vpu(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure gdf has a vpuId column, joining it from vpu_table.csv via outletRiverId if absent."""
    if schema.vpu_id in gdf.columns:
        return gdf
    vpu_map = (
        pd.read_csv(vpu_table_path)
        .set_index(schema.last_river_id)[schema.vpu_id]
        .to_dict()
    )
    gdf[schema.vpu_id] = gdf[schema.last_river_id].map(vpu_map)
    return gdf


def split_region(region_number: int, written_vpus: dict) -> None:
    src = os.path.join(modifications_root, f'{region_number}', f'streams_{region_number}.parquet')
    gdf = assign_vpu(gpd.read_parquet(src))

    missing = gdf[schema.vpu_id].isna()
    if missing.any():
        unmatched = gdf.loc[missing, schema.last_river_id].unique()
        raise ValueError(
            f'{int(missing.sum())} reach(es) in region {region_number} have no vpuId '
            f'(outletRiverId not in {os.path.basename(vpu_table_path)}), '
            f'e.g. outlets {unmatched[:10].tolist()}'
        )
    gdf[schema.vpu_id] = gdf[schema.vpu_id].astype(int)

    for vpu_id, vpu_gdf in gdf.groupby(schema.vpu_id):
        vpu_id = int(vpu_id)
        if vpu_id in written_vpus:
            raise ValueError(
                f'vpu {vpu_id} appears in both region {written_vpus[vpu_id]} and region '
                f'{region_number}; a VPU is expected to live in exactly one region. '
                f'Check vpu_table.csv before continuing or this would overwrite data.'
            )
        written_vpus[vpu_id] = region_number

        out_dir = os.path.join(vpu_root, str(vpu_id))
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f'streams_{vpu_id}.parquet')
        vpu_gdf.drop(columns=[schema.vpu_id]).to_parquet(out_file)
        print(f'region {region_number} -> vpu {vpu_id}: {len(vpu_gdf):,} reaches -> {out_file}')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        sys.exit('usage: 4_generate_groups.py [region_number]')

    if len(sys.argv) == 2:
        regions = [int(sys.argv[1])]
    else:
        regions = [
            int(os.path.basename(p)) for p in
            natsorted(glob(os.path.join(modifications_root, '*')))
            if os.path.basename(p).isdigit()
        ]

    os.makedirs(vpu_root, exist_ok=True)
    written_vpus: dict = {}
    for region in regions:
        split_region(region, written_vpus)
