import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from natsort import natsorted
from zarr.codecs import BloscCodec

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

# every output path hangs off the data root - see hydrography/paths.py and $RFS_DATA_ROOT
region_root = hy.paths.region_root
group_root = hy.paths.group_root
global_root = hy.paths.global_root
logs_root = hy.paths.logs_root

if __name__ == '__main__':
    global_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logs_root / 'concatenate_global.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    metadata_out = global_root / 'metadata.parquet'
    metadata_zarr_out = global_root / 'metadata.zarr'
    boundaries_out = global_root / 'groups.geo.parquet'

    outputs = [metadata_out, metadata_zarr_out, boundaries_out]
    if all(output.exists() for output in outputs):
        logging.info('All global outputs already exist, skipping')
        exit(0)

    # Said up front rather than only at the end: the boundaries are assembled last, and finding out
    # then that step 5 never ran is a wasted pass.
    group_dirs = sorted(group_root.glob('group=*'))
    boundary_files_present = sorted(group_root.glob('group=*/boundary_*.geo.parquet'))
    if not boundary_files_present:
        print(f'No group=*/boundary_*.geo.parquet under {group_root}, so {boundaries_out.name} '
              f'cannot be built in this run. Those come from 5_generate_groups.py, which writes a '
              f'boundary only once 4_create_catchments.py has produced the catchments. Everything '
              f'else below is unaffected.', flush=True)
    elif len(boundary_files_present) < len(group_dirs) - 1:  # -1 for group=0, which holds global files
        found_count = len(boundary_files_present)
        print(f'Only {found_count} of {len(group_dirs) - 1} groups '
              f'{"has" if found_count == 1 else "have"} a boundary file, so {boundaries_out.name} '
              f'will be partial. Run 4_create_catchments.py and 5_generate_groups.py for the '
              f'remaining regions.', flush=True)

    n_regions_expected = 50  # todo pull this from a file or config?
    metadata_frames = natsorted(list(region_root.glob('*/metadata_*.parquet')), key=str)
    if len(metadata_frames) != n_regions_expected:
        raise RuntimeError(
            f'Expected {n_regions_expected} region metadata files, found {len(metadata_frames)}'
        )


    def in_global_index_order(df: pd.DataFrame, label: str) -> pd.DataFrame:
        """
        Restore the global ordering that step 3 assigned.

        The region files are each internally ordered by riverIndex, but a region is not a contiguous
        block of it - the ordering is group-major and a region holds several groups - so
        concatenating them in path order interleaves. Sorting by riverIndex is what makes it true
        again that riverIndex IS the row number of this file, which is the whole contract.
        """
        df = df.sort_values(hy.schema.river_index).reset_index(drop=True)
        expected = np.arange(len(df), dtype=df[hy.schema.river_index].dtype)
        if not (df[hy.schema.river_index].to_numpy() == expected).all():
            gaps = int((df[hy.schema.river_index].to_numpy() != expected).sum())
            raise RuntimeError(
                f'{label}: riverIndex is not 0..{len(df) - 1:,} ({gaps:,} rows out of place). The '
                f'region files and the global ordering are out of step - rerun step 3.'
            )
        return df


    def assert_groups_are_self_contained(df: pd.DataFrame) -> None:
        group_of = pd.Series(df[hy.schema.group_id].to_numpy(), index=df[hy.schema.river_id].to_numpy())
        flowing = df[df[hy.schema.next_river_id] != -1]
        downstream_group = group_of.reindex(flowing[hy.schema.next_river_id].to_numpy()).to_numpy()
        crossing = int((downstream_group != flowing[hy.schema.group_id].to_numpy()).sum())
        dangling = int(pd.isna(downstream_group).sum())
        if crossing or dangling:
            raise RuntimeError(
                f'{crossing:,} reaches drain into another group and {dangling:,} have a missing '
                f'downstream reach. Per-group files would not be self-contained.'
            )

        groups = df[hy.schema.group_id].to_numpy()
        runs = int((groups[1:] != groups[:-1]).sum()) + 1
        if runs != df[hy.schema.group_id].nunique():
            raise RuntimeError(
                f'groupId occupies {runs:,} runs of riverIndex but there are '
                f'{df[hy.schema.group_id].nunique():,} groups, so at least one group is split across '
                f'the ordering and its rows are not a single slice. Rerun step 3.'
            )
        logging.info(f'checked: 0 cross-group edges, 0 dangling downstream ids, '
                     f'{runs} groups in {runs} contiguous riverIndex runs')


    metadata_frames = pd.concat([pd.read_parquet(f) for f in metadata_frames], ignore_index=True)
    metadata_frames = in_global_index_order(metadata_frames, 'metadata')
    assert_groups_are_self_contained(metadata_frames)
    # concat preserves the parts' dtypes, but this is the file most consumers read, so re-assert
    # rather than inherit whatever the region files happened to carry
    metadata_frames = hy.schema.enforce_int32(metadata_frames)
    hy.parquet.write_parquet(metadata_frames, metadata_out)
    logging.info(f'Wrote {len(metadata_frames):,} rows to {metadata_out}')

    # Make the dataframe a zarr chunked in groups of 50_000 rows along the df's index, 1 variable
    # per column. This is an allowlist, not everything the metadata carries: it is the projection a
    # client walking the network needs. The basin codes are deliberately not in it - they address
    # the catchment polygons at a zoom, which is a map concern, and metadata.parquet already has
    # them for anyone who wants to join on riverId.
    zarr_int_vars = [
        hy.schema.river_id,
        hy.schema.river_index,
        # riverIndex alone says where a reach is; with upstreamCount it also says where everything
        # draining into it is, as the block [riverIndex - upstreamCount, riverIndex]
        hy.schema.upstream_count,
        hy.schema.next_river_id,
        hy.schema.last_river_id,
    ]
    # the outlet point; float32 quantizes to ~1.5e-5 deg, finer than the 1/9 arcsec source grid
    zarr_float_vars = [hy.schema.lat_field, hy.schema.lon_field]
    (
        metadata_frames
        [zarr_int_vars + zarr_float_vars]
        .to_xarray()
        .chunk({'index': 10_000})
        .to_zarr(
            metadata_zarr_out,
            mode='w',
            zarr_format=3,
            consolidated=False,
            encoding={
                **{v: {'dtype': 'int32', 'compressors': BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")} for v in
                   zarr_int_vars},
                **{v: {'dtype': 'float32', 'compressors': BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")} for v
                   in zarr_float_vars},
            }
        )
    )

    # Assemble the exact group outlines. Step 5 dissolves each group's catchments into a one-row
    # boundary_<groupId>.geo.parquet while they are already in memory; all this does is stack them,
    # so the groupId comes from the filename the way every other per-group product identifies itself.
    # One row group per row, unlike the tables above: a row here is a whole continent's drainage
    # divide, so the usual 500 would put the entire world in a single fetch.
    expected_groups = sorted(int(g) for g in metadata_frames[hy.schema.group_id].unique())
    boundary_files = {g: hy.paths.group_dir(g) / f'boundary_{g}.geo.parquet' for g in expected_groups}
    found = {g: p for g, p in boundary_files.items() if p.exists()}
    missing = [g for g in expected_groups if g not in found]

    if not found:
        print(f'No group boundaries found for any of the {len(expected_groups)} groups, so '
              f'{boundaries_out.name} was NOT written. It is built from '
              f'group=<id>/boundary_<id>.geo.parquet, which 5_generate_groups.py writes only when the '
              f'catchments exist. Run 4_create_catchments.py then 5_generate_groups.py, then delete '
              f'this step\'s outputs and rerun it.', flush=True)
    else:
        if missing:
            print(f'{len(missing)} of {len(expected_groups)} groups have no boundary file, so '
                  f'{boundaries_out.name} covers only {len(found)} of them: missing '
                  f'{missing[:10]}{" ..." if len(missing) > 10 else ""}. Those groups have no '
                  f'catchments yet - run 4_create_catchments.py and 5_generate_groups.py for their '
                  f'regions, then delete this step\'s outputs and rerun it.', flush=True)
        parts = [gpd.read_parquet(p) for p in found.values()]
        boundaries = gpd.GeoDataFrame(
            {hy.schema.group_id: np.array(list(found), dtype='int32')},
            geometry=[part.geometry.values[0] for part in parts],
            crs=parts[0].crs,
        )
        hy.parquet.write_geoparquet(boundaries, boundaries_out, row_group_size=1)
        vertices = int(shapely.get_num_coordinates(boundaries.geometry.values).sum())
        logging.info(f'Wrote {len(boundaries)} group boundaries ({vertices:,} vertices) to '
                     f'{boundaries_out}')
