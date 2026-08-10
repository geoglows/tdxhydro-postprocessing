import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from natsort import natsorted
from zarr.codecs import BloscCodec

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

# The global streams table keeps default row groups, unlike the per-group ones. Small row groups
# exist to let a client fetch a few hundred reaches out of a file over HTTP; this file is the
# whole world in one piece, taken as a bulk download rather than subset, and the per-group files
# are what serve the subsetting case. Chunking it would only add footer. Everything else about
# how these are written is shared with the earlier steps - see hydrography/parquet.py.

# every output path hangs off the data root - see hydrography/paths.py and $RFS_DATA_ROOT
region_root = hy.paths.region_root
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
    streams_out = global_root / 'streams.geo.parquet'

    if metadata_out.exists() and metadata_zarr_out.exists() and streams_out.exists():
        logging.info('All global outputs already exist, skipping')
        exit(0)

    n_regions_expected = 50  # todo pull this from a file or config?
    # the digit keeps this off any stale streams_mapping_* left over from before that
    # product was removed, which would otherwise double every region
    streams_frames = region_root.glob('*/streams_[0-9]*.geo.parquet')
    metadata_frames = region_root.glob('*/metadata_*.parquet')

    streams_frames = natsorted(list(streams_frames), key=str)
    metadata_frames = natsorted(list(metadata_frames), key=str)
    if len(streams_frames) != n_regions_expected or len(metadata_frames) != n_regions_expected:
        raise RuntimeError(
            f'Expected {n_regions_expected}, only {len(streams_frames)} streams and {len(metadata_frames)} metadata'
        )

    metadata_frames = pd.concat([pd.read_parquet(f) for f in metadata_frames], ignore_index=True)
    # concat preserves the parts' dtypes, but this is the file most consumers read, so re-assert
    # rather than inherit whatever the region files happened to carry
    metadata_frames = hy.schema.enforce_int32(metadata_frames)
    hy.parquet.write_parquet(metadata_frames, metadata_out)
    logging.info(f'Wrote {len(metadata_frames):,} rows to {metadata_out}')

    # make the dataframe a zarr chunked in groups of 50_000 rows along the df's index, 1 variable per column
    zarr_int_vars = [
        hy.schema.river_id,
        hy.schema.river_index,
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
                **{v: {'dtype': 'int32', 'compressors': BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")} for v in zarr_int_vars},
                **{v: {'dtype': 'float32', 'compressors': BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")} for v in zarr_float_vars},
            }
        )
    )

    # concatenate every region's streams into one global table
    streams_frames = pd.concat([gpd.read_parquet(f) for f in streams_frames], ignore_index=True)
    streams_frames = hy.schema.enforce_int32(streams_frames)
    hy.parquet.write_geoparquet(
        gpd.GeoDataFrame(streams_frames, geometry=hy.schema.geometry, crs=streams_frames.crs),
        streams_out,
        row_group_size=None,
    )
    logging.info(f'Wrote {len(streams_frames):,} rows to {streams_out}')
