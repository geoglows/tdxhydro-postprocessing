import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from natsort import natsorted
from zarr.codecs import BloscCodec

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
import hydrography as hy

tdxregion_root = root / 'data' / 'TDXHydroGeoParquet'
region_root = root / 'data' / 'regions'
global_root = root / 'data' / 'groups' / 'group=0'
logs_root = root / 'data' / 'logs'

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
    streams_out = global_root / 'streams_mapping.geo.parquet'

    if metadata_out.exists() and metadata_zarr_out.exists() and streams_out.exists():
        logging.info('All global outputs already exist, skipping')
        exit(0)

    n_regions_expected = 50  # todo pull this from a file or config?
    streams_frames = region_root.glob('*/streams_mapping_*.geo.parquet')
    metadata_frames = region_root.glob('*/metadata_*.parquet')

    streams_frames = natsorted(list(streams_frames), key=str)
    metadata_frames = natsorted(list(metadata_frames), key=str)
    if len(streams_frames) != n_regions_expected or len(metadata_frames) != n_regions_expected:
        raise RuntimeError(
            f'Expected {n_regions_expected}, only {len(streams_frames)} streams and {len(metadata_frames)} metadata'
        )

    metadata_frames = pd.concat([pd.read_parquet(f) for f in metadata_frames], ignore_index=True)
    metadata_frames.to_parquet(metadata_out)
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

    # # concatenate the simplified-geometry tables into one global table
    streams_frames = pd.concat([gpd.read_parquet(f) for f in streams_frames], ignore_index=True)
    (
        gpd
        .GeoDataFrame(
            streams_frames,
            geometry=hy.schema.geometry,
            crs=streams_frames.crs,
        )
        .to_parquet(streams_out)
    )
    logging.info(f'Wrote {len(streams_frames):,} rows to {streams_out}')
