"""Reading and walking the raw TDX-Hydro geoparquet, shared by the raw_processing attribute scripts.

Each attribute script (koppen_classifications.py, country_classifications.py) measures something
over every raw catchment of a region and sums it down the raw network. This is the part they have
in common: finding a region's files, reading its basins in bounded batches, and the two walks over
LINKNO -> DSLINKNO (accumulating downstream, and finding each reach's outlet).
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyproj
import shapely

from . import schema

__all__ = ['region_of', 'streamnet_of', 'source_crs', 'id_column', 'batches', 'accumulate',
           'outlets']


def region_of(basins: Path) -> str:
    """The region id in a TDX_streamreach_basins_<region>_01.parquet name."""
    return basins.name.split('_')[-2]


def streamnet_of(basins: Path) -> Path:
    """The streamnet beside a region's basins file."""
    return basins.with_name(basins.name.replace('streamreach_basins', 'streamnet'))


def source_crs(path: Path) -> pyproj.CRS:
    """The CRS as the file's GeoParquet metadata states it; a null CRS is the spec's CRS84."""
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    geo = json.loads(metadata.get(b'geo', b'{}'))
    column = geo.get('primary_column', schema.geometry)
    crs = geo.get('columns', {}).get(column, {}).get('crs', 'missing')
    if crs == 'missing':
        raise RuntimeError(f'{path.name} carries no GeoParquet CRS metadata')
    return pyproj.CRS.from_json_dict(crs) if crs is not None else pyproj.CRS.from_user_input('OGC:CRS84')


def id_column(path: Path) -> str:
    """The column holding the globally unique raw id: TDXHydroLinkNo, or LINKNO once 1_translate_tdxhydro.py has
    stamped the global id into it."""
    names = pq.read_schema(path).names
    return schema.tdx_link_no_field if schema.tdx_link_no_field in names else schema.tdx_link_field


def batches(path: Path, ids: str, crs: pyproj.CRS, batch_size: int = 100_000):
    """The file as GeoDataFrames of ``batch_size`` rows, so a large region is never held whole."""
    parquet = pq.ParquetFile(path)
    geoarrow = parquet.schema_arrow.field(schema.geometry).metadata is not None
    for batch in parquet.iter_batches(batch_size=batch_size, columns=[ids, schema.geometry]):
        if geoarrow:
            geometry = gpd.GeoDataFrame.from_arrow(batch).geometry.values
        else:
            geometry = shapely.from_wkb(batch.column(schema.geometry).to_numpy(zero_copy_only=False))
        yield gpd.GeoDataFrame({ids: batch.column(ids).to_numpy().astype(np.int64)},
                               geometry=geometry, crs=crs)


def accumulate(link: np.ndarray, downstream: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Each reach's row of ``values`` plus those of every reach upstream of it.

    Processed in waves from the headwaters: a reach is added into the one below it only once
    everything above it has been added into it. A downstream id that is not in the table (-1 at
    an outlet) ends the path.
    """
    down = pd.Index(link).get_indexer(downstream)
    total = values.copy()
    waiting = np.bincount(down[down >= 0], minlength=len(link))
    frontier = np.flatnonzero(waiting == 0)
    done = 0
    while frontier.size:
        done += frontier.size
        frontier = frontier[down[frontier] >= 0]
        targets = down[frontier]
        np.add.at(total, targets, total[frontier])
        np.subtract.at(waiting, targets, 1)
        targets = np.unique(targets)
        frontier = targets[waiting[targets] == 0]
    if done != len(link):
        raise RuntimeError(f'{len(link) - done:,} reaches never drained - the network has a cycle')
    return total


def outlets(link: np.ndarray, downstream: np.ndarray) -> np.ndarray:
    """The row of each reach's outlet - the last reach on its path downstream.

    Pointer jumping: every pass doubles how far each reach has looked, so the deepest network
    resolves in a few dozen passes over the arrays rather than one walk per reach.
    """
    down = pd.Index(link).get_indexer(downstream)
    outlet = np.where(down >= 0, down, np.arange(len(link)))
    for _ in range(64):
        jumped = outlet[outlet]
        if np.array_equal(jumped, outlet):
            return outlet
        outlet = jumped
    raise RuntimeError('outlets did not converge - the network has a cycle')
