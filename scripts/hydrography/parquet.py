"""
How every step writes parquet.

The options here used to be copy-pasted into each step with a "must match" comment, which
is fragile: these files are split and concatenated across steps, so a codec that differs
between two of them re-encodes the data on the way through. They live in one place now.

Geometry is 92-96% of every file that has any, so it is the only part worth tuning:

    encoding    GeoParquet 1.0's WKB packs a coordinate into 16 interleaved bytes. The
                native geoarrow encoding added in GeoParquet 1.1 stores x and y as
                separate float64 child arrays instead, which is what lets a columnar
                codec see them as two smooth sequences rather than one shuffled blob.

    BYTE_STREAM_SPLIT   a standard parquet encoding that transposes each float64 into
                eight byte-planes, so the high-order bytes of neighbouring coordinates -
                nearly identical for vertices a few metres apart - end up adjacent and
                zstd can collapse them. It only pays once the low mantissa bytes are
                mostly zero; on full-precision floats it makes files *larger*, because
                the low planes are noise and splitting them costs the interleaving zstd
                was already exploiting. projection.to_web_mercator is what zeroes them,
                by snapping every published geometry to a 1 m grid.

Measured: streams for group 103 fall from 71.8 MB to 29.1 MB and the catchments of a test
region from 95.2 MB to 33.2 MB, both about 60%. Roughly two thirds of that is the snap and
one third the encoding — neither is worth much without the other.

Points are a separate case: geoarrow wins on confluences (~25%) no matter how precise the
coordinates are, because a WKB point spends 5 of its 21 bytes on a type header that the
native encoding does not repeat per row.

Two things about the pyarrow API are worth knowing before touching this:
``use_byte_stream_split=True`` is silently ignored on the nested child arrays geoarrow
writes - the columns have to be named by their full leaf path - and pyarrow refuses
``column_encoding`` unless dictionary encoding is off, which is why it is disabled here.
"""
import geopandas as gpd
import pandas as pd
import shapely

__all__ = [
    'WRITE_OPTS',
    'GEOMETRY_ROW_GROUP_SIZE',
    'write_geoparquet',
    'write_parquet',
]

COMPRESSION = 'zstd'
COMPRESSION_LEVEL = 3
WRITE_OPTS = {'compression': COMPRESSION, 'compression_level': COMPRESSION_LEVEL}

GEOMETRY_ROW_GROUP_SIZE = 500

# geoarrow nests coordinates one list level per geometry dimension: a point is a bare
# struct<x, y>, a linestring a list of those, a polygon a list of rings, a multipolygon a
# list of those again. The nesting is fixed by the geoarrow spec, so the parquet leaf
# paths follow from the geometry type alone and can be named without writing the file
# first to look at its schema.
_COORDINATE_NESTING = {
    'Point': 0,
    'LineString': 1,
    'MultiPoint': 1,
    'Polygon': 2,
    'MultiLineString': 2,
    'MultiPolygon': 3,
}


def coordinate_columns(gdf: gpd.GeoDataFrame) -> list[str]:
    """The parquet leaf paths of the x/y (/z) child arrays geoarrow will write for
    ``gdf``'s active geometry column."""
    column = gdf.geometry.name
    types = set(gdf[column].geom_type.dropna().unique())
    unknown = types - _COORDINATE_NESTING.keys()
    if unknown:
        raise ValueError(f'no geoarrow coordinate layout known for {sorted(unknown)}')
    # mixing LineString with MultiLineString promotes the whole column to the multi type,
    # one list level deeper, so the deepest type present is the one actually written
    depth = max(_COORDINATE_NESTING[t] for t in types)
    dims = 'xyz' if shapely.has_z(gdf[column].values).any() else 'xy'
    prefix = '.'.join([column] + ['list.element'] * depth)
    return [f'{prefix}.{d}' for d in dims]


def write_geoparquet(gdf: gpd.GeoDataFrame, path, row_group_size=GEOMETRY_ROW_GROUP_SIZE,
                     **kwargs) -> None:
    """Write a GeoDataFrame as GeoParquet 1.1 with byte-split geoarrow coordinates.

    ``row_group_size=None`` takes pyarrow's default, which is what the tables whose rows
    are a single point rather than a whole reach want - see GEOMETRY_ROW_GROUP_SIZE.

    Empty frames fall back to WKB: geopandas cannot build a geoarrow array without at
    least one geometry to take the type from, and the steps here deliberately write empty
    files so that every group has the same set of products."""
    opts = {**WRITE_OPTS, 'row_group_size': row_group_size, **kwargs}
    if len(gdf) == 0 or gdf.geometry.isna().all():
        gdf.to_parquet(path, **opts)
        return
    gdf.to_parquet(
        path,
        geometry_encoding='geoarrow',
        column_encoding={c: 'BYTE_STREAM_SPLIT' for c in coordinate_columns(gdf)},
        use_dictionary=False,
        **opts,
    )


def write_parquet(df: pd.DataFrame, path, **kwargs) -> None:
    """Write a table with no geometry. Nothing to tune: without a geometry column there is
    no float payload big enough for the encoding above to matter, and the default
    dictionary encoding is the right one for the id and attribute columns."""
    df.to_parquet(path, **{**WRITE_OPTS, **kwargs})
