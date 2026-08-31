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

Measured: streams for group 101 (103 when this was measured - see group_renumbering.csv)
fall from 71.8 MB to 29.1 MB and the catchments of a test
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
import json

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import shapely

__all__ = [
    'WRITE_OPTS',
    'GEOMETRY_ROW_GROUP_SIZE',
    'SOURCE_WRITE_OPTS',
    'SOURCE_ROW_GROUP_SIZE',
    'concat_geoparquet',
    'write_geoparquet',
    'write_parquet',
    'write_source_geoparquet',
]

COMPRESSION = 'zstd'
COMPRESSION_LEVEL = 3
WRITE_OPTS = {'compression': COMPRESSION, 'compression_level': COMPRESSION_LEVEL}

GEOMETRY_ROW_GROUP_SIZE = 500

# The raw TDX-Hydro geoparquet step 1 writes is the one file this pipeline does not snap, so it
# wants a different profile from everything above - see write_source_geoparquet.
SOURCE_COMPRESSION_LEVEL = 9
SOURCE_WRITE_OPTS = {'compression': COMPRESSION, 'compression_level': SOURCE_COMPRESSION_LEVEL}

# Large enough that the per-group overhead is nothing, small enough that a reader can take one
# group without taking the region. The size on disk does not move across 1, 2 and 12 groups of the
# same file, so this is chosen entirely for what it lets a reader do.
SOURCE_ROW_GROUP_SIZE = 20_000

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


def concat_geoparquet(paths: list, out) -> int:
    """Stack files ``write_geoparquet`` wrote into one, a row group at a time.

    Never holds more than one row group, so the total size does not matter, and the parts' row
    grouping carries through unchanged. The parts must share a schema - ``write_geoparquet`` picks
    the geoarrow nesting from the geometry types present, so a part with no multipolygon in it is
    written a level shallower than its neighbours - and the bbox in the output's ``geo`` metadata is
    the union of theirs.
    """
    handles = [pq.ParquetFile(path) for path in paths]
    template = handles[0].schema_arrow
    for path, handle in zip(paths[1:], handles[1:]):
        if not handle.schema_arrow.equals(template, check_metadata=False):
            raise ValueError(
                f'{path} cannot be stacked with {paths[0]}: the schemas differ.\n'
                f'{paths[0]}: {template}\n{path}: {handle.schema_arrow}'
            )

    metadata = dict(template.metadata or {})
    geo = json.loads(metadata[b'geo'])
    column = geo['primary_column']
    boxes = [json.loads(dict(h.schema_arrow.metadata)[b'geo'])['columns'][column].get('bbox')
             for h in handles]
    if all(box is not None for box in boxes):
        geo['columns'][column]['bbox'] = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                                          max(b[2] for b in boxes), max(b[3] for b in boxes)]
    metadata[b'geo'] = json.dumps(geo).encode()
    schema = template.with_metadata(metadata)

    # the coordinate leaves off the parquet schema, rather than rebuilt from the geometry type the
    # way coordinate_columns does it: the parts already say what nesting they were written at
    leaves = [handles[0].schema.column(i).path for i in range(len(handles[0].schema))]
    coordinates = [leaf for leaf in leaves
                   if leaf.startswith(f'{column}.') and leaf.rsplit('.', 1)[-1] in ('x', 'y', 'z')]

    rows = 0
    with pq.ParquetWriter(out, schema, use_dictionary=False,
                          column_encoding={c: 'BYTE_STREAM_SPLIT' for c in coordinates},
                          **WRITE_OPTS) as writer:
        for handle in handles:
            for group in range(handle.num_row_groups):
                table = handle.read_row_group(group)
                writer.write_table(table)
                rows += table.num_rows
    return rows


def write_source_geoparquet(gdf: gpd.GeoDataFrame, path,
                            row_group_size=SOURCE_ROW_GROUP_SIZE, **kwargs) -> None:
    """Write the raw TDX-Hydro geoparquet: geoarrow and zstd, but *not* BYTE_STREAM_SPLIT.

    This is the one product the pipeline writes on the wrong side of the 1 m mercator snap, and
    that changes the answer. ``projection.to_web_mercator`` is what zeroes the low mantissa bytes
    of a coordinate, and the encoding in write_geoparquet only pays once they are zero; the source
    is full-precision WGS84 degrees that has had no snap, so those bytes are noise and splitting
    them into their own planes costs the interleaving zstd was already exploiting. Measured on a
    102 MB basins file, against the same file written every other way:

        snappy, WKB (what this used to be)       101.7 MB   100%
        zstd, WKB                                 57.3 MB    56%
        zstd, geoarrow                            31.2 MB    31%
        zstd, geoarrow + BYTE_STREAM_SPLIT        55.0 MB    54%   <- nearly double

    So geoarrow is worth taking and the encoding on top of it is not. The gain is smaller on the
    big regions - a 1.8 GB basins file goes to 47%, a 767 MB streamnet file to 36% - which still
    puts the whole 144 GB source tree near 65 GB.

    zstd 9 rather than the 3 the published files use, because this is the only file written once
    and read on every pipeline run after that: it takes 1.7x the write time for another 5 points,
    and read speed is flat across zstd levels either way.

    One thing to know before diffing the output against an older copy. geoarrow's schema is fixed
    per column, so a mixed Polygon/MultiPolygon column - which every basins file is - comes back
    entirely as MultiPolygon. Coordinates are bit-identical and areas are exactly equal, so nothing
    downstream can tell (step 4 only unions them), but a WKB comparison will say the file changed.
    """
    opts = {**SOURCE_WRITE_OPTS, 'row_group_size': row_group_size, **kwargs}
    if len(gdf) == 0 or gdf.geometry.isna().all():
        # same reason write_geoparquet falls back: geoarrow needs one geometry to take its type from
        gdf.to_parquet(path, **opts)
        return
    gdf.to_parquet(path, geometry_encoding='geoarrow', **opts)


def write_parquet(df: pd.DataFrame, path, **kwargs) -> None:
    """Write a table with no geometry. Nothing to tune: without a geometry column there is
    no float payload big enough for the encoding above to matter, and the default
    dictionary encoding is the right one for the id and attribute columns."""
    df.to_parquet(path, **{**WRITE_OPTS, **kwargs})
