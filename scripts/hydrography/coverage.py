"""
Node the source coverage: make every shared basin edge one line instead of two.

The TDX-Hydro basins are a coverage in intent and not in fact. Neighbours agree on where a divide
runs, but one side carries a vertex the other does not, so two rings that are meant to be the same
line are not the same line - collinear, coincident, and un-noded. Measured on 5020055870, **4,519
of its 4,554 basins carry at least one invalid coverage edge**, and no region file tried is a valid
coverage.

Everything downstream pays for it, and the whole pipeline is built around not being able to fix it:

* ``coverage_simplify`` in step 4 is running on an invalid coverage, which is why it needs the
  bisection escape in ``geometry.simplify_coverage`` and why its output is routinely
  self-intersecting.
* ``coverage_union_all`` is unusable on the result - it raises a side-location conflict on the big
  groups, and worse, on the small ones it *returns*, with slivers in it: on group 122 its answer
  differs from the exact union by 390 m2 and ``symmetric_difference`` against that union throws
  ``unable to assign free hole to a shell``. So step 6 dissolved with ``hierarchical_union``, at
  5-9x the cost of the coverage operation it should be using.
* The unmatched vertex pairs leave gap slivers, and a dissolve turns a gap into a *hole*. Of group
  108's 180 outline holes, 42 are that and not a watershed; of group 122's 73, nine are.

GEOS 3.14 added ``GEOSCoverageClean``, which nodes the whole coverage at once and assigns any
overlap to one side. Shapely exposes it as ``coverage_clean`` from 2.2, which is **not released**
(2.1.2 is current and carries GEOS 3.13.1), so it is reached through the GDAL CLI instead:
``gdal vector clean-coverage``, GDAL >= 3.12 built against GEOS >= 3.14. That is a binary
dependency of the same kind as tippecanoe, and the same kind of check: the subcommand is only
registered when the GEOS behind it is new enough, so its presence in ``gdal vector --help`` is the
capability test, not a version number.

Measured on 5020055870 (4,554 basins, 4.6M vertices, 9.6 s):

    coverage_is_valid                    False -> True     (4,519 invalid polygons -> 0)
    vertices                             4,619,344 -> 4,627,788        (+0.18%)
    per-basin |area change|              <= 3.8e-17 deg2, i.e. float noise
    outline vertices dropped / added     0 / 0
    outline area change                  1.5e-13 deg2
    holes in the dissolved outline       54 -> 24

The outline is the line to watch. Cleaning a *sub*-coverage moves its outer edge - on group 122's
catchments, the clean added 29 vertices to the outline and shifted it by 2,121 m2 - because that
edge is an interior line of the real coverage, cut through ground the clean is entitled to renode.
On a whole source region it is the level-2 divide, nothing inside the file abuts it, and it comes
back untouched. **That is why this runs on the source region files and nothing smaller.**

Cost, measured on 8020022890 (40,830,451 source vertices): 91 s and 29.20 GB resident, so 447k
vertices/s on one core - GEOSCoverageClean is a single whole-coverage call and does not thread -
and 715 bytes resident per vertex. A 400M-vertex region is therefore ~15 min and ~286 GB. Regions
are independent, so the dial is memory: see $CLEAN_JOBS and ``schedule``.

**Four of the 62 source regions do not fit in 550 GB at that rate**, the largest being 5020049720
at 1,079,308,483 vertices and ~772 GB, and the throughput above is not a constant either - it
decays under memory pressure long before the wall, from 447k vertices/s at 40M to 140k at 417M.
Both problems have the same answer: cut the region up. See ``clean_tiled_in_place``.

**Tiling is exact, and that is a measured claim.** A region is split into strips of whole basins,
each strip padded with a halo of every basin whose bounding box touches one of its own, cleaned,
and only the strip's core kept. Verified against the whole-coverage clean on a 16,298-basin,
16.9M-vertex patch of 2020065840 (11,434 of its basins carrying an invalid coverage edge), split
eight ways:

    coverage_is_valid                    True, 0 invalid edges, same as the whole-region clean
    vertices                             16,929,594 = 16,929,594
    total area                           16.457779330749 deg2, equal to 12 decimals
    coverage_union_all vs whole's        symmetric difference 0.0, both 18,123 rings
    basins vertex-identical              16,282 / 16,298
    the other 16                         <= 9.1e-18 deg2 of area, i.e. ~0.1 um2

A two-ring halo moves five more basins into the identical column and is not worth its cost. The
halo runs ~13% of the input for a 12.5% core, and eight tiles took the same wall clock as the one
whole clean *before* any of them ran concurrently.

Why the halo works is the same argument as the region docstring above: a sub-coverage clean is
entitled to move the sub-coverage's outer edge, so the tile is drawn one ring wider than the part
that is kept and the moved edge is discarded with the halo. A tile on the region's own perimeter
has no halo there, which is right - that edge is the level-2 divide and nothing abuts it.

**The defaults are the point.** ``--snapping-distance`` and ``--maximum-gap-width`` are left alone
because the defect here is a missing vertex on a line both sides already agree on - a zero-width
gap, which noding closes without moving anything. A nonzero gap width would start swallowing real
holes, and the real holes are the endorheic sinks and dropped watersheds that 6_publish_basins.py
goes to some trouble to tell apart from slivers. The tiled path is the one exception, and only
because it has to be: GEOS derives the default snapping distance from the input extent, so it is
the one parameter that differs between a tile and the region it came from. See
TILE_SNAPPING_DISTANCE for what it is pinned to and the measurement saying the pinned values and
the defaults return the same coverage.
"""
import json
import logging
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from . import parquet
from . import schema

__all__ = [
    'binary',
    'clean_in_place',
    'clean_tiled_in_place',
    'estimated_peak',
    'is_clean',
    'marker_for',
    'needs_tiling',
    'schedule',
    'tile_vertices',
    'total_memory',
    'unavailable',
    'vertex_count',
]

# Maximum resident set size per input vertex. GEOS holds the whole coverage plus its noding
# structure and frees none of it until the call returns, so this is a peak and not an average, and
# it is *maximum resident* rather than the "peak memory footprint" the same tool reports: measured
# on 8020022890, 40,830,451 source vertices took 29.20 GB resident against a 20.71 GB footprint,
# and it is the resident figure that decides whether a run survives. 715 bytes is that measurement;
# the footprint numbers from the published catchments (3.32 GB over 6,807,717 vertices, 39.74 GB
# over 81,308,409) agree with each other at ~490 and are the same thing measured the softer way.
bytes_per_vertex = 715

# Vertices a second, one core. Measured on the same file: 40,830,451 in 91.25 s.
vertices_per_second = 447_000

# How much of the machine the clean is allowed to plan for. The rest is the rewrite that follows
# each clean, the OS, and being wrong about the line above.
memory_fraction = 0.8

# The subcommand GDAL only registers when it was built against GEOS >= 3.14, which makes asking for
# it by name the whole version check.
SUBCOMMAND = 'clean-coverage'

# What the cleaned file is written as on the way out of GDAL. It is read straight back and
# rewritten by parquet.write_source_geoparquet, so this profile is chosen to be cheap rather than
# final: geoarrow at the default zstd level keeps the intermediate near the source's size (14.1 MB
# against 14.0 MB on 5020055870, where GDAL's own defaults give 30.1 MB) without paying zstd 9
# twice, and the covering bbox column is off because the rewrite would drop it anyway.
CREATION_OPTIONS = (
    'COMPRESSION=ZSTD',
    'GEOMETRY_ENCODING=GEOARROW',
    'WRITE_COVERING_BBOX=NO',
)

# What a tile is cleaned with, and the one place the tiled path departs from the whole-region one.
#
# The snapping distance has to be pinned because GEOS derives its default *from the extent of the
# input*, which is a different number for every tile and for the region they came out of - the one
# parameter that would make a tiled clean disagree with a whole one by construction. Zero is the
# value to pin it to: the defect here is a missing vertex on a line both sides already agree on, so
# noding alone closes it, and on the 16.9M-vertex patch above `--snapping-distance 0
# --merge-strategy min-index` came back *bit-identical* to GDAL's defaults - same 16,298 basins,
# same 16,929,594 vertices, same area to 12 decimals, both valid coverages - in 23.6 s against
# 31.8 s.
#
# min-index rather than the default longest-border because it is the strategy that cannot disagree
# across a seam. Where two basins overlap, the overlap goes to whichever of them appears first in
# the input, and tiles are always built in source row order, so the two tiles that both contain a
# given pair rank that pair the same way. longest-border compares a measurement instead, and a
# measurement taken on either side of a seam is a measurement of two slightly different lines.
TILE_SNAPPING_DISTANCE = 0
TILE_MERGE_STRATEGY = 'min-index'

# Compression for the tile files on the way into GDAL. They are read once, by the process started
# on the next line, and deleted; level 3 writes them ~3x faster than the level 9 the source tree is
# kept at and nothing ever reads them twice.
TILE_WRITE_OPTS = {'compression': 'zstd', 'compression_level': 3}

# Beside the file, not inside it: the cleaned file is written by parquet.write_source_geoparquet
# like every other source file, and that writer has nowhere to put a key-value pair. The marker
# is what says the file has been cleaned; step 1 deletes it whenever it rewrites the file.
MARKER_SUFFIX = '.coverage-clean.json'


def binary() -> str:
    """The gdal CLI to call. $GDAL_BIN overrides, for a build that is not first on $PATH."""
    return os.environ.get('GDAL_BIN', 'gdal')


def unavailable() -> str:
    """Why this file cannot be cleaned here, or '' if it can.

    A string rather than a bool because every caller wants to say what is missing: the difference
    between no gdal at all and a gdal without the GEOS behind it is the difference between
    installing something and rebuilding it.
    """
    found = shutil.which(binary())
    if found is None:
        return (f'{binary()} is not on $PATH - the coverage clean needs GDAL >= 3.12 built '
                f'against GEOS >= 3.14 (brew install gdal), or $GDAL_BIN pointed at one')
    try:
        listing = subprocess.run([found, 'vector', '--help'], capture_output=True, text=True,
                                 timeout=120)
    except OSError as error:
        return f'{found} could not be run: {error}'
    if SUBCOMMAND not in listing.stdout + listing.stderr:
        version = subprocess.run([found, '--version'], capture_output=True, text=True)
        return (f'{found} ({version.stdout.strip() or "unknown version"}) has no '
                f'"gdal vector {SUBCOMMAND}" - it needs GDAL >= 3.12 built against GEOS >= 3.14')
    return ''


def vertex_count(path: Path) -> int:
    """How many coordinates a geoarrow geoparquet holds, read from the footer.

    The x child array has one value per vertex, and its row-group statistics are in the footer, so
    this is a metadata read rather than a pass over the geometry - which matters when the geometry
    is 2.78 GB. Returns 0 for a WKB-encoded file, where the count is not recorded anywhere.
    """
    handle = pq.ParquetFile(path)
    leaves = [handle.schema.column(i).path for i in range(len(handle.schema))]
    wanted = [i for i, leaf in enumerate(leaves)
              if leaf.startswith(f'{schema.geometry}.') and leaf.endswith('.x')]
    if not wanted:
        return 0
    metadata = handle.metadata
    return sum(metadata.row_group(group).column(column).num_values
               for group in range(metadata.num_row_groups) for column in wanted)


def estimated_peak(path: Path) -> int:
    """What cleaning this file is expected to cost in resident memory, from its vertex count."""
    return vertex_count(path) * bytes_per_vertex


def total_memory() -> int:
    """The machine's RAM, or 0 if it will not say. $CLEAN_MEMORY_BYTES overrides."""
    override = os.environ.get('CLEAN_MEMORY_BYTES')
    if override:
        return int(override)
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (AttributeError, ValueError, OSError):
        return 0


def tile_vertices(workers: int, memory: int = None) -> int:
    """How many vertices one tile may hold if ``workers`` of them are being cleaned at once.

    The whole memory model in one line: a clean is a single GEOS call whose peak is set by the
    vertex count it is handed, so the way to fit a region into a machine is to hand it less. A
    machine that will not say how much memory it has gets 0, which the callers read as "do not
    tile" - guessing a tile size is worse than running the file whole and finding out.
    """
    if memory is None:
        memory = total_memory()
    if memory <= 0:
        return 0
    return int(memory * memory_fraction / max(workers, 1) / bytes_per_vertex)


def needs_tiling(path: Path, workers: int, memory: int = None) -> bool:
    """Is this file too big to clean whole with ``workers`` cleans running?"""
    target = tile_vertices(workers, memory)
    return bool(target) and vertex_count(path) > target


def schedule(files, workers: int, memory: int = None) -> tuple:
    """Split the work into what can run ``workers`` at a time and what has to be tiled.

    Smallest first, and the split is the reason. A clean is one indivisible GEOS call holding
    everything it is given, so a region that does not fit in its share of the machine cannot share
    the machine with anything: it goes in the second list, is run on its own, and gets cut into
    tiles that do fit - ``workers`` of them at a time, which is why the size that decides the split
    is the same ``tile_vertices(workers)`` that sizes the tiles. Running the cheap ones first also
    means an interrupted run keeps most of the tree marked.

    ``memory`` defaults to ``total_memory()``, and a machine that will not say is treated as
    unknown rather than infinite - everything goes in the second list, where it is cleaned whole
    and alone, which is what this step did before tiling existed.
    """
    ordered = sorted(files, key=estimated_peak)
    if memory is None:
        memory = total_memory()
    if workers <= 1 or memory <= 0:
        return [], ordered
    budget = memory * memory_fraction
    cut = len(ordered)
    for index, path in enumerate(ordered):
        if estimated_peak(path) * workers > budget:
            cut = index
            break
    return ordered[:cut], ordered[cut:]


def marker_for(path: Path) -> Path:
    """The marker beside a source file."""
    return path.with_name(path.name + MARKER_SUFFIX)


def is_clean(path: Path) -> bool:
    """Has this file been cleaned? The marker beside it says so; delete it to clean again."""
    return marker_for(path).exists() and path.exists()


def _geoparquet_crs(path: Path):
    """A file's CRS as its GeoParquet metadata states it, or None for the spec's null = CRS84.

    Read and put back because GDAL does not carry it through: handed a file geopandas wrote as
    EPSG:4326, ``clean-coverage`` writes the geometry back with ``crs: null``, which is the same
    ground under a different name and would still show up as a changed CRS on every reader that
    compares one. The same rewrite is what restores the geoarrow extension metadata on the
    geometry field - 4_create_catchments.py's ``read_source_wkb`` decides between its geoarrow and
    WKB paths by whether that metadata is there, and GDAL leaves it off.
    """
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    geo = json.loads(metadata.get(b'geo', b'{}'))
    column = geo.get('primary_column', schema.geometry)
    return geo.get('columns', {}).get(column, {}).get('crs')


def _run_clean(source: Path, destination: Path, snapping=None, strategy=None) -> float:
    """One ``gdal vector clean-coverage`` call. Returns the seconds it took.

    ``snapping`` and ``strategy`` are left off entirely when they are None, which is not the same
    as passing GEOS's defaults explicitly: the default snapping distance is derived from the input
    extent and there is no number to write down for it. The tiled path pins both - see
    TILE_SNAPPING_DISTANCE.
    """
    command = [shutil.which(binary()), 'vector', SUBCOMMAND, '--overwrite', '--quiet',
               '--output-format', 'Parquet']
    for option in CREATION_OPTIONS:
        command += ['--lco', option]
    if snapping is not None:
        command += ['--snapping-distance', str(snapping)]
    if strategy is not None:
        command += ['--merge-strategy', strategy]
    command += [str(source), str(destination)]
    logging.info(' '.join(command))
    started = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f'{SUBCOMMAND} failed on {source.name}: '
                           f'{(result.stderr or result.stdout).strip()[:500]}')
    return time.time() - started


def _finish(path: Path, frame: gpd.GeoDataFrame, crs, columns: list) -> int:
    """Put a cleaned frame back where the source was, through the pipeline's own writer.

    Only the last line touches ``path``: the frame is written to a sibling and renamed, so a run
    killed here leaves the original in place and unmarked rather than half-rewritten.
    """
    if crs is not None:
        frame = frame.set_crs(crs, allow_override=True)
    # GDAL writes the geometry last; the source has it wherever step 1 put it
    frame = frame[[name for name in columns if name in frame.columns]]
    # and the index goes back to a range. The tiled path reassembles by putting source row numbers
    # in the index and sorting on them, and pandas writes any index that is not a RangeIndex out as
    # a real column - which would leave the cleaned file carrying an __index_level_0__ the source
    # never had, a schema change for every reader downstream. The index means nothing here either
    # way: these files are addressed by LINKNO.
    frame = frame.reset_index(drop=True)
    partial = path.with_name(path.name + '.partial')
    parquet.write_source_geoparquet(frame, partial)
    partial.replace(path)
    return len(frame)


def _write_marker(path: Path, record: dict) -> dict:
    """Write the record beside the file it describes."""
    record = {**record, 'size': path.stat().st_size}
    with open(marker_for(path), 'w') as f:
        json.dump(record, f, indent=2)
    return record


def clean_in_place(path: Path, workers: int = 1, keep_intermediate: bool = False) -> dict:
    """Node one source file's coverage, replacing it, and leave the marker that says so.

    Hands off to ``clean_tiled_in_place`` when the file is too big to clean whole with ``workers``
    cleans in flight - which is the only thing ``workers`` is consulted for on this path, since a
    whole-region clean is one single-threaded GEOS call whatever else is running.

    Three files exist at once and none of them is the one being replaced: GDAL writes the cleaned
    geometry aside, the rewrite through ``parquet.write_source_geoparquet`` writes a second file
    aside, and only a rename touches ``path``. A run killed anywhere in the middle leaves the
    original where it was, unmarked, and the next run does it again.
    """
    reason = unavailable()
    if reason:
        raise RuntimeError(reason)
    if needs_tiling(path, workers):
        return clean_tiled_in_place(path, workers, keep_intermediate=keep_intermediate)

    started = time.time()
    before = vertex_count(path)
    crs = _geoparquet_crs(path)
    columns = list(pq.ParquetFile(path).schema_arrow.names)

    # deliberately not named *.parquet, and so deliberately given the format by hand: a killed run
    # leaves this file behind, and the caller finds its work by globbing TDX_*_basins_*.parquet
    cleaned = path.with_name(path.name + '.cleaning')
    gdal_seconds = _run_clean(path, cleaned)
    rows = _finish(path, gpd.read_parquet(cleaned), crs, columns)
    if not keep_intermediate:
        cleaned.unlink(missing_ok=True)

    return _write_marker(path, {
        'cleaned': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'tool': f'gdal vector {SUBCOMMAND}',
        'tiles': 1,
        'rows': rows,
        'vertices_before': before,
        'vertices_after': vertex_count(path),
        'seconds': round(time.time() - started, 1),
        'gdal_seconds': round(gdal_seconds, 1),
    })


def _survey(path: Path) -> tuple:
    """Every basin's bounding box and vertex count, one row group at a time.

    The two numbers the tiling needs, and neither of them is in the footer: the box because the
    halo is decided by which basins touch, the vertex count because that is what sets a tile's
    memory and basins differ in it by three orders of magnitude - cutting on basin count would give
    tiles whose peaks differ by as much. It is a full pass over the geometry, which sounds
    expensive against a metadata read but is minutes against the hours the clean itself takes, and
    it never holds more than one row group (SOURCE_ROW_GROUP_SIZE = 20,000 basins).
    """
    handle = pq.ParquetFile(path)
    boxes, counts = [], []
    for group in range(handle.num_row_groups):
        geometry = gpd.GeoDataFrame.from_arrow(handle.read_row_group(group)).geometry.values
        boxes.append(shapely.bounds(geometry))
        counts.append(shapely.get_num_coordinates(geometry))
    return np.concatenate(boxes), np.concatenate(counts)


def _tile_plan(bounds, counts, target: int) -> tuple:
    """Cut the basins into contiguous strips of about ``target`` vertices each.

    Strips along the region's longer axis, which is the shape that keeps the halo cheap: halo cost
    is the length of the cuts, so cutting the long way round makes each cut as short as it can be
    and puts the fewest basins on it. Sorted by the box centre rather than by a corner so that a
    basin lands in the strip it mostly sits in.

    Equal *vertices* per strip, not equal basins, and the strips are balanced against each other
    rather than filled to the target in turn: ceil(total / target) strips of total/ceil each is the
    same number of tiles with the largest one smaller, which is the one that decides whether the
    run fits.

    Returns the axis it cut on and the strips, each a sorted array of source row numbers.
    """
    minx, miny, maxx, maxy = bounds.T
    # an empty geometry has no box, so it has no place in a strip and no neighbours to be a halo
    # for; better to say so here than to let a nan sort to the end and poison the STRtree
    if not np.isfinite(bounds).all():
        raise RuntimeError(f'{int((~np.isfinite(bounds)).any(axis=1).sum()):,} basins have no '
                           f'bounding box (empty or null geometry) and cannot be tiled')
    horizontal = (maxx.max() - minx.min()) >= (maxy.max() - miny.min())
    centre = (minx + maxx) / 2 if horizontal else (miny + maxy) / 2
    order = np.argsort(centre, kind='stable')
    total = int(counts.sum())
    tiles = max(1, math.ceil(total / target))
    running = np.cumsum(counts[order])
    cuts = np.searchsorted(running, np.arange(1, tiles) * (total / tiles))
    cores = [np.sort(piece) for piece in np.split(order, cuts) if len(piece)]
    return ('x' if horizontal else 'y'), cores


def _with_halo(bounds, cores: list) -> list:
    """Grow each strip by every basin whose box touches one of the strip's own.

    Box adjacency rather than geometric adjacency, which is a *superset* of it - two polygons that
    touch always have overlapping boxes - and a superset is the safe direction to be wrong in: the
    halo is thrown away, so an extra ring of it costs a little memory and nothing else, while a
    missing one would leave a seam that neither tile had the context to node.
    """
    boxes = shapely.box(*bounds.T)
    tree = shapely.STRtree(boxes)
    return [np.union1d(core, tree.query(boxes[core], predicate='intersects')[1])
            for core in cores]


def _write_tiles(path: Path, rows: list, directory: Path) -> list:
    """Write one parquet per tile, in a single pass over the source.

    One pass and not one per tile: the tiles overlap, so a row group is read once and dealt out to
    every tile that wants a row from it, and the peak is one row group rather than one region. Rows
    keep their source order inside each tile, which is what makes TILE_MERGE_STRATEGY's min-index
    mean the same thing in two tiles that both contain the same pair of basins.

    The source schema is carried through untouched, so the tiles are the same GeoParquet the source
    is and GDAL reads them the same way.
    """
    # from scratch: a killed run leaves a directory full of tiles cut to a plan that no longer
    # applies, and the tile count is free to change between runs
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    handle = pq.ParquetFile(path)
    membership = np.zeros((len(rows), handle.metadata.num_rows), dtype=bool)
    for index, tile in enumerate(rows):
        membership[index, tile] = True

    paths = [directory / f'tile{index:03d}.parquet' for index in range(len(rows))]
    writers = [None] * len(rows)
    offset = 0
    try:
        for group in range(handle.num_row_groups):
            table = handle.read_row_group(group)
            for index, mask in enumerate(membership):
                local = np.nonzero(mask[offset:offset + table.num_rows])[0]
                if not len(local):
                    continue
                if writers[index] is None:
                    writers[index] = pq.ParquetWriter(paths[index], table.schema,
                                                      **TILE_WRITE_OPTS)
                writers[index].write_table(table.take(pa.array(local)))
            offset += table.num_rows
    finally:
        for writer in writers:
            if writer is not None:
                writer.close()
    return paths


def _check_row_order(tile: Path, frame: gpd.GeoDataFrame) -> None:
    """Confirm ``clean-coverage`` gave the rows back in the order it was handed them.

    Everything about the reassembly rests on this - the core rows are picked out of the output by
    position - and it is not written down as a guarantee anywhere in GDAL, so it is checked rather
    than assumed. The attribute columns are the witness: they are carried through untouched, so if
    they come back in order the geometry did too. Costs one read of a column that is 4-8 bytes a
    row against a geometry that is thousands.
    """
    names = [name for name in pq.ParquetFile(tile).schema_arrow.names if name != schema.geometry]
    if not names:
        return
    before = pq.read_table(tile, columns=names)
    if before.num_rows != len(frame):
        raise RuntimeError(f'{SUBCOMMAND} returned {len(frame):,} rows for {tile.name}, which '
                           f'was handed {before.num_rows:,}')
    for name in names:
        if not np.array_equal(before.column(name).to_numpy(zero_copy_only=False),
                              frame[name].to_numpy()):
            raise RuntimeError(f'{SUBCOMMAND} reordered the rows of {tile.name} (column '
                               f'{name} came back in a different order). The tiled clean cannot '
                               f'reassemble a reordered tile - clean this region whole instead.')


def clean_tiled_in_place(path: Path, workers: int = 1, target: int = None,
                         keep_intermediate: bool = False) -> dict:
    """Node a source file's coverage in tiles, for regions too big to clean whole.

    Cut into strips of whole basins, each padded with a halo of everything touching it, cleaned
    ``workers`` at a time, and reassembled from the strips' cores. The result is the whole-region
    clean - see the module docstring for the measurements that say so - and the point is that the
    peak is now ``target`` vertices instead of the region's, so a 1.08-billion-vertex region asks
    for 97 GB in eight pieces instead of 772 GB in one.

    ``target`` defaults to ``tile_vertices(workers)``, i.e. an equal share of the machine each.

    The assembly at the end does hold the whole cleaned region at once, as a GeoDataFrame and again
    as the arrow table the writer builds - tens of GB on the largest region, against the hundreds
    the clean itself would have wanted. That is the reason the scheduler runs a tiled file alone.
    """
    reason = unavailable()
    if reason:
        raise RuntimeError(reason)

    started = time.time()
    before = vertex_count(path)
    crs = _geoparquet_crs(path)
    columns = list(pq.ParquetFile(path).schema_arrow.names)
    if target is None:
        target = tile_vertices(workers)
    if not target:
        raise RuntimeError('cannot size a tile: this machine will not say how much memory it has '
                           '- set $CLEAN_MEMORY_BYTES')

    bounds, counts = _survey(path)
    axis, cores = _tile_plan(bounds, counts, target)
    rows = _with_halo(bounds, cores)
    del bounds
    logging.info(f'{path.name}: {len(cores)} tiles on {axis}, '
                 f'largest {max(len(tile) for tile in rows):,} rows of {len(counts):,}')

    directory = path.with_name(path.name + '.tiles')
    tiles = _write_tiles(path, rows, directory)
    cleaned = [tile.with_name(tile.name.replace('.parquet', '.clean.parquet')) for tile in tiles]

    # threads, not processes: each one is waiting on a subprocess that owns its own memory, which
    # is exactly the memory `target` was sized against
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        gdal_seconds = sum(pool.map(
            lambda pair: _run_clean(pair[0], pair[1], TILE_SNAPPING_DISTANCE, TILE_MERGE_STRATEGY),
            zip(tiles, cleaned)))

    pieces = []
    for core, tile_rows, tile, output in zip(cores, rows, tiles, cleaned):
        frame = gpd.read_parquet(output)
        _check_row_order(tile, frame)
        # the core's rows sit inside the tile's at these positions, both being sorted source order
        piece = frame.iloc[np.searchsorted(tile_rows, core)]
        piece.index = core
        pieces.append(piece)
        del frame
        if not keep_intermediate:
            tile.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
    frame = gpd.GeoDataFrame(pd.concat(pieces).sort_index())
    del pieces

    halo_rows = sum(len(tile) for tile in rows) - len(frame)
    number_of_rows = _finish(path, frame, crs, columns)
    del frame
    if not keep_intermediate:
        shutil.rmtree(directory, ignore_errors=True)

    return _write_marker(path, {
        'cleaned': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'tool': f'gdal vector {SUBCOMMAND}',
        'tiles': len(cores),
        'tile_axis': axis,
        'tile_target_vertices': int(target),
        'halo_rows': int(halo_rows),
        'snapping_distance': TILE_SNAPPING_DISTANCE,
        'merge_strategy': TILE_MERGE_STRATEGY,
        'rows': number_of_rows,
        'vertices_before': before,
        'vertices_after': vertex_count(path),
        'seconds': round(time.time() - started, 1),
        'gdal_seconds': round(gdal_seconds, 1),
    })
