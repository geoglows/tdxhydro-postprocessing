"""
Build one region's leaf catchment polygons: one per surviving reach, in the published row order,
cut down to the resolution the source DEM actually has.

Two things happen here, and they are one step because the second wants the geometry the first is
already holding:

1. **The leaf catchments.** The source basins are one polygon per original TDX reach; step 2
   dissolved reaches into each other, so the surviving reach's catchment is the union of the basins
   that were folded into it. Which basins those are is read back out of the json journal step 2
   wrote, replayed in the same order.

2. **A coverage pass near the source's own resolution.** ``TOLERANCE_METERS`` is a statement about
   the DEM this data came off, not about any zoom. The source basins are polygonised 1/9 arcsec
   cells, so a boundary is a run of ~3.4 m stair treads -- 1,903 vertices per polygon on average,
   ~10.4 billion across the network, 12 GB on disk -- and a tolerance a few times that cell size
   takes the staircase and very little else.

   Generalizing for a zoom happens in the tiling stage instead -- step 8 cuts each band, including
   the leaf, at a tolerance derived from the zooms that band is drawn at. Nothing zoom-dependent is
   decided here, so what is published carries the resolution the data actually has.

**What this pass does NOT do is node the coverage.** It is worth being explicit, because the
opposite was believed for a while and it is the kind of thing that gets designed around. The raw
dissolve leaves a vertex present on one side of a shared edge and absent on the other, and
``coverage_simplify`` does not repair that: measured on 7020000010, an 8,000-polygon run in
published order comes out of this step with 7,999 polygons carrying invalid coverage edges and
``coverage_union_all`` raising a side-location conflict -- at 10, 20, 30, 50, 100 *and* 300 m alike.
Coarsening does not help and never did. What does node a coverage is snapping it onto a lattice
coarse enough to merge the mismatched pair, which is what step 8 does per band and why its dissolves
mostly take the fast path.

So the consumers carry the fallback rather than relying on this: ``union_catchments`` in step 5 and
``dissolve_by`` in step 8 both try ``coverage_union_all`` and fall back to
``hy.geometry.hierarchical_union``, which gives the same answer more slowly. The tolerance here is
therefore free to be chosen on resolution and file size alone.

Three properties of the geometry govern how the middle step is done, the first two measured in the
design note:

**Coverage simplification, not per-polygon.** Neighbouring catchments share a boundary.
Douglas-Peucker run on each polygon independently simplifies the shared stretch twice, from two
different ring start points, and leaves a sliver in the gap between the two answers. GEOS
CoverageSimplify simplifies each shared edge once and hands the same line to both neighbours, so
the coverage stays exact. It keeps ~2x more vertices than DP and that is the price of not shredding
the layer -- and it is what lets a consumer dissolve these polygons cheaply, because unioning an
exact coverage is edge cancellation (``coverage_union_all``) rather than a general overlay.

**The coverage boundary is held fixed, and the chunks are spatially coherent.**
``simplify_boundary=False`` leaves the outer edge of the input untouched, which is what lets this
run per region -- a region outline is a drainage divide shared vertex-for-vertex with the next
region's catchments -- and, for the same reason, what lets a region be simplified in chunks at all.
A pinned outline is only cheap if the chunk is a compact blob, so the catchments are put in the
published riverIndex order *before* they are chunked: Hilbert-ordered watersheds, DFS post-order
within each, so a contiguous run is a compact clump. Measured at a 30 m tolerance, chunking the same
region in riverId order keeps 78.8% of the vertices where this keeps 19.7% -- four times the
reduction for a reindex, and the gap only widens as the tolerance coarsens.

**The 1 m snap happens last, and is a rounding rather than a precision reduction.** Only the
*reprojection* has to precede the simplification -- the tolerance is in mercator metres. The snap
was going through the same call as the reprojection and so inherited its place, which cost twice:
it ran on the full-resolution dissolve, 4.4x the vertices it needed to see, and it ran as
``set_precision``, which is a GEOS precision reduction and prices like an overlay. Together that
was **48% of this step**, more than the dissolve and the simplification put together. Moved after
the simplification and done as ``x -> floor(x + 0.5)`` on the coordinate buffer -- exact, because
the grid is a power of two -- the same work is ~0.1%. ``set_precision`` is kept for the handful of
rings the rounding self-intersects, where being an overlay is the point. See
``projection.snap_to_grid``.

    RFS_DATA_ROOT=... TDXHYDRO_ROOT=... python 4_create_catchments.py <region>
"""
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyproj
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

# every output path hangs off the data root - see hydrography/paths.py and $RFS_DATA_ROOT
region_root = hy.paths.region_root
tdx_root = hy.paths.tdx_root
logs_root = hy.paths.logs_root

# How many published-order chunks are in flight at once, and how many threads each one's dissolve
# gets. The product is the core count; the split between them is what is tuned.
#
# Chunks used to be built one at a time, and for most of the step that left the machine idle. The
# reprojection and the coverage simplification are single array calls into GEOS, so a worker
# building one chunk uses exactly one core - at the `-P 3` this step used to run at, 3 cores of 16
# for ~87% of the step. Both calls release the GIL (measured: four chunks simplify in 2.34 s across
# four threads against 8.21 s one after another, 3.52x), so the chunks are what should be parallel,
# and the dissolve - which was already threaded, and is only 7% of the step - gives up its threads
# to pay for it.
#
# **The ceiling is memory, not cores**, and this is why the number is 2 rather than the 4 that is
# fastest per region. Every chunk in flight holds its own GEOS copies on top of the region's WKB.
# Measured on 1020000010 (135,159 catchments, 404M source vertices), against 566 s and ~24 GB
# before:
#
#     chunk_threads      wall     peak RSS      two of these at once
#         1              274 s    ~25 GB  (inferred)     ~50 GB
#         2              177 s     29.0 GB               ~58 GB
#         4              121 s     36.7 GB               ~73 GB   - does not fit in 64 GiB
#
# So the fastest single region is not the fastest run. Projecting those per-region times over the
# 50 regions: 4 threads would force `-P 1` and take ~55 min, where 2 threads at `-P 2` takes ~40.
# Process-level parallelism is also the better kind here - it overlaps one region's serial parts,
# the WKB read and the final write, with another region's parallel ones, which threads inside a
# single region cannot. Raise this and lower `-P` in pipeline.sh together, or the two multiply.
chunk_threads = int(os.environ.get('CATCHMENT_CHUNK_THREADS', 2))
dissolve_threads = max(1, (os.cpu_count() or 8) // chunk_threads)
# A few times the 3.4 m cell of the 1/9 arcsec DEM the source basins were polygonised from: enough
# to take the raster staircase, not enough to move a boundary anywhere the source could have told
# the difference. Sub-pixel until z13.
#
# Chosen by measurement rather than by argument, because the file size does NOT fall monotonically
# with the tolerance - it rises to a hump at 10 m and only then falls. Region 7020000010:
#
#     tolerance   vertices      kept     parquet    bytes/vertex
#         5 m     205,614,049   100.00%  228.8 MB   1.11
#        10 m     137,164,013    66.71%  250.2 MB   1.82
#        20 m      62,719,701    30.50%  162.6 MB   2.59
#       100 m      ~17,100,000     8.31%  ~48 MB    ~2.9   (what this used to be)
#
# The peak RSS this table used to also report (38-39 GB, barely moving across the three) was measured
# when the whole region was held as GEOS objects at once. It is not the peak any more - see
# read_source_wkb - but the conclusion it supported still holds: the tolerance is not what sets the
# memory, so it is chosen on resolution and file size alone.
#
# The staircase vertices are nearly free - consecutive deltas of +/-1 m on the integer grid
# projection.py snaps to, which zstd collapses - so removing them raises the per-vertex cost faster
# than it lowers the count until the count falls far enough to win again. 20 m is past the hump:
# a third of the vertices AND a smaller file than either finer setting.
TOLERANCE_METERS = 20.0
CHUNK_SIZE = 20_000


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def build_basin_edits(mods_dir: Path) -> tuple[dict, set]:
    """
    Replay, in the same order as 2_simplify_streams.py, the id-level edits that the
    stream simplification recorded in its json side-files, expressed for basins.

    Returns:
        redirect: {original_river_id: river_id it was merged into}. A basin's surviving
                  catchment is found by following redirect to a fixed point.
        deleted:  river_ids whose reach (and basin) were removed outright.
    """
    redirect: dict = {}
    deleted: set = set()

    # 1. lakes: every interior reach (including the inlet->outlet geometry path) collapses
    #    into the lake outlet, so its local catchment area belongs to the outlet's catchment.
    #    inlets and the outlet survive and keep their own basins.
    for outlet, edit in _load_json(mods_dir / 'lake_edits.json').items():
        outlet = int(outlet)
        for d in edit.get('delete', []):
            redirect[int(d)] = outlet

    # 2. zero-length streams: their basins are deleted in every case (1, 2, 3)
    zero_lengths = _load_json(mods_dir / 'zero_length_streams.json')
    for case in ('case1', 'case2', 'case3'):
        deleted.update(int(i) for i in zero_lengths.get(case, {}).get('ids', []))

    # 3. headwater dissolves, 4. branch pruning, 5. short consolidations:
    #    each {keeper: [members]} group folds every non-keeper member's basin into the keeper
    for fname in ('headwater_dissolves.json', 'branches_to_prune.json', 'short_consolidations.json'):
        for keeper, members in _load_json(mods_dir / fname).items():
            keeper = int(keeper)
            for member in members:
                redirect[int(member)] = keeper

    return redirect, deleted


def resolve_keeper(rid: int, redirect: dict) -> int:
    """Follow the redirect chain to the reach that ultimately absorbed rid."""
    seen = set()
    while rid in redirect and rid not in seen:
        seen.add(rid)
        rid = redirect[rid]
    return rid


def union_threaded(members: list, workers: int, groups_per_task: int = 300) -> list:
    """
    Union each list of geometries in ``members``, one geometry out per entry. Runs the per-group
    GEOS unions across worker threads (shapely.union_all drops the GIL during the union).
    """

    def union_chunk(start: int) -> list:
        return [a[0] if len(a) == 1 else shapely.union_all(a)
                for a in members[start:start + groups_per_task]]

    geometries = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(union_chunk, range(0, len(members), groups_per_task)):
            geometries.extend(part)
    return geometries


def source_crs(basins_src: Path):
    """The source's CRS, read from the file's GeoParquet metadata rather than from its geometry.

    Taking it off a real geometry would mean reading some, and the whole point here is not to. Both
    encodings carry it the same way: GeoParquet stores the CRS as PROJJSON under the geometry
    column, and a null means OGC:CRS84 by the spec.
    """
    metadata = pq.ParquetFile(basins_src).schema_arrow.metadata or {}
    geo = json.loads(metadata.get(b'geo', b'{}'))
    column = geo.get('primary_column', hy.schema.geometry)
    crs = geo.get('columns', {}).get(column, {}).get('crs', 'missing')
    if crs == 'missing':
        raise RuntimeError(f'{basins_src.name} carries no GeoParquet CRS metadata')
    return pyproj.CRS.from_json_dict(crs) if crs is not None else pyproj.CRS.from_user_input('OGC:CRS84')


def read_source_wkb(basins_src: Path, wanted: np.ndarray, batch_size: int = 20_000) -> np.ndarray:
    """The source basins' geometry as WKB bytes, indexed by source row, for the rows in ``wanted``.

    **This is the whole memory argument of this step.** A polygon costs ~78 bytes per vertex once
    GEOS has it and ~16 bytes as WKB, so a region held as shapely objects is ~5x what the same
    geometry costs as bytes -- and the peak here used to be holding all of it that way at once.
    Kept as WKB and converted one published-order chunk at a time, region 7020000010 (2.42 GB
    source, 117,948 catchments) goes **40.5 GB -> 19.7 GB peak RSS, 349 s -> 332 s, and the output
    is byte-identical** -- same rows, same order, same geometry bytes.

    What is left is still partly linear in the region: this array is the whole region's WKB (~5.6 GB
    there), and the working set of one chunk is several GEOS copies of that chunk -- raw members,
    the union, the reprojection, the simplified result -- times ``chunk_threads`` of them in flight.
    ``CHUNK_SIZE`` is the dial for the second, but it is not free: the design note measures 20,000
    keeping 7.96% of vertices against 5,000 keeping 10.38%, because every chunk outline is pinned.
    Getting the rest would mean sorting the WKB to published order on disk so it could be read a
    window at a time, which is a bigger change than this one.

    Both source encodings are handled, because whether the tree has been through
    recompress_tdxhydro.py decides which one is on disk: GeoParquet 1.0 stores the geometry as a
    WKB ``binary`` column, which is already the wanted form, while 1.1 stores geoarrow, which has to
    go through shapely to get back to bytes. That conversion is per batch, so it is bounded either
    way.
    """
    wkb = np.empty(len(wanted), dtype=object)
    parquet = pq.ParquetFile(basins_src)
    geoarrow = parquet.schema_arrow.field(hy.schema.geometry).metadata is not None
    row = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=[hy.schema.geometry]):
        stop = row + batch.num_rows
        keep = wanted[row:stop]
        if keep.any():
            column = batch.column(hy.schema.geometry)
            if geoarrow:
                block = shapely.to_wkb(gpd.GeoDataFrame.from_arrow(batch).geometry.values)
            else:
                block = column.to_numpy(zero_copy_only=False)
            wkb[row:stop] = np.where(keep, block, None)
        row = stop
    if row != len(wanted):
        raise RuntimeError(f'{basins_src.name} has {row:,} rows, expected {len(wanted):,}')
    return wkb


def simplify_chunk(geometries: np.ndarray, start: int) -> np.ndarray:
    """One chunk of the coverage, simplified. ``start`` only names the chunk in the error."""
    # simplify_boundary=False is what keeps this chunk's outline identical to its neighbours'
    clean = shapely.coverage_simplify(geometries, TOLERANCE_METERS, simplify_boundary=False)
    empty = int(shapely.is_empty(clean).sum())
    if empty:
        raise RuntimeError(f'chunk starting at row {start:,} simplified {empty} catchment(s) '
                           f'out of existence')
    return clean


def build_leaf_catchments(region_number: int, order: pd.DataFrame) -> gpd.GeoDataFrame:
    """The source basins dissolved onto the surviving reaches, in ``order``'s row order, simplified.

    ``order`` is the region's metadata: its row order IS the published order, and this is the only
    place the catchments are put into it. Everything downstream -- the group split, the basin
    dissolve below, a client reading the nth row of two products -- depends on that.
    """
    outputs_dir = region_root / f'{region_number}'
    mods_dir = outputs_dir / 'mods'

    basins_src = tdx_root / f'TDX_streamreach_basins_{region_number}_01.parquet'
    source = pq.read_table(basins_src, columns=[hy.schema.tdx_link_no_field])
    source_ids = source.column(0).to_numpy().astype(np.int64)
    del source
    logging.info(f'{len(source_ids):,} source basins in {basins_src.name}')

    # replay step 2's edits to map each original basin to the reach that absorbed it
    redirect, deleted = build_basin_edits(mods_dir)
    keeper_map = {rid: resolve_keeper(rid, redirect) for rid in np.unique(source_ids)}
    keeper = pd.Series(source_ids).map(keeper_map).to_numpy()

    river_ids = order[hy.schema.river_id].to_numpy()
    position = pd.Series(np.arange(len(river_ids), dtype=np.int64), index=river_ids)
    place = position.reindex(keeper).to_numpy(dtype=float, copy=True)
    place[np.isin(keeper, list(deleted))] = np.nan  # deleted outright (the zero-length cases)
    wanted = ~np.isnan(place)
    logging.info(f'{int((~wanted).sum()):,} source basins dropped (deleted or not in the metadata)')

    counts = np.bincount(place[wanted].astype(np.int64), minlength=len(river_ids))
    if (counts == 0).any():
        absent = river_ids[counts == 0]
        raise RuntimeError(f'{len(absent):,} reach(es) have no catchment, e.g. {absent[:5].tolist()}')

    # source rows grouped by published row, so a chunk of published rows is a slice of this
    by_place = np.argsort(place[wanted], kind='stable')
    rows = np.flatnonzero(wanted)[by_place]
    bounds = np.r_[0, np.cumsum(counts)]

    started = time.time()
    crs = source_crs(basins_src)
    wkb = read_source_wkb(basins_src, wanted)
    logging.info(f'source geometry held as WKB in {time.time() - started:.0f}s; '
                 f'{len(river_ids):,} catchments to build in chunks of {CHUNK_SIZE:,}')

    # Build the coverage one chunk of the published order at a time. Each chunk is dissolved,
    # reprojected and simplified on its own and then goes back to bytes, so the only geometry held
    # as GEOS objects at any moment is what the chunks in flight are holding. The chunks are
    # contiguous in the published order for the reason the coverage pass needs them to be - see the
    # note above on compactness - and independent of each other for the same reason, which is what
    # lets `chunk_threads` of them run at once.
    out = np.empty(len(river_ids), dtype=object)

    def build_chunk(start: int) -> tuple:
        stop = min(start + CHUNK_SIZE, len(river_ids))
        # one from_wkb for the whole chunk rather than one per catchment, then split on the group
        # bounds. Same arrays out; ~20,000 fewer python-level calls in, and the vertex count for the
        # log comes off the flat array in one call instead of one per group.
        block = rows[bounds[start]:bounds[stop]]
        flat = shapely.from_wkb(wkb[block])
        wkb[block] = None  # released as it is consumed
        before = int(shapely.get_num_coordinates(flat).sum())
        offsets = bounds[start:stop + 1] - bounds[start]
        members = [flat[offsets[i]:offsets[i + 1]] for i in range(stop - start)]
        del flat

        merged = union_threaded(members, dissolve_threads)
        del members
        # The reprojection has to happen before the simplification, whose tolerance is in mercator
        # metres. The 1 m snap does not - see projection.snap_to_grid - and it is four times cheaper
        # here, on a fifth of the vertices, than it was as part of the reprojection: as
        # set_precision on the full-resolution dissolve it was 48% of this step.
        chunk = hy.projection.to_web_mercator(gpd.GeoDataFrame(geometry=merged, crs=crs),
                                              round_meters=None)
        del merged
        clean = simplify_chunk(chunk[hy.schema.geometry].values, start)
        del chunk
        # Rounding moves linework without re-noding the ring, so a few come back self-intersecting -
        # 6 of 23,236 on 5020000010. set_precision used to repair those as part of doing the
        # rounding, and it is still the right tool for them: it is an overlay, so it both rounds and
        # repairs, and it lands on the same lattice. It is only the *whole array* that could not
        # afford it. Handing it the handful that rounding broke costs nothing and keeps every
        # coordinate on the grid, which plain repair does not - make_valid nodes the ring and the
        # intersection it computes is wherever the segments actually cross.
        snapped = hy.projection.snap_to_grid(clean)
        broken = ~shapely.is_valid(snapped)
        if broken.any():
            snapped[broken] = shapely.set_precision(clean[broken],
                                                    hy.projection.precision_meters)
        # fallback is the rounded geometry, never the unrounded one: a row that even set_precision
        # cannot fix keeps integer coordinates rather than reintroducing off-lattice ones.
        clean, held = hy.geometry.repair(snapped, fallback=snapped)
        del snapped
        after = int(shapely.get_num_coordinates(clean).sum())
        out[start:stop] = shapely.to_wkb(clean)
        return before, after, held

    starts = range(0, len(river_ids), CHUNK_SIZE)
    with ThreadPoolExecutor(max_workers=chunk_threads) as pool:
        counted = list(pool.map(build_chunk, starts))
    before = sum(c[0] for c in counted)
    after = sum(c[1] for c in counted)
    held = sum(c[2] for c in counted)

    logging.info(f'coverage pass at {TOLERANCE_METERS:g} m in chunks of {CHUNK_SIZE:,}, '
                 f'{chunk_threads} at a time: '
                 f'{before:,} -> {after:,} vertices ({100 * after / before:.2f}%), '
                 f'{time.time() - started:.0f}s')
    if held:
        logging.info(f'{held:,} catchment(s) kept their unsnapped geometry: rounding onto the '
                     f'{hy.projection.precision_meters:g} m grid broke them and the repair '
                     f'left no area')

    catchments = gpd.GeoDataFrame(
        {hy.schema.river_id: river_ids},
        geometry=shapely.from_wkb(out),
        crs=f'EPSG:{hy.projection.web_mercator_epsg}',
    )
    # riverIndex rides along so a leaf catchment carries the same id *and* index a reach does, which
    # is what lets one selector address the streams and every catchment layer alike
    catchments.insert(1, hy.schema.river_index, order[hy.schema.river_index].to_numpy())
    return hy.schema.enforce_int32(catchments)


if __name__ == '__main__':
    # find the ID of the region to process
    args = sys.argv[1:]
    if len(args) != 1:
        sys.exit('usage: 4_create_catchments.py <region_number>')
    region_number = int(args[0])
    # region_number = 1020000010  # Example region number

    outputs_dir = region_root / f'{region_number}'
    mods_dir = outputs_dir / 'mods'

    catchments_output = outputs_dir / f'catchments_{region_number}.geo.parquet'
    if catchments_output.exists():
        print(f'Catchments output {catchments_output} already exists, skipping region {region_number}')
        # os._exit, not sys.exit: pyarrow's thread pool destructor can hang at interpreter exit, and
        # this step runs under xargs, where one wedged process holds its slot and stalls the run.
        # See 5_generate_groups.py, where it happened. The flush is because print buffers to a pipe.
        sys.stdout.flush()
        os._exit(0)

    # prepare directories and logging
    mods_dir.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logs_root / f'create_catchments_{region_number}.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    metadata = pd.read_parquet(
        outputs_dir / f'metadata_{region_number}.parquet',
        columns=[hy.schema.river_id, hy.schema.river_index],
    )
    logging.info(f'{len(metadata):,} reaches in the published order')

    catchments = build_leaf_catchments(region_number, metadata)
    hy.parquet.write_geoparquet(catchments, catchments_output)
    logging.info(f'Catchments written to {catchments_output}')
    print(f'region {region_number}: {len(catchments):,} catchments -> {catchments_output}')
