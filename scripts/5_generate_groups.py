import logging
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

# These splits are what clients actually download, so how they are written matters more here than
# anywhere else in the pipeline — see hydrography/parquet.py, which every step shares so that a
# split does not re-encode what the step that produced it wrote.
#
# Only the tables whose rows carry a whole reach's geometry, ~5.6 KB each, need the small row
# groups: a row group is the smallest thing a reader can fetch, so a large one forces a client
# wanting a few hundred reaches to decompress hundreds of MB. Confluences are a single point per
# row, ~29 bytes — a whole file is about a megabyte, so there is nothing to subset out of and small
# groups would only add footer. Metadata is likewise light and stays on the default.
LARGE_GEOMETRY_KINDS = {'streams', 'catchments'}
# Threads for the group boundary dissolve, on the fallback path only. The driver fans this step out
# at -P 5, so taking every core in each of five processes would put 5x the machine's threads on the
# same work; a quarter each keeps the total near the core count. shapely releases the GIL during
# the union, so these are real.
DISSOLVE_THREADS = max(1, (os.cpu_count() or 8) // 4)
# every product here carries geometry except metadata, the attribute table
GEOMETRY_KINDS = LARGE_GEOMETRY_KINDS | {'confluences'}

# Everything split here carries geometry except metadata, the attribute table.
SUFFIXES = {'metadata': '.parquet'}


def out_name(kind: str, group_id: int) -> str:
    return f'{kind}_{group_id}{SUFFIXES.get(kind, ".geo.parquet")}'


def drop_interior_rings(outline):
    """Only the outermost boundary of each piece of the outline.

    The dissolve leaves holes: an endorheic sink or a lake drains nowhere, so no reach claims it and
    no catchment covers it, and the union closes around the gap. Those interior rings are not what
    this file is for - it answers "which group is this place in" and draws the group on a map, and a
    hole makes both answers wrong over the hole. Each piece is rebuilt from its exterior ring alone.

    The union afterwards is not cosmetic: a piece that sat inside a hole is now covered by the shell
    that used to surround it, and the two have to become one region rather than a self-overlapping
    multipolygon that no downstream consumer would accept as valid.
    """
    shells = shapely.polygons(shapely.get_exterior_ring(shapely.get_parts(outline)))
    return shapely.union_all(shells)


def union_catchments(geometries):
    """The outline of a group's catchments, as one polygon.

    ``coverage_union_all`` is the fast path: shared edges cancel and the union is one pass rather
    than a general overlay. It needs an edge-matched coverage, though, and step 4's output is not
    one - re-measured on 7020000010, 7,999 of 8,000 catchments carry a mismatched shared edge at
    every simplification tolerance tried, so GEOS refuses the set. The fallback is therefore the
    normal case here rather than the exception; it gives the same answer, and measured on group 718
    it is 10.9 s against 36.5 s for a plain union_all. See hydrography/geometry.py.
    """
    try:
        outline = shapely.make_valid(shapely.coverage_union_all(geometries))
    except shapely.errors.GEOSException:
        outline = hy.geometry.hierarchical_union(geometries, workers=DISSOLVE_THREADS)
    if outline is None or outline.is_empty:
        return None
    if outline.geom_type not in ('Polygon', 'MultiPolygon'):
        # degenerate input can leave stray lines or points in the union; keep only the areas
        outline = shapely.union_all([g for g in shapely.get_parts(outline)
                                     if g.geom_type in ('Polygon', 'MultiPolygon')])
        if outline.is_empty:
            return None
    outline = drop_interior_rings(outline)
    return None if outline.is_empty else outline


def sort_by_river_index(gdf, index_by_river):
    """
    Put a table that has no riverIndex of its own into riverIndex order anyway, via the reach each
    row belongs to. Confluences and catchments are one row per reach but do not carry the column, and
    the point is that every product in a group directory can be read row-for-row against the others.
    """
    sort_key = '_riverIndexSortKey'
    gdf = gdf.assign(**{sort_key: gdf[hy.schema.river_id].map(index_by_river)})
    return gdf.sort_values(sort_key, kind='stable').drop(columns=[sort_key]).reset_index(drop=True)


# every output path hangs off the data root - see hydrography/paths.py and $RFS_DATA_ROOT
region_root = hy.paths.region_root
group_root = hy.paths.group_root
logs_root = hy.paths.logs_root

if __name__ == '__main__':
    # find the ID of the region to process
    force = '--force' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--force']
    if len(args) != 1:
        sys.exit('usage: 5_generate_groups.py <region> [--force]')
    region_number = int(args[0])
    # region = 1020000010  # Example region number

    outputs_dir = region_root / f'{region_number}'
    streams_src = outputs_dir / f'streams_{region_number}.geo.parquet'
    confluences_src = outputs_dir / f'confluences_{region_number}.geo.parquet'
    catchments_src = outputs_dir / f'catchments_{region_number}.geo.parquet'
    metadata_src = outputs_dir / f'metadata_{region_number}.parquet'

    # fast path: if every per-group output already exists, skip before reading any geometry.
    # the datasets that get split are fixed by which region inputs are present, and the group
    # ids come from just the groupId column of the streams file (a cheap, geometry-free read).
    kinds = ['streams', 'confluences']
    if metadata_src.exists():
        kinds.append('metadata')
    if catchments_src.exists():
        # the boundary is derived from the catchments, so it is expected exactly when they are
        kinds.extend(['catchments', 'boundary'])
    # --force exists so that a change to the row ORDER can be republished. The skip below only knows
    # whether the files exist, and a reordering leaves every filename exactly where it was.
    if not force and streams_src.exists() and hy.schema.group_id in pq.read_schema(streams_src).names:
        existing_groups = pd.read_parquet(streams_src, columns=[hy.schema.group_id])[hy.schema.group_id]
        existing_groups = sorted(existing_groups.dropna().astype(int).unique().tolist())
        expected_outputs = [
            hy.paths.group_dir(g) / out_name(kind, g)
            for g in existing_groups for kind in kinds
        ]
        if expected_outputs and all(p.exists() for p in expected_outputs):
            print(f'All {len(expected_outputs)} group outputs for region {region_number} already exist, skipping')
            sys.exit(0)

    # prepare directories and logging
    logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logs_root / f'generate_groups_{region_number}.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    streams_gdf = gpd.read_parquet(streams_src)
    confluences_gdf = gpd.read_parquet(confluences_src)
    logging.info(f'Read {len(streams_gdf):,} reaches and {len(confluences_gdf):,} confluences')

    # use the groupId assigned previously in step 2
    if hy.schema.group_id not in streams_gdf.columns:
        sys.exit(f'{streams_src} has no {hy.schema.group_id} column; rerun step 2 to assign groupIds')
    missing = streams_gdf[hy.schema.group_id].isna()
    if missing.any():
        unmatched = streams_gdf.loc[missing, hy.schema.last_river_id].unique()
        raise ValueError(
            f'{int(missing.sum())} reach(es) in region {region_number} have no {hy.schema.group_id}; '
            f'e.g. outlets {unmatched[:10].tolist()}'
        )
    streams_gdf[hy.schema.group_id] = streams_gdf[hy.schema.group_id].astype('int32')

    # Every product here is one row per reach (confluences per junction), so they are all put in the
    # same order — the global riverIndex order step 3 assigned. Reading the nth row of any of them
    # then means the same reach without a join, and a group's streams end up a dense array whose
    # position is riverIndex - riverIndexStart. Only streams and metadata carry riverIndex; the
    # others are sorted by the riverIndex of the reach they belong to without gaining the column.
    streams_gdf = streams_gdf.sort_values(hy.schema.river_index).reset_index(drop=True)
    index_by_river = streams_gdf.set_index(hy.schema.river_id)[hy.schema.river_index]
    group_by_river = streams_gdf.set_index(hy.schema.river_id)[hy.schema.group_id]
    confluences_gdf[hy.schema.group_id] = confluences_gdf[hy.schema.river_id].map(group_by_river)
    dropped = int(confluences_gdf[hy.schema.group_id].isna().sum())
    confluences_gdf = confluences_gdf.dropna(subset=[hy.schema.group_id])
    confluences_gdf[hy.schema.group_id] = confluences_gdf[hy.schema.group_id].astype('int32')
    confluences_gdf = sort_by_river_index(confluences_gdf, index_by_river)
    if dropped:
        logging.info(f'Dropped {dropped} confluence row(s) with no groupId (e.g. the -1 outlet aggregate)')

    # streams and confluences are always split; metadata and catchments are split too if they exist
    datasets = {
        'streams': streams_gdf,
        'confluences': confluences_gdf,
    }

    if metadata_src.exists():
        metadata_df = pd.read_parquet(metadata_src)
        metadata_df[hy.schema.group_id] = metadata_df[hy.schema.group_id].astype('int32')
        metadata_df = metadata_df.sort_values(hy.schema.river_index).reset_index(drop=True)
        datasets['metadata'] = metadata_df
        logging.info(f'Read {len(metadata_df):,} metadata rows from {metadata_src}')
    else:
        logging.info(f'No metadata at {metadata_src}; skipping that split')

    # catchments carry no groupId; each catchment's id is its reach id, so it inherits the reach's groupId
    if catchments_src.exists():
        catchments_gdf = gpd.read_parquet(catchments_src)
        catchments_gdf[hy.schema.group_id] = catchments_gdf[hy.schema.river_id].map(group_by_river)
        dropped = int(catchments_gdf[hy.schema.group_id].isna().sum())
        catchments_gdf = catchments_gdf.dropna(subset=[hy.schema.group_id])
        catchments_gdf[hy.schema.group_id] = catchments_gdf[hy.schema.group_id].astype('int32')
        catchments_gdf = sort_by_river_index(catchments_gdf, index_by_river)
        datasets['catchments'] = catchments_gdf
        logging.info(f'Read {len(catchments_gdf):,} catchments from {catchments_src}')
        if dropped:
            logging.info(f'Dropped {dropped} catchment(s) whose reach has no groupId')
    else:
        logging.info(f'No catchments at {catchments_src}; skipping that split')

    # split every dataset into one parquet each per group
    group_ids = sorted(streams_gdf[hy.schema.group_id].unique().tolist())
    group_root.mkdir(parents=True, exist_ok=True)
    datasets_by_group = {
        kind: dict(tuple(gdf.groupby(hy.schema.group_id)))
        for kind, gdf in datasets.items()
    }
    for group_id in group_ids:
        out_dir = hy.paths.group_dir(group_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        counts = {}
        for kind, gdf in datasets.items():
            # a group with no rows for a dataset (e.g. no confluences) still gets an empty file for consistency
            part = datasets_by_group[kind].get(group_id, gdf.iloc[0:0])
            part = part.drop(columns=[hy.schema.group_id])
            # A group is a whole set of watersheds ordered together, so its reaches have to be one
            # unbroken run of riverIndex. That is what makes riverIndex - riverIndexStart a valid
            # array position within this file: the group's first riverIndex is row 0 of every
            # product here, so a client that read one row knows the offset for all of them.
            if hy.schema.river_index in part.columns and len(part):
                index_values = part[hy.schema.river_index].to_numpy()
                if not (np.diff(index_values) == 1).all():
                    raise ValueError(
                        f'{kind} for group {group_id} spans {int((np.diff(index_values) != 1).sum()) + 1} '
                        f'runs of riverIndex rather than one. The ordering from step 3 is not group-major.'
                    )
            out_path = out_dir / out_name(kind, group_id)
            if kind in GEOMETRY_KINDS:
                row_group_size = hy.parquet.GEOMETRY_ROW_GROUP_SIZE if kind in LARGE_GEOMETRY_KINDS else None
                hy.parquet.write_geoparquet(part, out_path, row_group_size=row_group_size)
            else:
                hy.parquet.write_parquet(part, out_path)
            counts[kind] = len(part)

        # The exact outline of the group: the union of the catchments just written. It is done here
        # rather than in a later step because the polygons are already in memory - a step that came
        # back for them would re-read the largest geometry in the dataset to produce 1 row. Step 6
        # concatenates these 125 one-row files into group=0/groups.geo.parquet.
        catchment_part = datasets_by_group.get('catchments', {}).get(group_id)
        if catchment_part is not None and len(catchment_part):
            started = time.time()
            outline = union_catchments(catchment_part[hy.schema.geometry].values)
            if outline is None:
                logging.warning(f'group {group_id}: {len(catchment_part):,} catchments dissolved to nothing')
            else:
                boundary = gpd.GeoDataFrame(geometry=[outline], crs=catchment_part.crs)
                hy.parquet.write_geoparquet(boundary, out_dir / out_name('boundary', group_id),
                                            row_group_size=None)
                counts['boundary'] = 1
                logging.info(f'group {group_id}: boundary dissolved from {len(catchment_part):,} '
                             f'catchments to {int(shapely.get_num_coordinates(outline)):,} vertices '
                             f'in {time.time() - started:.1f}s')

        msg = (f'region {region_number} -> group {group_id}: '
               + ', '.join(f'{n:,} {kind}' for kind, n in counts.items())
               + f' -> {out_dir}')
        logging.info(msg)
        print(msg)
