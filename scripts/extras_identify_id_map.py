"""
Build a lookup table mapping every original TDX-Hydro reach to its id in the
simplified GEOGLOWS v3 stream set.

Two columns, one row per original TDX-Hydro reach (all ~16M across every region,
including the regions this suite does not process):

    TDXHydroLinkNo : int64  - the original reach id (LINKNO + region spacer, the
                              globally-unique id this pipeline assigns; identical
                              to the source `TDXHydroLinkNo` column)
    v3RiverId      : Int64  - the id of the surviving v3 reach that now represents
                              this reach, or <NA> if the reach is not in v3

An original reach lands in one of three buckets:
  * survives unchanged            -> v3RiverId == TDXHydroLinkNo
  * merged/consolidated into a    -> v3RiverId == the surviving keeper's id
    surviving keeper                 (following the chain of edits transitively)
  * dropped, or in an unprocessed -> v3RiverId is <NA>
    region

The keeper mapping is reconstructed by replaying the edits recorded in each
processed region's `mods/` directory - see hydrography/edits.py, which owns that
replay and is also what resolves the ids in network_data/river_names.csv. Reaches
removed outright (dropped watersheds, sub-250 km^2 outlets, zero-length reaches)
are recorded nowhere as a keeper, so they correctly fall through to <NA>.

Run after 5_concatenate_global.py (needs hydrography/global/metadata.parquet under the data root).
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hydrography as hy

# Must match the earlier steps or else it will revert the work done there

# every output path hangs off the data root - see hydrography/paths.py and $RFS_DATA_ROOT
tdx_root = hy.paths.tdx_root
region_root = hy.paths.region_root
global_root = hy.paths.global_root
network_data_root = hy.paths.network_data_root
logs_root = hy.paths.logs_root

# canonical globally-unique original id column in the raw region parquet
orig_id_col = hy.schema.tdx_link_no_field  # 'TDXHydroLinkNo'
v3_id_col = 'v3RiverId'

def read_original_ids(region_file: Path) -> np.ndarray:
    """All original reach ids in a raw region parquet, as canonical global ids: TDXHydroLinkNo on
    the older converted tree, the already-stamped LINKNO on files step 1 writes now (step 1 owns
    the header arithmetic; nothing here re-derives it)."""
    try:
        return pd.read_parquet(region_file, columns=[orig_id_col])[orig_id_col].to_numpy()
    except (KeyError, ValueError):
        raw = pd.read_parquet(region_file, columns=[hy.schema.tdx_link_field])
        return raw[hy.schema.tdx_link_field].to_numpy().astype(np.int64)


if __name__ == '__main__':
    logs_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logs_root / 'tdxhydro_to_v3_id_map.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    lookup_out = global_root / 'tdxhydro_to_v3_id_map.parquet'

    metadata_path = global_root / 'metadata.parquet'
    if not metadata_path.exists():
        sys.exit(f'{metadata_path} not found - run 5_concatenate_global.py first')

    # v3 survivors: the reaches that exist in the final global stream set
    survivors = pd.read_parquet(metadata_path, columns=[hy.schema.river_id])[hy.schema.river_id].to_numpy()
    survivor_set = set(survivors.tolist())
    logging.info(f'{len(survivors):,} surviving v3 reaches')

    # reconstruct the original -> keeper mapping from every processed region's edits
    mods_dirs = hy.edits.mods_dirs(region_root)
    logging.info(f'Replaying edits from {len(mods_dirs)} processed regions')
    terminal_of = hy.edits.resolve_terminals(hy.edits.member_to_keeper(mods_dirs))

    # a member is only usable if its terminal keeper actually survives in v3; a
    # terminal that is missing means the keeper was itself dropped downstream, so
    # the member has no v3 representation (rare; reported below)
    members = np.fromiter(terminal_of.keys(), dtype=np.int64, count=len(terminal_of))
    terminals = np.fromiter(terminal_of.values(), dtype=np.int64, count=len(terminal_of))
    terminal_survives = np.isin(terminals, survivors)
    member_final_map = dict(zip(members[terminal_survives].tolist(), terminals[terminal_survives].tolist()))
    n_anomalies = int((~terminal_survives).sum())
    if n_anomalies:
        logging.warning(f'{n_anomalies} merged reaches resolve to a keeper that is absent from v3 (mapped to <NA>)')

    # every original reach across all regions, incl. those never processed
    raw_files = natsorted(list(tdx_root.glob('TDX_streamnet_*_01.parquet')), key=str)
    logging.info(f'Reading original ids from {len(raw_files)} raw region files')
    region_row_counts: dict[str, int] = {}
    orig_chunks = []
    for f in raw_files:
        ids = read_original_ids(f)
        orig_chunks.append(ids)
        region_row_counts[f.name.split('_')[2]] = len(ids)
    orig = np.concatenate(orig_chunks)

    if len(orig) != len(np.unique(orig)):
        raise RuntimeError('Original TDXHydroLinkNo ids are not globally unique; cannot build a lookup')

    # assemble the mapping:
    #   survivor        -> itself
    #   merged member   -> its surviving keeper
    #   everything else -> <NA>
    orig_series = pd.Series(orig)
    mapped = orig_series.map(member_final_map).to_numpy(dtype='float64')  # NaN where not a mapped member
    is_survivor = np.isin(orig, survivors)
    v3 = np.where(is_survivor, orig.astype('float64'), mapped)

    out = pd.DataFrame({orig_id_col: orig})
    out[v3_id_col] = pd.Series(v3).astype('Int64')  # ids < 2**53, so exact; NaN -> <NA>

    global_root.mkdir(parents=True, exist_ok=True)
    hy.parquet.write_parquet(out, lookup_out, index=False)

    # ---- summary / sanity checks ----------------------------------------------
    unprocessed = [r for r in region_row_counts if not (region_root / r / 'mods').is_dir()]
    n_self = int(is_survivor.sum())
    n_mapped = int(out[v3_id_col].notna().sum())
    n_merged = n_mapped - n_self
    n_null = int(out[v3_id_col].isna().sum())
    n_unprocessed_rows = sum(region_row_counts[r] for r in unprocessed)
    n_distinct_v3 = int(out[v3_id_col].dropna().nunique())

    # every survivor id is itself an original id, so it must appear exactly once as
    # a self-map, and the distinct v3 ids must be exactly the survivor set
    assert n_self == len(survivor_set), f'{n_self} self-maps != {len(survivor_set)} survivors'
    assert n_distinct_v3 == len(survivor_set), f'{n_distinct_v3} distinct v3 ids != {len(survivor_set)} survivors'

    summary = (
        f'Wrote {len(out):,} rows to {lookup_out}\n'
        f'  survive unchanged : {n_self:,}\n'
        f'  merged into keeper : {n_merged:,}\n'
        f'  no v3 reach (<NA>) : {n_null:,}\n'
        f'    of which in {len(unprocessed)} unprocessed regions : {n_unprocessed_rows:,}\n'
        f'    dropped in processed regions            : {n_null - n_unprocessed_rows:,}\n'
        f'  merged reaches whose keeper is absent from v3 : {n_anomalies:,}\n'
        f'  distinct v3 ids referenced : {n_distinct_v3:,} (== {len(survivor_set):,} survivors)'
    )
    logging.info(summary)
    print(summary)
    print(f'Unprocessed regions ({len(unprocessed)}): {", ".join(natsorted(unprocessed))}')
