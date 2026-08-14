import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical stream attribute columns (renamed or derived in this pipeline)
# ---------------------------------------------------------------------------
river_id = 'riverId'
river_index = 'riverIndex'
upstream_count = 'upstreamCount'
next_river_id = 'nextRiverId'
last_river_id = 'outletRiverId'
group_id = 'groupId'
topo_sort = 'topologySortedOrder'
strahler_order = 'strahlerOrder'
shreve_order = 'shreveOrder'
length = 'lengthMeters'
area = 'areaM2'
geometry = 'geometry'

# ---------------------------------------------------------------------------
# Original TDX-Hydro columns as they appear in the source gpkg / parquet.
# DSContArea, USContArea and Magnitude keep their source names, so they are
# both the original and the canonical column.
# ---------------------------------------------------------------------------
tdx_link_field = 'LINKNO'
tdx_ds_link_field = 'DSLINKNO'
tdx_us_link_1_field = 'USLINKNO1'
tdx_us_link_2_field = 'USLINKNO2'
tdx_ds_node_id_field = 'DSNODEID'
tdx_strm_order_field = 'strmOrder'
tdx_length_field = 'Length'
tdx_magnitude_field = 'Magnitude'  # this is the shreve order
tdx_ds_area_field = 'DSContArea'  # downstream contributing area
tdx_us_area_field = 'USContArea'  # upstream contributing area
tdx_strm_drop_field = 'strmDrop'
tdx_slope_field = 'Slope'
tdx_straight_length_field = 'StraightL'
tdx_ws_no_field = 'WSNO'
tdx_dout_end_field = 'DOUTEND'
tdx_dout_start_field = 'DOUTSTART'
tdx_dout_mid_field = 'DOUTMID'
# added by the tdx standardizer step
lon_field = 'lon'
lat_field = 'lat'
z_field = 'z'
tdx_geodesic_length_field = 'LengthGeodesicMeters'
tdx_region_field = 'TDXHydroRegion'
tdx_link_no_field = 'TDXHydroLinkNo'
basin_stream_id_field = 'streamID'

# ---------------------------------------------------------------------------
# Muskingum routing parameters (computed after the network is simplified)
# ---------------------------------------------------------------------------
static_velocity_factor = 'velocity_factor'
static_musk_k = 'musk_k'
static_musk_x = 'musk_x'

# ---------------------------------------------------------------------------
# Original -> canonical renames applied when ingesting the TDX streamnet
# ---------------------------------------------------------------------------
rename_map = {
    tdx_strm_order_field: strahler_order,
    tdx_magnitude_field: shreve_order,
    tdx_geodesic_length_field: length,
}

# columns kept from the source TDX streamnet when translating to parquet. Length stays because
# the published metadata carries it (final_columns_to_keep) - the planar TauDEM length beside the
# geodesic one this pipeline computes.
tdx_standardized_columns = [
    tdx_link_field,
    tdx_ds_link_field,
    tdx_strm_order_field,
    tdx_magnitude_field,
    tdx_us_area_field,
    tdx_ds_area_field,
    tdx_length_field,
    tdx_geodesic_length_field,
    tdx_region_field,
    lon_field,
    lat_field,
    geometry,
]

final_columns_to_keep = [
    river_id,
    next_river_id,
    last_river_id,
    # riverIndex and upstreamCount sit here, directly after the ids, so "the columns needed to
    # walk the network" read as one contiguous run of column chunks. Both come out of step 3's
    # own traversal. upstreamCount is region-local physics and identical everywhere. riverIndex
    # is SCOPED: region files carry the region-local position (0..n-1 within the region), and the
    # group-partitioned published files carry the globally unique position - step 5 re-values the
    # column with pure offset arithmetic while splitting, because a group is one contiguous run
    # of the region-local ordering and the global ordering is just the groups concatenated in
    # ascending groupId. See docs/river-index.md.
    river_index,
    upstream_count,
    strahler_order,
    shreve_order,
    tdx_us_area_field,
    tdx_ds_area_field,
    area,
    tdx_length_field,
    tdx_region_field,
    group_id,
    static_musk_k,
    static_musk_x,
    static_velocity_factor,
    geometry,
]

metadata_columns_to_keep = [c for c in final_columns_to_keep if c != geometry] + [lat_field, lon_field]

# ---------------------------------------------------------------------------
# lake_table.csv controlled vocabulary
# ---------------------------------------------------------------------------
inlet_field = 'inlet'
outlet_field = 'outlet'
lake_id_field = 'lake_id'
endorheic_field = 'endorheic'
trace_inlet_field = 'trace_inlet'


# ---------------------------------------------------------------------------
# Output dtypes
# ---------------------------------------------------------------------------
# Ids and indices are written as int32, not int64. Two reasons, both practical:
#
#  - JavaScript. A parquet int64 column decodes to BigInt in the browser, and BigInt does not
#    compare or hash equal to Number. A client that builds `Map` keys from one and looks them up
#    with the other gets no match and no error — the graph builds, every traversal returns only the
#    reach it started from, and nothing anywhere says why. int32 decodes to Number and the whole
#    class of bug disappears.
#  - It is what the v3 zarr stores already use for `riverId`, so the two agree.
#
# Every value fits with room to spare: the largest riverId is 820,422,448 against int32's
# 2,147,483,647, riverIndex tops out at the reach count, and groupId is three digits. enforce_int32
# checks rather than assumes, so a future id scheme that outgrows the range fails the build instead
# of silently wrapping into negative ids.
int32_columns = (
    river_id,
    next_river_id,
    last_river_id,
    river_index,
    upstream_count,
    group_id,
    strahler_order,
    shreve_order,
)


def enforce_int32(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast the id, index, group and order columns to int32 in place, refusing to wrap."""
    limits = np.iinfo(np.int32)
    for column in int32_columns:
        if column not in df.columns:
            continue
        values = df[column]
        if values.isna().any():
            raise ValueError(f'{column} has {int(values.isna().sum()):,} null(s); cannot be int32')
        low, high = int(values.min()), int(values.max())
        if low < limits.min or high > limits.max:
            raise ValueError(
                f'{column} ranges {low:,}..{high:,}, outside int32 '
                f'({limits.min:,}..{limits.max:,}). Widen the dtype rather than let it wrap.'
            )
        df[column] = values.astype('int32')
    return df
