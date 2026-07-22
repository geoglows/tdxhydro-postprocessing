# ---------------------------------------------------------------------------
# Canonical stream attribute columns (renamed or derived in this pipeline)
# ---------------------------------------------------------------------------
river_id = 'riverId'
river_index = 'riverIndex'
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

# columns kept from the source TDX streamnet when translating to parquet
tdx_standardized_columns = [
    tdx_link_field,
    tdx_ds_link_field,
    tdx_strm_order_field,
    tdx_magnitude_field,
    tdx_us_area_field,
    tdx_ds_area_field,
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
    # river_index is added after all regions are processed
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
