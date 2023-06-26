import glob
import json
import logging
import os
import types

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import Point, MultiPoint
from sklearn.cluster import KMeans

from .network import correct_0_length_streams
from .network import create_directed_graphs
from .network import find_branches_to_dissolve
from .network import find_branches_to_prune
from .network import identify_0_length
from .network import sort_topologically

# set up logging
logger = logging.getLogger(__name__)

__all__ = [
    'rapid_master_files',
    'dissolve_branches',
    'assign_vpu_by_kmeans',
    'make_quick_visuals',
    'fix_vpus',
    'rapid_input_csvs',
    'rapid_csvs_final',
]


def _calculate_geodesic_length(line) -> float:
    """
    Input is shapely geometry, should be all shapely LineString objects

    returns length in meters
    """
    length = Geod(ellps='WGS84').geometry_length(line)

    # This is for the outliers that have 0 length
    if length < 0.0000001:
        length = 0.01
    return length


def rapid_master_files(streams_gpq: str,
                       save_dir: str,
                       region_number: int,
                       id_field: str = 'LINKNO',
                       ds_id_field: str = 'DSLINKNO',
                       length_field: str = 'Length',
                       default_k: float = 0.35,
                       default_x: float = .3, ) -> None:
    """
    Create RAPID master files from a stream network

    Saves the following files to the save_dir:
        - rapid_inputs_master.parquet
    Args:
        streams_gpq:
        save_dir:
        region_number:
        id_field:
        ds_id_field:
        length_field:
        default_k:
        default_x:

    Returns:

    """
    sgdf = gpd.read_parquet(streams_gpq)

    logger.info('\tEnforcing data types')
    sgdf['LINKNO'] = sgdf['LINKNO'].astype(int)
    sgdf['DSLINKNO'] = sgdf['DSLINKNO'].astype(int)
    sgdf['USLINKNO1'] = sgdf['USLINKNO1'].astype(int)
    sgdf['USLINKNO2'] = sgdf['USLINKNO2'].astype(int)
    sgdf['strmOrder'] = sgdf['strmOrder'].astype(int)
    sgdf['Length'] = sgdf['Length'].astype(float)
    sgdf['lat'] = sgdf['lat'].astype(float)
    sgdf['lon'] = sgdf['lon'].astype(float)
    sgdf['z'] = sgdf['z'].astype(int)

    # length is in m, convert to km, then multiply by 3600 to get km/hr, then multiply by default k (velocity)
    logger.info('\tCalculating Muskingum k and x')
    sgdf["musk_k"] = sgdf['geometry'].apply(_calculate_geodesic_length) / 1000 * 3600 * default_k
    sgdf["musk_kfac"] = sgdf["musk_k"].values.flatten()
    sgdf["musk_x"] = default_x
    sgdf["musk_xfac"] = default_x
    sgdf['musk_k'] = sgdf['musk_k'].round(3)
    sgdf['musk_kfac'] = sgdf['musk_kfac'].round(3)
    sgdf['musk_x'] = sgdf['musk_x'].round(3)
    sgdf['musk_xfac'] = sgdf['musk_xfac'].round(3)

    # add globally unique ID numbers
    with open(os.path.join(os.path.dirname(__file__), 'network_data', 'tdx_header_numbers.json')) as f:
        tdx_header_numbers = json.load(f)
    sgdf['TDXHydroNumber'] = region_number
    sgdf['TDXHydroHeaderNumber'] = int(tdx_header_numbers[str(region_number)])
    sgdf['TDXHydroLeadingDigit'] = str(region_number)[0]
    sgdf['TDXHydroLinkNo'] = sgdf['TDXHydroHeaderNumber'] * 10_000_000 + sgdf['LINKNO']

    logger.info('\tLooking for 0 length segments')
    if 0 in sgdf[length_field].values:
        zero_length_fixes_df = identify_0_length(sgdf, id_field, ds_id_field, length_field)
        zero_length_fixes_df.to_csv(os.path.join(save_dir, 'mod_zero_length_streams.csv'), index=False)
        sgdf = correct_0_length_streams(sgdf, zero_length_fixes_df, id_field)

    # Fix basins with ID of 0
    if 0 in sgdf[id_field].values:
        logger.info('\tFixing basins with ID of 0')
        pd.DataFrame({
            id_field: [0, ],
            'centroid_x': sgdf[sgdf[id_field] == 0].centroid.x.values[0],
            'centroid_y': sgdf[sgdf[id_field] == 0].centroid.y.values[0]
        }).to_csv(os.path.join(save_dir, 'mod_basin_zero_centroid.csv'), index=False)

    logger.info('\tCreating Directed Graph')
    G = create_directed_graphs(sgdf, id_field, ds_id_field=ds_id_field)
    sorted_order = sort_topologically(G)
    sgdf = (
        sgdf
        .set_index(pd.Index(sgdf[id_field]))
        .reindex(sorted_order)
        .reset_index(drop=True)
        .set_index(pd.Index(range(1, len(sgdf) + 1)).rename('TopologicalOrder'))
        .reset_index()
        .dropna()
        .astype(sgdf.dtypes.to_dict())
    )

    # Drop trees with small total length/area
    logger.info('\tFinding and removing small trees')
    small_tree_outlet_ids = sgdf.loc[np.logical_and(
        sgdf[ds_id_field] == -1,
        sgdf['DSContArea'] < 100_000_000
    ), id_field].values
    small_tree_segments = [nx.ancestors(G, x) for x in small_tree_outlet_ids]
    small_tree_segments = set().union(*small_tree_segments).union(small_tree_outlet_ids)
    (
        pd
        .DataFrame(small_tree_segments, columns=['drop'])
        .to_csv(os.path.join(save_dir, 'mod_drop_small_trees.csv'), index=False)
    )
    sgdf = sgdf.loc[~sgdf[id_field].isin(small_tree_segments)]

    logger.info('\tCalculating RAPID connect columns')
    us_ids = sgdf[id_field].apply(lambda x: list(G.predecessors(x)))
    count_us = us_ids.apply(lambda x: len(x))
    max_num_upstream = np.max(count_us)
    us_ids = us_ids.apply(lambda x: x + [-1, ] * (max_num_upstream - len(x)))
    us_ids = pd.DataFrame(us_ids.tolist(), index=us_ids.index)
    upstream_columns = [f'USLINKNO{i}' for i in range(1, us_ids.shape[1] + 1)]
    us_ids.columns = upstream_columns
    us_ids['CountUS'] = count_us
    sgdf = sgdf.drop(columns=[x for x in sgdf.columns if x.startswith('USLINKNO')]).join(us_ids)
    sgdf[upstream_columns] = sgdf[upstream_columns].fillna(-1).astype(int)

    logger.info('\tFinding headwater streams to dissolve')
    branches_1 = find_branches_to_dissolve(sgdf, G, 2)
    sgdf = dissolve_branches(sgdf, branches_1, k_agg_func=_k_agg_order_2)
    branches_2 = find_branches_to_dissolve(sgdf, G, 3)
    sgdf = dissolve_branches(sgdf, branches_2, k_agg_func=_k_agg_order_3)
    (
        pd
        .concat([branches_1, branches_2])
        .fillna(-1)
        .astype(int)
        .to_csv(os.path.join(save_dir, 'mod_dissolve_headwater.csv'), index=False)
    )

    logger.info('\tFinding branches to prune')
    streams_to_prune = find_branches_to_prune(sgdf, G)
    sgdf = prune_branches(sgdf, streams_to_prune)
    streams_to_prune.to_csv(os.path.join(save_dir, 'mod_prune_streams.csv'), index=False)

    logger.info('\tLabeling watersheds by terminal node')
    for term_node in sgdf[sgdf[ds_id_field] == -1][id_field].values:
        sgdf.loc[sgdf[id_field].isin(list(nx.ancestors(G, term_node)) + [term_node, ]), 'TerminalNode'] = term_node
    sgdf['TerminalNode'] = sgdf['TerminalNode'].astype(int)

    # prepare attribute for clustering to make VPUs
    logger.info('\tAssigning VPUs')
    if sgdf.shape[0] > 100_000:
        sgdf = assign_vpu_by_kmeans(sgdf)
    else:
        sgdf['VPU'] = 101

    logger.info('\tWriting RAPID master parquets')
    sgdf.drop(columns=['geometry', ]).to_parquet(os.path.join(save_dir, "rapid_inputs_master.parquet"))
    return


def dissolve_branches(streams_df: pd.DataFrame,
                      head_to_dissolve: pd.DataFrame,
                      geometry_diss: types.FunctionType or str = 'last',
                      k_agg_func: types.FunctionType or str = 'last', ) -> pd.DataFrame:
    logger.info('\tDissolving headwater streams in inputs master')
    for streams_to_merge in head_to_dissolve.values:
        streams_df.loc[streams_df['LINKNO'].isin(streams_to_merge), 'LINKNO'] = streams_to_merge[0]
    agg_rules = {
        # 'LINKNO': 'last',
        'DSLINKNO': 'last',
        'DSNODEID': 'last',
        'strmOrder': 'last',
        'Length': lambda x: x.sum() if len(x) > 1 else x.iloc[0],
        'Magnitude': 'last',
        'DSContArea': 'last',
        'strmDrop': lambda x: x.iloc[-1] + x.iloc[:-1].max() if len(x) > 1 else x.iloc[0],
        'Slope': lambda x: -1,
        'StraightL': lambda x: -1,
        'USContArea': lambda x: x.iloc[:-1].sum() if len(x) > 1 else x.iloc[0],
        'WSNO': 'last',
        'DOUTEND': 'last',
        'DOUTSTART': lambda x: x.iloc[:-1].max() if len(x) > 1 else x.iloc[0],
        'DOUTMID': lambda x: x.mean() if len(x) > 1 else x.iloc[0],
        'lat': 'last',
        'lon': 'last',
        'z': 'last',
        'TDXHydroNumber': 'last',
        'TDXHydroHeaderNumber': 'last',
        'TDXHydroLeadingDigit': 'last',
        'TDXHydroLinkNo': 'last',
        'TopologicalOrder': 'last',
        'geometry': geometry_diss,
    }

    if all([x in streams_df.columns for x in ['musk_k', 'musk_x', 'musk_kfac', 'musk_xfac', 'CountUS', ]]):
        agg_rules.update({
            'musk_k': k_agg_func,
            'musk_kfac': k_agg_func,
            'musk_x': 'last',
            'musk_xfac': 'last',
            'CountUS': lambda x: 0 if len(x) else x,
        })
    agg_rules.update({
        col: (lambda x: -1 if len(x) > 1 else x) for col in sorted(streams_df.columns) if col.startswith('USLINKNO')
    })
    return streams_df.groupby('LINKNO').agg(agg_rules).reset_index().sort_values('TopologicalOrder')


def prune_branches(sdf: pd.DataFrame, streams_to_prune: pd.DataFrame) -> pd.DataFrame:
    return sdf[~sdf['LINKNO'].isin(streams_to_prune.iloc[:, 1].values.flatten())]


def _k_agg_order_3(x: pd.Series) -> np.ndarray:
    return x.mean() * 3.5 if len(x) > 1 else x.iloc[0]


def _k_agg_order_2(x: pd.Series) -> np.ndarray:
    return x.iloc[-1] + x.iloc[:-1].max() if len(x) > 1 else x.iloc[0]


def _geom_diss(x: pd.Series or gpd.GeoSeries):
    return gpd.GeoSeries(x).unary_union


def assign_vpu_by_kmeans(sgdf: gpd.GeoDataFrame) -> pd.DataFrame:
    logger.info('\tPreparing attributes for clustering')
    sgdf['geometry'] = sgdf['geometry'].apply(lambda x: Point(x.coords[0]))
    sgdf_grouped = sgdf.groupby('TerminalNode')
    n_samples = 20
    xdf = (
        sgdf_grouped
        .agg({'geometry': lambda x: MultiPoint(x.tolist())})
        .reset_index()
    )
    xdf = xdf.merge(
        pd
        .DataFrame(
            sgdf_grouped['geometry']
            .apply(lambda x: np.random.choice(x, n_samples))
            .apply(lambda x: np.array([[a.x, a.y] for a in x]).flatten()).values.tolist()
        ),
        left_index=True,
        right_index=True
    )
    xdf = xdf.set_index('TerminalNode')
    xdf['x_centroid'] = xdf['geometry'].apply(lambda x: x.centroid.x)
    xdf['y_centroid'] = xdf['geometry'].apply(lambda x: x.centroid.y)
    outlets = sgdf[sgdf['DSLINKNO'] == -1].set_index('TerminalNode')['geometry'].apply(lambda x: x.xy)
    outlets = pd.DataFrame(outlets.apply(lambda x: [x[0][0], x[1][0]]).values.tolist(), index=outlets.index)
    xdf = xdf.merge(outlets, left_index=True, right_index=True)
    xdf = xdf.drop(columns=['geometry'])

    # make VPU clusters
    logger.info('\tMaking KMeans VPU clusters')
    kmeans = KMeans(n_clusters=int(np.ceil(sgdf.shape[0] / 60_000)))
    xdf['VPU'] = kmeans.fit_predict(xdf.values).astype(int) + 101
    xdf = xdf[['VPU', ]]
    return sgdf.merge(xdf, left_on='TerminalNode', right_index=True)


def make_quick_visuals(save_dir: str, gpq: str) -> None:
    logger.info('Making VPU Exploration Datasets')
    labels = (
        pd
        .read_parquet(os.path.join(save_dir, "rapid_inputs_master.parquet"),
                      columns=['LINKNO', 'TerminalNode', 'VPU', ], )
        .set_index('LINKNO')
    )
    sgdf = gpd.read_parquet(gpq, columns=['LINKNO', 'geometry'])
    sgdf = sgdf[['LINKNO', 'geometry']].set_index('LINKNO')
    sgdf = sgdf.merge(labels, left_index=True, right_index=True, how='inner')
    sgdf['geometry'] = sgdf['geometry'].apply(lambda x: x.simplify(0.025))

    sgdf = (
        sgdf
        .merge(
            sgdf
            .groupby('VPU')
            .count()
            [['TerminalNode', ]]
            .rename(columns={'TerminalNode': 'Count'})
            .reset_index(),
            left_on='VPU',
            right_on='VPU',
        )
    )

    region_number = os.path.basename(gpq)
    region_number = region_number.split('_')[2]
    region_number = int(region_number)
    sgdf.to_file(os.path.join(save_dir, f'vpus_{region_number}.gpkg'), driver='GPKG')
    return


def fix_vpus(inputs_directory: str, final_inputs_dir: str, vpu_fixes_csv: str) -> None:
    """
    Use the vpu_csv to fix the tdx_table. Order of vpu_df is important, do not modify
    """
    input_dirs = [x for x in glob.glob(os.path.join(inputs_directory, '*')) if os.path.isdir(x)]
    vpu_df = pd.read_csv(vpu_fixes_csv)

    # Step 1: Fix VPUs and concat into a single master table
    all_dfs = []
    for input_dir in input_dirs:
        network_name = int(os.path.basename(input_dir))
        vpu_corrections = vpu_df[vpu_df['Hydrobasin'] == network_name]
        rapid_inputs_master = os.path.join(input_dir, 'rapid_inputs_master.parquet')

        tdx_table = pd.read_parquet(rapid_inputs_master)
        if vpu_corrections.shape[0] != 0:
            for _, row in vpu_corrections.iterrows():
                if pd.isnull(row['old_vpu']):
                    # Assign new_vpu to features with matching TERMINALNODE
                    tdx_table.loc[tdx_table['TerminalNode'] == row['TerminalNode'], 'VPU'] = row['new_vpu']
                elif pd.isnull(row['TerminalNode']):
                    # Assign new_vpu to features with matching old_vpu
                    tdx_table.loc[tdx_table['VPU'] == row['old_vpu'], 'VPU'] = row['new_vpu']
        else:
            print(f'No corrections made for {network_name}')

        all_dfs.append(tdx_table)

    master_table = pd.concat(all_dfs)
    all_dfs = []

    # Step 2: assign globally unqiue IDs
    unique_vpus_codes_df = []
    for tdxnumber in master_table['TDXHydroLeadingDigit'].unique():
        matching_rows = master_table['TDXHydroLeadingDigit'] == tdxnumber
        df = master_table.loc[matching_rows].groupby(['TDXHydroNumber', 'VPU']).count()
        df = df.reset_index()[['TDXHydroNumber', 'VPU']]
        df['VPUIndexNumber'] = df.reset_index().index.to_series() + 1
        unique_vpus_codes_df.append(df)
    master_table = master_table.merge(pd.concat(unique_vpus_codes_df), on=['TDXHydroNumber', 'VPU'], how='outer')
    master_table['VPUIndexNumber'] = master_table['VPUIndexNumber'].astype(int)

    master_table['VPUCode'] = (
            master_table['TDXHydroLeadingDigit'].astype(str) +
            master_table['VPUIndexNumber'].astype(str).str.pad(2, fillchar='0')
    )
    master_table['geoglowsID'] = (
            master_table['VPUCode'].astype(str) + '-' + master_table['TDXHydroLinkNo'].astype(str)
    )

    master_table = master_table.drop(columns=['TDXHydroLeadingDigit', 'VPUIndexNumber', 'VPU'])

    logger.info('Writing single master table')
    master_table.to_parquet(os.path.join(final_inputs_dir, 'master_table.parquet'))
    return


def rapid_input_csvs(sdf: pd.DataFrame,
                     save_dir: str,
                     id_field: str = 'LINKNO',
                     ds_id_field: str = 'DSLINKNO', ) -> None:
    """
    Create RAPID input csvs from a stream network dataframe

    Produces the following files:
        - rapid_connect.csv
        - riv_bas_id.csv
        - k.csv
        - x.csv
        - comid_lat_lon_z.csv
        - kfac.csv
        - xfac.csv

    Args:
        vpu_df:
        save_dir:
        id_field:
        ds_id_field:

    Returns:

    """
    # todo check change in arguments and usage
    logger.info('Creating RAPID input csvs')
    # upstream_columns = sorted([x for x in sdf.columns if 'USLINKNO' in x])
    # rapcon_columns = [id_field, ds_id_field, 'CountUS', ] + upstream_columns
    # rapcon_df = sdf[rapcon_columns].copy()
    # rapcon_df[upstream_columns] = rapcon_df[upstream_columns].replace(-1, 0)
    #
    # (
    #     rapcon_df
    #     .fillna(0)
    #     .astype(int)
    #     .to_csv(os.path.join(save_dir, "rapid_connect.csv"), index=False, header=False)
    # )

    downstream_field = ds_id_field
    rapid_connect = []
    max_count_upstream = 0
    for hydroid in sdf[id_field].values:
        # find the HydroID of the upstreams
        list_upstream_ids = sdf.loc[sdf[downstream_field] == hydroid, id_field].values
        # count the total number of the upstreams
        count_upstream = len(list_upstream_ids)
        if count_upstream > max_count_upstream:
            max_count_upstream = count_upstream
        next_down_id = sdf.loc[sdf[id_field] == hydroid, downstream_field].values[0]

        row_dict = {'HydroID': hydroid, 'NextDownID': next_down_id, 'CountUpstreamID': count_upstream}
        for i in range(count_upstream):
            row_dict[f'UpstreamID{i + 1}'] = list_upstream_ids[i]
        rapid_connect.append(row_dict)

        # Fill in NaN values for any missing upstream IDs
    for i in range(max_count_upstream):
        col_name = f'UpstreamID{i + 1}'
        for row in rapid_connect:
            if col_name not in row:
                row[col_name] = 0

    logger.info('\tWriting Rapid Connect CSV')
    df = pd.DataFrame(rapid_connect)
    df[df != 0] = df + int(sdf['TDXHydroHeaderNumber'].values.flatten()[0] * 10_000_000)
    df.to_csv(os.path.join(save_dir, 'rapid_connect.csv'), index=False, header=None)

    logger.info('\tWriting RAPID Input CSVS')
    sdf['TDXHydroLinkNo'].to_csv(os.path.join(save_dir, "riv_bas_id.csv"), index=False, header=False)
    sdf["musk_k"].to_csv(os.path.join(save_dir, "k.csv"), index=False, header=False)
    sdf["musk_x"].to_csv(os.path.join(save_dir, "x.csv"), index=False, header=False)
    sdf["musk_kfac"].to_csv(os.path.join(save_dir, "kfac.csv"), index=False, header=False)
    sdf["musk_xfac"].to_csv(os.path.join(save_dir, "xfac.csv"), index=False, header=False)
    sdf[['TDXHydroLinkNo', 'lat', 'lon', 'z']].to_csv(os.path.join(save_dir, "comid_lat_lon_z.csv"), index=False)

    return


def rapid_csvs_final(final_inputs_directory: str, tdxinputs_directory: str) -> None:
    df = pd.read_parquet(os.path.join(final_inputs_directory, 'master_table.parquet'))
    for vpu in sorted(df['VPUCode'].unique()):
        vpu_df = df.loc[df['VPUCode'] == vpu]
        vpu_dir = os.path.join(final_inputs_directory, vpu)
        os.makedirs(vpu_dir, exist_ok=True)
        rapid_input_csvs(vpu_df, vpu_dir)

        tdx_region = vpu_df['TDXHydroNumber'].unique()[0]
        weight_tables = glob.glob(os.path.join(tdxinputs_directory, str(tdx_region), f'weight*.csv'))
        weight_tables = [x for x in weight_tables if '_full.csv' not in x]
        for weight_table in weight_tables:
            a = pd.read_csv(weight_table)
            a = a[a.iloc[:, 0].astype(int).isin(vpu_df['LINKNO'].values)]
            a.to_csv(os.path.join(vpu_dir, os.path.basename(weight_table)), index=False)
    return
