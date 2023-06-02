import networkx as nx
import os

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import numpy as np
from time import time
import re
from shapely.geometry import Point, MultiPoint
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn import mixture
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool

from save import save_n

if __name__ == "__main__":
    # Set up variables
    start_time = time()
    g = nx.DiGraph()
    carib = '/Users/ricky/Downloads/tdxhydro_streams_70s_northamerica/TDX_streamnet_7020065090_01.gpkg'
    australia = "/Users/ricky/Downloads/tdxhydro_streams_50s_australia/TDX_streamnet_5020049720_01.parquet"
    next_down_id_col = 'DSLINKNO'
    stream_id_col = 'LINKNO'

    network_to_use = australia  # Assign
    # network = gpd.read_parquet(network_to_use)
    # print(f"    Read gpkg in {np.round(time()-start_time,2)} secs")

    #### DROP ORDER 1  and small catchments, this is for faster testing. REMOVE WHEN IMPLEMENTING WITH RILEY"S PREPROCESSING SCRIPT:
    # network = network.loc[~((network['DSLINKNO'] == -1) & (network['USLINKNO1'] == -1) | (network['DSContArea'] < 75000000)), :]
    ########## 

    # Use re to get the numbers we want to make the computational id
    pattern = re.compile(r'.*streamnet_(\d{10})_0.*')  # Get 10 digit hydrobasin id from path
    try:
        hydrobasin = pattern.findall(network_to_use)[-1]
    except:
        raise ValueError(f"Hydrobasin got is '{pattern.findall(network_to_use)}', which is not valid")
    hydrobasin = hydrobasin[0] + '-' + hydrobasin[-5:] + '-'

    # # # Create network from dataframe
    # for next_down_id, stream_id in zip(network[next_down_id_col].values, network[stream_id_col].values):
    #     g.add_edge(next_down_id, stream_id)

    # print(f"    Made networkx in {np.round(time()-start_time,2)} secs")

    # network['TERMINALID'] = ""

    # for outlet in network[network.DSLINKNO == -1]['LINKNO']:
    #     # get a list of upstream streams, including this outlet
    #     descendants = list(map(int, nx.descendants(g, outlet))) + [outlet]
    #     # Give all streams that go to this node the same terminal id (the end stream id)
    #     network.loc[network['LINKNO'].isin(descendants), 'TERMINALID'] = outlet

    # print("DONE 1")
    # filtered_network = network[(network.DSLINKNO == -1) & (network.DSContArea < 100000000)]
    # filtered_terminalids = filtered_network['TERMINALID'].tolist()
    # network = network[~network['TERMINALID'].isin(filtered_terminalids)]
    # print("DONE 2")
    # for outlet in network[network.DSLINKNO == -1]['LINKNO']:
    #     # get a list of upstream streams, including this outlet
    #     descendants = list(map(int, nx.descendants(g, outlet))) + [outlet]
    #     # Give all streams that go to this node the same terminal id (the end stream id)
    #     network.loc[network['LINKNO'].isin(descendants), 'TERMINALID'] = outlet
    # print(network)
    # print(network.columns)
    # network.to_parquet('cached_network_aus.parquet')
    # exit()

    ###### METHOD 1 MEANS ########
    # compute the x- and y-coordinates of the centroids of each line, average them, sort by the absolute average and proximity of features, 
    # leaves us with a geodataframe sorted by geometry
    # cols = ['geometry', 'LINKNO', 'DSLINKNO']
    # sorted_net = (
    #     network.loc[network['DSLINKNO'] == -1, cols]
    #     .to_crs({'proj':'cea'})
    #     .assign(**{'x':lambda df: np.abs(df['geometry'].centroid.x), 
    #                 'y':lambda df: np.abs(df['geometry'].centroid.y),
    #                 'rep_val':lambda df: df[['x', 'y']].mean(axis=1),
    #                 'prev':lambda df: df['rep_val'].shift(1),
    #                 'closeness': lambda df: df['rep_val'] - df['prev']}) 
    #     .sort_values(by=['rep_val','closeness'])
    # )
    # print(f"    Sorted gpkg in {np.round(time()-start_time,2)} secs")
    ########    ##########

    ############## READ IN / PREPROCESS ##############
    network = gpd.read_parquet('cached_network_aus.parquet')

    # # Create network from dataframe
    for next_down_id, stream_id in zip(network[next_down_id_col].values, network[stream_id_col].values):
        g.add_edge(next_down_id, stream_id)

    print(f"    Made networkx in {np.round(time() - start_time, 2)} secs")

    # Assign Terminal IDS
    network['TERMINALID'] = ""

    for outlet in network[network.DSLINKNO == -1]['LINKNO']:
        # get a list of upstream streams, including this outlet
        descendants = list(map(int, nx.descendants(g, outlet))) + [outlet]
        # Give all streams that go to this node the same terminal id (the end stream id)
        network.loc[network['LINKNO'].isin(descendants), 'TERMINALID'] = outlet

    grouped = network.groupby('TERMINALID')['geometry'].apply(lambda x: Point(x.unary_union.centroid))
    df = pd.DataFrame(index=grouped.index, data={'x': grouped.x.values, 'y': grouped.y.values})
    df = df.sort_values(by=['x', 'y'])

    group_size = 50000
    count = 0
    computation_id = 1

    # network = gpd.read_parquet('cached_simple_aus.parquet')
    network = network[['LINKNO', 'DSLINKNO', 'geometry', 'TERMINALID', 'x', 'y']]
    # network = network[['LINKNO', 'DSLINKNO', 'geometry', 'TERMINALID']]
    df = pd.read_parquet('centroids_sorted.parquet')
    df['used'] = False
    print(f"    Read {np.round(time() - start_time, 2)} secs")

    network['vpu'] = ""
    # network.geometry = network['geometry'].map(lambda geom: geom.simplify(tolerance=0.001))
    # network.to_parquet('cached_simple_aus.parquet')

    ############## End PREPROCESS ##############

    network['out_x'] = network[network.LINKNO.isin(df.index)].geometry.apply(lambda x: x.coords[0][0])
    network['out_y'] = network[network.LINKNO.isin(df.index)].geometry.apply(lambda x: x.coords[0][1])
    net = network.groupby('TERMINALID')
    random_rows = net.apply(lambda x: x.sample(25 , random_state=np.random.RandomState(), replace=True)).set_index(
        'TERMINALID')
    random_rows = random_rows.drop(columns=['out_x', 'out_y'])
    relevant_piece = network[network.LINKNO.isin(df.index)][['TERMINALID', 'out_x', 'out_y']].set_index('TERMINALID')
    df = df.join(relevant_piece)

    termid = random_rows.index[0]
    count = 0
    termid = None
    count = 0

    for row in random_rows.itertuples():
        if row[0] != termid:
            count = 0
        df.loc[row[0], f"x_{count}"] = row[4]
        df.loc[row[0], f"y_{count}"] = row[5]
        termid = row[0]
        count += 1

    ##################### ML #####################
    num_features = network.shape[0]

    geometry = [
        (x, y, out_x, out_y, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10,
         x11, y11, x12, y12, x13, y13, x14, y14, x15, y15, x16, y16, x17, y17, x18, y18, x19, y19, x20, y20, x21, y21,
         x22, y22, x23, y23, x24, y24, x25, y25, x26, y26, x27, y27, x28, y28, x29, y29, x30, y30, x31, y31, x32, y32,
         x33, y33, x34, y34, x35, y35, x36, y36, x37, y37, x38, y38, x39, y39)
        for
        x, y, out_x, out_y, x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9, x10, y10, x11, y11, x12, y12, x13, y13, x14, y14, x15, y15, x16, y16, x17, y17, x18, y18, x19, y19, x20, y20, x21, y21, x22, y22, x23, y23, x24, y24, x25, y25, x26, y26, x27, y27, x28, y28, x29, y29, x30, y30, x31, y31, x32, y32, x33, y33, x34, y34, x35, y35, x36, y36, x37, y37, x38, y38, x39, y39
        in zip(df.x, df.y, df.out_x, df.out_y, df.x_0, df.y_0, df.x_1, df.y_1, df.x_2, df.y_2, df.x_3, df.y_3, df.x_4,
               df.y_4, df.x_5, df.y_5, df.x_6, df.y_6, df.x_7, df.y_7, df.x_8, df.y_8, df.x_9, df.y_9, df.x_10, df.y_10,
               df.x_11, df.y_11, df.x_12, df.y_12, df.x_13, df.y_13, df.x_14, df.y_14, df.x_15, df.y_15, df.x_16,
               df.y_16, df.x_17, df.y_17, df.x_18, df.y_18, df.x_19, df.y_19, df.x_20, df.y_20, df.x_21, df.y_21,
               df.x_22, df.y_22, df.x_23, df.y_23, df.x_24, df.y_24, df.x_25, df.y_25, df.x_26, df.y_26, df.x_27,
               df.y_27, df.x_28, df.y_28, df.x_29, df.y_29, df.x_30, df.y_30, df.x_31, df.y_31, df.x_32, df.y_32,
               df.x_33, df.y_33, df.x_34, df.y_34, df.x_35, df.y_35, df.x_36, df.y_36, df.x_37, df.y_37, df.x_38,
               df.y_38, df.x_39, df.y_39)]

    # gaussian = mixture.BayesianGaussianMixture(n_components=num_features // group_size)
    # cluster = SpectralClustering(n_clusters=(num_features // group_size))
    kmeans = KMeans(n_clusters=num_features // group_size, n_init='auto')
    df['cluster'] = kmeans.fit_predict(geometry)
    df = df.sort_values(by=['cluster', 'x', 'y'])

    print(f"    Prepared {np.round(time() - start_time, 2)} secs")

    current_clust = df['cluster'].values[0]
    for outlet in df.index:
        # Add to count the amount of features that have the current terminalid
        count += network[network.TERMINALID == outlet].shape[0]
        if df.loc[outlet, 'cluster'] != current_clust:
            computation_id += 1
            count = 0
            current_clust = df.loc[outlet, 'cluster']
        network.loc[network['TERMINALID'] == outlet, 'vpu'] = hydrobasin + str(computation_id)
    ############################ #####################

    print(f"    Assigned gpkg in {np.round(time() - start_time, 2)} secs")
    ############### METHOD 2
    # def get_centroid(group):
    #     return group.geometry.centroid

    # for outlet in network.loc[network['DSLINKNO'] == -1, 'LINKNO']:
    #     # get a list of upstream streams, including this outlet
    #     descendants = list(map(int, nx.descendants(g, outlet))) + [outlet]
    #     # Give all streams that go to this node the same terminal id (the end stream id)
    #     network.loc[network['LINKNO'].isin(descendants), 'TERMINALID'] = outlet

    # # Compute the centroids of each group
    # centroids = network.to_crs({'proj':'cea'}).groupby('TERMINALID')['geometry'].apply(get_centroid).reset_index()

    # # Dissolve the GeoDataFrame to create a single geometry object
    # df_new = centroids.dissolve(by='TERMINALID')
    # df_new['x'] = df_new['geometry'].apply(
    #     lambda geom: geom.centroid.x if isinstance(geom, MultiPoint) else geom.x
    # )
    # df_new['y'] = df_new['geometry'].apply(
    #     lambda geom: geom.centroid.y if isinstance(geom, MultiPoint) else geom.y
    # )

    # # Add the centroids to the original GeoDataFrame using a join operation
    # gdf = network[['LINKNO', 'DSLINKNO','TERMINALID']].join(df_new, on=['TERMINALID'])

    # cols = ['LINKNO', 'DSLINKNO','TERMINALID','x','y']
    # sorted_net = (
    #     gdf.loc[network['DSLINKNO'] == -1, cols + ['geometry']]
    #     .assign(**{'rep_val':lambda df: df[['x', 'y']].mean(axis=1),
    #                 'prev':lambda df: df['rep_val'].shift(1),
    #                 'closeness': lambda df: df['rep_val'] - df['prev']}) 
    #     .sort_values(by=['rep_val', 'closeness'])
    # )

    # print(f"    Sorted gpkg in {np.round(time()-start_time,2)} secs")

    # group_size = 100000
    # count = 0
    # computation_id = 0

    # # Add to count the amount of features that have the current terminalid
    # for termid in sorted_net['TERMINALID']:
    #     count += network[network.TERMINALID == termid].shape[0]
    #     if count > group_size: # Too many features for current coputational_id - change it
    #         computation_id += 1
    #         count = 0
    #     network.loc[network['TERMINALID'] == termid, 'computation_id'] = hydrobasin + str(computation_id)

    # print(f"    Assigned gpkg in {np.round(time()-start_time,2)} secs")

    ##################

    # Save the updated GeoDataFrame to a new GeoPackage
    # print(network.shape)

    network['geometry'] = network['geometry'].apply(lambda x: Point(x.coords[0]))
    network.to_file('outputs/kmeans_5.gpkg', driver='GPKG')

    print(f"    Saved gpkg in {np.round(time() - start_time, 2)} secs")
