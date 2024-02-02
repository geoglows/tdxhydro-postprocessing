import glob
import os
from multiprocessing import Pool

import geopandas as gpd
import natsort
import networkx as nx
import pandas as pd
import tqdm

countries_gpkg = '/Users/rchales/Downloads/geoBoundariesCGAZ_ADM0.gpkg'
lakes_gpkg = ''
gpqdir = '/Volumes/T9Hales4TB/TDXHydroGeoParquet'
propsdir = '/Volumes/T9Hales4TB/TDXHydroProperties'

countries = (
    gpd
    .read_file(countries_gpkg)
    .to_crs(epsg=4326)
    .rename(columns={'shapeName': 'RiverCountry'})
)

# lakes = (
#     gpd
#     .read_file(lakes_gpkg)
#     .to_crs(epsg=4326)
# )


def process_tdxpq(tdxpq):
    print(tdxpq)
    tdxnumber = os.path.basename(tdxpq).split('_')[2]
    gdf = gpd.read_parquet(tdxpq, columns=['LINKNO', 'DSLINKNO', 'geometry'])
    print(f'{tdxnumber} read {gdf.shape[0]} rows')

    print(f'{tdxnumber} graph')
    G = nx.DiGraph()
    G.add_edges_from(gdf[['LINKNO', 'DSLINKNO']].values)

    # # make a df with LINKNO and a lon, lat column
    # gdf[['lon', 'lat']] = [[x, y] for x, y in gdf.geometry.apply(lambda x: x.coords[0]).values]
    # gdf[['LINKNO', 'lon', 'lat']].to_parquet(f'{propsdir}/outletcoords_{tdxnumber}.parquet')

    print(f'{tdxnumber} sjoin with country boundaries')
    river_countries_within = (
        gpd
        .sjoin(countries, gdf, predicate='intersects', how='right', )
        [['LINKNO', 'RiverCountry']]
    )
    river_countries_within = (
        river_countries_within
        [~river_countries_within['RiverCountry'].isna()]
        .groupby('LINKNO')
        .first()
        .reset_index()
    )
    print(f'{tdxnumber} sjoin nearest for {river_countries_within.shape[0]} rivers not within a country boundary')
    river_countries_nearest = (
        gpd
        .sjoin_nearest(
            countries,
            gdf[~gdf['LINKNO'].isin(river_countries_within['LINKNO'])],
            how='right',
            max_distance=0.1,
        )
        [['LINKNO', 'RiverCountry']]
    )
    river_countries_nearest = (
        river_countries_nearest
        [~river_countries_nearest['RiverCountry'].isna()]
        .groupby('LINKNO')
        .first()
        .reset_index()
    )
    river_countries = pd.concat([river_countries_within, river_countries_nearest])
    river_without_country = gdf[~gdf['LINKNO'].isin(river_countries['LINKNO'])]
    river_without_country['RiverCountry'] = 'Unknown'
    river_countries = pd.concat([river_countries, river_without_country[['LINKNO', 'RiverCountry']]])

    river_countries['OutletCountry'] = ''
    outlet_links = gdf[gdf['DSLINKNO'] == -1]['LINKNO']
    for outlet_link in tqdm.tqdm(outlet_links, desc=f'{tdxnumber} outlet links processed'):
        outlet_country = river_countries.loc[river_countries['LINKNO'] == outlet_link, 'RiverCountry'].values[0]
        watershed_rivers = nx.ancestors(G, outlet_link)
        watershed_rivers.add(outlet_link)
        river_countries.loc[river_countries['LINKNO'].isin(watershed_rivers), 'OutletCountry'] = outlet_country
    river_countries.to_parquet(f'{propsdir}/countries_{tdxnumber}.parquet')

    # print(f'{tdxnumber} with lake boundaries')
    # lake_rivers = (
    #     gpd
    #     .sjoin(lakes, gdf, predicate='intersects', how='right')
    #     [['LINKNO', ]]
    # )
    # lake_rivers.to_parquet(f'{propsdir}/lakes_{tdxnumber}.parquet')


def merge_outletcoords():
    (
        pd
        .concat(
            [pd.read_parquet(x) for x in
             natsort.natsorted(glob.glob(f'{propsdir}/outletcoords_*.parquet'))]
        )
        .to_parquet(f'{propsdir}/outletcoords.parquet')
    )
    os.system(f'rm {propsdir}/outletcoords_*.parquet')


def merge_lakes():
    (
        pd
        .concat(
            [pd.read_parquet(x) for x in
             natsort.natsorted(glob.glob(f'{propsdir}/lakes_*.parquet'))]
        )
        .to_parquet(f'{propsdir}/lake_rivers.parquet')
    )
    os.system(f'rm {propsdir}/lakes_*.parquet')


def merge_countries():
    (
        pd
        .concat(
            [pd.read_parquet(x) for x in
             natsort.natsorted(glob.glob(f'{propsdir}/countries_*.parquet'))]
        )
        .to_parquet(f'{propsdir}/river_countries.parquet')
    )
    # os.system(f'rm {propsdir}/countries_*.parquet')


if __name__ == '__main__':
    jobs = natsort.natsorted(glob.glob(f'{gpqdir}/TDX_streamnet*.parquet'))
    # with Pool(min(11, len(jobs))) as p:
    #     p.map(process_tdxpq, jobs)
    # for job in jobs:
    #     process_tdxpq(job)

    # merge_outletcoords()
    # merge_lakes()
    merge_countries()
