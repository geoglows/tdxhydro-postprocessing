import glob
import os
from multiprocessing import Pool

import tqdm
import pandas as pd
import networkx as nx
import geopandas as gpd
from pyproj.crs.crs import CRS
from shapely.geometry import MultiLineString, LineString

from tdxhydrorapid.network import create_directed_graphs

stream_files = glob.glob(r"D:\geoglows_v3\geoglows_v3\hydrography\*\streams_*.gpkg")
original_stream_pqs = glob.glob(r"D:\geoglows_v3\parquets\TDX_streamnet_*_01.parquet")
alterations_dir = r"D:\geoglows_v3\geoglows_v3\hydrography-tdxhydro-sources\alterations"
lake_table_path = os.path.join(os.path.dirname(__file__), 'tdxhydrorapid', 'network_data', 'lake_table.csv')
OVERWRITE = True

lake_df = pd.read_csv(lake_table_path)

def identify_downstream_segments(multi_line_geom: MultiLineString,) -> list[LineString]:
    # # Extract individual LineStrings from the MultiLineString
    line_segments = list(multi_line_geom.geoms)

    # Build graph
    G = nx.DiGraph()
    for i, segment in enumerate(line_segments):
        G.add_edge(segment.coords[-1], segment.coords[0], segment=segment)
        G.add_node(segment.coords[0])
        G.add_node(segment.coords[-1])

    # We want to get the downstream segments: these are segments with a max of 1 up or downstream segment
    # Fortunately, the line segments are in topological order so we can append freely
    downstream_segments = []
    for segment in line_segments:
        if G.in_degree(segment.coords[0]) <= 1 and G.out_degree(segment.coords[-1]) <= 1:
            downstream_segments.append(segment)

    return downstream_segments

def return_lake_trunk(gdf: gpd.GeoSeries,
                      G: nx.DiGraph,
                      OG_G: nx.DiGraph,
                      og_gdf: gpd.GeoDataFrame,
                      CRS: CRS) -> LineString:
    # Find endorheic lakes because this way for now because I forgot to update the JSON.... something for V3
    # Also, we cannot assume that the lake dataset labels endorheic lakes correctly (Utah lake was labeled as endorheic)
    if len(list(G.successors(gdf['LINKNO']))) == 0 and ((gdf['LINKNO'] in lake_df['outlet'].values and lake_df.loc[lake_df['outlet'] == gdf['LINKNO'], 'endorheic'].values[0]) or \
        (lake_df['inlet'].isin(set(G.predecessors(gdf['LINKNO']))).any() and lake_df.loc[lake_df['inlet'].isin(set(G.predecessors(gdf['LINKNO']))), 'endorheic'].values[0])):
        # Do not display endorheic lakes
        return LineString()

    # Find the inlet with the largest upstream area
    inlets = set(G.predecessors(gdf['LINKNO']))

    if len(inlets) == 0:
        # Return just the outlet's geom
        return og_gdf[og_gdf['LINKNO'] == gdf['LINKNO']]['geometry'].to_crs(CRS).iloc[0]

    # Find the inlet with the largest upstream area
    max_dscontarea = 0
    for inlet in inlets:
        area = G.edges()[inlet, gdf['LINKNO']]['DSContArea']
        if area > max_dscontarea:
            max_dscontarea = area
            largest_inlet = inlet

    # Find the path from the largest inlet to the outlet
    path: list[int] = nx.shortest_path(OG_G, source=largest_inlet, target=gdf['LINKNO'])

    # Extract the geometries of the path, in that order
    path_geoms = [og_gdf[og_gdf['LINKNO'] == link]['geometry'].to_crs(CRS).iloc[0] for link in path[1:]] # Ignore inlet
    multicoords = [list(line.coords) for line in path_geoms]
    
    # The multicoords may be in opposing directions (i.e., up to down for the first one, but down to up for the second)
    # We need to make sure they are in the same direction
    for i in range(1, len(path_geoms)):
        if multicoords[i-1][-1] == multicoords[i][0]:
            continue # These match
        if multicoords[i-1][0] == multicoords[i][0]:
            multicoords[i-1] = multicoords[i-1][::-1]
        elif multicoords[i-1][0] == multicoords[i][-1]:
            multicoords[i-1] = multicoords[i-1][::-1]
            multicoords[i] = multicoords[i][::-1]
        elif multicoords[i-1][-1] == multicoords[i][-1]:
            multicoords[i] = multicoords[i][::-1]
        else:
            raise NotImplementedError("The geometries are not connected")
    return LineString([item for sublist in multicoords for item in sublist])

def return_all_big_paths(gdf: gpd.GeoSeries, 
                         G: nx.DiGraph,
                         OG_G: nx.DiGraph,
                         og_gdf: gpd.GeoDataFrame,
                         CRS: CRS,
                         min_strm_order: int) -> MultiLineString: 
    # Find all inlets
    inlets = set(G.predecessors(gdf['LINKNO']))

    # Filter inlets to only include stream orders min_strm_order+
    inlets = og_gdf[og_gdf['LINKNO'].isin(inlets) & (og_gdf['strmOrder'] >= min_strm_order)]['LINKNO']

    if len(inlets) == 0:
        # This shouldn't happen, but if it does, we probably want to return just the outlet's geom
        return og_gdf[og_gdf['LINKNO'] == gdf['LINKNO']]['geometry'].iloc[0]

    # Get paths from each inlet to the outlet
    paths: dict[int, list[int]] = nx.shortest_path(OG_G, target=gdf['LINKNO'])

    # Get all ids in the paths, ignoring the inlet
    to_dissolve = set()
    for inlet in inlets:
        to_dissolve.update(paths[inlet][1:])

    # Dissolve the geometries
    return og_gdf[og_gdf['LINKNO'].isin(to_dissolve)]['geometry'].to_crs(CRS).union_all()

def purge_dissolved(gdf: gpd.GeoSeries, 
                    lake_outlets: set, 
                    og_gdf: gpd.GeoDataFrame, 
                    G: nx.DiGraph, 
                    OG_G: nx.DiGraph, 
                    CRS: CRS) -> LineString:
    g = gdf['geometry']

    if gdf['LINKNO'] in lake_outlets:
        assert isinstance(g, MultiLineString), f"Expected MultiLineString for a lake, got {type(g)}"
        return return_lake_trunk(gdf, G, OG_G, og_gdf, CRS)
        # return return_all_big_paths(gdf, G, OG_G, og_gdf, CRS, 5)
    
    if isinstance(g, MultiLineString):
        if len(g.geoms) >= 3 and len(list(G.predecessors(gdf['LINKNO']))) == 0:
            if len(g.geoms) == 3:
                return g.geoms[-1]
            
            geoms = identify_downstream_segments(g)
            return LineString([item for line in geoms for item in line.coords])
    
        return LineString([item for line in g.geoms for item in line.coords])
    
    if isinstance(g, LineString):
        return g # Return the geometry unchanged if no condition is met
    
    raise ValueError(f"Could not resolve geometry: {gdf}")

def main(file: str):
    vpu = file.split('.')[0].split('_')[-1]

    out_file = os.path.join(os.path.dirname(file), f'streams_mapping_{vpu}.gpkg')
    if os.path.exists(out_file) and not OVERWRITE:
        return

    df: gpd.GeoDataFrame = gpd.read_file(file)
    CRS = df.crs
    G: nx.DiGraph = nx.from_pandas_edgelist(df[df['DSLINKNO'] != -1], source='LINKNO', target='DSLINKNO', edge_attr='DSContArea', create_using=nx.DiGraph)
    G.add_nodes_from(df['LINKNO'].values)

    region = str(df['TDXHydroRegion'].iloc[0])
    og_gdf = gpd.read_parquet([f for f in original_stream_pqs if region in f][0])
    OG_G = create_directed_graphs(og_gdf)

    lake_csv = os.path.join(alterations_dir, region, 'mod_dissolve_lakes.json')
    if os.path.exists(lake_csv):
        lake_streams_df = pd.read_json(lake_csv, orient='index', convert_axes=False, convert_dates=False)
        lake_outlets = set(lake_streams_df.index.astype(int))
        
    df.geometry = df.apply(purge_dissolved, axis=1, args=(lake_outlets, og_gdf, G, OG_G, CRS))

    df.to_file(out_file, driver='GPKG')

if __name__ == '__main__':
    stream_files = [f for f in stream_files if 'mapping' not in os.path.basename(f)]

    # I suggest 1 process for each 16 GB of total RAM
    with Pool(2) as p:
        list(tqdm.tqdm(p.imap_unordered(main, stream_files), total=len(stream_files)))

