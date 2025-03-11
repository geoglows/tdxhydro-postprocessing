import logging
import sys
import glob
import os
import networkx as nx
import tqdm

import geopandas as gpd
import pandas as pd
import dask_geopandas as dgpd
from shapely.geometry import Polygon, MultiPolygon


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)

lakes_parquet = r"D:\geoglows_v3\all_lakes_filtered.parquet" # https://garslab.com/?p=234 August 24, 2022
gpq_dir = r"D:\geoglows_v3\parquets"
save_dir = os.path.join('.', 'tdxhydrorapid', 'network_data')
MIN_LAKE_AREA = 10
DEBUG = False

def create_directed_graphs(df: gpd.GeoDataFrame,
                           id_field='LINKNO',
                           ds_id_field='DSLINKNO', ) -> nx.DiGraph:
    G: nx.DiGraph = nx.from_pandas_edgelist(df[df[ds_id_field] != -1], source=id_field, target=ds_id_field, create_using=nx.DiGraph)
    G.add_nodes_from(df[id_field].values)
    return G

def fill_holes(geometry):
    """
    Removes holes from a Polygon or MultiPolygon geometry.
    """
    if isinstance(geometry, Polygon):
        # Remove holes by keeping only the exterior
        return Polygon(geometry.exterior)
    elif isinstance(geometry, MultiPolygon):
        # Process each Polygon in the MultiPolygon
        return MultiPolygon([Polygon(poly.exterior) for poly in geometry.geoms])
    return geometry  # Return as-is for non-polygon geometries (if any)

if __name__ == "__main__":
    logging.info('Getting Lake Polygons')
    
    lakes_gdf = gpd.read_parquet(lakes_parquet)
    lakes_gdf = lakes_gdf[lakes_gdf['Area_PW'] > MIN_LAKE_AREA]
    lakes_gdf['geometry'] = lakes_gdf['geometry'].apply(fill_holes) # Remove holes from lakes, could lead to issues...

    values_list = []
    pqs = glob.glob(os.path.join(gpq_dir, 'TDX_streamnet**.parquet'))
    for pq in tqdm.tqdm(pqs):
        global_outlets = set()
        global_inlets = set()
        global_inside = set()
        gdf = gpd.read_parquet(pq)
        G = create_directed_graphs(gdf)
        
        # Get all streams that intersect with a lake
        bounds = gdf.total_bounds
        lakes_subset = lakes_gdf.cx[ bounds[0]:bounds[2], bounds[1]:bounds[3]]
        dgdf = dgpd.from_geopandas(gdf, npartitions=os.cpu_count())
        intersect: gpd.GeoDataFrame = dgpd.sjoin(dgdf, dgpd.from_geopandas(lakes_subset), how='inner', predicate='intersects').compute()

        if intersect.empty:
            continue

        intersect = intersect.drop(columns=['index_right'])
        lakes_g = create_directed_graphs(intersect)
        intersect = intersect.set_index('LINKNO')

        # Make looking these up O(1)
        lake_id_dict: dict[int, int] = intersect['Lake_id'].to_dict()
        lake_polygon_dict: dict[int, Polygon] = lakes_subset.set_index('Lake_id')['geometry'].to_dict() 
        geom_dict: dict[int, Polygon] = gdf.set_index('LINKNO')['geometry'].to_dict()
        
        # Get connected components
        check=False
        for lake_ids in nx.weakly_connected_components(lakes_g):
            extra_inlets = set()
            # Get sink
            outlets = [node for node in lake_ids if lakes_g.out_degree(node) == 0]
            if len(outlets) != 1:
                raise RuntimeError(f'Lake has {len(outlets)} outlets')
            
            outlet = outlets[0]
            # Move the outlet, as needed
            predecessors = set(G.predecessors(outlet))
            if len(predecessors) == 2 and all([pred in lake_ids for pred in predecessors]):
                if gdf.loc[gdf['LINKNO'] == outlet, 'LengthGeodesicMeters'].values[0] <= 0.01:
                    # Find first non-zero length stream
                    downstream = set(G.successors(outlet))
                    while downstream:
                        extra_inlets.update({pred for pred in predecessors if pred not in lake_ids})
                        outlet = downstream.pop()
                        if gdf.loc[gdf['LINKNO'] == outlet, 'LengthGeodesicMeters'].values[0] > 0.01:
                            break
                    else:
                        # This means that there was no downstream. We must choose an outlet from the upstreams
                        # First, we must get all preds that are not 0 length that are in the lake (Depth-first Search)
                        visited = set()
                        queue = [outlet]
                        outlet = []
                        while queue:
                            node = queue.pop()
                            if node in visited:
                                continue
                            visited.add(node)
                            preds = set(G.predecessors(node))
                            for pred in preds:
                                if pred in lake_ids and gdf.loc[gdf['LINKNO'] == pred, 'LengthGeodesicMeters'].values[0] > 0.01:
                                    outlet.append(pred)
                                else:
                                    queue.append(pred)     
                else:
                    # Outlet can stay in this situation
                    pass
            else:
                if gdf.loc[gdf['LINKNO'] == outlet, 'LengthGeodesicMeters'].values[0] <= 0.01:
                    # This is a confluence. Let's choose the directly downstream segment if it exists
                    downstream = set(G.successors(outlet))
                    if downstream:
                        outlet = downstream.pop()
                        extra_inlets.update({pred for pred in predecessors if pred not in lake_ids})
                    else:
                        # Let 0-len stream be the outlet
                        pass
                else:# Set the upstream predecessor that touches the lake as the outlet
                    for pred in predecessors:
                        if pred in lake_ids:
                            outlet = pred
                            break       

            if not isinstance(outlet, list):
                outlet_list = [outlet]
            else:
                outlet_list = outlet

            for outlet in outlet_list:
                if outlet in global_inlets:
                    # Ah! The outlet really should be the outlet of another lake
                    new_outlets = [v for v in values_list if v[0] == outlet]
                    if len(new_outlets) != 1 and not all(x[:2] == new_outlets[0][:2] for x in new_outlets):
                        # raise RuntimeError(f'Outlet {outlet} is in multiple lakes')
                        pass
                    
                    elif new_outlets:
                        # Remove the inlet from the list
                        values_list = [v for v in values_list if v[0] != outlet]
                        
                        outlet = new_outlets[0][1]
                        check = True
                        
                if (outlet in global_inside or outlet in global_outlets) and not check:
                    # This is already taken care of
                    continue

                # Get all inlets
                inlets = {node for node in lake_ids if lakes_g.in_degree(node) == 0} | extra_inlets
                if outlet in inlets or len(lake_ids - inlets - {outlet}) == 0:
                    # Small lake, skip
                    continue

                # Choosing inlets based on in_degree can miss segments that have one upstream segment that is an inlet, but another upstream that was not in the intersection.
                # Let's find and add these inlets
                _continue = False
                to_test = lake_ids - inlets
                for river in to_test:
                    preds = set(G.predecessors(river))
                    preds_remaining = list(preds - lake_ids)
                    if len(preds) == 2 and len(preds_remaining) == 1 and 0 < len(preds - inlets) <= 2 and preds_remaining[0] != outlet and outlet not in preds:
                        # Add the other pred
                        inlet_to_be = preds_remaining[0]

                        if inlet_to_be in global_inside:
                            # This stream is already in a lake
                            _continue = True
                        inlets.add(inlet_to_be)
                        other_pred = (preds - {inlet_to_be}).pop()
                        lake_id_dict[inlet_to_be] = lake_id_dict.get(river, lake_id_dict.get(other_pred))

                if _continue:  
                    continue

                # Add any direct predecessors of inlets that have strmOrder == 1
                new_inlets = set()
                _continue = False
                for inlet in inlets:
                    preds = set(G.predecessors(inlet))
                    if not preds:
                        # Inlet is a headwater touching lake
                        pass
                    else:
                        lake_id = lake_id_dict.get(inlet, lake_id_dict.get(outlet))
                        if lake_id is None:
                            # This usually indicates that this outlet is far removed from a real lake.
                            # This tends to happen for tiny lakes that happened to have a 3+ river confluence
                            _continue = True
                            continue

                        intersection = geom_dict[inlet].intersection(lake_polygon_dict[lake_id])
                        intersection_length = intersection.length if not intersection.is_empty else 0

                        # Calculate the length outside the polygon
                        outside = geom_dict[inlet].difference(lake_polygon_dict[lake_id])
                        outside_length = outside.length if not outside.is_empty else 0
                        if intersection_length > outside_length:
                            # This stream is mostly inside the lake, so lets add preds
                            new_inlets.update(preds)
                            for pred in preds:
                                lake_id_dict[pred] = lake_id
                        else:
                            # This stream is mostly outside the lake, so lets maintain it as an inlet
                            new_inlets.add(inlet)

                if _continue:
                    continue

                if len(lake_ids - new_inlets - {outlet}) == 0 or not new_inlets:
                    # Not worth making a lake
                    continue

                outlet_ans = nx.ancestors(G, outlet)
                for inlet in new_inlets:
                    outlet_ans -= nx.ancestors(G, inlet) | {inlet}
                if check:
                    for inlet in [v[0] for v in values_list if v[1] == outlet]:
                        # Remove other inlets for this lake
                        outlet_ans -= nx.ancestors(G, inlet) | {inlet}

                global_inlets.update(new_inlets)
                global_outlets.add(outlet)
                global_inside.update(outlet_ans)

                # TODO I think this dataset flags endorheic lakes incorrectly sometimes. In V3, just check if downstream exists...
                try:
                    endorheic = bool(lakes_gdf.loc[lakes_gdf['Lake_id'] == lake_id_dict.get(outlet), 'Endo_flag'].values[0])
                except IndexError:
                    endorheic = bool(intersect.loc[intersect.index.isin(predecessors), 'Endo_flag'].values[0])

                for inlet in new_inlets:
                    values_list.append((inlet, outlet, lake_id_dict.get(inlet, lake_id_dict.get(outlet)), endorheic))

    df = pd.DataFrame(values_list, columns=['inlet', 'outlet', 'lake_id', 'endorheic'])
    df = df.drop_duplicates() # Sometimes we get duplicate entries, not sure why

    compare = pd.read_csv(os.path.join(save_dir, 'lake_table.csv'))
    if not compare.equals(df):
        print('Dataframes are not equal')

    exit()
    out_name = os.path.join(save_dir, 'lake_table.csv')
    df.to_csv(out_name, index=False)
    logging.info(f'Saved lakes to {out_name}')

    if DEBUG:
        gdf['type'] = ""
        gdf.loc[gdf['LINKNO'].isin(df['inlet']), 'type'] = 'inlet'
        gdf.loc[gdf['LINKNO'].isin(df['outlet']), 'type'] = 'outlet'
        gdf.loc[gdf['LINKNO'].isin(global_inside), 'type'] = 'inside'
        gdf[gdf['type'] != ''].to_parquet('inlets_outlets.parquet')
