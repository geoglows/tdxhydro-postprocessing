"""The country boundaries and the area vote that picks a reach's country, shared by
raw_processing/country_classifications.py and simplify_streams.py.

Boundaries: geoBoundaries CGAZ (Comprehensive Global Administrative Zones) ADM0, 218 features that
partition the land with no overlaps. 198 are countries (shapeType ADM0, shapeGroup the ISO 3166
alpha-3 code). 20 are disputed areas (shapeType DISP) cut out of the countries around them; each is
labelled by its own name (Western Sahara, Aksai Chin, Abyei, West Bank, ...) and flagged disputed,
so no claim is taken. Their shapeGroup is a placeholder number (111-129) except Western Sahara's,
which is its real ISO code ESH, so only that one keeps a code.

One exception, by decision: CGAZ's "No Man's Land" (the 1949 Latrun armistice strip, 52 km2) is
labelled Israel, which administers it, rather than by its own name.

A catchment's land in each country is the area of its intersection with that country, and every
label is a vote by that area: the country holding the most land wins, a tie going to the name
that sorts first.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

__all__ = ['NAME', 'ISO3', 'DISPUTED', 'OVERRIDES', 'load_boundaries', 'split_pieces',
           'planar_km2', 'dominant']

NAME = 'country'
ISO3 = 'countryIso3'
DISPUTED = 'countryDisputed'

# CGAZ shapeName -> (label, iso3, disputed) for the areas not labelled as CGAZ has them
OVERRIDES = {
    "No Man's Land": ('Israel', 'ISR', False),
}

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEGREE = EARTH_RADIUS_KM * np.pi / 180


def load_boundaries(path: Path) -> gpd.GeoDataFrame:
    """The boundaries as one row per label: name, iso3, disputed, geometry (EPSG:4326, valid).

    The gpkg's CRS is an unnamed geographic one on the WGS84 ellipsoid with lon/lat values, so it
    is declared EPSG:4326 rather than transformed. An override that relabels an area as an
    existing country is dissolved into that country's row.
    """
    raw = gpd.read_file(path)
    raw = raw.set_crs(4326, allow_override=True)
    raw['geometry'] = shapely.make_valid(raw.geometry.values)
    disputed = raw['shapeType'].eq('DISP')
    table = pd.DataFrame({
        NAME: raw['shapeName'],
        ISO3: np.where(disputed & ~raw['shapeGroup'].str.fullmatch('[A-Z]{3}'), None, raw['shapeGroup']),
        DISPUTED: disputed,
    })
    for source, (label, iso3, is_disputed) in OVERRIDES.items():
        hit = table[NAME].eq(source)
        if not hit.any():
            raise RuntimeError(f'{path.name} has no area named {source!r} to relabel')
        table.loc[hit, [NAME, ISO3, DISPUTED]] = [label, iso3, is_disputed]
    gdf = gpd.GeoDataFrame(table, geometry=raw.geometry.values, crs=4326)
    gdf = gdf.dissolve(by=[NAME, ISO3, DISPUTED], dropna=False, as_index=False)
    if gdf[NAME].duplicated().any():
        raise RuntimeError(f'{path.name}: a label names two different areas')
    return gdf[[NAME, ISO3, DISPUTED, 'geometry']].sort_values(NAME, ignore_index=True)


def _polygonal(geometry) -> list:
    parts = shapely.get_parts(geometry)
    return [p for p in parts if p.geom_type in ('Polygon', 'MultiPolygon') and not p.is_empty]


def split_pieces(boundaries: gpd.GeoDataFrame, max_vertices: int = 1000) -> gpd.GeoDataFrame:
    """Every label's polygons cut into pieces of at most ``max_vertices``, by quadtree.

    A country is one polygon of up to millions of vertices; tested whole, every catchment that
    touches it pays for all of them. Cut into small rectangles, a catchment meets only the few
    pieces around it, and a catchment entirely within one piece needs no intersection at all.
    """
    labels, pieces = [], []

    def cut(geometry, label: int, depth: int = 0):
        if shapely.get_num_coordinates(geometry) <= max_vertices or depth > 24:
            for part in _polygonal(geometry):
                labels.append(label)
                pieces.append(part)
            return
        x0, y0, x1, y1 = geometry.bounds
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        for box in ((x0, y0, xm, ym), (xm, y0, x1, ym), (x0, ym, xm, y1), (xm, ym, x1, y1)):
            part = shapely.clip_by_rect(geometry, *box)
            if not part.is_empty:
                cut(part, label, depth + 1)

    for label, geometry in enumerate(boundaries.geometry):
        for polygon in shapely.get_parts(geometry):
            cut(polygon, label)
    return gpd.GeoDataFrame({'label': np.asarray(labels, dtype=np.int16)}, geometry=pieces, crs=4326)


def planar_km2(geometries: np.ndarray) -> np.ndarray:
    """Area in km2 of lon/lat geometries, each scaled by the cosine of its own centroid latitude.

    Exact enough for a vote between the pieces of one small catchment; catchments are small
    enough that the latitude barely changes across one.
    """
    latitude = shapely.get_y(shapely.centroid(geometries))
    return shapely.area(geometries) * KM_PER_DEGREE ** 2 * np.cos(np.radians(latitude))


def dominant(keys: np.ndarray, names: np.ndarray, km2: np.ndarray) -> pd.Series:
    """The name holding the most km2 per key, from a long (key, name, km2) table.

    Rows of the same (key, name) are summed first. A tie goes to the name that sorts first.
    """
    table = pd.DataFrame({'key': keys, 'name': names, 'km2': km2})
    table = table.groupby(['key', 'name'], sort=False, observed=True)['km2'].sum().reset_index()
    table = table[table['km2'] > 0].sort_values(['key', 'km2', 'name'], ascending=[True, False, True])
    return table.drop_duplicates('key').set_index('key')['name']
