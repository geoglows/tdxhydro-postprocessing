import geopandas as gpd
import shapely

from . import schema

__all__ = [
    'to_mapping_geometry',
]

web_mercator_epsg = 3857
mapping_simplify_meters = 1.0
mapping_precision_meters = 1.0


def to_mapping_geometry(
        gdf: gpd.GeoDataFrame,
        simplify_meters: float = mapping_simplify_meters,
        round_meters: float = mapping_precision_meters,
) -> gpd.GeoDataFrame:
    """Reproject full-resolution streams to EPSG:3857, Douglas-Peucker simplify to
    ``simplify_meters`` tolerance, and round coordinates to ``round_meters``. Produces the
    'mapping' geometry the tile step consumes -- roughly half the vertices of the source at
    sub-metre error. Set ``simplify_meters=0`` to keep full resolution."""
    mapping = gdf.to_crs(epsg=web_mercator_epsg)
    geom = mapping[schema.geometry].values
    if simplify_meters:
        geom = shapely.simplify(geom, simplify_meters, preserve_topology=True)
    mapping[schema.geometry] = shapely.set_precision(geom, round_meters)
    return mapping
