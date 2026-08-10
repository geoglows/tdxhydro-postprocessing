import geopandas as gpd
import shapely

from . import schema

__all__ = [
    'web_mercator_epsg',
    'precision_meters',
    'to_web_mercator',
]

web_mercator_epsg = 3857

# The grid every published geometry is snapped to. It is 1 m rather than something finer for
# a reason that is entirely about file size: a power-of-two metre grid is exactly
# representable in float64, so snapping to it leaves ~30 trailing zero mantissa bits in every
# coordinate, which is what the BYTE_STREAM_SPLIT encoding in parquet.py needs in order to
# pay off. In degrees there is no equivalent - 1e-7 is not a power of two, so snapping to it
# zeroes nothing and the same files come out a third larger.
#
# The cost is bounded and uniform: snapping to a 1 m grid moves a vertex at most sqrt(2)/2 =
# 0.71 m. Mercator metres are inflated by 1/cos(lat), so the true ground error is that times
# cos(lat) - worst at the equator, finer toward the poles. The TDX-Hydro source is a 1/9
# arcsec DEM, ~3.4 m, so 0.71 m is well inside the grid the data was derived from.
precision_meters = 1.0


def to_web_mercator(gdf: gpd.GeoDataFrame, round_meters: float = precision_meters) -> gpd.GeoDataFrame:
    """Reproject to EPSG:3857 and snap coordinates to a ``round_meters`` grid.

    Every published geometry goes through here, so the whole dataset shares one CRS and one
    grid. Reaches north of 85.05 degrees would fall outside what web mercator can represent;
    the network's northernmost outlet is at 80.3, so there is nothing near that edge.
    """
    out = gdf.to_crs(epsg=web_mercator_epsg)
    out[schema.geometry] = shapely.set_precision(out[schema.geometry].values, round_meters)
    return out
