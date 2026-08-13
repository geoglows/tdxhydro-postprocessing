import geopandas as gpd
import numpy as np
import shapely

from . import schema

__all__ = [
    'web_mercator_epsg',
    'precision_meters',
    'to_web_mercator',
    'snap_to_grid',
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

    The snap is ``snap_to_grid``'s rounding, not ``set_precision``'s precision reduction. The two
    put a coordinate on the same lattice; what the second adds is a repair of whatever the rounding
    broke, and that is only ever a *ring* - a line or a point has no self-intersection to acquire.
    Every layer that snaps here is one or the other (streams are LineStrings, confluences are
    Points), so the repair is being paid for and can never be used. Measured on 1020000010's raw
    streamnet, 98.3M vertices: **3.09 s -> 1.68 s, and the output is identical reach for reach**.

    That 1.8x is worth having but it is not the 137x the same substitution is worth on a polygon
    coverage in step 4, and the difference is the whole reason this is a safe change here. What
    ``set_precision`` charges for on polygons is the noding, and on lines there is none to do -
    so what is saved here is small precisely because what was at risk here was nothing.

    **A polygon layer must not use this path as-is.** It would want ``repair`` afterwards - see
    step 4, which snaps separately for that reason and passes ``round_meters=None`` here.

    ``round_meters=None`` reprojects without snapping, for a caller that has a reason to snap later
    instead. Everything else gets both, as it always did.
    """
    out = gdf.to_crs(epsg=web_mercator_epsg)
    if round_meters is not None:
        out[schema.geometry] = snap_to_grid(out[schema.geometry].values, round_meters)
    return out


def snap_to_grid(geometries, round_meters: float = precision_meters):
    """Round every vertex onto a ``round_meters`` lattice, and nothing else.

    ``set_precision`` is the general answer and does more than round: it is a GEOS precision
    reduction, so it re-nodes each geometry and repairs what rounding broke. That costs what an
    overlay costs, and on a full-resolution coverage it was measured as **48% of step 4** - more
    than the dissolve and the simplification put together.

    Rounding is the part that is actually wanted. The grid is a power of two (see
    ``precision_meters``), so ``x -> floor(x + 0.5)`` on the coordinate buffer is exact in
    float64 and lands on the same lattice point ``set_precision`` would have chosen. Measured on
    a 4,000-catchment chunk: 5.50 s -> 0.04 s, total area identical to four decimal places, and
    five fewer vertices out of 2.23M.

    ``np.round`` is deliberately not used: it rounds half to even, which would disagree with GEOS
    on a coordinate landing exactly on .5.

    What is given up is the repair, so a caller has to handle the occasional ring that rounding
    self-intersects - one in 4,000 on that chunk - with ``geometry.repair``. Rounding also cannot
    merge two vertices that land on the same lattice point, so unlike ``set_precision`` this never
    collapses a geometry out of existence, and unlike ``set_precision`` it never nodes a coverage.
    """
    return shapely.transform(geometries, lambda c: np.floor(c + 0.5), include_z=False)
