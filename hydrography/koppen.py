"""The Koppen-Geiger legend and the area-vote that picks a class, shared by the classifier and simplify_streams.py.

raw_processing/koppen_classifications.py records each raw catchment's land area in every class (one column per
class, ``AREA_COLUMNS``). simplify_streams.py sums those rows over the raw catchments that make up a published
reach's catchment and calls ``dominant`` on the sums, so a merged reach is labelled by all the land
it drains directly, not by whichever source reach happened to keep its id.
"""
import numpy as np
import pandas as pd

__all__ = ['CLASS_CODES', 'CLASS_COUNT', 'AREA_COLUMNS', 'GROUPS', 'dominant', 'group_fractions']

# the raster's values 1-30 in Beck et al. (2023) legend.txt order; index 0 is ocean/nodata
CLASS_CODES = (None,
               'Af', 'Am', 'Aw', 'BWh', 'BWk', 'BSh', 'BSk', 'Csa', 'Csb', 'Csc',
               'Cwa', 'Cwb', 'Cwc', 'Cfa', 'Cfb', 'Cfc', 'Dsa', 'Dsb', 'Dsc', 'Dsd',
               'Dwa', 'Dwb', 'Dwc', 'Dwd', 'Dfa', 'Dfb', 'Dfc', 'Dfd', 'ET', 'EF')
CLASS_COUNT = len(CLASS_CODES) - 1

# the per-class land area columns of the parts files, km2, in raster-value order 1..30
AREA_COLUMNS = tuple(f'km2_{code}' for code in CLASS_CODES[1:])

# the five main groups, the first letter of every class
GROUPS = ('A', 'B', 'C', 'D', 'E')


def dominant(areas: np.ndarray) -> pd.array:
    """The class with the most land per row of an (n, 30) area array, as letters; null with no land.

    argmax keeps the lower legend number on a tie.
    """
    codes = np.array(CLASS_CODES[1:], dtype=object)
    winner = codes[np.argmax(areas, axis=1)] if len(areas) else codes[:0]
    return pd.array(np.where(areas.sum(axis=1) > 0, winner, None), dtype='string')


def group_fractions(areas: np.ndarray) -> np.ndarray:
    """Each row's land fraction in each main group, an (n, 5) array in ``GROUPS`` order; NaN with no land."""
    first = np.array([code[0] for code in CLASS_CODES[1:]])
    by_group = np.stack([areas[:, first == g].sum(axis=1) for g in GROUPS], axis=1)
    total = by_group.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(total > 0, by_group / total, np.nan)
