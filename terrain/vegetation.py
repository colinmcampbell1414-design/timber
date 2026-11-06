"""Vegetation indices from the orthophoto's red, green and blue colors.

Healthy plants reflect more green than red or blue. These indices turn that
into a number for each cell. Inputs are rows x cols x 3 arrays with colors
from 0 to 1 (NaN where there is no photo).
"""

import numpy as np


def _channels(rgb: np.ndarray):
    rgb = np.asarray(rgb, dtype="float64")
    return rgb[..., 0], rgb[..., 1], rgb[..., 2]


def vari(rgb: np.ndarray) -> np.ndarray:
    """Visible Atmospherically Resistant Index: VARI = (G - R) / (G + R - B).

    Above 0 = greener than red (living plants); below 0 = redder/browner
    (dry grass, soil, dead leaves). Where the bottom of the fraction is
    almost 0 the result would blow up, so it is set to NaN ("safe division"),
    and results are limited to the range -1 to 1.
    """
    r, g, b = _channels(rgb)
    denominator = g + r - b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (g - r) / denominator
    out[np.abs(denominator) < 1e-6] = np.nan
    return np.clip(out, -1.0, 1.0)


def saturation(rgb: np.ndarray) -> np.ndarray:
    """How colorful each cell is: (brightest channel - dimmest) / brightest, 0 to 1.

    Gray, white and black things (roofs, pavement, gravel) are near 0. Plants
    are colorful in every season: green in summer, red/orange/yellow in fall.
    Black cells (brightest = 0) give NaN instead of dividing by zero.
    """
    rgb = np.asarray(rgb, dtype="float64")
    hi, lo = rgb.max(axis=-1), rgb.min(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (hi - lo) / hi
    out[hi < 1e-6] = np.nan
    return out


def exg(rgb: np.ndarray) -> np.ndarray:
    """Excess Green: ExG = 2G - R - B.

    How much greener a cell is than it is red and blue. Plants are well above
    0; bare soil, pavement and dead material are near 0 or below.
    (No division, so it is always safe.)
    """
    r, g, b = _channels(rgb)
    return 2 * g - r - b
