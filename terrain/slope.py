"""Slope, aspect and hillshade from an elevation grid.

Rows of the grid run north to south and columns run west to east. All inputs
are in meters: `z` is height, `cell_x`/`cell_y` are cell width/height.
"""

import numpy as np
import pandas as pd

# Default slope class boundaries in degrees. These are common screening
# ranges, not an official standard; the app lets the user change them.
# (Many landslide guides start to worry above roughly 25 degrees.)
DEFAULT_SLOPE_BREAKS = (5.0, 15.0, 25.0, 35.0)
SLOPE_CLASS_NAMES = ("Flat", "Gentle", "Moderate", "Steep", "Very steep")


def surface_gradients(z: np.ndarray, cell_x: float, cell_y: float):
    """How fast height changes going east and going north.

    Returns (dz/d_east, dz/d_north), each in meters of rise per meter.
    np.gradient compares each cell with its neighbors on both sides
    ("central differences") and divides by the distance between them.
    """
    if z.ndim != 2 or min(z.shape) < 2:
        raise ValueError("Need a grid at least 2 x 2 cells to compute slope.")
    dz_drow, dz_dcol = np.gradient(z, cell_y, cell_x)
    # Central differences skip the cell itself, so a missing cell between two
    # good neighbors would still get a value. Keep missing cells missing.
    missing = np.isnan(z)
    dz_drow[missing] = np.nan
    dz_dcol[missing] = np.nan
    # Moving down one row means moving SOUTH, so the north change is negative.
    return dz_dcol, -dz_drow


def slope_degrees(z: np.ndarray, cell_x: float, cell_y: float) -> np.ndarray:
    """Steepness of the ground at each cell, in degrees (0 = flat, 90 = cliff).

    The total rise per meter is the length of the gradient vector:
        rise = sqrt((dz/dx)^2 + (dz/dy)^2)
    and the angle is arctan(rise), like the angle of a ramp.
    """
    dz_de, dz_dn = surface_gradients(z, cell_x, cell_y)
    rise = np.hypot(dz_de, dz_dn)
    return np.degrees(np.arctan(rise))


def aspect_degrees(z: np.ndarray, cell_x: float, cell_y: float) -> np.ndarray:
    """Compass direction the slope faces (downhill direction), in degrees.

    0 = faces north, 90 = east, 180 = south, 270 = west. Flat cells get NaN
    because flat ground doesn't face any direction.
    """
    dz_de, dz_dn = surface_gradients(z, cell_x, cell_y)
    # Downhill is opposite to the uphill gradient. arctan2(east, north) gives a
    # compass angle measured clockwise from north.
    aspect = np.degrees(np.arctan2(-dz_de, -dz_dn)) % 360.0
    aspect[(dz_de == 0) & (dz_dn == 0)] = np.nan  # (NaN cells stay NaN)
    return aspect


def hillshade(
    z: np.ndarray,
    cell_x: float,
    cell_y: float,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Shaded-relief image (0 = dark, 1 = fully lit) as if lit by a low sun.

    azimuth: compass direction the light comes FROM (315 = northwest).
    altitude: sun angle above the horizon in degrees.
    z_factor: vertical exaggeration so gentle bumps are easier to see.

    Brightness = how directly the ground faces the sun:
        cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth - aspect)
    """
    dz_de, dz_dn = surface_gradients(z * z_factor, cell_x, cell_y)
    slope = np.arctan(np.hypot(dz_de, dz_dn))
    aspect = np.arctan2(-dz_de, -dz_dn)  # radians, clockwise from north
    zenith = np.radians(90.0 - altitude)  # angle of the sun from straight up
    sun = np.radians(azimuth)
    shade = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(
        sun - aspect
    )
    return np.clip(shade, 0.0, 1.0)


def cast_shadow_from_south(z: np.ndarray, cell_y: float, altitude: float) -> np.ndarray:
    """True where something to the SOUTH blocks a sun shining from due south.

    The sun is `altitude` degrees above the horizon. Looking from a cell
    toward the sun, a cell k rows to the south (k x cell_y meters away) blocks
    the sun if it is higher than the sun ray at that distance:
        height there > height here + distance x tan(altitude)
    We check every distance up to the point where even the highest spot in
    the map can no longer block the sun.
    """
    rows = z.shape[0]
    rise_per_m = np.tan(np.radians(altitude))
    relief = np.nanmax(z) - np.nanmin(z)
    max_steps = min(rows - 1, int(np.ceil(relief / (cell_y * rise_per_m))) if rise_per_m > 0 else rows - 1)
    shadow = np.zeros(z.shape, dtype=bool)
    for k in range(1, max_steps + 1):
        south = np.full(z.shape, np.nan)
        south[:-k] = z[k:]  # the cell k rows further south (row numbers grow southward)
        with np.errstate(invalid="ignore"):
            shadow |= south > z + k * cell_y * rise_per_m
    return shadow


def classify_slope(slope: np.ndarray, breaks=DEFAULT_SLOPE_BREAKS) -> np.ndarray:
    """Put each cell into a slope class: 0, 1, 2, ... (len(breaks) classes + 1).

    A cell goes in class k if breaks[k-1] <= slope < breaks[k].
    Cells with no data get -1.
    """
    breaks = np.asarray(breaks, dtype=float)
    if np.any(np.diff(breaks) <= 0):
        raise ValueError("Slope class breaks must be in increasing order.")
    classes = np.digitize(slope, breaks)
    classes[~np.isfinite(slope)] = -1
    return classes


def slope_class_table(
    slope: np.ndarray, breaks=DEFAULT_SLOPE_BREAKS, cell_area_m2: float = 1.0, names=SLOPE_CLASS_NAMES
) -> pd.DataFrame:
    """Area and percent of the mapped ground in each slope class."""
    classes = classify_slope(slope, breaks)
    edges = [0.0, *breaks, 90.0]
    total = int((classes >= 0).sum())
    rows = []
    for k in range(len(breaks) + 1):
        count = int((classes == k).sum())
        rows.append(
            {
                "Class": names[k] if k < len(names) else f"Class {k}",
                "Range (deg)": f"{edges[k]:g} - {edges[k + 1]:g}",
                "Area (m²)": count * cell_area_m2,
                "Percent": 100.0 * count / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plan_curvature(z: np.ndarray, cell_x: float, cell_y: float) -> np.ndarray:
    """How the contour lines bend: do flow lines come together or spread apart?

    Negative = concave, like a valley or hollow: water flowing downhill is
               squeezed together, so the ground collects water.
    Positive = convex, like a ridge or a mound: water spreads out.
    Units are 1/meters (1 / radius of the bend).

    Formula (Evans/Krcho), with p, q = first derivatives (east, north) and
    r, s, t = second derivatives (east-east, east-north, north-north):
        plan curvature = -(q²·r - 2·p·q·s + p²·t) / (p² + q²)^1.5
    On flat ground (p = q = 0) there are no contour lines, so the value is 0.
    """
    p, q = surface_gradients(z, cell_x, cell_y)  # dz/d_east, dz/d_north
    r, s = surface_gradients(p, cell_x, cell_y)  # d²z/d_east², d²z/(d_east d_north)
    _, t = surface_gradients(q, cell_x, cell_y)  # d²z/d_north²
    steepness_sq = p**2 + q**2
    with np.errstate(invalid="ignore", divide="ignore"):
        curv = -(q**2 * r - 2 * p * q * s + p**2 * t) / steepness_sq**1.5
    curv[steepness_sq < 1e-12] = 0.0  # flat: no contour lines to bend
    curv[~np.isfinite(z)] = np.nan
    return curv
