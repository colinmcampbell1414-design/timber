"""Use a better ground model (DTM) with a survey's surface model (DSM).

Photogrammetry can't see the ground under leaves, so a leaf-on DTM is poor
under canopy. A DTM from a leaf-off flight, or from public lidar, is better.
This module puts such a DTM onto the survey's grid, converts feet to meters,
lines it up vertically on stable open ground, and compares DTM sources.
"""

import numpy as np
import pandas as pd

from terrain import io, storm
from terrain import slope as slope_mod

# Meters in one unit of height.
UNIT_FACTORS = {"meters": 1.0, "feet": 0.3048, "US survey feet": 1200 / 3937}  # US survey foot = 1200/3937 m exactly
UNIT_CHOICES = ["auto (read from the file)"] + list(UNIT_FACTORS)


def vertical_factor(crs, units: str = "auto") -> float:
    """Meters per height unit of a raster.

    "auto" reads the coordinate system's length unit (rasterio reports it as
    meters per unit, e.g. 0.3048006 for the US survey foot); anything else is
    the manual override.
    """
    if units in UNIT_FACTORS:
        return UNIT_FACTORS[units]
    try:
        if crs is not None and crs.is_projected:
            return float(crs.linear_units_factor[1])
    except Exception:
        pass
    return 1.0


def to_grid(raster: io.Raster, target: io.Raster, units: str = "auto") -> io.Raster:
    """Put a DTM from somewhere else onto the survey's grid, in meters.

    The raster is reprojected (its coordinate system may differ) and resampled
    with bilinear interpolation (a blend of the 4 nearest cells), then its
    heights are multiplied by the meters-per-unit factor.
    """
    moved = io.resample_to_grid(raster, target, resampling="bilinear")
    return io.Raster(data=moved.data * vertical_factor(raster.crs, units), transform=target.transform, crs=target.crs)


def align_dtm(other: np.ndarray, reference: np.ndarray, dsm: np.ndarray, cell: float, building: np.ndarray = None):
    """Line another DTM up vertically with the survey's own DTM.

    Two surveys have different absolute heights (and a slight tilt). On stable
    open ground (canopy under 0.5 m with BOTH ground models, smooth, slope
    under 10 degrees, not a building) a plane is fitted to (other - reference)
    and subtracted, the same plane fit the storm check uses.
    Returns (aligned DTM, report with the offset before and the leftover RMS after).
    """
    rough = np.nanmedian(other - reference)  # so the canopy heights below make sense before the real fit
    chm_ref = storm._chm(dsm, reference)
    chm_other = storm._chm(dsm, other - rough)
    slope_deg = slope_mod.slope_degrees(reference, cell, cell)
    stable = storm.stable_ground(chm_ref, chm_other, slope_deg, building, surfaces=(dsm, dsm), cell=cell)
    diff = other - reference
    before = storm._stable_rms(diff - np.nanmedian(diff[stable]) if stable.any() else diff, stable)
    plane, report = storm.fit_plane(diff, stable, cell)
    aligned = other - plane
    report.update({"offset_before_m": report.pop("offset_m"), "rms_before_m": before, "rms_after_m": report.pop("rms_m"),
                   "stable_share": float(stable.mean())})
    return aligned, report


def compare_dtms(sources: dict, reference: str, dsm: np.ndarray, cell: float, min_tree: float = 5.0) -> pd.DataFrame:
    """Each DTM source against the reference DTM, split into open ground and under canopy.

    sources: {"name": DTM grid (already aligned)}. Canopy = where the reference
    canopy height (DSM - reference DTM) is tree-high. Also the canopy heights
    each DTM produces: a ground model that follows the treetops gives trees
    that look too short.
    """
    ref = sources[reference]
    chm_ref = storm._chm(dsm, ref)
    canopy = np.nan_to_num(chm_ref) >= min_tree
    open_ground = np.nan_to_num(chm_ref, nan=99) < 0.5
    rows = []
    for name, dtm in sources.items():
        diff = dtm - ref
        chm = storm._chm(dsm, dtm)

        def stat(mask):
            v = diff[mask & np.isfinite(diff)]
            return (float(np.mean(v)), float(np.sqrt(np.mean(v ** 2)))) if len(v) else (np.nan, np.nan)

        (open_mean, open_rms), (can_mean, can_rms) = stat(open_ground), stat(canopy)
        trees = chm[np.nan_to_num(chm) >= min_tree]
        rows.append({"DTM source": name + (" (reference)" if name == reference else ""),
                     "open ground: mean difference (m)": open_mean, "open ground: RMS (m)": open_rms,
                     "under canopy: mean difference (m)": can_mean, "under canopy: RMS (m)": can_rms,
                     "median canopy height (m)": float(np.median(trees)) if len(trees) else np.nan,
                     "tree-high area (%)": float(100 * (np.nan_to_num(chm) >= min_tree).mean())})
    return pd.DataFrame(rows)


def heights_at(x, y, dsm: np.ndarray, dtm: np.ndarray, transform, radius_m: float = 2.0) -> np.ndarray:
    """Tree height at given positions with a given DTM: the highest (DSM - DTM)
    within `radius_m` of each position. Used to compare field heights with every DTM source."""
    from scipy import ndimage

    chm = np.nan_to_num(storm._chm(dsm, dtm))
    cell = abs(transform.a)
    top = ndimage.maximum_filter(chm, footprint=storm._disk(round(radius_m / cell)), mode="nearest")
    r = np.clip(((np.asarray(y, dtype=float) - transform.f) / transform.e).astype(int), 0, chm.shape[0] - 1)
    c = np.clip(((np.asarray(x, dtype=float) - transform.c) / transform.a).astype(int), 0, chm.shape[1] - 1)
    return top[r, c]
