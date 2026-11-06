"""Accuracy checker: compare model heights with hand-measured check points.

The idea: at each check point we know the true height (measured by hand).
We read the height the drone model gives at the same spot and look at the
differences.

Because the model's absolute height is off by an unknown amount (no ground
control points), we first remove the *average* difference ("mean offset").
What is left over shows how well the model captures the *shape* of the ground.
"""

import math

import numpy as np
import pandas as pd

from terrain.io import Raster

METERS_PER_INCH = 0.0254
REQUIRED_COLUMNS = ("id", "x", "y", "measured_z")


# ---------------------------------------------------------------------------
# Getting the check points
# ---------------------------------------------------------------------------
def read_checkpoints(file) -> pd.DataFrame:
    """Read a check point CSV with columns id, x, y, measured_z.

    `file` can be a path or an open file. Column names are not case-sensitive.
    Rows with a missing or non-numeric x, y or measured_z are dropped.
    """
    df = pd.read_csv(file)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The CSV is missing column(s): {', '.join(missing)}. "
            f"It needs: {', '.join(REQUIRED_COLUMNS)}."
        )
    df = df[list(REQUIRED_COLUMNS)].copy()
    for c in ("x", "y", "measured_z"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["x", "y", "measured_z"]).reset_index(drop=True)


def grid_points(
    first_x, first_y, end_x, end_y, cols, rows, spacing, side="left", order="rows", start_id=1
) -> pd.DataFrame:
    """Map coordinates of every point in a rectangular grid of markers.

    first_x, first_y: map coordinates of the first point (column 1, row 1).
    end_x, end_y: map coordinates of the last point of that same row (column `cols`, row 1).
    spacing: distance between rows, in meters.
    side: "left" or "right" - which side of the first row the other rows are
        on, when standing on the first point looking toward the end of the row.
    order: how the points are numbered:
        "rows"  = 1, 2, 3 ... along each row, always starting at column 1;
        "snake" = zig-zag, every other row runs back the other way.

    The two corners can be at any angle, so the grid does not have to line up
    with north.
    """
    if cols < 2 or rows < 1:
        raise ValueError("The grid needs at least 2 columns and 1 row.")
    row_length = math.hypot(end_x - first_x, end_y - first_y)
    if row_length == 0:
        raise ValueError("The two corner points are the same point.")

    # Unit vector pointing along a row (from first point to end of row).
    ux, uy = (end_x - first_x) / row_length, (end_y - first_y) / row_length
    # Unit vector pointing from one row to the next: the along-row direction
    # turned 90 degrees to the left (-uy, ux) or to the right (uy, -ux).
    vx, vy = (-uy, ux) if side == "left" else (uy, -ux)
    col_step = row_length / (cols - 1)  # column spacing comes from the corners

    records = []
    number = start_id
    for r in range(rows):
        col_order = range(cols)
        if order == "snake" and r % 2 == 1:
            col_order = reversed(range(cols))
        for c in col_order:
            x = first_x + c * col_step * ux + r * spacing * vx
            y = first_y + c * col_step * uy + r * spacing * vy
            records.append({"id": number, "x": x, "y": y, "col": c + 1, "row": r + 1})
            number += 1
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Reading model heights at the points
# ---------------------------------------------------------------------------
def sample_bilinear(raster: Raster, x, y) -> np.ndarray:
    """Model height at map coordinates (x, y) using bilinear interpolation.

    A point usually falls between cell centers. Bilinear interpolation takes the
    4 surrounding cell centers and blends them, giving more weight to closer
    ones (like a weighted average). Points outside the map, or next to a
    missing cell, get NaN.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    t = raster.transform
    rows, cols = raster.shape

    # Fractional column/row, measured from the center of the top-left cell.
    col_f = (x - t.c) / t.a - 0.5
    row_f = (y - t.f) / t.e - 0.5

    result = np.full(x.shape, np.nan)
    inside = (col_f >= 0) & (col_f <= cols - 1) & (row_f >= 0) & (row_f <= rows - 1)
    if rows < 2 or cols < 2 or not inside.any():
        return result

    cf, rf = col_f[inside], row_f[inside]
    # Top-left of the 4 surrounding cells (clipped so a point exactly on the
    # last row/column still has a cell to its right/below).
    c0 = np.clip(np.floor(cf).astype(int), 0, cols - 2)
    r0 = np.clip(np.floor(rf).astype(int), 0, rows - 2)
    fx, fy = cf - c0, rf - r0  # how far across the cell (0 to 1)

    z = raster.data
    top = z[r0, c0] * (1 - fx) + z[r0, c0 + 1] * fx
    bottom = z[r0 + 1, c0] * (1 - fx) + z[r0 + 1, c0 + 1] * fx
    result[inside] = top * (1 - fy) + bottom * fy  # NaN if any corner is NaN
    return result


# ---------------------------------------------------------------------------
# Errors and statistics
# ---------------------------------------------------------------------------
def align_errors(model_z, measured_z):
    """Errors after removing the mean offset.

    difference_i = model_z_i - measured_z_i
    mean_offset  = average of all differences (the unknown height shift)
    error_i      = difference_i - mean_offset

    Returns (errors, mean_offset). Points with a NaN height give NaN errors and
    are left out of the mean.
    """
    diff = np.asarray(model_z, dtype=float) - np.asarray(measured_z, dtype=float)
    if not np.isfinite(diff).any():
        return diff, float("nan")
    offset = float(np.nanmean(diff))
    return diff - offset, offset


def error_stats(errors) -> dict:
    """Summary statistics for a set of errors (meters). NaNs are ignored.

    mean   = average error (0 after mean alignment, shown as a check)
    std    = standard deviation: typical spread of the errors around the mean
    rmse   = root mean square error: sqrt(average of error squared)
    max_abs = biggest error, ignoring its sign
    le90   = 90% of points have an error smaller than this ("linear error 90%")
    """
    e = np.asarray(errors, dtype=float)
    e = e[np.isfinite(e)]
    n = e.size
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "rmse": np.nan, "max_abs": np.nan, "le90": np.nan}
    return {
        "n": n,
        "mean": float(e.mean()),
        "std": float(e.std(ddof=1)) if n > 1 else 0.0,  # sample standard deviation
        "rmse": float(np.sqrt(np.mean(e**2))),
        "max_abs": float(np.max(np.abs(e))),
        "le90": float(np.percentile(np.abs(e), 90)),
    }


def stats_table(named_stats: dict) -> pd.DataFrame:
    """Turn {"label": stats_dict, ...} into a table in meters and inches."""
    labels = {
        "mean": "Mean error",
        "std": "Standard deviation",
        "rmse": "RMSE",
        "max_abs": "Max absolute error",
        "le90": "LE90",
    }
    rows = []
    for key, text in labels.items():
        row = {"Statistic": text}
        for name, s in named_stats.items():
            row[f"{name} (m)"] = s[key]
            row[f"{name} (in)"] = s[key] / METERS_PER_INCH
        rows.append(row)
    count = {"Statistic": "Number of points"}
    for name, s in named_stats.items():
        count[f"{name} (m)"] = s["n"]
        count[f"{name} (in)"] = s["n"]
    rows.append(count)
    return pd.DataFrame(rows)


def evenly_spread_subset(x, y, size: int) -> np.ndarray:
    """Pick `size` points that are spread out evenly. Returns their positions.

    Method ("farthest point sampling"): start with the first point, then keep
    adding whichever point is farthest from all points picked so far. This
    spreads the picks over the whole area instead of bunching them up.
    """
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n = len(pts)
    size = max(0, min(int(size), n))
    if size == 0:
        return np.array([], dtype=int)
    chosen = [0]
    # Distance from every point to its nearest chosen point.
    nearest = np.hypot(*(pts - pts[0]).T)
    while len(chosen) < size:
        nxt = int(np.argmax(nearest))
        chosen.append(nxt)
        nearest = np.minimum(nearest, np.hypot(*(pts - pts[nxt]).T))
    return np.sort(np.array(chosen))


def compare_points(points: pd.DataFrame, models: dict, slope_raster: Raster | None = None) -> tuple:
    """Sample each model at the check points and work out the errors.

    points: DataFrame with id, x, y, measured_z.
    models: {"DTM": raster, "DSM": raster, ...}
    Returns (table, offsets): one row per point with each model's height,
    raw difference and aligned error; and the mean offset for each model.
    """
    table = points[list(REQUIRED_COLUMNS)].copy()
    offsets = {}
    for name, raster in models.items():
        model_z = sample_bilinear(raster, table["x"], table["y"])
        errors, offset = align_errors(model_z, table["measured_z"])
        table[f"{name}_z"] = model_z
        table[f"{name}_diff"] = model_z - table["measured_z"]  # before alignment
        table[f"{name}_error"] = errors  # after removing the mean offset
        offsets[name] = offset
    if slope_raster is not None:
        table["slope_deg"] = sample_bilinear(slope_raster, table["x"], table["y"])
    return table, offsets


# ---------------------------------------------------------------------------
# Typing in hand-measured heights
# ---------------------------------------------------------------------------
# How many meters are in one of each unit.
SKIP_WORDS = {"x", "-", "na", "n/a", "nan", "skip"}
UNIT_TO_METERS = {"meters": 1.0, "centimeters": 0.01, "millimeters": 0.001, "inches": 0.0254, "feet": 0.3048}


def parse_height_list(text: str) -> list[float]:
    """Turn pasted/typed text like "12.5, 13.0\n11.75" into a list of numbers.

    Numbers can be separated by commas, spaces, tabs, semicolons or new lines.
    A marker that couldn't be measured can be written as "x", "-", "na" or
    "skip"; it becomes NaN (missing) so the rest stay in the right order.
    Raises ValueError naming the first thing that isn't a number.
    """
    values = []
    for token in text.replace(",", " ").replace(";", " ").split():
        if token.lower() in SKIP_WORDS:
            values.append(float("nan"))
            continue
        try:
            values.append(float(token))
        except ValueError:
            raise ValueError(f"'{token}' is not a number (entry {len(values) + 1}).") from None
    return values


def measured_to_meters(values, unit: str = "meters", measured_downward: bool = False) -> np.ndarray:
    """Convert hand measurements to heights in meters, with up = positive.

    unit: the unit the numbers were written down in (see UNIT_TO_METERS).
    measured_downward: True if each number is a distance measured DOWN from a
        level line or reference (a bigger number means LOWER ground). These are
        flipped to negative so that higher ground has a bigger height, which
        is how the drone model works.
    """
    meters = np.asarray(values, dtype=float) * UNIT_TO_METERS[unit]
    return -meters if measured_downward else meters
