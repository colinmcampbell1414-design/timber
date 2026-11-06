"""Accuracy checker tests using synthetic DEMs and points with known errors."""

import io as std_io
import math

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin

from terrain import accuracy
from terrain.io import Raster

LEFT, TOP, CELL = 500000.0, 4000000.0, 0.05


def plane_raster(a=0.3, b=-0.2, c=100.0, rows=60, cols=80):
    """A tilted plane z = a*(x - LEFT) + b*(y - TOP) + c, stored at cell centers."""
    transform = from_origin(LEFT, TOP, CELL, CELL)
    col_x = LEFT + (np.arange(cols) + 0.5) * CELL
    row_y = TOP - (np.arange(rows) + 0.5) * CELL
    xx, yy = np.meshgrid(col_x, row_y)
    data = a * (xx - LEFT) + b * (yy - TOP) + c
    return Raster(data=data, transform=transform, crs="EPSG:32618")


def plane_z(x, y, a=0.3, b=-0.2, c=100.0):
    return a * (np.asarray(x) - LEFT) + b * (np.asarray(y) - TOP) + c


def test_bilinear_is_exact_on_a_plane():
    r = plane_raster()
    rng = np.random.default_rng(0)
    # Random points well inside the map (not at cell centers).
    x = LEFT + rng.uniform(0.1, 3.8, 50)
    y = TOP - rng.uniform(0.1, 2.8, 50)
    np.testing.assert_allclose(accuracy.sample_bilinear(r, x, y), plane_z(x, y), atol=1e-9)


def test_bilinear_outside_and_nodata_give_nan():
    r = plane_raster()
    r.data[10, 10] = np.nan
    x_hole = LEFT + 10.5 * CELL  # right on the missing cell's center
    y_hole = TOP - 10.5 * CELL
    out = accuracy.sample_bilinear(r, [LEFT - 1.0, x_hole, LEFT + 1.0], [TOP - 1.0, y_hole, TOP - 1.0])
    assert np.isnan(out[0])  # outside the map
    assert np.isnan(out[1])  # touches a missing cell
    assert np.isfinite(out[2])


def test_constant_offset_is_fully_removed():
    # The model is exactly the ground, just shifted up by 1.234 m (like GPS error).
    r = plane_raster()
    x = LEFT + np.linspace(0.5, 3.5, 20)
    y = TOP - np.linspace(0.5, 2.5, 20)
    measured = plane_z(x, y) - 1.234
    points = pd.DataFrame({"id": range(20), "x": x, "y": y, "measured_z": measured})

    table, offsets = accuracy.compare_points(points, {"DTM": r})
    assert offsets["DTM"] == pytest.approx(1.234)
    np.testing.assert_allclose(table["DTM_error"], 0.0, atol=1e-9)
    stats = accuracy.error_stats(table["DTM_error"])
    assert stats["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_known_errors_give_exact_rmse():
    # Errors of +0.03, -0.03, +0.04, -0.04 m on top of a 5 m offset.
    # Their mean is 0, so after alignment they stay the same:
    # RMSE = sqrt((0.03² + 0.03² + 0.04² + 0.04²) / 4) = sqrt(0.00125)
    r = plane_raster()
    true_errors = np.array([0.03, -0.03, 0.04, -0.04])
    x = LEFT + np.array([0.5, 1.5, 2.5, 3.5])
    y = TOP - np.array([0.5, 1.0, 1.5, 2.0])
    measured = plane_z(x, y) - 5.0 - true_errors  # model - measured = 5 + error
    points = pd.DataFrame({"id": [1, 2, 3, 4], "x": x, "y": y, "measured_z": measured})

    table, offsets = accuracy.compare_points(points, {"DTM": r})
    assert offsets["DTM"] == pytest.approx(5.0)
    np.testing.assert_allclose(table["DTM_error"], true_errors, atol=1e-9)

    s = accuracy.error_stats(table["DTM_error"])
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(0.0, abs=1e-12)
    assert s["rmse"] == pytest.approx(math.sqrt(0.00125))
    assert s["max_abs"] == pytest.approx(0.04)
    assert s["std"] == pytest.approx(np.std(true_errors, ddof=1))


def test_le90_and_stats_ignore_nan():
    errors = np.array([*np.arange(1, 11) / 100.0, np.nan])  # 0.01 ... 0.10
    s = accuracy.error_stats(errors)
    assert s["n"] == 10
    assert s["le90"] == pytest.approx(np.percentile(np.arange(1, 11) / 100.0, 90))


def test_read_checkpoints_accepts_any_case_and_drops_bad_rows():
    csv = "ID, X ,Y,Measured_Z\n1,10,20,0.5\n2,11,21,\n3,12,22,0.7\n"
    df = accuracy.read_checkpoints(std_io.StringIO(csv))
    assert list(df.columns) == ["id", "x", "y", "measured_z"]
    assert df["id"].tolist() == [1, 3]


def test_read_checkpoints_reports_missing_columns():
    with pytest.raises(ValueError, match="measured_z"):
        accuracy.read_checkpoints(std_io.StringIO("id,x,y\n1,2,3\n"))


def test_grid_points_17_by_6_layout():
    # Grid along the east direction, rows going north (to the left), 0.5 m spacing.
    g = accuracy.grid_points(100.0, 200.0, 108.0, 200.0, cols=17, rows=6, spacing=0.5, side="left")
    assert len(g) == 102
    assert g["id"].tolist() == list(range(1, 103))
    last = g.iloc[-1]
    assert (last["x"], last["y"]) == pytest.approx((108.0, 202.5))
    assert g.iloc[17][["x", "y"]].tolist() == pytest.approx([100.0, 200.5])  # start of row 2


def test_grid_points_rotated_and_snake_order():
    # Row runs north; rows step to the right (east). Snake numbering.
    g = accuracy.grid_points(0.0, 0.0, 0.0, 2.0, cols=3, rows=2, spacing=1.0, side="right", order="snake")
    np.testing.assert_allclose(
        g[["x", "y"]].values, [[0, 0], [0, 1], [0, 2], [1, 2], [1, 1], [1, 0]], atol=1e-12
    )


def test_evenly_spread_subset():
    g = accuracy.grid_points(0.0, 0.0, 8.0, 0.0, cols=17, rows=6, spacing=0.5)
    idx = accuracy.evenly_spread_subset(g["x"], g["y"], 30)
    assert len(idx) == 30 and len(set(idx)) == 30
    # Same answer every time (no randomness).
    np.testing.assert_array_equal(idx, accuracy.evenly_spread_subset(g["x"], g["y"], 30))
    # Spread out: the subset should reach all four corners of the grid area.
    sub = g.iloc[idx]
    assert sub["x"].min() == 0.0 and sub["x"].max() == 8.0
    assert sub["y"].min() == 0.0 and sub["y"].max() == 2.5
    # Asking for more points than exist just returns all of them.
    assert len(accuracy.evenly_spread_subset(g["x"], g["y"], 500)) == 102


def test_parse_height_list_handles_separators_and_skips():
    values = accuracy.parse_height_list("12.5, 13\n-1.25;x\t7")
    assert values[:3] == [12.5, 13.0, -1.25]
    assert math.isnan(values[3])  # "x" = marker that couldn't be measured
    assert values[4] == 7.0


def test_parse_height_list_reports_bad_entry():
    with pytest.raises(ValueError, match="'5cm' is not a number"):
        accuracy.parse_height_list("1 2 12.5 5cm")


def test_measured_to_meters_units_and_direction():
    np.testing.assert_allclose(accuracy.measured_to_meters([100, 250], "centimeters"), [1.0, 2.5])
    np.testing.assert_allclose(accuracy.measured_to_meters([12], "inches"), [0.3048])
    # Measured DOWN from a level line: 10 cm down is higher ground than 30 cm down.
    down = accuracy.measured_to_meters([10, 30], "centimeters", measured_downward=True)
    assert down[0] > down[1]
    np.testing.assert_allclose(down, [-0.1, -0.3])
