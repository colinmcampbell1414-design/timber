"""Water flow tests using synthetic DEMs whose correct answers we know."""

import numpy as np
import pytest

from terrain import hydrology

CELL = 0.05


def bowl(n=41, radius=0.6):
    """A cone-shaped pit in the middle of flat ground.

    Height = distance from the center, but never more than `radius`. So the
    ground outside the bowl is flat at height `radius`, and the center of the
    bowl is exactly `radius` meters below the rim.
    """
    r_idx, c_idx = np.indices((n, n))
    dist = np.hypot(r_idx - n // 2, c_idx - n // 2) * CELL
    return np.minimum(dist, radius)


def v_valley(rows=30, cols=21, side_slope=0.5, down_slope=0.1):
    """A V-shaped valley along the middle column, running downhill to the south."""
    r_idx, c_idx = np.indices((rows, cols))
    across = np.abs(c_idx - cols // 2) * CELL * side_slope  # up the valley sides
    along = (rows - 1 - r_idx) * CELL * down_slope  # higher to the north
    return across + along


def test_bowl_ponds_at_center_with_known_depth():
    z = bowl()
    result = hydrology.analyze_flow(z, CELL, CELL)
    depth = result["depth"]
    center = z.shape[0] // 2
    assert depth[center, center] == pytest.approx(0.6)  # filled up to the rim
    assert depth.max() == pytest.approx(0.6)
    assert np.allclose(result["filled"], 0.6)  # the whole bowl fills level with the rim

    labels, ponds = hydrology.find_ponds(depth, min_depth=0.01, cell_area=CELL**2)
    assert len(ponds) == 1
    assert ponds.loc[0, "max_depth_m"] == pytest.approx(0.6)
    assert labels[center, center] == 1
    # Cells deeper than 1 cm are within 0.59 m of the center: about pi * 0.59^2 m².
    assert ponds.loc[0, "area_m2"] == pytest.approx(np.pi * 0.59**2, rel=0.05)


def test_v_valley_flow_collects_along_valley_line():
    z = v_valley()
    rows, cols = z.shape
    mid = cols // 2
    result = hydrology.analyze_flow(z, CELL, CELL)
    acc = result["acc_cells"]

    assert np.allclose(result["depth"], 0.0)  # nothing to fill in a valley
    # In every row, the valley line has the most water.
    assert np.all(np.argmax(acc, axis=1) == mid)
    # Going downstream (south), water along the valley line only increases.
    assert np.all(np.diff(acc[:, mid]) > 0)
    # All the water on the map leaves through the bottom of the valley.
    assert acc[-1, mid] == rows * cols


def test_flat_plane_does_not_crash_and_all_water_drains():
    z = np.full((25, 30), 7.0)
    result = hydrology.analyze_flow(z, CELL, CELL)
    assert np.allclose(result["depth"], 0.0)
    acc = result["acc_cells"]
    assert np.all(np.isfinite(acc)) and acc.min() >= 1
    # Water is never lost: everything that flows off the map adds up to every cell.
    outlets = result["receiver"] < 0
    assert acc.ravel()[outlets].sum() == z.size


def test_nodata_stays_nodata_and_counts_as_outlet():
    z = bowl()
    z[:, :20] = np.nan  # cut the bowl in half: its water can now escape into the no-data area
    result = hydrology.analyze_flow(z, CELL, CELL)
    assert np.all(np.isnan(result["depth"][:, :20]))
    assert np.all(np.isnan(result["acc_cells"][:, :20]))
    assert np.nanmax(result["depth"]) == pytest.approx(0.0)  # no pond: it drains away
    outlets = result["receiver"] < 0
    assert np.nansum(result["acc_cells"].ravel()[outlets]) == np.isfinite(z).sum()


def test_remove_tilt_recovers_known_tilt():
    z = bowl()
    rows, cols = z.shape
    r_idx, c_idx = np.indices(z.shape)
    east = c_idx * CELL
    tilted = z + np.tan(np.radians(5.0)) * east  # whole model tilted 5 degrees, rising to the east
    detrended, info = hydrology.remove_tilt(tilted, CELL, CELL)
    # The bowl itself is symmetric, so only the added tilt should be found.
    assert info["tilt_deg"] == pytest.approx(5.0, abs=0.01)
    assert info["downhill_toward_deg"] == pytest.approx(270.0, abs=0.5)  # slopes down to the west
    np.testing.assert_allclose(detrended - detrended.mean(), z - z.mean(), atol=1e-9)


def test_tilt_can_hide_a_pond():
    # The same bowl on a 30-degree tilt spills out, so far less ponding.
    z = bowl()
    tilted = z + np.tan(np.radians(30.0)) * np.indices(z.shape)[1] * CELL
    plain = hydrology.analyze_flow(tilted, CELL, CELL)
    fixed = hydrology.analyze_flow(tilted, CELL, CELL, detrend=True)
    assert plain["depth"].max() < 0.6
    assert fixed["depth"].max() == pytest.approx(0.6, abs=1e-6)


def test_find_ponds_min_area_and_numbering():
    depth = np.zeros((10, 10))
    depth[1:3, 1:3] = 0.05  # small pond: 4 cells
    depth[5:9, 5:9] = 0.02  # bigger, shallower pond: 16 cells
    labels, ponds = hydrology.find_ponds(depth, min_depth=0.01, cell_area=1.0)
    assert len(ponds) == 2
    assert ponds["pond"].tolist() == [1, 2]
    assert ponds.loc[0, "volume_m3"] == pytest.approx(0.32)  # biggest volume is pond 1
    assert labels[6, 6] == 1 and labels[1, 1] == 2

    _, big_only = hydrology.find_ponds(depth, min_depth=0.01, cell_area=1.0, min_area=5.0)
    assert len(big_only) == 1


@pytest.mark.parametrize("make", ["bowl", "valley", "noisy", "flat", "nodata"])
def test_fast_method_matches_simple_method(make):
    rng = np.random.default_rng(3)
    if make == "bowl":
        z = bowl()
    elif make == "valley":
        z = v_valley()
    elif make == "noisy":  # bumpy slope with many small pits and flats
        r, c = np.indices((60, 70))
        z = 0.02 * r + np.round(rng.normal(0, 0.03, (60, 70)), 2)
    elif make == "flat":
        z = np.full((30, 40), 5.0)
    else:
        z = bowl()
        z[:, :12] = np.nan
    simple = hydrology.analyze_flow(z, CELL, CELL, method="simple")
    fast = hydrology.analyze_flow(z, CELL, CELL, method="fast")
    np.testing.assert_allclose(fast["filled"], simple["filled"], atol=1e-12)  # same ponding
    valid = np.isfinite(z)
    # No water lost: everything leaving the map adds up to the number of cells.
    outlets = fast["receiver"] < 0
    assert np.nansum(fast["acc_cells"].ravel()[outlets & valid.ravel()]) == pytest.approx(valid.sum())
    # Every valid cell got water routed somewhere (no endless loops, no stuck cells).
    assert np.all(fast["acc_cells"][valid] >= 1)
    if make == "valley":  # no flats at all: identical flow
        np.testing.assert_array_equal(fast["acc_cells"], simple["acc_cells"])


def test_fast_method_speed_on_big_grid():
    import time

    r, c = np.indices((1500, 1500))
    z = 0.01 * r + 0.3 * np.sin(c / 40) + np.random.default_rng(0).normal(0, 0.02, r.shape)
    start = time.time()
    hydrology.analyze_flow(z, 0.25, 0.25)  # 2.25 million cells -> "auto" picks the fast method
    assert time.time() - start < 30


def test_flow_segments_point_downhill():
    z = v_valley()
    result = hydrology.analyze_flow(z, CELL, CELL)
    mask = np.zeros(z.shape, dtype=bool)
    mask[5, 3] = True  # a cell on the west side of the valley
    seg = hydrology.flow_segments(result["receiver"], mask, CELL, CELL)
    assert seg.shape == (1, 2, 2)
    (x0, y0), (x1, y1) = seg[0]
    assert x1 > x0  # flows east, toward the valley line


def test_suggested_path_threshold_keeps_paths_to_two_percent():
    # 10,000 cells; 1% drain at least 100 m², 5% at least 10 m², the rest 1 m².
    acc = np.ones((100, 100))
    acc.flat[:500] = 10.0
    acc.flat[:100] = 100.0
    options = [1.0, 10.0, 100.0, 1000.0]
    assert hydrology.suggest_path_threshold(acc, options) == 100.0
    assert hydrology.suggest_path_threshold(acc, options, max_fraction=0.06) == 10.0
