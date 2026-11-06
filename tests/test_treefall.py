"""Tests for the upgraded tree fall hazard app, using synthetic forests with known answers."""

import time

import numpy as np
import pytest

from conftest import FakeStreamlit, local_grid, make_raster, to_map
from terrain import treefall
from terrain.apps import base, trees

CELL = 0.25


def cone(x, y, cx, cy, height, radius):
    """A cone-shaped 'tree': `height` at (cx, cy), 0 at `radius` and beyond."""
    return np.clip(height * (1 - np.hypot(x - cx, y - cy) / radius), 0, None)


def defaults(**overrides):
    """The app's default settings (overrides use the slider labels)."""
    return trees.params(FakeStreamlit(overrides))


def analyze(dtm_z, chm, p=None, ortho=None, targets=None):
    """Run the full tree analysis on synthetic grids."""
    layers = base.Layers(dtm=make_raster(dtm_z, CELL), dsm=make_raster(dtm_z + chm, CELL), ortho=ortho)
    grids = {"chm": np.maximum(chm, 0), "dtm": dtm_z, "slope": layers.slope_deg, "twi": layers.twi,
             "acc_area": layers.flow["acc_area_m2"], "vari": layers.vari, "exg": layers.exg}
    return treefall.analyze(grids, layers.dtm.transform, layers.dtm.crs, CELL, p or defaults(), targets)


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


def test_two_cone_trees_heights_and_height_factor():
    rows, cols = 160, 240  # 60 m x 40 m
    x, y = local_grid((rows, cols), CELL)
    chm = cone(x, y, 15.125, 20.125, 10.0, 3.0) + cone(x, y, 40.125, 20.125, 20.0, 5.0)
    out = analyze(np.zeros((rows, cols)), chm)["trees"].sort_values("height").reset_index(drop=True)
    assert len(out) == 2
    assert out.loc[0, "height"] == pytest.approx(10.0, rel=0.05)
    assert out.loc[1, "height"] == pytest.approx(20.0, rel=0.05)
    assert out.loc[1, "f_height"] > out.loc[0, "f_height"]


def test_slope_factor_and_fall_direction_downhill():
    rows, cols = 120, 120  # 30 m x 30 m
    x, y = local_grid((rows, cols), CELL)
    chm = cone(x, y, 15.125, 15.125, 15.0, 4.0)
    p = defaults(**{"Include prevailing wind": False})
    flat = analyze(np.zeros((rows, cols)), chm, p)["trees"].iloc[0]
    ground = np.tan(np.radians(30)) * x  # rises to the east, so downhill is west (270 degrees)
    steep = analyze(ground, chm, p)["trees"].iloc[0]
    assert steep["f_slope"] > flat["f_slope"]
    assert steep["slope_deg"] == pytest.approx(30, abs=1)
    assert angle_diff(steep["downhill_dir"], 270) < 5
    assert angle_diff(steep["fall_dir"], 270) <= 20


def test_lopsided_crown_leans_east():
    rows, cols = 120, 120
    x, y = local_grid((rows, cols), CELL)
    cx, cy = 13.125, 15.125
    # The crown reaches 6 m to the east but only 3 m to the west of the top.
    radius = np.where(x > cx, 6.0, 3.0)
    chm = np.clip(12.0 * (1 - np.hypot(x - cx, y - cy) / radius), 0, None)
    tree = analyze(np.zeros((rows, cols)), chm)["trees"]
    assert len(tree) == 1
    assert angle_diff(tree.iloc[0]["lean_dir"], 90) <= 20
    t = tree.iloc[0]
    assert t["lean_ratio"] > 0.2  # the crown middle is well off to the east
    p = defaults()
    assert t["f_lean"] == pytest.approx(treefall.ramp(t["lean_ratio"], p["lean_start"], p["lean_full"]))


def test_ramp_start_middle_full():
    assert treefall.ramp([10, 15, 20, 25, 40], 15, 25).tolist() == [0.0, 0.0, 0.5, 1.0, 1.0]


def planted_forest(clearing=False):
    """80 m x 80 m of identical 15 m cone trees, 6 m apart, with a 6 m open strip
    around the outside. The small holes between neighboring crowns are under 5 m
    wide. clearing=True cuts a 20 m x 20 m clearing in the middle."""
    rows = cols = 320
    x, y = local_grid((rows, cols), CELL)
    chm = np.zeros((rows, cols))
    for cx in np.arange(10, 71, 6.0):
        for cy in np.arange(10, 71, 6.0):
            if clearing and abs(cx - 40) < 10 and abs(cy - 40) < 10:
                continue
            chm = np.maximum(chm, cone(x, y, cx + 0.125, cy + 0.125, 15.0, 4.0))
    return analyze(np.zeros((rows, cols)), chm)["trees"]


def tree_at(t, x, y):
    """The tree whose top is nearest to local position (x, y) in meters."""
    tx, ty = t["x"] - 500000.0, t["y"] - (4000000.0 - 80.0)
    return t.loc[np.argmin(np.hypot(tx - x, ty - y))]


def test_ordinary_interior_tree_scores_low_and_small_holes_are_not_openings():
    t = planted_forest()
    middle = tree_at(t, 40, 40)
    assert middle["f_exposure"] == 0  # the holes between crowns are too narrow to count
    assert middle["hazard"] < 10 and middle["class"] == "low"
    corner = tree_at(t, 10, 10)
    assert corner["f_exposure"] == 1  # but the open strip around the forest is a real opening


def test_large_clearing_makes_its_neighbors_edge_trees():
    before = tree_at(planted_forest(), 52, 40)
    after = tree_at(planted_forest(clearing=True), 52, 40)
    assert before["f_exposure"] == 0
    assert after["f_exposure"] == 1
    assert after["hazard"] > before["hazard"]


def test_percent_classes_give_exact_counts():
    hazard = np.random.default_rng(3).uniform(0, 100, 100)
    p = {"class_mode": "percent", "high_pct": 5.0, "moderate_pct": 15.0}
    cls = treefall.assign_classes(hazard, p)
    assert (cls == "high").sum() == 5 and (cls == "moderate").sum() == 15
    assert hazard[cls == "high"].min() > hazard[cls == "moderate"].max() > hazard[cls == "low"].max()
    score = treefall.assign_classes(np.array([10.0, 35.0, 61.0]), {"moderate": 35, "high": 60})
    assert score.tolist() == ["low", "moderate", "high"]


def test_fall_exposure_is_a_disk_of_radius_height():
    rows, cols = 120, 120
    x, y = local_grid((rows, cols), CELL)
    chm = cone(x, y, 15.125, 15.125, 10.0, 3.0)
    p = defaults(**{"Include prevailing wind": False, "Fall exposure map cell size (m)": CELL})
    res = analyze(np.zeros((rows, cols)), chm, p)
    t = res["trees"].iloc[0]
    dist = np.hypot(x - (t["col"] + 0.5) * CELL, y - (rows - t["row"] - 0.5) * CELL)
    h = t["height"]
    assert h == pytest.approx(10.0, rel=0.05)
    inside, outside = dist <= h - CELL, dist >= h + CELL
    assert np.all(res["reach"][inside] == 1) and np.all(res["reach"][outside] == 0)
    assert np.all(res["exposure"][inside] > 0) and np.all(res["exposure"][outside] == 0)


def test_trail_strip_found_but_round_clearing_is_not():
    rows, cols = 240, 320  # 80 m x 60 m forest
    x, y = local_grid((rows, cols), CELL)
    chm = np.full((rows, cols), 15.0)
    strip = (np.abs(y - 15.0) < 1.0) & (x > 10) & (x < 50)  # 2 m wide, 40 m long
    clearing = np.hypot(x - 60, y - 40) < 6.0  # round clearing, 12 m across
    chm[strip | clearing] = 0.0
    grids = {"chm": chm, "dtm": np.zeros((rows, cols)), "acc_area": np.zeros((rows, cols)), "exg": None, "vari": None}
    masks, info = treefall.detect_features(grids, CELL, defaults())
    found = masks["trail"] | masks["road"]
    assert found[strip].mean() > 0.9  # the strip is a trail
    assert not found[clearing].any()  # the round clearing is not
    assert len(info) == 1 and info[0]["length_m"] == pytest.approx(40, rel=0.15)
    assert masks["clearing"][clearing].all()  # but it is a clearing


def test_building_found_but_cone_tree_of_same_height_is_not():
    rows, cols = 160, 200  # 50 m x 40 m
    x, y = local_grid((rows, cols), CELL)
    box = (x > 5) & (x < 15) & (y > 10) & (y < 18)  # 10 m x 8 m
    chm = np.where(box, 4.0, 0.0) + cone(x, y, 35, 20, 4.0, 3.0)
    gray = np.full((rows, cols, 3), 0.5)  # nothing is green: only the shape can tell them apart
    from terrain import vegetation
    grids = {"chm": chm, "dtm": np.zeros((rows, cols)), "acc_area": np.zeros((rows, cols)),
             "exg": vegetation.exg(gray), "vari": vegetation.vari(gray)}
    masks, _ = treefall.detect_features(grids, CELL, defaults())
    building = masks["building"]
    assert building[box].mean() > 0.9
    assert not building[np.hypot(x - 35, y - 20) < 3].any()


def test_downed_log_axis_angle():
    rows, cols = 120, 120
    x, y = local_grid((rows, cols), CELL)
    angle = 30.0  # compass: 30 degrees east of north
    ux, uy = np.sin(np.radians(angle)), np.cos(np.radians(angle))
    along = (x - 15) * ux + (y - 15) * uy
    across = (x - 15) * uy - (y - 15) * ux
    log = (np.abs(along) <= 3.0) & (np.abs(across) <= 0.3)  # 6 m x 0.6 m
    chm = np.where(log, 0.5, 0.0)
    logs, _ = treefall.detect_logs(chm, make_raster(chm, CELL).transform, CELL)
    assert len(logs) == 1
    assert angle_diff(logs.iloc[0]["axis_deg"], angle) <= 10 or angle_diff(logs.iloc[0]["axis_deg"] + 180, angle) <= 10
    assert logs.iloc[0]["length_m"] == pytest.approx(6.0, rel=0.15)


def test_tree_that_can_reach_building_ranks_first():
    rows, cols = 160, 320  # 80 m x 40 m
    x, y = local_grid((rows, cols), CELL)
    chm = cone(x, y, 15.125, 20.125, 15.0, 4.0) + cone(x, y, 65.125, 20.125, 15.0, 4.0)  # identical trees
    chm += np.where((x > 22) & (x < 32) & (y > 16) & (y < 24), 4.0, 0.0)  # building 7 m east of the left tree
    p = defaults(**{"Weight: wetness": 0.0, "Find streams": False})
    out = analyze(np.zeros((rows, cols)), chm, p)["trees"]
    assert len(out) == 2
    assert out.loc[0, "hazard"] == pytest.approx(out.loc[1, "hazard"], abs=1)
    assert out.loc[0, "x"] < out.loc[1, "x"]  # the left tree (next to the building) is first
    assert out.loc[0, "can reach building"] and not out.loc[1, "can reach building"]


def test_app_runs_and_has_five_tabs():
    rows, cols = 160, 240
    x, y = local_grid((rows, cols), CELL)
    chm = cone(x, y, 15.125, 20.125, 10.0, 3.0) + cone(x, y, 40.125, 20.125, 20.0, 5.0)
    ortho = np.zeros((rows, cols, 3))
    ortho[...] = (0.2, 0.6, 0.2)
    shed = to_map(25.0, 20.0, rows, CELL)
    layers = base.Layers(dtm=make_raster(np.zeros((rows, cols)), CELL), dsm=make_raster(chm, CELL), ortho=ortho,
                         shapes={"targets": [{"label": "shed", "xy": [shed]}]})
    result = trees.run(layers, defaults())
    assert [s.title for s in result.sections] == ["Trees & hazard", "Where trees could fall", "Features at risk",
                                                 "Downed logs", "Field check"]
    assert result.metrics["Trees found"] == "2"
    assert "can reach marked target" in result.table


def test_field_check_round_trip():
    rows, cols = 160, 400
    x, y = local_grid((rows, cols), CELL)
    rng = np.random.default_rng(1)
    chm = np.zeros((rows, cols))
    for cx in np.arange(8, 95, 9.0):
        chm = np.maximum(chm, cone(x, y, cx + 0.125, 20.125, rng.uniform(8, 20), 3.5))
    res = analyze(np.zeros((rows, cols)), chm)
    tr = res["trees"]
    sheet = treefall.field_check_export(tr, "EPSG:32618", n_top=3)
    assert {"latitude", "longitude", "observed_height", "dead_or_damaged", "notes"} <= set(sheet.columns)
    sheet["observed_height"] = sheet["height_m"] + 1.0  # field heights 1 m taller
    sheet["dead_or_damaged"] = "n"
    summary, table = treefall.field_check_compare(tr, sheet)
    assert summary["Trees matched"].startswith(f"{len(sheet)} of")
    assert "mean -1.0 m" in summary["Height error (app - field)"]


def test_performance_2000_by_2000_with_1000_trees():
    n = 2000
    rng = np.random.default_rng(7)
    r, c = np.indices((n, n))
    dtm = 0.05 * r * CELL + 2 * np.sin(c * CELL / 40)  # gentle hills
    chm = np.zeros((n, n))
    heights = rng.uniform(8, 30, 1000)
    centers = rng.uniform(5, n * CELL - 5, (1000, 2))
    for (cx, cy), h in zip(centers, heights):  # stamp each cone tree in its own small window
        rad = 0.2 * h + 1.5
        c0, c1 = int((cx - rad) / CELL), int((cx + rad) / CELL) + 1
        r0, r1 = int((cy - rad) / CELL), int((cy + rad) / CELL) + 1
        yy, xx = np.mgrid[max(r0, 0):min(r1, n), max(c0, 0):min(c1, n)]
        chm[yy, xx] = np.maximum(chm[yy, xx], cone((xx + 0.5) * CELL, (yy + 0.5) * CELL, cx, cy, h, rad))
    ortho = np.empty((n, n, 3))
    ortho[...] = (0.25, 0.55, 0.2)
    layers = base.Layers(dtm=make_raster(dtm, CELL), dsm=make_raster(dtm + chm, CELL), ortho=ortho)
    start = time.time()
    result = trees.run(layers, defaults())
    elapsed = time.time() - start
    found = int(result.metrics["Trees found"])
    assert 800 <= found <= 1100  # some cones overlap and merge
    assert elapsed < 60, f"took {elapsed:.0f} s"
