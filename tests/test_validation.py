"""Tests for the Phase 6 validation tools (sensitivity, blind field check, check plots,
downed logs, DTM fusion, season, design criteria), with made-up data whose answers are known."""

import time

import numpy as np
import pandas as pd
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from conftest import FakeStreamlit, local_grid, make_raster
from terrain import fusion, io, storm, treefall, validation
from terrain.apps import base, trees

WEIGHTS = dict(treefall.DEFAULT_WEIGHTS)
NAMES = list(WEIGHTS)


def fake_trees(n=300, seed=1):
    """A made-up tree table with every column the validation tools read."""
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({"tree": np.arange(1, n + 1), "x": rng.uniform(0, 500, n), "y": rng.uniform(0, 500, n),
                      "height": rng.uniform(6, 35, n), "fall_dir": 90.0, "reasons": "made up",
                      "possibly dead": False, "gap_dist_m": rng.uniform(0, 30, n),
                      "latitude": 39.2, "longitude": -76.9, "trail_dist_m": rng.uniform(0, 80, n),
                      "trail_bearing_deg": rng.uniform(0, 360, n)})
    for name in NAMES:
        t[f"f_{name}"] = rng.uniform(0, 1, n)
    t["hazard"] = treefall.hazard_score(t, WEIGHTS)
    return validation.add_baselines(t)


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------
def test_scores_for_weights_match_the_hazard_score():
    t = fake_trees()
    w = np.array([WEIGHTS[n] for n in NAMES])
    scores = validation.scores_for_weights(validation.factor_matrix(t, NAMES), (w / w.sum())[None, :])
    assert np.allclose(scores[:, 0], t["hazard"])


def test_sensitivity_when_only_height_matters():
    # Every factor except height is 0 for every tree: any weights give the same order.
    t = fake_trees(200)
    for name in NAMES[1:]:
        t[f"f_{name}"] = 0.0
    F = validation.factor_matrix(t, NAMES)
    one = validation.sensitivity_one_at_a_time(F, WEIGHTS)
    assert len(one) == 2 * len(NAMES)
    assert np.allclose(one["Spearman correlation"], 1.0) and (one["original top 20 kept"] == 20).all()
    rand = validation.sensitivity_random(F, WEIGHTS, n_sets=200)
    assert (rand["overlap"] == 20).all() and rand["weight_stability_top20"] == 1.0
    assert rand["summary"].startswith("20 of the top 20")


def test_sensitivity_changes_with_real_factors_and_is_seeded():
    F = validation.factor_matrix(fake_trees(), NAMES)
    a = validation.sensitivity_random(F, WEIGHTS, seed=42)
    b = validation.sensitivity_random(F, WEIGHTS, seed=42)
    c = validation.sensitivity_random(F, WEIGHTS, seed=7)
    assert np.array_equal(a["overlap"], b["overlap"]) and not np.array_equal(a["overlap"], c["overlap"])
    assert 0.3 < a["weight_stability_top20"] < 1.0  # some, but not all, of the top 20 survive
    assert a["stability_percent"].max() <= 100 and a["stability_percent"][a["base_top"]].mean() > 50
    tight = validation.sensitivity_random(F, WEIGHTS, concentration=2000)
    assert tight["weight_stability_top20"] > a["weight_stability_top20"]  # weights closer to the original change less


def test_sensitivity_is_fast():
    rng = np.random.default_rng(0)
    F = rng.uniform(0, 1, (5000, 8))
    start = time.time()
    validation.sensitivity_random(F, WEIGHTS, n_sets=500)
    validation.sensitivity_one_at_a_time(F, WEIGHTS)
    assert time.time() - start < 10


# ---------------------------------------------------------------------------
# Blind field check
# ---------------------------------------------------------------------------
def test_blind_sheet_hides_everything_about_the_prediction():
    t = fake_trees()
    sheet, key = validation.blind_field_sample(t, n=30)
    assert list(sheet.columns) == validation.SHEET_COLUMNS
    hidden = {"hazard", "score", "group", "height", "reasons", "tree", "class", "hazard_rank"}
    assert not hidden & set(sheet.columns)
    assert sheet["code"].is_unique and sheet["code"].between(1000, 9999).all()
    assert sorted(key["code"]) == sorted(sheet["code"])  # every code appears once in the key
    assert (key["group"] == "high").sum() == 30 and (key["group"] == "low").sum() == 30
    assert (key["height"] > 10).all()
    assert key[key["group"] == "high"]["hazard"].min() > key[key["group"] == "low"]["hazard"].max()
    # Same seed, same sheet; the reachable-on-foot limit is respected.
    assert validation.blind_field_sample(t, n=30)[0]["code"].tolist() == sheet["code"].tolist()
    near = validation.blind_field_sample(t, n=10, max_trail_m=30)[1]
    assert (t.set_index("tree").loc[near["tree"], "trail_dist_m"] <= 30).all()


def filled_sheet(sheet, key, defect_groups=("high",)):
    filled = sheet.copy()
    filled["status"] = "found"
    group = filled["code"].map(key.set_index("code")["group"])
    filled["dead_or_dying_crown"] = np.where(group.isin(defect_groups), "y", "n")
    filled["observed_height_m"] = filled["code"].map(key.set_index("code")["height"]) - 1.0  # app reads 1 m too tall
    return filled


def test_field_import_all_high_and_no_low_defects():
    sheet, key = validation.blind_field_sample(fake_trees(), n=30)
    res = validation.field_import(key, filled_sheet(sheet, key), n_boot=50)
    d = res["defects"]
    assert d["ratio"] == float("inf") and d["ratio_text"] == "all high, none low"
    assert d["fisher_p"] < 0.001 and d["table_2x2"].loc["high hazard", "defect"] == 30
    auc = res["auc"].set_index("ranking")["AUC"]
    assert auc["full hazard score"] == pytest.approx(1.0)
    heights = res["heights"].set_index("trees")
    assert heights.loc["all", "bias_m"] == pytest.approx(1.0) and heights.loc["all", "rmse_m"] == pytest.approx(1.0)
    assert heights.loc["open / edge", "n"] + heights.loc["interior", "n"] == heights.loc["all", "n"] == 60
    assert res["criteria"] == {"height_rmse_m": pytest.approx(1.0), "defect_rate_ratio": "inf",
                               "fisher_p": pytest.approx(d["fisher_p"]), "beats_baseline": True}


def test_field_import_equal_defects_and_not_found_trees():
    sheet, key = validation.blind_field_sample(fake_trees(), n=30)
    everyone = filled_sheet(sheet, key, defect_groups=("high", "low"))
    everyone.loc[:4, "status"] = "not_found"
    res = validation.field_import(key, everyone, n_boot=50)
    assert res["checked"] == 55
    assert np.isnan(res["auc"].set_index("ranking")["AUC"]["full hazard score"])  # nothing to tell apart
    assert res["defects"]["ratio"] == pytest.approx(1.0)
    nobody = filled_sheet(sheet, key, defect_groups=())
    assert validation.field_import(key, nobody, n_boot=50)["defects"]["ratio_text"] == "no defects found in either group"
    # Half the trees in each group defective at random: the score can't tell them apart (AUC about 0.5).
    rng = np.random.default_rng(0)
    coin = filled_sheet(sheet, key, defect_groups=())
    coin["trunk_crack_or_cavity"] = np.where(rng.random(len(coin)) < 0.5, "y", "n")
    auc = validation.field_import(key, coin, n_boot=50)["auc"].set_index("ranking")["AUC"]["full hazard score"]
    assert auc == pytest.approx(0.5, abs=0.2)


def test_snag_flag_table():
    sheet, key = validation.blind_field_sample(fake_trees(), n=10)
    key["possibly dead"] = key["group"] == "high"
    res = validation.field_import(key, filled_sheet(sheet, key), n_boot=20)
    assert res["snag_2x2"].loc["app: possibly dead", "observed dead or dying"] == 10
    assert res["snag_2x2"].loc["app: not flagged", "observed healthy"] == 10


# ---------------------------------------------------------------------------
# Detection check plots
# ---------------------------------------------------------------------------
def cone_plot_trees(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return pd.DataFrame({"tree": range(n), "x": 100 + 8 * np.cos(angles), "y": 100 + 8 * np.sin(angles), "height": 15.0})


def test_detection_rate_per_plot_and_overall():
    plots = pd.DataFrame({"plot_id": ["A"], "x": [100.0], "y": [100.0], "radius_m": [12.0], "field_count_over_10m": [10]})
    assert validation.detection_rates(plots, cone_plot_trees(10))["overall"] == pytest.approx(1.0)
    res = validation.detection_rates(plots, cone_plot_trees(8))
    assert res["overall"] == pytest.approx(0.8) and res["criteria"]["detection_rate"] == pytest.approx(0.8)
    assert res["table"].loc[0, "app count"] == 8 and res["table"].loc[0, "field count"] == 10
    # Short trees and trees outside the circle don't count.
    extra = pd.concat([cone_plot_trees(8), pd.DataFrame({"tree": [98, 99], "x": [100, 130.0], "y": [100, 100.0],
                                                         "height": [6.0, 20.0]})])
    assert validation.detection_rates(plots, extra)["overall"] == pytest.approx(0.8)
    sheet = validation.plot_sheet(pd.DataFrame({"plot_id": ["A"], "latitude": [39.0], "longitude": [-77.0], "radius_m": [12.0]}))
    assert list(sheet.columns) == ["plot_id", "latitude", "longitude", "radius_m", "field_count_over_10m", "notes"]


# ---------------------------------------------------------------------------
# Downed logs
# ---------------------------------------------------------------------------
def test_axial_statistics():
    # Axes at 0 and 180 degrees on a slope facing 0: the same axis, perfectly lined up.
    same = validation.rayleigh_axial([0, 180, 0, 180, 0, 180], 0)
    assert same["mean_resultant_length"] == pytest.approx(1.0) and same["mean_offset_deg"] == pytest.approx(0, abs=1e-6)
    rng = np.random.default_rng(3)
    spread = validation.rayleigh_axial(rng.uniform(0, 180, 200), 0)
    assert spread["p"] > 0.05 and spread["mean_resultant_length"] < 0.2
    tight = validation.rayleigh_axial(40 + rng.uniform(-10, 10, 30), 40)
    assert tight["p"] < 0.01 and tight["mean_offset_deg"] < 10
    across = validation.rayleigh_axial(130 + rng.uniform(-5, 5, 30), 40)  # lying across the slope
    assert across["p"] < 0.01 and across["mean_offset_deg"] > 80


def make_logs(x, y):
    return pd.DataFrame({"log": np.arange(len(x)), "x": x, "y": y, "length_m": 6.0, "width_m": 0.5, "axis_deg": 45.0})


def test_log_factor_test_finds_logs_on_steep_ground():
    cell, n = 1.0, 200
    transform = from_origin(0, n * cell, cell, cell)
    xx, _ = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    slope = np.where(xx > 100, 30.0, 5.0)  # the east half is steep
    forest = np.ones((n, n), dtype=bool)
    rng = np.random.default_rng(1)
    steep_logs = make_logs(rng.uniform(120, 190, 25), rng.uniform(10, 190, 25))
    random_logs = make_logs(rng.uniform(10, 190, 25), rng.uniform(10, 190, 25))
    steep = validation.log_factor_test(steep_logs, {"slope": slope}, forest, transform, cell)
    rand = validation.log_factor_test(random_logs, {"slope": slope}, forest, transform, cell)
    p_steep = steep["table"]["Mann-Whitney p (logs higher)"].iloc[0]
    assert p_steep < 0.001 and p_steep < rand["table"]["Mann-Whitney p (logs higher)"].iloc[0]
    assert steep["criteria"]["log_factor_test_p"] == pytest.approx(p_steep)  # one factor: no correction
    assert steep["table"]["median at logs"].iloc[0] == pytest.approx(30)
    two = validation.log_factor_test(steep_logs, {"slope": slope, "flat": np.zeros((n, n))}, forest, transform, cell)
    assert two["criteria"]["log_factor_test_p"] == pytest.approx(min(1.0, 2 * p_steep))  # Bonferroni x 2


def test_logs_near_trails_are_excluded_and_manual_logs_matched():
    cell, n = 1.0, 100
    transform = from_origin(0, n * cell, cell, cell)
    trail = np.zeros((n, n), dtype=bool)
    trail[50, :] = True  # a trail across the middle (y = 49.5)
    logs = make_logs(np.array([20.0, 60.0, 80.0]), np.array([52.0, 90.0, 10.0]))  # the first is 2.5 m from the trail
    kept = validation.logs_away_from(logs, trail, transform, 5.0)
    assert kept["log"].tolist() == [1, 2]
    manual = pd.DataFrame({"x": [61.0, 30.0], "y": [91.0, 30.0]})
    assert validation.match_logs(manual, logs) == 1


# ---------------------------------------------------------------------------
# DTM fusion
# ---------------------------------------------------------------------------
def test_other_dtm_is_aligned_on_stable_ground():
    cell = 0.5
    x, y = local_grid((200, 240), cell)
    ground = 0.03 * x + 0.01 * y
    trees_chm = np.where(np.hypot(x - 60, y - 50) < 20, 15.0, 0.0)  # a block of forest
    dsm = ground + trees_chm
    tilt = np.tan(np.radians(0.5)) * (y - y.mean())
    # The other survey's DTM: 3.2 m higher, tilted 0.5 degrees, and better under the trees.
    other = ground + 3.2 + tilt
    own = ground + 0.6 * trees_chm  # the leaf-on DTM rides up into the canopy
    aligned, report = fusion.align_dtm(other, own, dsm, cell)
    assert report["offset_before_m"] == pytest.approx(3.2, abs=0.05) and report["tilt_deg"] == pytest.approx(0.5, abs=0.02)
    assert report["rms_after_m"] < 0.02
    open_ground = trees_chm == 0
    assert np.abs(aligned - ground)[open_ground].max() < 0.02
    table = fusion.compare_dtms({"leaf-on": own, "leaf-off": aligned}, "leaf-on", dsm, cell)
    off = table.set_index("DTM source").loc["leaf-off"]
    assert abs(off["open ground: mean difference (m)"]) < 0.02
    assert off["median canopy height (m)"] == pytest.approx(15, abs=0.1)
    assert table["median canopy height (m)"].iloc[0] == pytest.approx(6, abs=0.1)  # leaf-on DTM: trees look 6 m tall


def test_us_survey_feet_are_converted_to_meters():
    target = make_raster(np.zeros((40, 40)), 1.0)  # UTM 18N, meters
    assert fusion.vertical_factor(CRS.from_epsg(2248)) == pytest.approx(1200 / 3937)  # Maryland state plane, US survey ft
    assert fusion.vertical_factor(CRS.from_epsg(32618)) == 1.0
    assert fusion.vertical_factor(CRS.from_epsg(32618), "feet") == 0.3048
    feet = io.Raster(data=np.full((40, 40), 100.0), transform=target.transform, crs=target.crs)
    moved = fusion.to_grid(feet, target, "US survey feet")
    assert np.nanmax(np.abs(moved.data - 100 * 1200 / 3937)) < 1e-6
    # A real state-plane raster (coordinates and heights in US survey feet) lands on the meter grid.
    lon, lat = io.map_to_latlon([500020.0], [3999980.0], target.crs)
    from rasterio.warp import transform as to_crs

    fx, fy = to_crs("EPSG:4326", "EPSG:2248", lon, lat)
    lidar = io.Raster(data=np.full((60, 60), 328.0833), transform=from_origin(fx[0] - 150, fy[0] + 150, 5, 5),
                      crs=CRS.from_epsg(2248))
    on_grid = fusion.to_grid(lidar, target)
    assert np.nanmedian(on_grid.data) == pytest.approx(100.0, abs=0.01)  # 328.0833 US survey ft = 100 m


def test_heights_at_positions():
    dsm = np.zeros((50, 50))
    dsm[20, 30] = 18.0
    t = from_origin(0, 50, 1, 1)
    assert fusion.heights_at([30.5], [29.5], dsm, np.zeros((50, 50)), t)[0] == 18.0
    assert fusion.heights_at([10.5], [10.5], dsm, np.zeros((50, 50)), t)[0] == 0.0


# ---------------------------------------------------------------------------
# Season
# ---------------------------------------------------------------------------
def season_run(season):
    rows, cols, cell = 160, 240, 0.25
    x, y = local_grid((rows, cols), cell)
    chm = np.zeros((rows, cols))
    for cx in (10, 25, 40, 50):
        chm = np.maximum(chm, np.clip(12.0 * (1 - np.hypot(x - cx, y - 20) / 3.0), 0, None))
    ortho = np.zeros((rows, cols, 3))
    ortho[...] = (0.2, 0.6, 0.2)
    ortho[np.hypot(x - 50, y - 20) < 3] = (0.45, 0.4, 0.35)  # one brown crown
    layers = base.Layers(dtm=make_raster(np.zeros((rows, cols)), cell), dsm=make_raster(chm, cell), ortho=ortho)
    p = trees.params(FakeStreamlit(), {"width_m": 60, "season": season})
    return p, trees.run(layers, p)


def test_season_changes_the_health_settings():
    p_summer, summer = season_run("summer leaf-on")
    p_autumn, autumn = season_run("autumn color")
    p_off, off = season_run("leaf-off")
    assert p_summer["weights"]["health"] == pytest.approx(0.20) and p_autumn["weights"]["health"] == pytest.approx(0.05)
    assert summer.raw["trees"]["f_health"].max() > 0 and not any("season" in w for w in summer.warnings)
    assert any("autumn color" in w for w in autumn.warnings)
    # Leaf-off: the health factor is switched off (unknown) and no tree is "possibly dead".
    assert off.raw["trees"]["f_health"].isna().all() and not off.raw["trees"]["possibly dead"].any()
    assert any("leaf-off" in w for w in off.warnings)
    assert off.raw["trees"]["hazard"].notna().all()  # the other weights were rescaled


# ---------------------------------------------------------------------------
# Storm check details added by the audit
# ---------------------------------------------------------------------------
def test_unexplained_gap_is_listed():
    t = from_origin(0, 60, 1, 1)
    before = np.full((60, 60), 12.0)
    after = before.copy()
    after[10:16, 10:16] = 0.0  # a 36 m² hole where no tree was mapped
    after[40, 40] = 0.0  # one cell: too small
    crowns = np.zeros((60, 60), dtype=int)
    crowns[30:60, :] = 1  # a mapped crown elsewhere (changes inside it are not "unexplained")
    gaps = storm.unexplained_gaps(crowns, before, after, t, 1.0, threshold=0.5, min_height=5.0, min_area=10.0)
    assert len(gaps) == 1 and gaps["area_m2"].iloc[0] == 36 and gaps["mean_drop_m"].iloc[0] == pytest.approx(12)
    assert gaps["x"].iloc[0] == pytest.approx(13, abs=0.6) and gaps["y"].iloc[0] == pytest.approx(47, abs=0.6)


def test_top_only_mode_ignores_crown_thinning():
    trees_ = pd.DataFrame({"tree": [1, 2], "row": [20, 20], "col": [20, 60], "height": [15.0, 15.0]})
    crowns = np.zeros((40, 80), dtype=int)
    crowns[10:30, 10:30], crowns[10:30, 50:70] = 1, 2
    before = np.where(crowns > 0, 15.0, 0.0)
    after = before.copy()
    after[10:30, 10:19] = 0.0  # tree 1 lost one side of its crown; its top is untouched
    after[10:30, 50:70] = 1.0  # tree 2 is down
    normal, _ = storm.detect_changes(trees_, crowns, before, after, 1.0, {"edge_m": 0})
    top_only, info = storm.detect_changes(trees_, crowns, before, after, 1.0, {"edge_m": 0, "top_only": True})
    assert normal["damage"].tolist() == ["crown damage", "fallen"]
    assert top_only["damage"].tolist() == ["no change", "fallen"] and info["top_only"]


def test_exposure_under_new_logs_is_compared_with_random_points():
    n = 100
    t = from_origin(0, n, 1, 1)
    exposure = np.zeros((n, n))
    exposure[:, 50:] = 1.0  # the east half is exposed
    logs = pd.DataFrame({"log": range(20), "x": np.linspace(60, 95, 20), "y": np.linspace(5, 95, 20), "length_m": 5.0,
                         "width_m": 0.5, "axis_deg": 0.0, "tree": np.nan})
    table = pd.DataFrame({"tree": [], "observed_fall_deg": []})
    out = storm.landing_metrics(table, logs, {"exposure": exposure, "grid_factor": np.array(1)}, np.full((n, n), 10.0),
                                t, 1.0, {})
    e = out["exposure"]
    assert e["mean exposure under new logs"] == 1.0 and e["mean exposure at random forest points"] == pytest.approx(0.5, abs=0.06)
    assert e["Mann-Whitney p (logs higher)"] < 0.001
