"""Tests for the storm check, using synthetic forests where the answer is known."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from conftest import FakeStreamlit, local_grid, make_raster
from terrain import storm, treefall, validation
from terrain.apps import base, trees

CELL = 0.25
ROWS, COLS = 240, 320  # 80 m x 60 m
# Tree positions (meters east, north), all 15 m tall with a 4 m crown radius.
SPOTS = [(15, 15), (15, 45), (30, 30), (45, 15), (45, 45), (62, 30), (70, 50), (70, 12)]


def cone(x, y, cx, cy, height=15.0, radius=4.0):
    return np.clip(height * (1 - np.hypot(x - cx, y - cy) / radius), 0, None)


def forest_chm(skip=()):
    x, y = local_grid((ROWS, COLS), CELL)
    chm = np.zeros((ROWS, COLS))
    for i, (cx, cy) in enumerate(SPOTS):
        if i not in skip:
            chm = np.maximum(chm, cone(x, y, cx + 0.125, cy + 0.125))
    return chm


def defaults(**overrides):
    return trees.params(FakeStreamlit(overrides))


def analyze(chm, p=None):
    """The normal tree pipeline on a flat-ground synthetic forest."""
    dtm = np.zeros(chm.shape)
    layers = base.Layers(dtm=make_raster(dtm, CELL), dsm=make_raster(chm, CELL))
    grids = {"chm": chm, "dtm": dtm, "slope": layers.slope_deg, "twi": layers.twi,
             "acc_area": layers.flow["acc_area_m2"], "vari": None, "exg": None}
    p = p or defaults()
    return treefall.analyze(grids, layers.dtm.transform, layers.dtm.crs, CELL, p), p, layers.dtm


def tree_near(table, x, y):
    """The tree whose top is nearest to local (x, y) meters."""
    tx, ty = table["x"] - 500000.0, table["y"] - (4000000.0 - ROWS * CELL)
    return table.iloc[int(np.argmin(np.hypot(tx - x, ty - y)))]


def compare(chm_after, noise=0.0, seed=0, **change_p):
    res, p, dtm = analyze(forest_chm())
    rng = np.random.default_rng(seed)
    before = {"dsm": forest_chm() + rng.normal(0, noise, (ROWS, COLS)), "dtm": np.zeros((ROWS, COLS)), "ortho": None}
    after = {"dsm": chm_after + rng.normal(0, noise, (ROWS, COLS)), "dtm": np.zeros((ROWS, COLS)), "ortho": None}
    out = storm.compare_surveys(before, after, res["trees"], res["crowns"], dtm.transform, CELL, p, change_p)
    return out, res


# ---------------------------------------------------------------------------
# Wind data
# ---------------------------------------------------------------------------
def test_gust_weighted_direction_wraps_around_north():
    d = storm.weighted_wind_direction([350, 10], [10, 10])
    assert min(d, 360 - d) < 1e-6  # 0 (north), not 180
    # A strong gust from the west outweighs light wind from the east.
    assert abs(storm.weighted_wind_direction([270, 90], [20, 5]) - 270) < 1e-6


def test_wind_stats_from_station_csv(tmp_path):
    f = tmp_path / "station.csv"
    f.write_text("valid,sknt,gust,drct\n2026-11-04 20:00,10,20,350\n2026-11-04 23:00,20,40,10\n"
                 "2026-11-05 02:00,10,20,350\n2026-11-08 00:00,50,90,180\n")
    readings = storm.read_station_csv(f, "knots")
    s = storm.wind_stats(readings, "2026-11-04", "2026-11-05")
    assert s["peak_gust_mps"] == pytest.approx(40 * 0.514444, abs=0.05)  # the Nov 8 reading is outside the storm
    assert s["readings"] == 3 and min(s["wind_from_deg"], 360 - s["wind_from_deg"]) <= 5


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def test_alignment_recovers_shift_offset_and_tilt():
    x, y = local_grid((ROWS, COLS), CELL)
    dtm = 0.02 * x  # gentle hillside
    box = np.where((abs(x - 30) < 5) & (abs(y - 12) < 4), 4.0, 0.0)  # a building
    dsm = dtm + np.maximum(forest_chm(skip=(3,)), box)
    east, north = 1.3, 0.4
    tilt = np.tan(np.radians(0.3)) * (x - x.mean())

    def moved(grid):  # content appears 1.3 m east and 0.4 m north, 2.1 m higher, tilted 0.3 degrees
        return ndimage.shift(grid, (-north / CELL, east / CELL), order=1, mode="nearest") + 2.1 + tilt

    before = {"dsm": dsm, "dtm": dtm, "ortho": None}
    after = {"dsm": moved(dsm), "dtm": moved(dtm), "ortho": None}
    aligned, report = storm.align_surveys(before, after, CELL)
    assert report["shift_east_m"] == pytest.approx(east, abs=0.1)
    assert report["shift_north_m"] == pytest.approx(north, abs=0.1)
    assert report["dsm"]["offset_m"] == pytest.approx(2.1, abs=0.05)
    assert report["dsm"]["tilt_deg"] == pytest.approx(0.3, abs=0.05)
    assert report["dsm"]["rms_m"] < 0.03 and report["ok"]
    inner = (slice(20, -20), slice(20, -20))
    open_ground = (dsm - dtm)[inner] < 0.1
    # (95% of open-ground cells: the few cells right next to a wall or crown edge blur a little.)
    assert np.percentile(np.abs((aligned["dsm"] - dsm)[inner][open_ground]), 95) < 0.1


def test_poor_alignment_gives_a_warning():
    x, y = local_grid((ROWS, COLS), CELL)
    waves = 0.5 * np.sin(x / 2.0) * np.sin(y / 2.0)  # the after model is wavy: +-0.5 m bumps about 6 m across
    before = {"dsm": forest_chm(), "dtm": np.zeros((ROWS, COLS)), "ortho": None}
    after = {"dsm": forest_chm() + waves, "dtm": waves, "ortho": None}
    _, report = storm.align_surveys(before, after, CELL)
    assert not report["ok"] and "too poor" in report["warnings"][0]


def test_shift_that_changes_across_the_map_is_fitted():
    # Blocks whose shift grows steadily toward the east (a 1% stretch), plus one bad block.
    xs, ys = np.meshgrid(np.arange(50, 500, 100.0), -np.arange(50, 400, 100.0))
    blocks = pd.DataFrame({"x": xs.ravel(), "y": ys.ravel()})
    blocks["east"], blocks["north"] = 0.01 * blocks["x"] - 2.0, 0.5
    blocks.loc[3, ["east", "north"]] = [-8.0, -35.0]  # a bad match at the ragged edge
    coef, used = storm.fit_shift_model(blocks)
    assert used == len(blocks) - 1
    assert coef[1, 0] == pytest.approx(0.01, abs=1e-4) and coef[0, 0] == pytest.approx(-2.0, abs=0.02)
    assert coef[0, 1] == pytest.approx(0.5, abs=0.02)
    assert storm.fit_shift_model(blocks.head(4)) == (None, 0)  # too few blocks: a plain shift is used
    # Undoing the fitted shift puts a marker back where it belongs.
    grid = np.zeros((40, 400))
    grid[20, 300] = 1.0  # at x = 300 m (1 m cells) the content is 0.01 x 300 - 2 = 1 m east
    back = storm.warp_grid(grid, coef, 1.0)
    assert np.unravel_index(np.nanargmax(back), back.shape) == (20, 299) or back[20, 299] > 0.4


# ---------------------------------------------------------------------------
# Change classes, new logs and fall direction
# ---------------------------------------------------------------------------
def storm_damaged_forest():
    """Tree 2 (30, 30) fell toward 120 degrees leaving a log; tree 4 (45, 45) lost
    the top 40%; tree 0 (15, 15) lost its east side."""
    x, y = local_grid((ROWS, COLS), CELL)
    chm = forest_chm(skip=(2,))
    ux, uy = np.sin(np.radians(120)), np.cos(np.radians(120))
    along = (x - 30.125) * ux + (y - 30.125) * uy
    across = (x - 30.125) * uy - (y - 30.125) * ux
    chm = np.maximum(chm, np.where((along >= 0) & (along <= 12) & (np.abs(across) <= 0.3), 0.6, 0.0))
    snapped = cone(x, y, 45.125, 45.125)
    chm = np.where(snapped > 0, np.minimum(chm, 0.6 * 15.0), chm)
    side = (cone(x, y, 15.125, 15.125) > 0) & (x > 15.6)
    return np.where(side, 0.0, chm)


def test_fallen_snapped_crown_damage_and_no_change():
    out, res = compare(storm_damaged_forest(), noise=0.1)
    table = res["trees"].merge(out["changes"], on="tree")
    assert tree_near(table, 30, 30)["damage"] == "fallen"
    assert tree_near(table, 45, 45)["damage"] == "snapped top"
    assert tree_near(table, 15, 15)["damage"] == "crown damage"
    untouched = [tree_near(table, *SPOTS[i])["damage"] for i in (1, 3, 5)]
    assert untouched == ["no change"] * 3
    # The log's root end is at the trunk, so the fall direction is 120, not 300.
    fell = tree_near(table, 30, 30)
    assert storm.angle_error(fell["observed_fall_deg"], 120) <= 10
    assert fell["observed_log_m"] == pytest.approx(12, rel=0.2)
    assert len(out["new_logs"]) == 1 and out["new_logs"]["tree"].iloc[0] == fell["tree"]


def test_trees_at_the_map_edge_are_uncertain():
    out, res = compare(forest_chm())
    table = res["trees"].merge(out["changes"], on="tree")
    assert tree_near(table, 70, 12)["damage"] == "no change"
    out2, _ = compare(forest_chm(), edge_m=12.0)  # (70, 12) is 10 m from the east edge
    table2 = res["trees"].merge(out2["changes"], on="tree")
    assert tree_near(table2, 70, 12)["damage"] == "uncertain"


def test_calm_pair_has_no_false_changes():
    out, _ = compare(forest_chm(), noise=0.1, seed=3)
    counts = storm.change_counts(out["changes"])
    assert counts["any_damage"] == 0 and counts["compared"] >= 6
    assert len(out["new_logs"]) == 0 and not out["info"]["leaf_drop_suspected"]
    storm_out, _ = compare(storm_damaged_forest(), noise=0.1)
    table = storm.calm_comparison(storm.change_counts(storm_out["changes"]), counts)
    assert table.set_index("class").loc["any damage", "calm rate (%)"] == 0


def test_leaf_drop_guard():
    out, _ = compare(forest_chm() * 0.7)  # every crown 30% lower
    assert out["info"]["leaf_drop_suspected"]
    assert any("leaf drop" in w for w in out["warnings"])


def test_log_already_there_before_is_not_new():
    logs = pd.DataFrame({"log": [1, 2], "x": [10.0, 50.0], "y": [10.0, 50.0], "length_m": [8.0, 8.0],
                         "width_m": [0.5, 0.5], "axis_deg": [30.0, 100.0]})
    after = logs.assign(x=logs["x"] + 0.5, axis_deg=[35.0, 150.0])  # second one turned by 50 degrees
    new = storm.find_new_logs(logs, after)
    assert new["log"].tolist() == [2]


# ---------------------------------------------------------------------------
# Frozen predictions
# ---------------------------------------------------------------------------
def frozen_storm(tmp_path):
    res, p, dtm = analyze(forest_chm())
    table = validation.add_baselines(res["trees"])
    grids = {"crowns": res["crowns"], "exposure": res["exposure"], "grid_factor": np.array(res["grid_factor"])}
    storm.freeze_predictions(tmp_path, table, res["probs"], p, crs=dtm.crs, grids=grids)
    return res, p, dtm


def test_frozen_predictions_are_read_back_unchanged_and_tampering_is_detected(tmp_path):
    res, p, _ = frozen_storm(tmp_path)
    status = storm.verify_frozen(tmp_path)
    assert status["ok"] and len(status["combined"]) == 64
    frozen, probs, info = storm.load_frozen(tmp_path)
    assert np.allclose(frozen["hazard"], res["trees"]["hazard"]) and np.allclose(probs, res["probs"], atol=1e-5)
    assert {"baseline_height", "baseline_height_target", "baseline_random", "latitude", "longitude"} <= set(frozen)
    assert info["settings"]["weights"]["height"] == pytest.approx(0.20)

    # Changing the hazard weights later doesn't change what the analysis reads.
    new_p = defaults(**{"Weight: height": 1.0, "Weight: health": 0.0})
    assert analyze(forest_chm(), new_p)[1]["weights"]["height"] == 1.0
    assert np.allclose(storm.load_frozen(tmp_path)[0]["hazard"], res["trees"]["hazard"])

    # Editing the frozen file is detected.
    csv = tmp_path / "predictions_frozen.csv"
    csv.write_text(csv.read_text().replace(",low,", ",high,", 1))
    assert not storm.verify_frozen(tmp_path)["ok"]
    with pytest.raises(storm.FrozenError):
        storm.load_frozen(tmp_path)


def test_actual_wind_copy_changes_only_the_direction(tmp_path):
    res, _, _ = frozen_storm(tmp_path)
    storm.freeze_actual_wind(tmp_path, 0.0, CELL)  # wind from the north: trees fall south
    prevailing, _, _ = storm.load_frozen(tmp_path, storm.PREVAILING)
    actual, _, info = storm.load_frozen(tmp_path, storm.ACTUAL)
    assert np.allclose(actual["hazard"], prevailing["hazard"])
    assert (actual["fall_dir"] == 180).all() and (prevailing["fall_dir"] == 90).all()
    assert info["label"] == storm.ACTUAL and info["settings"]["wind_from"] == 0.0


def test_nothing_frozen_gives_a_clear_error(tmp_path):
    assert not storm.verify_frozen(tmp_path)["exists"]
    with pytest.raises(storm.FrozenError):
        storm.load_frozen(tmp_path)


# ---------------------------------------------------------------------------
# Baselines and prediction metrics
# ---------------------------------------------------------------------------
def fake_trees(n=400, seed=0):
    rng = np.random.default_rng(seed)
    hazard = rng.uniform(0, 100, n)
    t = pd.DataFrame({"tree": np.arange(1, n + 1), "hazard": hazard, "height": rng.uniform(5, 35, n),
                      "class": np.where(hazard > 60, "high", np.where(hazard >= 35, "moderate", "low")),
                      "judged": True})
    return validation.add_baselines(t)


def test_baselines():
    t = pd.DataFrame({"tree": [1, 2, 3], "height": [15.0, 25.0, 20.0], "hazard": [10.0, 20.0, 30.0],
                      "can reach trail": [True, False, False]})
    b = validation.add_baselines(t, seed=42)
    assert b["baseline_height"].tolist() == [3, 1, 2]  # the 25 m tree ranks above the 15 m tree
    assert b["baseline_height_target"].tolist() == [1, 2, 3]  # the tree that can reach the trail comes first
    assert b["baseline_random"].tolist() == validation.add_baselines(t, seed=42)["baseline_random"].tolist()
    big = pd.DataFrame({"tree": range(50), "height": 10.0, "hazard": 1.0})
    assert (validation.add_baselines(big, 1)["baseline_random"] != validation.add_baselines(big, 2)["baseline_random"]).any()


def test_auc_matches_a_hand_worked_example():
    # Damaged scores 3 and 5; undamaged 1, 2, 4. Pairs won by the damaged tree: 3>1, 3>2, 5>1, 5>2, 5>4 = 5 of 6.
    assert validation.auc([1, 2, 3, 4, 5], [0, 0, 1, 0, 1]) == pytest.approx(5 / 6)
    assert validation.auc([1, 1, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.5)  # all tied
    assert np.isnan(validation.auc([1, 2, 3], [0, 0, 0]))


def test_auc_is_one_when_the_top_scores_are_the_damaged_trees():
    t = fake_trees()
    t["severe"] = t["hazard_rank"] <= 20
    t["any_damage"] = t["severe"]
    m = storm.prediction_metrics(t, n_boot=200)["outcomes"]["severe"]
    auc = m["auc"].set_index("ranking")
    assert auc.loc["full hazard score", "AUC"] == pytest.approx(1.0)
    assert m["top_n"]["share caught (%)"].iloc[0] == 100  # all 20 are in the top 10% (40 trees)
    assert m["ratio"]["ratio"] == float("inf") and m["ratio"]["fisher_p"] < 0.001
    assert m["odds"]["odds_ratio_per_10"] > 2 and not m["small_sample"]


def test_random_damage_gives_auc_near_half_and_the_interval_covers_it():
    aucs = []
    for seed in range(30):
        t = fake_trees(seed=seed)
        damaged = np.random.default_rng(100 + seed).random(len(t)) < 0.1
        aucs.append(validation.auc(t["hazard"], damaged))
    assert abs(np.mean(aucs) - 0.5) < 0.1
    t = fake_trees(seed=1)
    damaged = np.random.default_rng(7).random(len(t)) < 0.1
    value, low, high = validation.bootstrap_auc(t["hazard"], damaged, n_boot=500, seed=0)
    assert low <= 0.5 <= high and low <= value <= high


def test_few_damaged_trees_gives_a_small_sample_flag():
    t = fake_trees()
    t["severe"] = t["hazard_rank"] <= 4
    t["any_damage"] = t["severe"]
    assert storm.prediction_metrics(t, n_boot=50)["outcomes"]["severe"]["small_sample"]


# ---------------------------------------------------------------------------
# Fall direction
# ---------------------------------------------------------------------------
def direction_case(observed):
    n = len(observed)
    frozen = pd.DataFrame({"tree": np.arange(1, n + 1)})
    # Like the real forecast: 40% "any direction" + 60% a bump toward the east.
    probs = 0.4 / treefall.N_BINS + 0.6 * treefall.von_mises(np.full(n, 90.0), 2.0)
    table = frozen.assign(observed_fall_deg=observed, judged=True, downhill_dir=np.nan, lean_dir=np.nan)
    return storm.direction_metrics(table, {storm.PREVAILING: (frozen, probs)}, wind_from_deg=270.0)


def test_direction_perfect_prediction():
    d = direction_case(np.full(25, 90.0))
    row = d["table"].set_index("model").loc[storm.PREVAILING]
    assert row["median error (°)"] == 0 and row["within 45° (%)"] == 100
    like = d["likelihood"].iloc[0]
    assert like["log-likelihood"] > like["uniform guess"] and like["better than guessing"]
    assert d["table"].set_index("model").loc["wind only (actual wind)", "median error (°)"] == 0


def test_direction_random_falls_are_no_better_than_guessing():
    observed = np.random.default_rng(0).uniform(0, 360, 2000)
    d = direction_case(observed)
    like = d["likelihood"].iloc[0]
    # Per tree, the log-likelihood is close to (slightly below) the uniform value.
    assert (like["log-likelihood"] - like["uniform guess"]) / 2000 == pytest.approx(0, abs=0.3)
    assert not like["better than guessing"]
    assert d["table"].set_index("model").loc[storm.PREVAILING, "median error (°)"] == pytest.approx(90, abs=8)


# ---------------------------------------------------------------------------
# Whole storm, results files and design criteria
# ---------------------------------------------------------------------------
def test_storm_results_files_and_design_criteria(tmp_path):
    work = tmp_path
    criteria = validation.load_criteria(work / "config")
    table = validation.evaluate_criteria(criteria, work)
    assert {"storm_auc_severe", "storm_beats_baseline", "fall_direction_mae_deg",
            "calm_false_change_rate"} <= set(table["criterion"]) and len(table) == 12
    assert (table["status"] == "no data yet").all()  # no storms analyzed: no crash

    storms_dir = work / "results" / "storms"
    gale = {**storm.STORM_TEMPLATE, "name": "Test gale", "peak_gust_mps": 25.0, "wind_from_deg": 300.0}
    calm = {**storm.STORM_TEMPLATE, "name": "Calm pair", "is_calm_control": True}
    for s, chm_after in ((gale, storm_damaged_forest()), (calm, forest_chm())):
        folder = storms_dir / storm.storm_slug(s["name"])
        res, p, dtm = frozen_storm(folder)
        storm.freeze_actual_wind(folder, 300.0, CELL)
        frozen, probs, info = storm.load_frozen(folder)
        actual, probs_actual, _ = storm.load_frozen(folder, storm.ACTUAL)
        grids = storm.load_frozen_grids(folder)
        rng = np.random.default_rng(1)
        before = {"dsm": forest_chm() + rng.normal(0, 0.05, (ROWS, COLS)), "dtm": np.zeros((ROWS, COLS)), "ortho": None}
        after = {"dsm": chm_after + rng.normal(0, 0.05, (ROWS, COLS)), "dtm": np.zeros((ROWS, COLS)), "ortho": None}
        out = storm.compare_surveys(before, after, frozen, grids["crowns"], dtm.transform, CELL, info["settings"])
        joined = storm.join_outcomes(frozen, out["changes"])
        prediction = storm.prediction_metrics(joined, s["peak_gust_mps"], n_boot=50)
        direction = storm.direction_metrics(joined, {storm.PREVAILING: (frozen, probs),
                                                     storm.ACTUAL: (actual, probs_actual)}, s["wind_from_deg"])
        landing = storm.landing_metrics(joined, out["new_logs"], grids, out["chm_before"], dtm.transform, CELL,
                                        info["settings"])
        storm.save_storm_results(folder, s, out, joined, prediction, direction, landing, storm.verify_frozen(folder))
        if s is gale:
            # Wind from 300 pushes trees toward 120, which is where the tree fell.
            row = direction["table"].set_index("model").loc[storm.ACTUAL]
            assert row["median error (°)"] <= 10
            assert landing["strip"]["within reach (any direction)"] == 1

    results = json.loads((storms_dir / "test_gale" / "storm_results.json").read_text())
    assert results["counts"]["fallen"] == 1 and results["counts"]["snapped top"] == 1 and results["new_logs"] == 1
    pooled = storm.pool_results(storms_dir, [gale, calm], n_boot=50)
    assert pooled["storms"] == ["Test gale"] and pooled["criteria"]["calm_false_change_rate"] == 0
    assert pooled["criteria"]["fall_direction_mae_deg"] <= 10
    table = validation.evaluate_criteria(criteria, work).set_index("criterion")
    assert table.loc["calm_false_change_rate", "status"] == "pass"
    assert table.loc["fall_direction_mae_deg", "status"] == "pass"
    assert table.loc["storm_auc_severe", "status"] in ("pass", "fail")


def test_storms_yaml_round_trip(tmp_path):
    assert storm.load_storms(tmp_path) == []
    s = {**storm.STORM_TEMPLATE, "name": "Nov 4 gale", "date_start": "2026-11-04", "before_survey": "forest_2026-11-01",
         "after_survey": "forest_2026-11-06", "peak_gust_mps": 22.5}
    storm.save_storms(tmp_path, [s])
    back = storm.load_storms(tmp_path)
    assert back[0]["name"] == "Nov 4 gale" and back[0]["peak_gust_mps"] == 22.5 and back[0]["is_calm_control"] is False
    assert storm.storm_slug("Nov 4 gale!") == "nov_4_gale"


# ---------------------------------------------------------------------------
# Field confirmation
# ---------------------------------------------------------------------------
def test_blind_field_sheet_and_comparison():
    rng = np.random.default_rng(0)
    n = 200
    table = pd.DataFrame({"tree": np.arange(n), "damage": ["fallen"] * 6 + ["snapped top"] * 4 + ["no change"] * (n - 10),
                          "hazard": rng.uniform(0, 100, n), "height": 20.0, "x": 0.0, "y": 0.0, "fall_dir": 90.0,
                          "observed_fall_deg": np.nan, "latitude": 39.0, "longitude": -77.0})
    sheet, key = storm.field_sheet(table)
    assert len(sheet) == 20 and sheet["code"].is_unique and sheet["code"].between(1000, 9999).all()
    assert not {"damage", "hazard", "tree", "height"} & set(sheet.columns)  # blind: nothing about the app's answer
    assert (key["damage"] != "no change").sum() == 10

    filled = sheet.copy()
    filled["status"] = "found"
    truth = key.set_index("code")["damage"].map({"fallen": "uprooted", "snapped top": "top broken", "no change": "none"})
    filled["storm_damage"] = filled["code"].map(truth)
    one_wrong = key[key["damage"] == "fallen"]["code"].iloc[0]
    filled.loc[filled["code"] == one_wrong, "storm_damage"] = "none"  # the app was wrong about one tree
    filled.loc[filled["code"] == key[key["damage"] == "fallen"]["code"].iloc[1], "fall_azimuth_deg"] = "135"
    out = storm.field_compare(key, filled)
    t = out["table"].set_index("class")
    assert t.loc["fallen", "precision (%)"] == pytest.approx(100 * 5 / 6) and t.loc["fallen", "recall (%)"] == 100
    assert t.loc["snapped top", "precision (%)"] == 100
    assert (out["trees"]["best_fall_deg"] == 135).sum() == 1
