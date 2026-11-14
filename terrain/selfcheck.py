"""End-to-end self-check on a small made-up forest.

Runs the whole chain (find trees -> hazard -> fall direction -> exposure ->
features -> sensitivity -> blind field sheet -> fake field results -> check
plots -> downed logs -> frozen predictions -> fake storm and calm pair ->
design criteria -> summary) and reports which output file each step produced.
Used by tests/test_end_to_end.py and scripts/audit.py, so a break anywhere in
the chain is caught without needing real data.
"""

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.transform import from_origin

from terrain import io, storm, validation


class _Tab:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class DefaultWidgets:
    """Stands in for `streamlit` when calling an app's params(st) without a window.

    Every widget simply returns its starting value (or an override given by its
    label), so tests and the self-check use the same starting values as the app.
    Text output (markdown, caption...) is ignored.
    """

    def __init__(self, overrides=None):
        self.overrides = overrides or {}

    def _value(self, label, default):
        return self.overrides.get(label, default)

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, **kw):
        return self._value(label, value)

    def number_input(self, label, min_value=None, max_value=None, value=None, step=None, **kw):
        return self._value(label, value)

    def select_slider(self, label, options=(), value=None, **kw):
        return self._value(label, value if value is not None else options[0])

    def selectbox(self, label, options, index=0, **kw):
        return self._value(label, list(options)[index])

    def radio(self, label, options, index=0, **kw):
        return self._value(label, list(options)[index])

    def checkbox(self, label, value=False, **kw):
        return self._value(label, value)

    def text_input(self, label, value="", **kw):
        return self._value(label, value)

    def file_uploader(self, label, **kw):
        return self._value(label, None)

    def columns(self, spec, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [self] * n

    @contextlib.contextmanager
    def expander(self, *a, **kw):
        yield self

    @contextlib.contextmanager
    def form(self, *a, **kw):
        yield self

    def tabs(self, labels, **kw):
        return [_Tab() for _ in labels]  # each tab is a "with" block that does nothing

    def __getattr__(self, name):  # markdown, caption, write, info, warning, ...
        return lambda *a, **kw: None


CELL = 0.5
SHAPE = (160, 200)  # 100 m x 80 m
CRS = "EPSG:32618"
SPOTS = [(x, y) for x in range(12, 95, 11) for y in range(12, 75, 12)]  # 8 x 6 = 48 trees


def synthetic_forest(skip=()):
    """Cone-shaped trees 12-28 m tall on a gentle slope. Returns (ground, canopy height)."""
    rows, cols = SHAPE
    r, c = np.indices(SHAPE)
    x, y = (c + 0.5) * CELL, (rows - r - 0.5) * CELL
    ground = 0.05 * x
    chm = np.zeros(SHAPE)
    for i, (cx, cy) in enumerate(SPOTS):
        if i in skip:
            continue
        height = 12 + (i * 7) % 17  # 12 to 28 m, mixed up across the map
        chm = np.maximum(chm, np.clip(height * (1 - np.hypot(x - cx - 0.25, y - cy - 0.25) / 4.0), 0, None))
    return ground, chm


def _raster(data):
    return io.Raster(data=np.asarray(data, dtype=float), transform=from_origin(500000.0, 4000000.0, CELL, CELL), crs=CRS)


def run(work_dir, n_boot: int = 100) -> dict:
    """Run every step in `work_dir` (config/ and results/ are made inside it).

    Returns {step name: output path}; a step that failed has the value None.
    Raises nothing for a missing output, so the caller can list every failure.
    """
    from terrain.apps import base, trees

    work = Path(work_dir)
    results = work / "results"
    steps = {}

    def done(step, path):
        steps[step] = path if path is not None and Path(path).exists() else None

    # --- the tree pipeline: detect -> hazard -> direction -> exposure -> features
    ground, chm = synthetic_forest()
    layers = base.Layers(dtm=_raster(ground), dsm=_raster(ground + chm))
    p = trees.params(DefaultWidgets(), {"width_m": 100.0, "season": "summer leaf-on"})
    result = trees.run(layers, p)
    raw = result.raw
    table = raw["trees"]
    tree_csv = results / "selfcheck_trees.csv"
    results.mkdir(parents=True, exist_ok=True)
    table.to_csv(tree_csv, index=False)
    ok = (len(table) >= 40 and table["hazard"].between(0, 100).all() and table["fall_dir"].between(0, 360).all()
          and raw["exposure"].max() > 0 and [s.title for s in result.sections] == trees.TREE_TABS[:5])
    done("tree pipeline (detect, hazard, direction, exposure, features)", tree_csv if ok else None)

    # --- sensitivity
    F = validation.factor_matrix(table, list(p["weights"]))
    rand = validation.sensitivity_random(F, p["weights"])
    done("weight sensitivity", validation.save_result(work, "results/sensitivity.json",
                                                      {"weight_stability_top20": rand["weight_stability_top20"]}, [rand["summary"]],
                                                      {"overlap": rand["overlap"]}))

    # --- blind field sheet, then a made-up field visit (defects in the high group, heights 1 m lower)
    field_dir = results / "field_check"
    field_dir.mkdir(parents=True, exist_ok=True)
    sheet, key = validation.blind_field_sample(table, n=8)
    sheet.to_csv(field_dir / "field_sheet.csv", index=False)
    key.to_csv(field_dir / "field_key.csv", index=False)
    done("blind field export", field_dir / "field_sheet.csv" if not {"hazard", "group", "height", "tree"} & set(sheet) else None)
    filled = sheet.copy()
    filled["status"] = "found"
    group = filled["code"].map(key.set_index("code")["group"])
    filled["broken_or_dead_top"] = np.where(group == "high", "y", "n")
    filled["observed_height_m"] = filled["code"].map(key.set_index("code")["height"]) - 1.0
    field = validation.field_import(key, filled, n_boot=n_boot)
    done("field import", validation.save_result(work, "results/field_check_results.json", field["criteria"], field["sentences"],
                                                {"auc": field["auc"], "height_pairs": field["height_pairs"]}))

    # --- detection check plot: a 15 m circle in the middle; the "field count" is the true number of trees in it
    plots = pd.DataFrame({"plot_id": ["A"], "x": [500000.0 + 50.0], "y": [4000000.0 - 40.0], "radius_m": [15.0]})
    truth = sum(np.hypot(cx + 0.25 - 50.0, cy + 0.25 - 40.0) <= 15.0 for cx, cy in SPOTS)
    plots["field_count_over_10m"] = truth
    det = validation.detection_rates(plots, table)
    done("detection check plots", validation.save_result(work, "results/detection_plots.json", det["criteria"], det["sentences"]))

    # --- downed logs: made-up logs on the steeper east side, tested against random points
    rng = np.random.default_rng(42)
    logs = pd.DataFrame({"log": range(12), "x": 500000.0 + rng.uniform(60, 95, 12), "y": 4000000.0 - rng.uniform(10, 70, 12),
                         "length_m": 6.0, "width_m": 0.5, "axis_deg": 90.0})
    x_grid = (np.indices(SHAPE)[1] + 0.5) * CELL
    log_test = validation.log_factor_test(logs, {"east-ness": x_grid}, np.ones(SHAPE, dtype=bool), layers.dtm.transform, CELL)
    done("downed log test", validation.save_result(work, "results/log_validation.json", log_test["criteria"], log_test["sentences"],
                                                   {"axes_deg": logs["axis_deg"].tolist(),
                                                    "values": {k: [a.tolist(), b.tolist()] for k, (a, b) in log_test["values"].items()}}))

    # --- timing
    speed = validation.hectares_per_hour(0.8, 0.05, 2.0)
    done("timing", validation.save_result(work, "results/timing.json", {"hectares_per_hour": speed}, ["Self-check timing."]))

    # --- storm: freeze, then a made-up storm (one tree gone, one top snapped) and a calm pair
    storms_dir = results / "storms"
    gale = {**storm.STORM_TEMPLATE, "name": "Selfcheck gale", "peak_gust_mps": 25.0, "wind_from_deg": 270.0}
    calm = {**storm.STORM_TEMPLATE, "name": "Selfcheck calm", "is_calm_control": True}
    _, after_storm = synthetic_forest(skip=(20,))
    rows, cols = SHAPE
    r, c = np.indices(SHAPE)
    x, y = (c + 0.5) * CELL, (rows - r - 0.5) * CELL
    cx, cy = SPOTS[30]
    after_storm = np.where(np.hypot(x - cx, y - cy) < 4.2, np.minimum(after_storm, 6.0), after_storm)  # snapped at 6 m
    for s, chm_after in ((gale, after_storm), (calm, chm)):
        folder = storms_dir / storm.storm_slug(s["name"])
        grids = {"crowns": raw["crowns"].astype("int32"), "exposure": raw["exposure"], "grid_factor": np.array(raw["grid_factor"])}
        grids.update({f"mask_{k}": v.astype("uint8") for k, v in raw["masks"].items()})
        storm.freeze_predictions(folder, table, raw["probs"], p, crs=CRS, grids=grids)
        storm.freeze_actual_wind(folder, 270.0, CELL)
        frozen, probs, info = storm.load_frozen(folder)
        actual, probs_actual, _ = storm.load_frozen(folder, storm.ACTUAL)
        frozen_grids = storm.load_frozen_grids(folder)
        noise = np.random.default_rng(1)
        before = {"dsm": ground + chm + noise.normal(0, 0.03, SHAPE), "dtm": ground, "ortho": None}
        after = {"dsm": ground + chm_after + noise.normal(0, 0.03, SHAPE), "dtm": ground, "ortho": None}
        out = storm.compare_surveys(before, after, frozen, frozen_grids["crowns"], layers.dtm.transform, CELL, info["settings"])
        joined = storm.join_outcomes(frozen, out["changes"])
        prediction = storm.prediction_metrics(joined, s["peak_gust_mps"], n_boot=n_boot)
        direction = storm.direction_metrics(joined, {storm.PREVAILING: (frozen, probs), storm.ACTUAL: (actual, probs_actual)},
                                            s["wind_from_deg"])
        landing = storm.landing_metrics(joined, out["new_logs"], frozen_grids, out["chm_before"], layers.dtm.transform, CELL,
                                        info["settings"])
        saved = storm.save_storm_results(folder, s, out, joined, prediction, direction, landing, storm.verify_frozen(folder))
        if s is gale:
            done("frozen predictions", folder / "predictions_frozen.csv" if storm.verify_frozen(folder)["ok"] else None)
            done("fake storm pair", folder / "storm_results.json" if saved["counts"]["severe"] == 2 else None)
        else:
            done("calm pair", folder / "storm_results.json" if saved["counts"]["any_damage"] == 0 else None)
    storm.pool_results(storms_dir, [gale, calm], n_boot=n_boot)
    done("pooled storm results", storms_dir / "pooled_results.json")

    # --- design criteria and the summary
    criteria = validation.load_criteria(work / "config")
    evaluated = validation.evaluate_criteria(criteria, work)
    evaluated.to_csv(results / "selfcheck_criteria.csv", index=False)
    done("design criteria", results / "selfcheck_criteria.csv" if len(evaluated) == len(validation.CRITERIA) else None)
    text = validation.write_summary(work, work / "config")
    done("summary", results / "summary.md" if validation.GOAL in text else None)
    return {"steps": steps, "criteria": evaluated, "trees": len(table)}
