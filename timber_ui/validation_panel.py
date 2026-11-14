"""Screens for the validation tools: weight sensitivity, the blind field check,
detection check plots, downed-log tests and the design criteria.

The math is in terrain/validation.py. Each tool saves a results file with its
measured design criteria, which the Design criteria table reads.
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import ndimage

from terrain import io, treefall, validation
from terrain.apps.base import ShapeSpec
from terrain.apps.trees import canopy_height
from timber_ui import charts, maps
from timber_ui.paths import CONFIG_DIR, RESULTS_DIR, WORK_DIR
from timber_ui.storm_panel import _card_image
from timber_ui.widgets import shape_editor

FIELD_DIR = RESULTS_DIR / "field_check"
CARD_STYLE = ("<style>body{font-family:sans-serif}.card{display:inline-block;width:320px;margin:8px;padding:8px;"
              "border:1px solid #999;page-break-inside:avoid}.card img{width:100%}h2{margin:2px 0}"
              "p{margin:4px 0;font-size:13px}</style>")


# ---------------------------------------------------------------------------
# Sensitivity (shown under the Trees & hazard part)
# ---------------------------------------------------------------------------
def sensitivity_box(survey, layers, p, result):
    with st.expander("Sensitivity: do the results depend on the weights I chose?"):
        trees = result.raw["trees"]
        names = list(p["weights"])
        F = validation.factor_matrix(trees, names)
        c1, c2 = st.columns(2)
        concentration = c1.slider("How close the random weight sets stay to mine (concentration)", 5.0, 200.0, 20.0, 5.0,
                                  help="Bigger = closer to the chosen weights. 20 is the starting value.")
        seed = c2.number_input("Seed", 0, 9999, 42, 1, key="sens_seed")
        one = validation.sensitivity_one_at_a_time(F, p["weights"])
        rand = validation.sensitivity_random(F, p["weights"], 500, concentration, 20, int(seed))
        st.markdown(f"**{rand['summary']}**")
        left, right = st.columns(2)
        with left:
            st.markdown("**One weight at a time** (x 0.5 and x 1.5, the rest rescaled)")
            st.dataframe(one, hide_index=True, width="stretch")
        with right:
            maps.show_fig(charts.overlap_histogram(rand["overlap"]))
            stable = pd.DataFrame({"tree": trees["tree"], "hazard": trees["hazard"].round(1),
                                   "top-20 stability (%)": rand["stability_percent"].round(0)})
            st.markdown("**How often each tree lands in the top 20** (500 random weight sets)")
            st.dataframe(stable.sort_values("top-20 stability (%)", ascending=False).head(30), hide_index=True, width="stretch")
        validation.save_result(WORK_DIR, "results/sensitivity.json",
                               {"weight_stability_top20": rand["weight_stability_top20"]}, [rand["summary"]],
                               {"overlap": rand["overlap"], "one_at_a_time": one, "concentration": concentration,
                                "survey": survey.name})


# ---------------------------------------------------------------------------
# Blind field check + detection check plots (replaces the old Field check part)
# ---------------------------------------------------------------------------
def _tree_booklet(key, sheet, survey, crowns) -> str:
    cards = []
    for (_, k), (_, s) in zip(key.iterrows(), sheet.iterrows()):
        img = _card_image(survey, k["x"], k["y"], crowns, int(k["tree"]))
        bearing = (f"{s['trail_distance_m']:.0f} m at {s['trail_bearing_deg']:.0f}° from the nearest trail point"
                   if pd.notna(s["trail_distance_m"]) else "no trail found nearby")
        cards.append(f'<div class="card"><h2>Code {int(k["code"])}</h2><img src="data:image/jpeg;base64,{img}">'
                     f"<p>{s['latitude']:.6f}, {s['longitude']:.6f}</p><p>{bearing}</p></div>")
    return (f"<html><head><meta charset='utf-8'>{CARD_STYLE}</head><body><h1>Blind field check</h1><p>The tree is "
            "outlined in yellow (40 m x 40 m photo, north up, 10 m scale bar). No hazard information is shown on "
            "purpose.</p>" + "".join(cards) + "</body></html>")


def field_check(survey, layers, p, result):
    trees, crowns = result.raw["trees"], result.raw["crowns"]
    st.markdown("**Blind field check.** When you look at a tree in the woods you must not know whether the app "
                "called it high or low hazard. The sheet you take along has only a code and a position.")
    with st.expander("1. Export the blind field sheet", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        n = c1.slider("Trees per group (N high + N low)", 5, 100, 30, 5)
        near = c2.checkbox("Only trees near a trail", True, help="So every tree can be reached on foot.")
        dist = c3.slider("Within this distance of a trail or road (m)", 5.0, 200.0, 30.0, 5.0, disabled=not near)
        seed = c4.number_input("Seed", 0, 9999, 42, 1, key="field_seed")
        if near and not np.isfinite(trees["trail_dist_m"]).any():
            st.caption("No trail or road was detected, so the distance limit is not used.")
            near = False
        sheet, key = validation.blind_field_sample(trees, n, 10.0, dist if near else None, int(seed))
        st.caption(f"{(key['group'] == 'high').sum()} high-hazard trees (the highest scores) and "
                   f"{(key['group'] == 'low').sum()} low-hazard trees (random, from the lowest third), all taller than "
                   "10 m, shuffled.")
        if st.button("Save the sheet, key and photo booklet", type="primary"):
            FIELD_DIR.mkdir(parents=True, exist_ok=True)
            sheet.to_csv(FIELD_DIR / "field_sheet.csv", index=False)
            key.to_csv(FIELD_DIR / "field_key.csv", index=False)
            with st.spinner("Drawing one photo card per tree..."):
                (FIELD_DIR / "field_booklet.html").write_text(_tree_booklet(key, sheet, survey, crowns), encoding="utf-8")
            st.success(f"Saved in `{FIELD_DIR}`: field_sheet.csv (take along), field_booklet.html (print), "
                       "field_key.csv (stays home).")
        if (FIELD_DIR / "field_sheet.csv").exists():
            c1, c2 = st.columns(2)
            c1.download_button("Field sheet (CSV)", (FIELD_DIR / "field_sheet.csv").read_bytes(), "field_sheet.csv", "text/csv")
            if (FIELD_DIR / "field_booklet.html").exists():
                c2.download_button("Photo booklet (HTML)", (FIELD_DIR / "field_booklet.html").read_bytes(),
                                   "field_booklet.html", "text/html")
        st.caption("Fill in: status (found / not_found / inaccessible), observed_height_m (clinometer), and y or n "
                   "for each defect column.")

    with st.expander("2. Import the filled sheet"):
        upload = st.file_uploader("Filled field_sheet.csv", type=["csv"], key="blind_upload")
        key_upload = st.file_uploader("field_key.csv (only if it is not in the results folder)", type=["csv"], key="key_upload")
        key_path = FIELD_DIR / "field_key.csv"
        if upload is not None:
            if key_upload is None and not key_path.exists():
                st.error("The key is missing. Upload field_key.csv too.")
            else:
                try:
                    res = validation.field_import(pd.read_csv(key_upload if key_upload is not None else key_path),
                                                  pd.read_csv(upload))
                    _show_field_results(res)
                except Exception as err:
                    st.error(f"Could not read the sheet: {err}")

    with st.expander("3. Detection check plots (does the app find the trees that are there?)"):
        _check_plots(survey, trees)


def _show_field_results(res):
    d = res["defects"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trees found", res["checked"])
    c2.metric("Defects: high group", f"{d['high'][0]} of {d['high'][1]}")
    c3.metric("Defects: low group", f"{d['low'][0]} of {d['low'][1]}")
    c4.metric("Ratio high / low", d["ratio_text"])
    for sentence in res["sentences"]:
        st.write(sentence)
    left, right = st.columns(2)
    with left:
        st.markdown("**Defects by group** (Fisher's exact test uses this table)")
        st.dataframe(d["table_2x2"], width="stretch")
        st.markdown("**Tree height: app vs field**")
        st.dataframe(res["heights"], hide_index=True, width="stretch")
        if "snag_2x2" in res:
            st.markdown("**'Possibly dead' flag vs what was seen**")
            st.dataframe(res["snag_2x2"], width="stretch")
    with right:
        if len(res["auc"]):
            maps.show_fig(charts.auc_figure(res["auc"], "Which ranking finds the defective trees?"))
        if len(res["height_pairs"]):
            maps.show_fig(charts.height_scatter(res["height_pairs"]))
    validation.save_result(WORK_DIR, "results/field_check_results.json", res["criteria"], res["sentences"],
                           {"checked": res["checked"], "auc": res["auc"], "heights": res["heights"],
                            "height_pairs": res["height_pairs"], "defects": {k: v for k, v in d.items() if k != "table_2x2"}})
    res["trees"].to_csv(RESULTS_DIR / "field_check_results.csv", index=False)
    st.caption(f"Saved: `{RESULTS_DIR / 'field_check_results.json'}` and `.csv`.")


def _plot_booklet(plots, survey) -> str:
    cards = []
    for _, pl in plots.iterrows():
        img = _card_image(survey, pl["x"], pl["y"], None, None, size_m=max(40.0, 3 * pl["radius_m"]), circle_m=pl["radius_m"])
        cards.append(f'<div class="card"><h2>Plot {pl["plot_id"]}</h2><img src="data:image/jpeg;base64,{img}">'
                     f"<p>{pl['latitude']:.6f}, {pl['longitude']:.6f} · radius {pl['radius_m']:g} m</p>"
                     "<p>trees over 10 m inside the circle: ______</p></div>")
    return (f"<html><head><meta charset='utf-8'>{CARD_STYLE}</head><body><h1>Detection check plots</h1><p>Count every "
            "tree taller than 10 m whose trunk is inside the circle.</p>" + "".join(cards) + "</body></html>")


def _check_plots(survey, trees):
    st.write("Choose 3-5 circular plots, count the trees taller than 10 m in each in the field, and compare with the "
             "app's count. Click plot centers on the map, or upload a CSV with plot_id, latitude, longitude, radius_m.")
    radius = st.slider("Plot radius (m)", 5.0, 30.0, 12.0, 1.0)
    clicked = shape_editor("check_plots", ShapeSpec("points", "Check plot centers", "Each click is one plot center."),
                           survey, survey.ortho is not None)
    upload = st.file_uploader("...or a plots CSV", type=["csv"], key="plots_csv")
    crs = survey.dtm.crs
    if upload is not None:
        plots = pd.read_csv(upload)
        plots.columns = [str(c).strip().lower() for c in plots.columns]
        if "radius_m" not in plots:
            plots["radius_m"] = radius
        xs, ys = io.latlon_to_map(plots["longitude"].to_numpy(), plots["latitude"].to_numpy(), crs)
        plots["x"], plots["y"] = xs, ys
    else:
        xy = np.array([s["xy"][0] for s in clicked], dtype=float).reshape(-1, 2)
        plots = pd.DataFrame({"plot_id": np.arange(1, len(xy) + 1), "x": xy[:, 0], "y": xy[:, 1], "radius_m": radius})
        if len(plots):
            lon, lat = io.map_to_latlon(plots["x"].to_numpy(), plots["y"].to_numpy(), crs)
            plots["latitude"], plots["longitude"] = np.round(lat, 7), np.round(lon, 7)
    if plots.empty:
        return
    if st.button("Save the plot sheet and cards"):
        FIELD_DIR.mkdir(parents=True, exist_ok=True)
        validation.plot_sheet(plots).to_csv(FIELD_DIR / "plot_sheet.csv", index=False)
        (FIELD_DIR / "plot_booklet.html").write_text(_plot_booklet(plots, survey), encoding="utf-8")
        st.success(f"Saved plot_sheet.csv and plot_booklet.html in `{FIELD_DIR}`.")
    st.download_button("Plot sheet (CSV)", validation.plot_sheet(plots).to_csv(index=False), "plot_sheet.csv", "text/csv")
    filled = st.file_uploader("Filled plot_sheet.csv", type=["csv"], key="plots_filled")
    if filled is not None:
        counts = pd.read_csv(filled)
        counts.columns = [str(c).strip().lower() for c in counts.columns]
        xs, ys = io.latlon_to_map(counts["longitude"].to_numpy(), counts["latitude"].to_numpy(), crs)
        counts["x"], counts["y"] = xs, ys
        res = validation.detection_rates(counts, trees)
        st.dataframe(res["table"], hide_index=True, width="stretch")
        st.metric("Overall detection rate (sum of app counts / sum of field counts)", f"{res['overall']:.2f}")
        st.caption("This compares counts, so a tree the app missed and an extra tree it invented cancel out. "
                   "Look at the app and field counts side by side as well.")
        validation.save_result(WORK_DIR, "results/detection_plots.json", res["criteria"], res["sentences"],
                               {"plots": res["table"]})


# ---------------------------------------------------------------------------
# Downed log validation (shown under the Downed logs part)
# ---------------------------------------------------------------------------
def log_validation(survey, layers, p, result):
    raw = result.raw
    masks, transform, cell = raw["masks"], layers.dtm.transform, layers.cell[0]
    with st.expander("Do downed logs lie where the model says trees are at risk?", expanded=True):
        shape = layers.dtm.shape
        paths = masks.get("trail", np.zeros(shape, dtype=bool)) | masks.get("road", np.zeros(shape, dtype=bool))
        skip = st.checkbox("Leave out logs on or within 5 m of trails and roads", True,
                           help="Trail crews cut and move those logs, so where they lie says little.")
        manual_file = st.file_uploader("Logs measured in the field (CSV: latitude, longitude, axis_azimuth_deg, length_m)",
                                       type=["csv"], key="manual_logs")
        detected = raw["logs"]
        sets = {"Detected logs": validation.logs_away_from(detected, paths, transform) if skip else detected}
        if manual_file is not None:
            m = pd.read_csv(manual_file)
            m.columns = [str(c).strip().lower() for c in m.columns]
            xs, ys = io.latlon_to_map(m["longitude"].to_numpy(), m["latitude"].to_numpy(), layers.dtm.crs)
            manual = pd.DataFrame({"log": np.arange(1, len(m) + 1), "x": xs, "y": ys, "length_m": m["length_m"],
                                   "width_m": 0.5, "axis_deg": m["axis_azimuth_deg"] % 180})
            sets["Field-measured logs"] = manual
            st.caption(f"The detector also found {validation.match_logs(manual, detected)} of the {len(manual)} "
                       "field-measured logs (a detected log within 3 m).")
        chm = canopy_height(layers)
        # Edge exposure: 1 right at a detected opening (clearing, trail, road), 0 from 10 m inside the forest.
        openings = paths | masks.get("clearing", np.zeros(shape, dtype=bool))
        edge = np.clip(1 - ndimage.distance_transform_edt(~openings, sampling=cell) / 10.0, 0, 1) if openings.any() \
            else np.zeros(shape)
        factors = {"slope (°)": layers.slope_deg, "wetness (TWI)": layers.twi, "edge exposure (0-1)": edge,
                   "canopy height (m)": chm}
        forbidden = paths | masks.get("building", np.zeros(shape, dtype=bool)) | (raw["log_labels"] > 0)
        forest = np.isfinite(chm) & ~forbidden & ~masks.get("clearing", np.zeros(shape, dtype=bool))
        for title, logs in sets.items():
            st.markdown(f"#### {title} ({len(logs)})")
            if len(logs) < 3:
                st.info("Fewer than 3 logs: not enough to test.")
                continue
            test = validation.log_factor_test(logs, factors, forest, transform, cell)
            left, right = st.columns([3, 2])
            with left:
                st.dataframe(test["table"], hide_index=True, width="stretch")
                maps.show_fig(charts.log_boxplots(test["values"]))
            # Direction: only logs on ground steeper than 5 degrees have a meaningful "downhill".
            r = np.clip(((logs["y"] - transform.f) / transform.e).astype(int), 0, shape[0] - 1)
            c = np.clip(((logs["x"] - transform.c) / transform.a).astype(int), 0, shape[1] - 1)
            smooth = treefall._smooth(layers.dtm.data, 2.0, cell)
            from terrain import slope as slope_mod

            steep = slope_mod.slope_degrees(smooth, cell, cell)[r, c] > 5
            aspect = slope_mod.aspect_degrees(smooth, cell, cell)[r, c]
            axial = validation.rayleigh_axial(logs["axis_deg"].to_numpy()[steep], aspect[steep])
            with right:
                maps.show_fig(charts.rose_figure(logs["axis_deg"], title="Log directions", axial=True))
            how = "along the slope (up/downhill)" if axial["mean_offset_deg"] < 45 else "across the slope"
            verdict = (f"logs line up {how} more than chance" if axial["p"] < 0.05 else
                       "no clear link between log direction and the slope")
            sentence = (f"{title}: {axial['n']} logs on ground steeper than 5°; {verdict} (mean resultant length "
                        f"{axial['mean_resultant_length']:.2f}, Rayleigh p = {axial['p']:.3g}).")
            st.write(sentence)
            if title == "Detected logs":
                validation.save_result(WORK_DIR, "results/log_validation.json", test["criteria"],
                                       test["sentences"] + [sentence],
                                       {"table": test["table"], "axial": axial, "axes_deg": logs["axis_deg"].tolist(),
                                        "values": {k: [np.round(a, 3).tolist(), np.round(b, 3).tolist()]
                                                   for k, (a, b) in test["values"].items()}})


# ---------------------------------------------------------------------------
# Design criteria
# ---------------------------------------------------------------------------
def design_criteria(survey, layers, p, result):
    st.markdown(f"**Engineering goal.** {validation.GOAL}")
    criteria = validation.load_criteria(CONFIG_DIR)

    with st.expander("Survey speed: my time for this survey"):
        saved = (validation.read_result(WORK_DIR, "results/timing.json") or {}).get("hours", {})
        c1, c2, c3 = st.columns(3)
        flight = c1.number_input("Flying (hours)", 0.0, 100.0, float(saved.get("flight", 0.0)), 0.25)
        processing = c2.number_input("WebODM processing I had to attend (hours)", 0.0, 100.0, float(saved.get("processing", 0.0)), 0.25)
        field = c3.number_input("Field inspection of the flagged trees (hours)", 0.0, 100.0, float(saved.get("field", 0.0)), 0.25)
        area = float(np.isfinite(survey.dtm.data).sum() * layers.cell_area / 10000)
        compute = float(st.session_state.get("tree_seconds", 0.0)) + float(survey.info.get("load_seconds", 0.0))
        speed = validation.hectares_per_hour(area, flight + processing + field, compute)
        st.caption(f"Area mapped: {area:.1f} ha. Computer time for this survey: {compute:.0f} s. "
                   + (f"Speed: **{speed:.1f} ha per hour**." if speed else "Enter your hours to get the speed."))
        if st.button("Save timing"):
            validation.save_result(WORK_DIR, "results/timing.json", {"hectares_per_hour": speed},
                                   [f"Survey speed: {area:.1f} ha in {flight + processing + field:.2f} person-hours plus "
                                    f"{compute:.0f} s of computing = {speed:.1f} ha per hour."] if speed else [],
                                   {"hours": {"flight": flight, "processing": processing, "field": field},
                                    "area_ha": area, "compute_seconds": compute, "survey": survey.name})
            st.rerun()

    table = validation.evaluate_criteria(criteria, WORK_DIR)
    icon = {"pass": "✅ PASS", "fail": "❌ FAIL", "no data yet": "… NO DATA YET"}
    st.dataframe(table.assign(status=table["status"].map(icon)), hide_index=True, width="stretch")
    done = (table["status"] != "no data yet").sum()
    st.caption(f"{(table['status'] == 'pass').sum()} pass, {(table['status'] == 'fail').sum()} fail, "
               f"{len(table) - done} have no data yet. Targets can be edited in `{CONFIG_DIR / 'design_criteria.yaml'}`.")
    background = validation.read_result(WORK_DIR, "results/accuracy.json")
    st.caption("Background (not pass/fail): " + (background["sentences"][0] if background and background.get("sentences")
               else "best-case open-ground vertical error: no data yet (run the Accuracy checker on the yard survey)."))

    if st.button("Export results for board", type="primary"):
        validation.write_summary(WORK_DIR, CONFIG_DIR)
        with st.spinner("Saving every chart at 300 dpi..."):
            saved_figures = charts.export_figures(WORK_DIR)
            for key, args in st.session_state.items():
                if str(key).startswith("storm_map_args_"):
                    name = f"storm_{str(key)[len('storm_map_args_'):]}_change_map.png"
                    (RESULTS_DIR / "figures" / name).write_bytes(maps.render_png(*args, True, 0.6, dpi=300))
                    saved_figures.append(name)
        st.success(f"Saved `{RESULTS_DIR / 'summary.md'}` and {len(saved_figures)} figures in `{RESULTS_DIR / 'figures'}`.")
        st.caption("A storm's change map is included for storms opened in Storm check since TIMBER started.")
    summary = RESULTS_DIR / "summary.md"
    if summary.exists():
        with st.expander("results/summary.md"):
            st.markdown(summary.read_text(encoding="utf-8"))
