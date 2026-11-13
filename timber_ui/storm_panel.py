"""The "Storm check" and "Design criteria" parts of the Tree hazard page.

Storm check: freeze the predictions before a storm, then compare the before
and after surveys and score the frozen predictions against what really
happened. All the math is in terrain/storm.py and terrain/validation.py; this
file only draws the screens and saves the charts.
"""

import base64
import json
from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from terrain import io, storm, validation
from terrain import survey as survey_mod
from terrain.apps.base import Overlay
from timber_ui import maps
from timber_ui.paths import CACHE_DIR, CONFIG_DIR, DATA_ROOT, RESULTS_DIR, WORK_DIR, survey_folders
from timber_ui.style import DISCLAIMER
from timber_ui.widgets import show_table

STORMS_DIR = RESULTS_DIR / "storms"

WHY = ("The field check asks whether high-hazard trees **look** defective. The storm check asks whether they "
       "actually **fail**. The scores are frozen before the storm, so this is a true prediction, not a fit made "
       "after the damage was known.")

FLIGHT_CHECKLIST = """
- **Same flight every time:** same flight plan, altitude, overlap, camera angle and exposure settings for every
  survey of a site. Save the plan file in the survey's folder.
- **Before:** fly as close to the storm as is safely possible (ideally 1-3 days before).
  **After:** fly as soon as winds are within the drone's limits and it is safe.
  **Never fly in or near storm conditions.**
- **Similar time of day** (shadows change the photo and the surface model).
- **At least one calm pair per season:** two flights with no storm between them, as the null test.
- **Note the leaves** (on / turning / off) in *Flight notes* for every flight.
- **Cover the same area:** the after survey must fully cover the before area. Damage at the edge of the map
  is "uncertain".
"""


# ---------------------------------------------------------------------------
# Storm list
# ---------------------------------------------------------------------------
def _storm_editor(storms):
    """Add or edit an entry of config/storms.yaml (wind numbers can come from a station CSV)."""
    names = [s["name"] for s in storms]
    surveys = [p.name for p in survey_folders()]
    with st.expander("Add or edit a storm", expanded=not storms):
        st.caption(f"Storms are kept in `{CONFIG_DIR / 'storms.yaml'}` (you can also edit that file by hand).")
        which = st.selectbox("Storm to edit", ["New storm..."] + names, key="storm_edit_which")
        s = dict(storm.STORM_TEMPLATE) if which == "New storm..." else dict(storms[names.index(which)])
        with st.form(f"storm_form_{which}"):
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Name", s["name"], placeholder="e.g. Nov 4 gale")
            start = c2.text_input("Start (YYYY-MM-DD or YYYY-MM-DD HH:MM)", str(s["date_start"] or ""))
            end = c3.text_input("End", str(s["date_end"] or ""))
            c1, c2, c3 = st.columns(3)

            def pick(col, label, value):
                options = [""] + surveys
                return col.selectbox(label, options, index=options.index(value) if value in options else 0)

            before = pick(c1, "Before survey (folder)", s["before_survey"])
            after = pick(c2, "After survey (folder)", s["after_survey"])
            calm = c3.checkbox("Calm pair (no storm between the flights)", bool(s["is_calm_control"]),
                               help="The null test: measures how much 'damage' is just noise.")
            c1, c2, c3 = st.columns(3)
            station = s.get("weather_station") or {}
            station_id = c1.text_input("Weather station id", station.get("id", ""), placeholder="e.g. BWI")
            station_name = c2.text_input("Weather station name", station.get("name", ""))
            c1, c2, c3 = st.columns(3)
            gust = c1.number_input("Peak gust (m/s)", 0.0, 100.0, float(s["peak_gust_mps"] or 0.0), 0.5,
                                   help="1 m/s = 2.24 mph = 1.94 knots.")
            mean = c2.number_input("Mean wind during the storm (m/s)", 0.0, 100.0, float(s["mean_wind_mps"] or 0.0), 0.5)
            wind_known = c3.checkbox("Wind direction known", s["wind_from_deg"] is not None)
            wind_from = c3.number_input("Wind FROM (degrees, 270 = west)", 0.0, 359.0, float(s["wind_from_deg"] or 270.0), 5.0)
            st.markdown("**Or read the wind numbers from a station CSV** (columns: time, wind speed, gust, direction)")
            c1, c2 = st.columns([2, 1])
            csv = c1.file_uploader("Station CSV", type=["csv", "txt"], key=f"station_csv_{which}")
            unit = c2.selectbox("Speed unit in the file", list(storm.SPEED_TO_MPS), index=1)
            notes = st.text_area("Notes", s["notes"], height=70)
            if st.form_submit_button("Save storm", type="primary"):
                if not name.strip():
                    st.error("Give the storm a name.")
                    return
                new = {**s, "name": name.strip(), "date_start": start.strip(), "date_end": end.strip(),
                       "before_survey": before, "after_survey": after, "is_calm_control": calm,
                       "weather_station": {"id": station_id.strip(), "name": station_name.strip()},
                       "peak_gust_mps": gust or None, "mean_wind_mps": mean or None,
                       "wind_from_deg": wind_from if wind_known else None, "notes": notes}
                if csv is not None:
                    try:
                        stats_ = storm.wind_stats(storm.read_station_csv(csv, unit), start.strip(), end.strip() or start.strip())
                        new.update({k: stats_[k] for k in ("peak_gust_mps", "mean_wind_mps", "wind_from_deg")})
                        st.success(f"From {stats_['readings']} readings: peak gust {stats_['peak_gust_mps']} m/s, mean "
                                   f"wind {stats_['mean_wind_mps']} m/s, wind from {stats_['wind_from_deg']:.0f}°.")
                    except Exception as err:
                        st.error(f"Could not read the wind CSV: {err}")
                        return
                others = [x for x in storms if x["name"] not in (which, new["name"])]
                storm.save_storms(CONFIG_DIR, others + [new])
                st.session_state["storm_pick"] = new["name"]
                st.rerun()


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------
def _frozen_before_storm(status, s) -> bool | None:
    """Was the freeze made before the storm's start? (None if a date is missing.)"""
    try:
        frozen = pd.Timestamp(status["frozen_at_utc"]).tz_convert(None)
        return bool(frozen < pd.Timestamp(s["date_start"]) + pd.Timedelta(hours=12))  # 12 h of slack for time zones
    except (KeyError, ValueError, TypeError):
        return None


def _freeze_section(s, folder, survey, p, result):
    status = storm.verify_frozen(folder)
    with st.expander("1. Freeze predictions", expanded=not status["exists"]):
        st.write("Freezing runs the tree analysis on the **before** survey with the current settings and saves every "
                 "tree's scores and predicted fall direction, with a fingerprint (SHA-256) and the time. The storm "
                 "analysis only ever reads this frozen file, so the predictions provably existed before the damage "
                 "was known. **Do this before the storm (or at least before looking at the after survey).**")
        if status["exists"]:
            (st.success if status["ok"] else st.error)(
                f"Frozen {status['frozen_at_utc']} UTC. {status['message']}  \nFingerprint: `{status['combined']}`")
            early = _frozen_before_storm(status, s)
            if early is False and not s["is_calm_control"]:
                st.warning("These predictions were frozen AFTER the storm's start date. They are still read-only, "
                           "but say so when you present them: they are not a forecast made in advance.")
        on_before = survey.name == s["before_survey"]
        if not on_before:
            st.info(f"To freeze, choose the before survey **{s['before_survey'] or '(not set)'}** in the sidebar "
                    f"(now: {survey.name}).")
        again = st.checkbox("Freeze again (replaces the frozen files; the log keeps every freeze)",
                            key=f"again_{folder.name}") if status["exists"] else True
        if st.button("Freeze predictions now", type="primary", disabled=not (on_before and again),
                     key=f"freeze_{folder.name}"):
            raw = result.raw
            grids = {"crowns": raw["crowns"].astype("int32"), "exposure": raw["exposure"].astype("float32"),
                     "grid_factor": np.array(raw["grid_factor"])}
            grids.update({f"mask_{k}": v.astype("uint8") for k, v in raw["masks"].items()})
            settings = {**p, "survey": {"name": survey.name, "key": survey.key, "shape": list(survey.dtm.shape),
                                        **survey.settings}}
            entry = storm.freeze_predictions(folder, raw["trees"], raw["probs"], settings, crs=survey.dtm.crs,
                                             grids=grids, note=f"storm: {s['name']}")
            st.success(f"Frozen {len(raw['trees'])} trees. Fingerprint: `{entry['combined']}`")
            st.rerun()

        st.markdown("**Second copy with the storm's actual wind direction** (for the fall direction test only)")
        actual = storm.verify_frozen(folder, storm.ACTUAL)
        if actual["exists"]:
            st.caption(f"Actual-wind copy made {actual['frozen_at_utc']} UTC. {actual['message']}")
        wind = s["wind_from_deg"]
        if st.button(f"Make the actual-wind copy (wind from {wind:.0f}°)" if wind is not None else
                     "Make the actual-wind copy (fill in the storm's wind direction first)",
                     disabled=wind is None or not (status["exists"] and status["ok"]), key=f"actual_{folder.name}"):
            info = json.loads((folder / storm.FROZEN_FILES[storm.PREVAILING][1]).read_text(encoding="utf-8"))
            storm.freeze_actual_wind(folder, wind, info["settings"]["survey"]["cell_size"])
            st.rerun()
        log = storm.freeze_log(folder)
        if log:
            st.dataframe(pd.DataFrame([{"what": e["label"], "frozen (UTC)": e["frozen_at_utc"],
                                        "fingerprint": e["combined"][:16] + "...", "note": e.get("note", "")} for e in log]),
                         hide_index=True, width="stretch")
    return status


# ---------------------------------------------------------------------------
# Loading the pair of surveys and running the analysis
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False, max_entries=2)
def _load_pair(before_name: str, after_name: str, survey_settings: str, after_fingerprint: str):
    """The before survey (prepared exactly as when it was frozen) and the after survey on the same grid."""
    cfg = json.loads(survey_settings)
    before = survey_mod.prepare(DATA_ROOT / before_name, cfg["cell_size"], cfg["pit_depth"], cfg["detrend"],
                                cache_dir=CACHE_DIR)
    before.key = cfg["key"]
    files = survey_mod.find_files(DATA_ROOT / after_name)
    if files["dsm"] is None:
        raise FileNotFoundError(f"The after survey '{after_name}' has no DSM file.")
    after = {"dsm": io.resample_to_grid(io.load_raster(files["dsm"]), before.dtm).data,
             "dtm": io.resample_to_grid(io.load_raster(files["dtm"]), before.dtm).data,
             "ortho": io.read_rgb_on_grid(files["ortho"], before.dtm) if files["ortho"] is not None else None}
    return before, after


@st.cache_resource(show_spinner=False, max_entries=4)
def _analyze(slug: str, frozen_fingerprint: str, pair_key: str, options: str, _s, _before, _after):
    """The whole storm analysis for one storm (cached per frozen file, survey pair and thresholds)."""
    folder = STORMS_DIR / slug
    opts = json.loads(options)
    frozen, probs, info = storm.load_frozen(folder)
    settings = info["settings"]
    grids = storm.load_frozen_grids(folder)
    models = {storm.PREVAILING: (frozen, probs)}
    if storm.verify_frozen(folder, storm.ACTUAL).get("ok"):
        actual, probs_actual, _ = storm.load_frozen(folder, storm.ACTUAL)
        models[storm.ACTUAL] = (actual, probs_actual)
    cell = _before.dtm.pixel_size[0]
    seasons = (survey_mod.read_info(DATA_ROOT / _s["before_survey"])["season"],
               survey_mod.read_info(DATA_ROOT / _s["after_survey"])["season"])
    building = grids["mask_building"].astype(bool) if "mask_building" in grids else None
    before = {"dsm": _before.dsm.data, "dtm": _before.dtm.data, "ortho": _before.ortho}
    out = storm.compare_surveys(before, _after, frozen, grids["crowns"], _before.dtm.transform, cell, settings,
                                opts["change"], building, seasons, opts["trust_alignment"])
    table = storm.join_outcomes(frozen, out["changes"])
    prediction = storm.prediction_metrics(table, _s.get("peak_gust_mps"))
    direction = storm.direction_metrics(table, models, _s.get("wind_from_deg"))
    landing = storm.landing_metrics(table, out["new_logs"], grids, out["chm_before"], _before.dtm.transform, cell, settings)
    status = storm.verify_frozen(folder)
    status["frozen_before_storm"] = _frozen_before_storm(status, _s)
    results = storm.save_storm_results(folder, _s, out, table, prediction, direction, landing, status)
    pooled = storm.pool_results(STORMS_DIR, storm.load_storms(CONFIG_DIR))
    return {"out": out, "table": table, "prediction": prediction, "direction": direction, "landing": landing,
            "results": results, "pooled": pooled, "grids": grids, "settings": settings}


def _threshold_form(slug, rms_hint=None):
    """Sliders for the change detection (in a form, so nothing re-runs until the button is pressed)."""
    d = storm.CHANGE_DEFAULTS
    with st.form(f"thresholds_{slug}", border=False):
        c1, c2, c3, c4 = st.columns(4)
        auto = c1.checkbox("Change threshold: automatic", True,
                           help="3 x the leftover height noise on stable ground, at least 0.5 m.")
        thr = c1.slider("Change threshold if not automatic (m)", 0.2, 5.0, 0.5, 0.1)
        fallen_top = c2.slider("Fallen: top dropped more than (% of height)", 30, 95, int(d["fallen_top"] * 100), 5)
        fallen_crown = c2.slider("Fallen: crown lost more than (%)", 30, 95, int(d["fallen_crown"] * 100), 5)
        snap_min = c3.slider("Snapped top: top dropped more than (m)", 0.5, 10.0, d["snap_min_m"], 0.5)
        snap_frac = c3.slider("Snapped top: and more than (% of height)", 5, 60, int(d["snap_frac"] * 100), 5)
        crown_frac = c4.slider("Crown damage: crown lost more than (%)", 5, 70, int(d["crown_frac"] * 100), 5)
        edge = c4.slider("Uncertain within this distance of the map edge (m)", 0.0, 30.0, d["edge_m"], 1.0)
        c1, c2 = st.columns(2)
        top_only = c1.checkbox("Leaf-drop mode: classify with the top drop and new logs only", False,
                               help="Use when leaves fell between the flights: crown thinning is then ignored.")
        trust = c2.checkbox("Trust the alignment even if it gives a warning", False,
                            help="Otherwise every tree is 'uncertain' when the alignment is poor.")
        run = st.form_submit_button("Run storm analysis", type="primary")
    change = {"change_threshold": None if auto else thr, "fallen_top": fallen_top / 100, "fallen_crown": fallen_crown / 100,
              "snap_min_m": snap_min, "snap_frac": snap_frac / 100, "crown_frac": crown_frac / 100, "edge_m": edge,
              "top_only": top_only}
    return {"change": change, "trust_alignment": trust}, run


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def auc_figure(auc: pd.DataFrame, title: str):
    """AUC of every ranking with its 95% range. The dashed line (0.5) is pure guessing."""
    auc = auc.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(auc) + 1.4))
    colors = ["#2e7d32" if r == "full hazard score" else "#7f8c8d" if r.startswith("baseline") else "#90a4ae"
              for r in auc["ranking"]]
    low = (auc["AUC"] - auc["low 95%"]).clip(lower=0).fillna(0)
    high = (auc["high 95%"] - auc["AUC"]).clip(lower=0).fillna(0)
    ax.barh(auc["ranking"], auc["AUC"].fillna(0), xerr=[low, high], color=colors, ecolor="black", capsize=3)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC (0.5 = guessing, 1.0 = perfect); lines = 95% range")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def rose_figure(observed_deg, wind_from_deg=None):
    """Rose diagram: how many trees fell toward each direction (north up, clockwise)."""
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    counts, edges = np.histogram(np.asarray(observed_deg, dtype=float) % 360, bins=np.arange(0, 361, 20))
    ax.bar(np.radians(edges[:-1] + 10), counts, width=np.radians(20), color="#c0392b", edgecolor="white", alpha=0.85)
    if wind_from_deg is not None and np.isfinite(wind_from_deg):
        top = max(counts.max(), 1)
        to = np.radians((wind_from_deg + 180) % 360)
        ax.annotate("", xy=(to, top), xytext=(np.radians(wind_from_deg), top),
                    arrowprops=dict(arrowstyle="-|>", color="#1565c0", lw=2.5))
        ax.set_title(f"Fall directions of storm-felled trees\n(blue arrow: wind from {wind_from_deg:.0f}°)", fontsize=11)
    else:
        ax.set_title("Fall directions of storm-felled trees", fontsize=11)
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    return fig


def _png(fig, dpi=300) -> bytes:
    out = BytesIO()
    fig.savefig(out, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Change map
# ---------------------------------------------------------------------------
def _change_layers(a, before):
    """Overlays and annotations for the change map."""
    out, table = a["out"], a["table"]
    crowns = a["grids"]["crowns"]
    order = ["fallen", "snapped top", "crown damage", "uncertain"]
    lookup = np.full(int(crowns.max()) + 1, np.nan)
    number = table["damage"].map({name: i + 1 for i, name in enumerate(order)})
    lookup[table["tree"].to_numpy()] = number.to_numpy(dtype=float)
    damage_grid = lookup[crowns]
    damage_grid[crowns == 0] = np.nan
    drop = np.nan_to_num(out["chm_before"]) - np.nan_to_num(out["chm_after"])
    thr = out["thresholds"]["change_threshold"]
    overlays = [
        Overlay("Damage class (crowns)", damage_grid,
                classes=[(i + 1, name.capitalize(), storm.CLASS_COLORS[name]) for i, name in enumerate(order)]),
        Overlay("Canopy height lost (m)", np.where(drop > thr, drop, np.nan), cmap="Reds", vmin=0, vmax=15, units="m"),
    ]
    annotations = []
    hit = table[table["damage"].isin(storm.ANY_DAMAGE)]
    if len(hit):
        hover = [f"Tree {int(t)}: {d}<br>frozen hazard {h:.0f} (rank {int(r)})<br>top dropped {td:.1f} m, crown lost {100 * cl:.0f}%"
                 for t, d, h, r, td, cl in zip(hit["tree"], hit["damage"], hit["hazard"], hit["hazard_rank"],
                                               hit["top_drop_m"], hit["crown_loss_frac"].fillna(0))]
        annotations.append({"type": "scatter", "x": hit["x"].to_numpy(), "y": hit["y"].to_numpy(), "size": 12,
                            "colors": [storm.CLASS_COLORS[d] for d in hit["damage"]], "hover": hover,
                            "name": "Damaged trees"})
    logs = out["new_logs"]
    linked = logs[logs["tree"].notna()] if len(logs) else logs
    loose = logs[logs["tree"].isna()] if len(logs) else logs
    if len(linked):  # observed fall: arrow from the root end along the log
        ang = np.radians(linked["fall_deg"].to_numpy())
        length = linked["length_m"].to_numpy()
        annotations.append({"type": "arrows", "x": linked["root_x"].to_numpy(), "y": linked["root_y"].to_numpy(),
                            "dx": length * np.sin(ang), "dy": length * np.cos(ang), "colors": ["#00e5ff"] * len(linked)})
    if len(loose):
        annotations.append({"type": "segments", "segments": storm.log_ends(loose), "color": "#00e5ff", "width": 2.5,
                            "name": "New logs (cyan; arrows = root to top)"})
    fell = table[table["damage"] == "fallen"]
    if len(fell):  # predicted direction for the same trees, in a different color
        ang = np.radians(fell["fall_dir"].to_numpy())
        length = fell["height"].to_numpy()
        annotations.append({"type": "arrows", "x": fell["x"].to_numpy(), "y": fell["y"].to_numpy(),
                            "dx": length * np.sin(ang), "dy": length * np.cos(ang), "colors": ["#ffd400"] * len(fell)})
    return overlays, annotations


# ---------------------------------------------------------------------------
# Field confirmation booklet
# ---------------------------------------------------------------------------
def _card_image(before, x, y, crowns, tree_id, size_m=40.0) -> str:
    """A 40 m x 40 m crop of the BEFORE photo around a tree, with the crown outlined,
    a north arrow and a 10 m scale bar (as an image for the HTML booklet)."""
    t = before.dtm.transform
    cell = t.a
    r, c = int((y - t.f) / t.e), int((x - t.c) / t.a)
    half = int(size_m / 2 / cell)
    r0, r1, c0, c1 = max(r - half, 0), min(r + half, crowns.shape[0]), max(c - half, 0), min(c + half, crowns.shape[1])
    bg = maps.background_rgb(before.key, True, before)[r0:r1, c0:c1]
    fig, ax = plt.subplots(figsize=(4, 4))
    extent = (0, (c1 - c0) * cell, 0, (r1 - r0) * cell)
    ax.imshow(bg, extent=extent, interpolation="bilinear")
    ax.contour(np.flipud(crowns[r0:r1, c0:c1] == tree_id), levels=[0.5], colors="#ffd400", linewidths=1.5, extent=extent)
    w, h = extent[1], extent[3]
    ax.plot([0.05 * w, 0.05 * w + 10], [0.05 * h, 0.05 * h], color="white", linewidth=3)
    ax.text(0.05 * w + 5, 0.08 * h, "10 m", color="white", ha="center", fontsize=8, weight="bold")
    ax.annotate("N", xy=(0.92 * w, 0.95 * h), xytext=(0.92 * w, 0.80 * h), color="white", ha="center", fontsize=10,
                weight="bold", arrowprops=dict(arrowstyle="-|>", color="white", lw=2))
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    out = BytesIO()
    fig.savefig(out, format="jpg", dpi=110)
    plt.close(fig)
    return base64.b64encode(out.getvalue()).decode()


def booklet_html(key: pd.DataFrame, sheet: pd.DataFrame, before, crowns, storm_name: str) -> str:
    """One card per code: photo crop and position. Nothing about what the app found."""
    cards = []
    for (_, k), (_, s) in zip(key.iterrows(), sheet.iterrows()):
        img = _card_image(before, k["x"], k["y"], crowns, int(k["tree"]))
        where = f"{s['latitude']:.6f}, {s['longitude']:.6f}" if "latitude" in s else ""
        cards.append(f'<div class="card"><h2>Code {int(k["code"])}</h2><img src="data:image/jpeg;base64,{img}">'
                     f"<p>{where}</p><p>status: ______ &nbsp; damage: ______ &nbsp; fall azimuth: ____°</p></div>")
    return ("<html><head><meta charset='utf-8'><style>body{font-family:sans-serif}"
            ".card{display:inline-block;width:320px;margin:8px;padding:8px;border:1px solid #999;page-break-inside:avoid}"
            ".card img{width:100%}h2{margin:2px 0}p{margin:4px 0;font-size:13px}</style></head><body>"
            f"<h1>Storm field check: {storm_name}</h1><p>Photos are from BEFORE the storm. The tree is outlined in "
            "yellow. storm_damage: none / branches / top broken / fallen / uprooted. Fall azimuth: phone compass, "
            "standing at the roots and pointing at the top.</p>" + "".join(cards) + "</body></html>")


# ---------------------------------------------------------------------------
# The Storm check part
# ---------------------------------------------------------------------------
def storm_check(survey, layers, p, result):
    st.markdown(WHY)
    with st.expander("Flight checklist (read before every storm flight)"):
        st.markdown(FLIGHT_CHECKLIST)
    storms = storm.load_storms(CONFIG_DIR)
    _storm_editor(storms)
    if not storms:
        st.info("No storms yet. Add one above: give it a name and its before survey, then freeze the predictions.")
        return
    names = [s["name"] for s in storms]
    if st.session_state.get("storm_pick") not in names:
        st.session_state["storm_pick"] = names[0]
    name = st.selectbox("Storm", names, key="storm_pick",
                        format_func=lambda n: n + ("  (calm pair)" if storms[names.index(n)]["is_calm_control"] else ""))
    s = storms[names.index(name)]
    slug = storm.storm_slug(name)
    folder = STORMS_DIR / slug
    gust = f"{s['peak_gust_mps']} m/s peak gust" if s["peak_gust_mps"] else "wind not filled in yet"
    st.caption(f"{s['date_start'] or '?'} to {s['date_end'] or '?'} · before: **{s['before_survey'] or '?'}** · after: "
               f"**{s['after_survey'] or '?'}** · {gust}"
               + (f" from {s['wind_from_deg']:.0f}°" if s["wind_from_deg"] is not None else ""))

    status = _freeze_section(s, folder, survey, p, result)
    if not status["exists"]:
        return
    if not status["ok"]:
        st.error("The frozen predictions were changed after freezing, so the storm analysis is blocked.")
        return
    if not s["after_survey"] or not (DATA_ROOT / s["after_survey"]).exists():
        st.info("Predictions are frozen. After the storm, put the after-flight in its own folder in the data folder, "
                "set it as this storm's **after survey** (Add or edit a storm), and come back here.")
        return

    st.subheader("Compare the before and after surveys")
    options, run = _threshold_form(slug)
    if run:
        st.session_state[f"storm_ran_{slug}"] = True
    if not st.session_state.get(f"storm_ran_{slug}"):
        st.caption("Press **Run storm analysis** (about a minute on a forest map the first time).")
        return
    info = json.loads((folder / storm.FROZEN_FILES[storm.PREVAILING][1]).read_text(encoding="utf-8"))
    cfg = info["settings"]["survey"]
    after_files = survey_mod.find_files(DATA_ROOT / s["after_survey"])
    after_print = survey_mod.fingerprint(after_files, cfg["cell_size"], None, False)
    try:
        with st.spinner("Loading both surveys onto one grid..."):
            before, after = _load_pair(s["before_survey"], s["after_survey"], json.dumps(cfg, sort_keys=True), after_print)
        if list(before.dtm.shape) != list(cfg["shape"]):
            st.error("The before survey's files changed since the predictions were frozen (different grid size).")
            return
        with st.spinner("Aligning the surveys and measuring every tree..."):
            a = _analyze(slug, status["combined"], after_print, json.dumps(options, sort_keys=True), s, before, after)
    except Exception as err:
        st.error(f"The storm analysis could not run: {err}")
        return
    _show_results(s, slug, folder, a, before)
    st.caption(DISCLAIMER)


def _calm_counts(storms, current):
    """Change counts of the analyzed calm pairs (other than the current entry), added up."""
    total = None
    for other in storms:
        path = STORMS_DIR / storm.storm_slug(other["name"]) / "storm_results.json"
        if other["is_calm_control"] and other["name"] != current and path.exists():
            counts = json.loads(path.read_text(encoding="utf-8"))["counts"]
            total = counts if total is None else {k: total[k] + counts[k] for k in total}
    return total


def _show_results(s, slug, folder, a, before):
    out, table, counts = a["out"], a["table"], a["results"]["counts"]
    severe = a["prediction"]["outcomes"]["severe"]
    auc = severe["auc"].set_index("ranking")["AUC"].get("full hazard score", np.nan)
    calm = _calm_counts(storm.load_storms(CONFIG_DIR), s["name"])
    for w in out["warnings"]:
        st.warning(w, icon="⚠️")

    cols = st.columns(7)
    cols[0].metric("Trees compared", counts["compared"])
    cols[1].metric("Fallen", counts["fallen"])
    cols[2].metric("Snapped top", counts["snapped top"])
    cols[3].metric("Crown damage", counts["crown damage"])
    cols[4].metric("New logs", len(out["new_logs"]))
    cols[5].metric("Calm false change rate", f"{100 * calm['any_damage'] / calm['compared']:.2f}%"
                   if calm and calm["compared"] else "no calm pair yet")
    cols[6].metric("Storm AUC (severe)", f"{auc:.2f}" if np.isfinite(auc) else "no damage")
    if counts["uncertain"]:
        st.caption(f"{counts['uncertain']} trees are 'uncertain' (map edge, missing after data or poor alignment) "
                   "and are left out of every score.")

    with st.expander("2. Alignment"):
        r = out["alignment"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sideways shift found", f"{r['shift_m']:.2f} m", f"{r['shift_east_m']:+.2f} E, {r['shift_north_m']:+.2f} N",
                  delta_color="off")
        c2.metric("Shift left after aligning", f"{r['residual_shift_m']:.2f} m")
        c3.metric("Height offset (surface model)", f"{r['dsm']['offset_m']:+.2f} m", f"tilt {r['dsm']['tilt_deg']:.2f}°",
                  delta_color="off")
        c4.metric("Leftover noise on stable ground", f"{r['dsm']['rms_m']:.2f} m")
        if r["shift_blocks_used"]:
            st.caption(f"The sideways shift changed across the map by up to {r['shift_variation_m']:.1f} m (the two "
                       f"models differ a little in scale or rotation), so it was corrected with a stretch-and-rotate "
                       f"fit over {r['shift_blocks_used']} blocks of 100 m.")
        st.caption(f"Bend between the two models that was removed (typical height difference within 40 m): up to "
                   f"{r['warp_m']:.2f} m over 95% of the map.")
        st.caption(f"Stable open ground used: {100 * r['stable_share']:.1f}% of the map. Ground model (DTM): offset "
                   f"{r['dtm']['offset_m']:+.2f} m, tilt {r['dtm']['tilt_deg']:.2f}°, leftover {r['dtm']['rms_m']:.2f} m. "
                   f"Change threshold used: {out['thresholds']['change_threshold']:.2f} m. Both canopy heights use the "
                   "BEFORE ground model, because the ground doesn't move in a storm.")

    with st.expander("3. Change map", expanded=True):
        overlays, annotations = _change_layers(a, before)
        c1, c2 = st.columns(2)
        layer = c1.radio("Map layer", [o.name for o in overlays], horizontal=True, key=f"storm_layer_{slug}")
        photos = ["Before photo", "After photo"] if out["after_ortho"] is not None and before.ortho is not None else ["Before"]
        which = c2.radio("Background", photos, horizontal=True, key=f"storm_bg_{slug}")
        shown = before if which.startswith("Before") else SimpleNamespace(
            dtm=before.dtm, ortho=out["after_ortho"], key=f"{before.key}_after_{slug}_{a['results']['analyzed_at_utc']}")
        overlay = overlays[[o.name for o in overlays].index(layer)]
        # Remembered so "Export results for board" can draw this map at 300 dpi.
        st.session_state[f"storm_map_args_{slug}"] = (before, overlays[0], annotations)
        fig = maps.interactive_map(shown, overlay, annotations, True, 0.6, view_key=f"storm_{slug}")
        maps.show_interactive(fig, f"storm_map_{slug}")
        st.caption("Dots: damaged trees. Cyan arrows: new logs, from the root end to the top (how the tree really fell). "
                   "Yellow arrows: the frozen predicted direction for the same trees. Cyan lines: new logs not "
                   "linked to a tree.")
        damaged = table[table["damage"].isin(storm.ANY_DAMAGE)].sort_values("hazard_rank")
        err = a["direction"]["per_tree"]
        shown_cols = {"tree": "tree", "damage": "class", "hazard": "frozen hazard", "hazard_rank": "frozen rank",
                      "reasons": "reasons (frozen)", "fall_dir": "predicted direction (°)",
                      "observed_fall_deg": "observed direction (°)", "top_drop_m": "top drop (m)",
                      "crown_loss_frac": "crown lost (share)"}
        t = damaged[[c for c in shown_cols if c in damaged]].rename(columns=shown_cols)
        if len(err) and f"error ({storm.PREVAILING})" in err:
            t = t.merge(err[["tree", f"error ({storm.PREVAILING})"]].rename(
                columns={f"error ({storm.PREVAILING})": "direction error (°)"}), on="tree", how="left")
        show_table(t, f"Damaged trees ({len(t)}), best frozen rank first", head=25, key=f"storm_damaged_{slug}")
        if len(out["gaps"]):
            show_table(out["gaps"], "New canopy gaps not explained by any tree", key=f"storm_gaps_{slug}")

    with st.expander("4. Did we predict it?"):
        _prediction_section(a, slug)
    with st.expander("5. Fall direction"):
        _direction_section(a, s, slug)
    with st.expander("6. Calm control"):
        _calm_section(s, counts, calm)
    with st.expander("7. Field confirmation"):
        _field_section(a, s, slug, before)
    st.caption(f"Results saved in `{folder}` (storm_results.json, storm_trees.csv, new_logs.csv) and pooled in "
               f"`{STORMS_DIR / 'pooled_results.json'}`.")


def _prediction_section(a, slug):
    st.write("Everything here uses only the **frozen** scores. AUC = the chance that a damaged tree had a higher "
             "frozen score than an undamaged one (0.5 = guessing).")
    labels = {"severe": "Severe damage (fallen or snapped top)", "any_damage": "Any damage (incl. crown damage)"}
    key = st.radio("Outcome", list(labels), format_func=labels.get, horizontal=True, key=f"storm_outcome_{slug}")
    o = a["prediction"]["outcomes"][key]
    if o["damaged"] == 0:
        st.info("No trees in this class, so there is nothing to score. A storm with no damage is still useful: "
                "it shows the change detection doesn't invent damage.")
        return
    if o["small_sample"]:
        st.warning(f"Only {o['damaged']} damaged trees. That is too few for the percentages and p-values to mean "
                   "much: look at the exact counts, and pool more storms.", icon="⚠️")
    auc = o["auc"].set_index("ranking")
    full, base = auc.loc["full hazard score"], auc["AUC"].get("baseline: tallest first", np.nan)
    st.markdown(f"**In plain words:** {o['damaged']} of {a['prediction']['trees']} trees were damaged. The frozen hazard "
                f"score's AUC is **{full['AUC']:.2f}** (95% range {full['low 95%']:.2f} to {full['high 95%']:.2f}); "
                f"'tallest tree first' scores {base:.2f}. "
                + ("The range includes 0.5, so this is not yet clearly better than guessing."
                   if full["low 95%"] <= 0.5 else "The whole range is above 0.5: better than guessing."))
    fig = auc_figure(o["auc"], labels[key])
    st.session_state[f"storm_auc_png_{slug}"] = _png(auc_figure(o["auc"], labels[key]))
    maps.show_fig(fig)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Damage rate by frozen hazard class**")
        st.dataframe(o["rates"], hide_index=True, width="stretch")
        r = o["ratio"]
        ratio = "all in the upper group, none in low" if r["ratio"] == float("inf") else f"{r['ratio']:.1f}x"
        st.caption(f"{r['groups']}: {r['counts'][0]} of {r['counts'][1]} vs {r['counts'][2]} of {r['counts'][3]} "
                   f"damaged. Ratio {ratio}; Fisher's exact test p = {r['fisher_p']:.3g}.")
    with c2:
        st.markdown("**Were the damaged trees near the top of the frozen ranking?**")
        st.dataframe(o["top_n"], hide_index=True, width="stretch")
        odds = o["odds"]
        if np.isfinite(odds["odds_ratio_per_10"]):
            st.caption(f"Logistic regression: each 10 hazard points multiply the odds of this damage by "
                       f"{odds['odds_ratio_per_10']:.2f} (95% range {odds['low 95%']:.2f} to {odds['high 95%']:.2f}; "
                       "1.0 = no effect).")
    pooled = a["pooled"]
    if pooled.get("prediction") and len(pooled["storms"]) > 1:
        st.markdown(f"**All {len(pooled['storms'])} storms pooled**")
        st.dataframe(pd.DataFrame(pooled["prediction"]["outcomes"][key]["auc"]), hide_index=True, width="stretch")


def _direction_section(a, s, slug):
    d = a["direction"]
    if d["trees"] == 0:
        st.info("No fallen tree has a new log linked to it, so no fall direction is known yet. (Field-measured "
                "directions can be added under Field confirmation.)")
        return
    st.write(f"{d['trees']} fallen trees have a new log whose root end is at the old trunk, so their fall direction "
             "is known. Error = angle between the observed direction and each model's most likely direction "
             "(a blind guess is 90° off on average).")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(d["table"], hide_index=True, width="stretch")
        st.markdown("**Log-likelihood** (higher = the 36-direction forecast put more chance on what happened)")
        st.dataframe(d["likelihood"], hide_index=True, width="stretch")
        landing = a["landing"]
        if landing["strip"]:
            k = landing["strip"]
            st.caption(f"Landing zone: {k['within reach (any direction)']} of {k['trees']} logs ended within the tree's "
                       f"predicted reach; {k['inside the most likely strip']} lay inside the strip of the single most "
                       "likely direction.")
        if landing["exposure"]:
            e = landing["exposure"]
            st.caption(f"Frozen fall exposure: mean {e['mean exposure under new logs']:.3f} under the {e['new logs']} new "
                       f"logs vs {e['mean exposure at random forest points']:.3f} at random forest points "
                       f"(Mann-Whitney p = {e['Mann-Whitney p (logs higher)']:.3g}).")
        if len(landing["features"]):
            st.markdown("**New logs that landed on a detected feature**")
            st.dataframe(landing["features"], hide_index=True, width="stretch")
    with c2:
        fig = rose_figure(d["per_tree"]["observed_fall_deg"], s.get("wind_from_deg"))
        st.session_state[f"storm_rose_png_{slug}"] = _png(rose_figure(d["per_tree"]["observed_fall_deg"], s.get("wind_from_deg")))
        maps.show_fig(fig)


def _calm_section(s, counts, calm):
    if s["is_calm_control"]:
        rate = counts["any_damage"] / counts["compared"] if counts["compared"] else float("nan")
        st.write(f"This entry is a **calm pair**: no storm happened, so every change found is a false alarm of the "
                 f"method. **{counts['any_damage']} of {counts['compared']} trees** were falsely flagged "
                 f"({100 * rate:.2f}%): {counts['fallen']} fallen, {counts['snapped top']} snapped top, "
                 f"{counts['crown damage']} crown damage.")
        return
    if calm is None:
        st.info("No calm pair has been analyzed yet. Fly the same area twice with no storm in between, add it as a "
                "storm with **Calm pair** ticked, freeze and run it. Then this storm's damage can be compared with "
                "the method's false alarm rate.")
        return
    st.write("The storm's change rate next to the false change rate of the calm pair(s). A small Fisher p means "
             "the storm damage is real and not processing noise.")
    st.dataframe(storm.calm_comparison(counts, calm), hide_index=True, width="stretch")


def _field_section(a, s, slug, before):
    st.write("Remote change detection can be wrong. This blind sheet mixes every fallen or snapped tree (up to 30) "
             "with the same number of random 'no change' trees. The sheet doesn't say which is which.")
    sheet, key = storm.field_sheet(a["table"])
    if sheet.empty:
        st.info("No fallen or snapped trees to check.")
        return
    c1, c2, c3 = st.columns(3)
    c1.download_button("Field sheet (take to the woods)", sheet.to_csv(index=False), f"storm_field_sheet_{slug}.csv",
                       "text/csv", key=f"sheet_{slug}")
    c2.download_button("Key (keep at home)", key.to_csv(index=False), f"storm_field_key_{slug}.csv", "text/csv",
                       key=f"key_{slug}")
    if c3.button("Prepare the photo booklet", key=f"mk_booklet_{slug}"):
        with st.spinner("Drawing one card per tree..."):
            st.session_state[f"booklet_{slug}"] = booklet_html(key, sheet, before, a["grids"]["crowns"], s["name"])
    if f"booklet_{slug}" in st.session_state:
        c3.download_button("⬇ Save booklet (HTML, print it)", st.session_state[f"booklet_{slug}"],
                           f"storm_booklet_{slug}.html", "text/html", key=f"booklet_dl_{slug}")
    st.caption("Fill in: status (found / not_found / inaccessible), storm_damage (none / branches / top broken / "
               "fallen / uprooted), fall_azimuth_deg (phone compass from the roots toward the top), notes.")
    upload = st.file_uploader("Filled field sheet (CSV)", type=["csv"], key=f"field_up_{slug}")
    if upload is not None:
        try:
            result = storm.field_compare(key, pd.read_csv(upload))
        except Exception as err:
            st.error(f"Could not compare: {err}")
            return
        st.markdown(f"**{result['checked']} trees checked in the field**")
        st.dataframe(result["table"], hide_index=True, width="stretch")
        st.caption("Precision: of the trees the app put in a class, the share the field agreed with. Recall: of the "
                   "trees the field put in a class, the share the app had found.")
        result["trees"].to_csv(STORMS_DIR / slug / "field_confirmation.csv", index=False)


# ---------------------------------------------------------------------------
# Design criteria
# ---------------------------------------------------------------------------
def write_summary(table: pd.DataFrame, storms: list) -> str:
    """results/summary.md: the criteria table and a plain-sentence storm section."""
    lines = ["# TIMBER results summary", "",
             f"Written {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC.", "",
             "Goal: use only a consumer drone and free software to find which trees in a forest pose the greatest "
             "fall risk to trails, roads and buildings.", "", "## Design criteria", "",
             "| Criterion | Target | Measured | Status |", "|---|---|---|---|"]
    lines += [f"| {r['what it means']} | {r['target']} | {r['measured']} | {r['status']} |" for _, r in table.iterrows()]
    lines += ["", "## Storm check", "", WHY.replace("**", ""), ""]
    done = 0
    for s in storms:
        path = STORMS_DIR / storm.storm_slug(s["name"]) / "storm_results.json"
        if not path.exists():
            continue
        done += 1
        r = json.loads(path.read_text(encoding="utf-8"))
        c = r["counts"]
        kind = "Calm pair" if s["is_calm_control"] else "Storm"
        wind = f", peak gust {s['peak_gust_mps']} m/s" if s["peak_gust_mps"] else ""
        lines.append(f"### {kind}: {s['name']} ({s['date_start']}{wind})")
        lines.append(f"- Predictions frozen {r['frozen'].get('frozen_at_utc', '?')} UTC (fingerprint "
                     f"{str(r['frozen'].get('combined', ''))[:16]}...); frozen before the storm: "
                     f"{r['frozen'].get('frozen_before_storm')}.")
        lines.append(f"- Alignment: sideways shift {r['alignment']['shift_m']:.2f} m, leftover height noise on stable "
                     f"ground {r['alignment']['dsm']['rms_m']:.2f} m.")
        lines.append(f"- Of {c['compared']} trees compared: {c['fallen']} fallen, {c['snapped top']} snapped top, "
                     f"{c['crown damage']} crown damage; {r['new_logs']} new logs.")
        sev = r["prediction"]["outcomes"]["severe"]
        full = next((x for x in sev["auc"] if x["ranking"] == "full hazard score"), None)
        if not s["is_calm_control"] and full and full["AUC"] is not None:
            note = " (fewer than 10 damaged trees: treat as a count, not a statistic)" if sev["small_sample"] else ""
            lines.append(f"- Frozen hazard score vs severe damage: AUC {full['AUC']:.2f} (95% range "
                         f"{full['low 95%']:.2f} to {full['high 95%']:.2f}){note}.")
        for row in r["direction"]["table"][:2]:
            lines.append(f"- Fall direction, {row['model']}: median error {row['median error (°)']:.0f}° over "
                         f"{row['trees']} trees.")
        lines.append("")
    if not done:
        lines.append("No storm has been analyzed yet.")
    lines += ["", f"_{DISCLAIMER}_", ""]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines)
    (RESULTS_DIR / "summary.md").write_text(text, encoding="utf-8")
    return text


def design_criteria(survey, layers, p, result):
    st.write("The targets the system is judged against. Edit the targets in "
             f"`{CONFIG_DIR / 'design_criteria.yaml'}`. A criterion shows **no data yet** until the analysis that "
             "measures it has been run.")
    table = validation.evaluate_criteria(validation.load_criteria(CONFIG_DIR), WORK_DIR)
    icon = {"pass": "✅ pass", "fail": "❌ fail", "no data yet": "… no data yet"}
    st.dataframe(table.assign(status=table["status"].map(icon)), hide_index=True, width="stretch")
    st.caption("Storm criteria pool every analyzed storm (not calm pairs). The other project criteria (detection "
               "rate, height error, field check, timing) are not built yet.")

    if st.button("Export results for board", type="primary"):
        storms = storm.load_storms(CONFIG_DIR)
        write_summary(table, storms)
        figures = RESULTS_DIR / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        saved = ["summary.md"]
        for s in storms:
            slug = storm.storm_slug(s["name"])
            for kind in ("auc", "rose"):
                png = st.session_state.get(f"storm_{kind}_png_{slug}")
                if png:
                    (figures / f"storm_{slug}_{kind}.png").write_bytes(png)
                    saved.append(f"figures/storm_{slug}_{kind}.png")
            args = st.session_state.get(f"storm_map_args_{slug}")
            if args:
                with st.spinner(f"Drawing the change map of {s['name']}..."):
                    (figures / f"storm_{slug}_change_map.png").write_bytes(maps.render_png(*args, True, 0.6, dpi=300))
                saved.append(f"figures/storm_{slug}_change_map.png")
        st.success(f"Saved in `{RESULTS_DIR}`: " + ", ".join(saved))
        st.caption("Charts (300 dpi) are saved for the storms you opened in Storm check since TIMBER started: open "
                   "each storm there once (and its 'Did we predict it?' and 'Fall direction' boxes), then export.")
    summary = RESULTS_DIR / "summary.md"
    if summary.exists():
        with st.expander("results/summary.md"):
            st.markdown(summary.read_text(encoding="utf-8"))
