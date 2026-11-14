"""App 2: Tree fall hazard (works automatically on any dataset; clicking is optional).

The math is in terrain/treefall.py. This file only draws the sliders
(params) and turns the results into maps, tables and tabs (run).
"""

import numpy as np
import pandas as pd
from affine import Affine
from scipy import ndimage

from terrain import io, treefall, validation, vegetation
from terrain.apps.base import (LOW_MOD_HIGH, M_TO_FT, Overlay, Result, Section, ShapeSpec, distance_to_shapes,
                               outline_annotations, shape_labels, shape_list, shapes_mask)

NAME = "Tree fall hazard"
DESCRIPTION = (
    "Finds every tree from the surface model minus the ground model, gives each one a hazard score for how "
    "likely it is to fall, and predicts which way it would most likely fall and how far it could reach. "
    "It also finds trails, roads, buildings, streams and clearings by itself, so you can see what falling trees "
    "could hit; clicking targets on the map is optional."
)
REQUIRED_INPUTS = ["dtm", "dsm"]
SHAPES = {"targets": ShapeSpec("points", "Extra targets (optional: house, shed, play area...)",
                               "Things the app can't find by itself. They are added to the automatically "
                               "detected features.")}
TEST_AT_HOME = (
    "Measure 8-10 trees with a clinometer phone app (or the shadow/stick method), including some deep in the "
    "canopy and some at the edge of clearings, and compare with the heights in the table (use the Field check "
    "tab). Visit the trees flagged as possibly dead or leaning and see whether they really are. Walk the area "
    "and compare the downed logs the app found with real fallen trees: are they where the app's fall exposure "
    "is high?"
)

CLASS_COLORS = {"low": "#1a9850", "moderate": "#fdae61", "high": "#d73027"}
DOT_SIZE = {"low": 5, "moderate": 9, "high": 12}  # tree-top dot size on interactive maps (pixels)
FEATURE_STYLE = [("clearing", "Clearing", "#fff2a8"), ("trail", "Trail", "#ff9900"), ("road", "Road / driveway", "#e8174a"),
                 ("stream", "Stream / channel", "#1e88e5"), ("building", "Building", "#9c27b0"),
                 ("marked target", "Marked target", "#ffd400")]


SETTING_TABS = ["Basic", "Tree detection", "Hazard weights", "Factor start points", "Fall direction", "Features"]


def params(st, info=None) -> dict:
    """Sliders, grouped in tabs: "Basic" has the few settings most people change,
    the other tabs hold the advanced ones. `info` = {"width_m": map width, "season":
    the orthophoto's season} sets forest starting values for big maps and the
    health settings for the season."""
    big_map = bool(info and info.get("width_m", 0) > 100)
    season = (info or {}).get("season") or "summer leaf-on"
    p = {"season": season}
    basic, detect, weights, points, direction, features = st.tabs(SETTING_TABS)

    with basic:
        c1, c2, c3 = st.columns(3)
        p["min_height"] = c1.slider("Minimum tree height (m)", 0.5, 20.0, 5.0 if big_map else 2.0, 0.5,
                                    help="Anything shorter is not counted as a tree.")
        p["wind_on"] = c2.checkbox("Include prevailing wind", value=True)
        p["wind_from"] = c2.slider("Wind comes FROM (degrees, 270 = west)", 0, 359, 270, 5,
                                   help="Used when 'Include prevailing wind' is on.")
        p["safety"] = c3.slider("Fall reach safety factor", 1.0, 1.5, 1.0, 0.05,
                                help="Reach = tree height x this. Above 1 allows for sliding and bouncing.")
        mode = st.radio("Classes by", ["hazard score", "percent of trees"], horizontal=True,
                        help="Hazard score: fixed score limits, so a safe forest gets few flagged trees. "
                             "Percent of trees: always flags the riskiest share of trees in this map.")
        p["class_mode"] = "percent" if mode == "percent of trees" else "score"
        c1, c2, c3, c4 = st.columns(4)
        p["moderate"] = c1.slider("Moderate hazard from score", 0, 100, 35, 1, help="Used with 'hazard score'.")
        p["high"] = c2.slider("High hazard above score", 0, 100, 60, 1, help="Used with 'hazard score'.")
        p["high_pct"] = c3.slider("High hazard: the riskiest (% of trees)", 0.0, 50.0, 5.0, 0.5,
                                  help="Used with 'percent of trees'.")
        p["moderate_pct"] = c4.slider("Moderate hazard: the next (% of trees)", 0.0, 50.0, 15.0, 0.5,
                                      help="Used with 'percent of trees'.")

    with detect:
        if big_map:
            st.caption("Map wider than 100 m: forest starting values are used (min height 5 m, smoothing 0.5 m, "
                       "crown edge 2 m).")
        c1, c2, c3 = st.columns(3)
        p["smooth"] = c1.slider("Canopy smoothing (m)", 0.0, 2.0, 0.5 if big_map else 0.2, 0.05,
                                help="Smooths the canopy so one tree doesn't get several tops.")
        p["crown_edge"] = c2.slider("Crown edge height (m)", 0.2, 5.0, 2.0 if big_map else 1.0, 0.1,
                                    help="Canopy lower than this is not part of any crown.")
        p["exposure_cell"] = c3.slider("Fall exposure map cell size (m)", 0.05, 2.0, 0.5, 0.05,
                                       help="A coarser exposure map is much faster on big maps.")
        c1, c2, c3 = st.columns(3)
        p["variable"] = c1.checkbox("Variable search window (grows with tree height)", value=True)
        p["win_a"] = c2.slider("Window = a + b x height: a (m)", 0.5, 5.0, 1.5, 0.1)
        p["win_b"] = c3.slider("b (per m of height)", 0.0, 0.3, 0.1, 0.01)
        c1, c2, c3 = st.columns(3)
        p["window"] = c1.slider("Fixed search window (m)", 0.5, 10.0, 2.0, 0.25,
                                help="Used when the variable search window is off.")
        c1, c2, c3 = st.columns(3)
        p["stress_k"] = c1.slider("'Possibly stressed' if VARI this unusual (spreads)", 0.5, 3.0, 1.5, 0.1)
        p["snag_z"] = c2.slider("'Possibly dead': greenness below (spreads)", -5.0, 0.0, -2.0, 0.1)
        p["snag_ratio"] = c3.slider("'Possibly dead': height / crown width above", 1.0, 8.0, 3.0, 0.1)

    with weights:
        p["baseline_seed"] = st.slider("Seed for the random baseline ranking", 0, 999, 42, 1,
                                       help="The 'random order' baseline used when checking the hazard score "
                                            "against simple rules. Same seed = same order.")
        st.caption("How much each factor counts. They are rescaled to add up to 1.")
        cols = st.columns(4)
        p["weights"] = {}
        for i, name in enumerate(treefall.FACTORS):
            p["weights"][name] = cols[i % 4].slider(f"Weight: {name}", 0.0, 1.0, default_weights(season)[name], 0.05,
                                                    key=f"weight_{name}_{season}")
        if season != "summer leaf-on":
            st.caption(SEASON_NOTES[season])

    with points:
        st.caption("Each factor is 0 until its start point and rises to 1 at its full point, so an ordinary "
                   "tree adds nothing to the score.")
        d = treefall.FACTOR_POINTS
        c1, c2, c3, c4 = st.columns(4)
        p["height_start"] = c1.slider("Height: start (m)", 0.0, 40.0, d["height_start"], 1.0)
        p["height_full"] = c1.slider("Height: full (m)", 5.0, 60.0, d["height_full"], 1.0)
        p["slope_start"] = c2.slider("Slope: start (degrees)", 0.0, 45.0, d["slope_start"], 1.0)
        p["slope_full"] = c2.slider("Slope: full (degrees)", 5.0, 70.0, d["slope_full"], 1.0)
        p["slope_smooth"] = c2.slider("Smooth the ground before measuring slope (m)", 0.0, 10.0, d["slope_smooth"], 0.5,
                                      help="The ground model under thick canopy is bumpy; smoothing measures the "
                                           "hillside instead of the bumps.")
        p["wet_start"] = c3.slider("Wetness: start (percentile, 100 = wettest tree)", 0.0, 99.0, d["wet_start"], 1.0)
        p["gap_height"] = c3.slider("Opening = canopy lower than (m)", 0.5, 10.0, 2.0, 0.5)
        p["gap_width"] = c3.slider("Real opening: at least this wide (m)", 0.0, 30.0, d["gap_width"], 0.5,
                                   help="Narrow spaces between neighboring crowns don't make a tree an edge tree.")
        p["gap_area"] = c3.slider("Real opening: at least (m²)", 1.0, 2000.0, d["gap_area"], 10.0,
                                  help="Small holes in the canopy don't make a tree an edge tree.")
        p["edge_dist"] = c3.slider("Edge tree: still counts this far inside (m)", 1.0, 30.0, d["edge_dist"], 1.0)
        p["taller_start"] = c4.slider("Taller than neighbors: start (m)", 0.0, 20.0, d["taller_start"], 0.5)
        p["taller_full"] = c4.slider("Taller than neighbors: full (m)", 1.0, 30.0, d["taller_full"], 0.5)
        c1, c2, c3, c4 = st.columns(4)
        p["slender_start"] = c1.slider("Slenderness: start (height / crown width)", 0.0, 10.0, d["slender_start"], 0.5)
        p["slender_full"] = c1.slider("Slenderness: full", 1.0, 15.0, d["slender_full"], 0.5)
        p["health_start"] = c2.slider("Health: start (spreads less green)", 0.0, 5.0, d["health_start"], 0.1)
        p["health_full"] = c2.slider("Health: full (spreads less green)", 0.5, 8.0, d["health_full"], 0.1)
        p["lean_start"] = c3.slider("Lean: start (crown offset / radius)", 0.0, 1.0, d["lean_start"], 0.05)
        p["lean_full"] = c3.slider("Lean: full (crown offset / radius)", 0.1, 2.0, d["lean_full"], 0.05)
        p["bank_area"] = c4.slider("Stream bank: stream drains at least (m²)", 50.0, 50000.0, d["bank_area"], 50.0,
                                   help="Smaller channels are just rills after rain and can't undercut a bank.")

    with direction:
        c1, c2, c3, c4 = st.columns(4)
        p["dir_weights"] = {
            "uniform": c1.slider("Any direction", 0.0, 1.0, 0.4, 0.05),
            "downhill": c2.slider("Downhill", 0.0, 1.0, 0.3, 0.05),
            "lean": c3.slider("Heavy side of crown", 0.0, 1.0, 0.15, 0.05),
            "wind": c4.slider("Wind", 0.0, 1.0, 0.15, 0.05),
        }
        p["kappa"] = st.slider("How focused each preference is (kappa)", 0.5, 10.0, 2.0, 0.5,
                               help="Bigger = falls more tightly in the preferred direction.")

    with features:
        c1, c2, c3, c4 = st.columns(4)
        p["find_clearings"] = c1.checkbox("Find clearings", value=True)
        p["find_trails"] = c2.checkbox("Find trails and roads", value=True)
        p["find_buildings"] = c3.checkbox("Find buildings", value=True)
        p["find_streams"] = c4.checkbox("Find streams", value=True)
        c1, c2, c3, c4 = st.columns(4)
        p["open_height"] = c1.slider("Open ground: canopy below (m)", 0.1, 2.0, 0.5, 0.1)
        p["open_roughness"] = c1.slider("Open ground: surface roughness below (m)", 0.02, 1.0, 0.15, 0.01,
                                        help="Lawns and trails are smooth; tree crowns are bumpy even where the "
                                             "ground model failed under dense canopy.")
        p["clearing_area"] = c2.slider("Clearing: bigger than (m²)", 5.0, 500.0, 20.0, 5.0)
        p["trail_length"] = c3.slider("Trail: longer than (m)", 5.0, 100.0, 15.0, 1.0)
        p["trail_elongation"] = c3.slider("Trail: length at least x its width", 1.0, 20.0, 5.0, 0.5)
        p["trail_straightness"] = c3.slider("Trail: straightness at least (0-1)", 0.0, 1.0, 0.5, 0.05,
                                            help="Straight-line extent / length along the trail. 0.9 = straight, "
                                                 "low = curls back on itself.")
        p["stream_area"] = c4.slider("Stream: area draining in above (m²)", 50.0, 10000.0, 500.0, 50.0)
        c1, c2, c3, c4 = st.columns(4)
        p["trail_min_width"] = c1.slider("Trail width at least (m)", 0.25, 5.0, 1.0, 0.25)
        p["trail_max_width"] = c2.slider("Trail width at most (m)", 2.0, 20.0, 8.0, 0.5)
        p["paved_roughness"] = c3.slider("Paved: surface roughness below (m)", 0.005, 0.3, 0.03, 0.005)
        p["green_exg"] = c4.slider("Green if ExG at least", -0.2, 0.5, 0.05, 0.01)
        c1, c2, c3, c4 = st.columns(4)
        p["green_vari"] = c1.slider("Green if VARI at least", -0.3, 0.5, 0.05, 0.01)
        p["gray_saturation"] = c1.slider("Gray (roof/pavement) if color saturation below", 0.05, 0.6, 0.25, 0.01,
                                         help="0 = pure gray/white/black, 1 = pure color. Plants are colorful in "
                                              "every season, including red and yellow fall leaves.")
        p["building_height"] = c2.slider("Building: taller than (m)", 1.0, 10.0, 2.0, 0.5)
        p["roof_brightness"] = c2.slider("Building: roof brightness at least (0-1)", 0.0, 1.0, 0.6, 0.05,
                                         help="Set to 0 if roofs are dark (black shingles).")
        p["building_roughness"] = c3.slider("Building: roof roughness below (m)", 0.01, 0.5, 0.15, 0.01)
        p["building_area"] = c4.slider("Building: bigger than (m²)", 5.0, 200.0, 15.0, 5.0)
        p["roof_pitch"] = c4.slider("Building: roof slope at most (degrees)", 10.0, 80.0, 45.0, 1.0)
        st.markdown("**Consequence weights** (priority = hazard x (1 + consequence))")
        c = st.columns(4)
        p["consequence"] = {
            "building": c[0].slider("Building", 0.0, 2.0, 1.0, 0.1),
            "road": c[1].slider("Road", 0.0, 2.0, 0.7, 0.1),
            "trail": c[2].slider("Trail", 0.0, 2.0, 0.5, 0.1),
            "stream": c[3].slider("Stream", 0.0, 2.0, 0.3, 0.1),
            "clearing": c[0].slider("Clearing", 0.0, 2.0, 0.0, 0.1),
            "marked target": c[1].slider("Marked target", 0.0, 2.0, 1.0, 0.1),
            "per other tree": c[2].slider("Per other tree it could hit", 0.0, 0.5, 0.1, 0.05),
            "other trees cap": c[3].slider("Other trees: at most", 0.0, 2.0, 0.5, 0.1),
        }
        st.markdown("**Downed logs**")
        c1, c2, c3, c4 = st.columns(4)
        p["log_low"] = c1.slider("Log: higher than (m)", 0.05, 1.0, 0.2, 0.05)
        p["log_high"] = c2.slider("Log: lower than (m)", 0.5, 3.0, 1.5, 0.1)
        p["log_length"] = c3.slider("Log: longer than (m)", 1.0, 20.0, 4.0, 0.5)
        p["log_width"] = c4.slider("Log: narrower than (m)", 0.2, 3.0, 1.0, 0.1)
    return p


# Every part of the Tree hazard page, in order. The first five are built in run();
# "Design criteria" and "Storm check" are added by the page (timber_ui).
TREE_TABS = ["Trees & hazard", "Where trees could fall", "Features at risk", "Downed logs", "Field check",
             "Design criteria", "Storm check"]

SEASON_NOTES = {
    "autumn color": "Orthophoto season: autumn color. Healthy trees look 'stressed' when their leaves turn, so the "
                    "health weight starts at 0.05 and 'possibly dead' uses only the snag rule (among the least "
                    "green 5% of crowns and a small crown for its height).",
    "leaf-off": "Orthophoto season: leaf-off. Bare crowns can't show health, so the health factor and 'possibly "
                "dead' are switched off (the other weights are rescaled). Use a leaf-on photo to get them back.",
}


def default_weights(season: str = "summer leaf-on") -> dict:
    """Starting hazard weights for the photo's season: in autumn the health weight drops to 0.05."""
    weights = dict(treefall.DEFAULT_WEIGHTS)
    if season == "autumn color":
        weights["health"] = 0.05
    return weights


def canopy_height(layers) -> np.ndarray:
    """Canopy height model: CHM = DSM - DTM (height of things above the ground)."""
    chm = layers.dsm.data - layers.dtm.data
    return np.where(np.isfinite(chm), np.maximum(chm, 0.0), np.nan)  # negative = noise -> 0


def _targets_mask(layers):
    """Cells covered by the user's extra targets (points count within 0.5 m)."""
    shapes = shape_list(layers, "targets")
    if not shapes:
        return None
    polygons = [s for s in shapes if len(s) >= 3]
    mask = shapes_mask(layers.dtm, polygons) if polygons else np.zeros(layers.dtm.shape, dtype=bool)
    small = [s for s in shapes if len(s) < 3]
    if small:
        x, y = layers.xy
        mask |= distance_to_shapes(x, y, small, closed=False) <= 0.5
    return mask


def _crown_grid(crowns, trees, column):
    """Paint each crown with one value from the tree table (NaN outside crowns)."""
    lookup = np.full(int(crowns.max()) + 1, np.nan)
    lookup[trees["tree"].to_numpy()] = trees[column].to_numpy(dtype=float)
    out = lookup[crowns]
    out[crowns == 0] = np.nan
    return out


def run(layers, p) -> Result:
    cell = layers.cell[0]
    warnings = ["The hazard score is a relative screening score inside this map, not a certified arborist "
                "assessment. Features (trails, buildings, streams) are found automatically and can be wrong."]
    chm = canopy_height(layers)
    grids = {"chm": chm, "dtm": layers.dtm.data, "slope": layers.slope_deg, "twi": layers.twi,
             "acc_area": layers.flow["acc_area_m2"], "vari": layers.vari, "exg": layers.exg,
             "has_photo": np.isfinite(layers.ortho).all(axis=2) if layers.ortho is not None else None,
             "saturation": vegetation.saturation(layers.ortho) if layers.ortho is not None else None,
             "brightness": np.max(layers.ortho, axis=2) if layers.ortho is not None else None}
    if layers.ortho is None:
        warnings.append("No orthophoto: tree health, 'possibly dead', green/paved checks are left out.")
    elif p.get("season", "summer leaf-on") in SEASON_NOTES:
        warnings.append(SEASON_NOTES[p["season"]])
    targets = _targets_mask(layers)
    # The slow part. Remembered on disk per survey + settings + marked targets (see Layers.remember).
    res = layers.remember("trees", (sorted(p.items(), key=str), repr(layers.shapes.get("targets", []))),
                          lambda: treefall.analyze(grids, layers.dtm.transform, layers.dtm.crs, cell, p, targets))
    trees = res["trees"]
    if len(trees) == 0:
        return Result(warnings=warnings + ["No trees found. Try a lower minimum height or less smoothing."])
    # Simple rankings to compare the hazard score with (tallest first, random order...).
    trees = validation.add_baselines(trees, p.get("baseline_seed", 42))
    # Where each tree is: latitude/longitude, and how to walk to it from the nearest trail or road.
    if "latitude" not in trees and layers.dtm.crs is not None:
        lon, lat = io.map_to_latlon(trees["x"].to_numpy(), trees["y"].to_numpy(), layers.dtm.crs)
        trees["latitude"], trees["longitude"] = np.round(lat, 7), np.round(lon, 7)
    paths = res["masks"].get("trail", np.zeros(chm.shape, dtype=bool)) | res["masks"].get("road", np.zeros(chm.shape, dtype=bool))
    trees["trail_dist_m"], trees["trail_bearing_deg"] = validation.nearest_on_mask(trees["x"], trees["y"], paths,
                                                                                    layers.dtm.transform)
    res["trees"] = trees

    trees["class_num"] = trees["class"].map({"low": 0, "moderate": 1, "high": 2})
    table = _tree_table(trees, list(res["masks"]))
    masks = res["masks"]
    reach_road = trees[[c for c in ("can reach road", "can reach trail") if c in trees]].any(axis=1) \
        if any(c in trees for c in ("can reach road", "can reach trail")) else pd.Series(False, index=trees.index)
    metrics = {
        "Trees found": str(len(trees)),
        "High-hazard trees": str(int((trees["class"] == "high").sum())),
        "Trees that can reach a road/trail": str(int(reach_road.sum())),
        "Trees that can reach a building": str(int(trees["can reach building"].sum())) if "can reach building" in trees else "0",
        "Downed logs found": str(len(res["logs"])),
    }

    sections = [
        _section_trees(layers, res, trees, table),
        _section_fall(layers, res, trees),
        _section_features(layers, res, trees),
        _section_logs(layers, res, trees, grids, cell),
        _section_field(layers, trees),
    ]
    overlay = Overlay("Hazard class (crowns)", _crown_grid(res["crowns"], trees, "class_num"), classes=LOW_MOD_HIGH)
    return Result(overlays=[overlay], metrics=metrics, table=table, warnings=warnings, sections=sections, raw=res)


def _tree_table(trees, feature_names):
    """The tree table shown to the user (plain column names, sorted by priority)."""
    cols = {"tree": "tree", "priority": "priority", "hazard": "hazard (0-100)", "class": "class",
            "reasons": "reasons", "height": "height (m)", "crown_diam": "crown width (m)",
            "fall_dir_word": "likely fall direction", "fall_dir": "fall direction (°)",
            "possibly dead": "possibly dead", "trees it could hit": "trees it could hit"}
    out = trees[list(cols)].rename(columns=cols)
    out.insert(6, "height (ft)", trees["height"] * M_TO_FT)
    for name in feature_names:
        out[f"can reach {name}"] = trees[f"can reach {name}"]
    for name in treefall.FACTORS:
        out[f"{name} factor"] = trees[f"f_{name}"]
    if "possibly stressed" in trees:
        out["possibly stressed"] = trees["possibly stressed"]
    out["lean direction"] = [treefall.compass_word(d) for d in trees["lean_dir"]]
    out["x"], out["y"] = trees["x"], trees["y"]
    return out


def _tops_scatter(trees, only=None):
    """Tree tops as dots, colored by class. "ids" and "hover" let an interactive
    map show a tree's details when you point at or click it."""
    t = trees if only is None else trees[only]
    hover = [f"Tree {int(i)}: {c} hazard ({h:.0f})<br>{ht:.0f} m tall, falls most likely {d}<br>{why}"
             for i, c, h, ht, d, why in zip(t["tree"], t["class"], t["hazard"], t["height"], t["fall_dir_word"],
                                            t["reasons"])]
    # Low-hazard trees are small and see-through ("88" = about half opacity), so the
    # moderate and high ones stand out among thousands of dots.
    colors = [CLASS_COLORS[c] + ("88" if c == "low" else "") for c in t["class"]]
    sizes = [DOT_SIZE[c] for c in t["class"]]
    return {"type": "scatter", "x": t["x"].to_numpy(), "y": t["y"].to_numpy(), "colors": colors, "size": 10,
            "sizes": sizes, "ids": t["tree"].to_numpy(), "hover": hover, "name": "Tree tops"}


def _section_trees(layers, res, trees, table):
    """The 'Trees & hazard' part: crowns colored by class or score, tree tops, the priority table."""
    crowns = res["crowns"]
    canopy = canopy_height(layers)
    overlays = [
        Overlay("Hazard class (crowns)", _crown_grid(crowns, trees, "class_num"), classes=LOW_MOD_HIGH),
        Overlay("Hazard score (crowns)", _crown_grid(crowns, trees, "hazard"), cmap="RdYlGn_r", vmin=0, vmax=100),
        Overlay("Canopy height (m)", np.where(canopy >= 0.5, canopy, np.nan), cmap="YlGn", vmin=0, units="m"),
    ]
    annotations = [_tops_scatter(trees)]
    targets = shape_list(layers, "targets")
    if targets:
        annotations += outline_annotations(targets, color="#ffd400")
        for t, name in zip(targets, shape_labels(layers, "targets")):
            if name:
                annotations.append({"type": "point", "x": t[0][0], "y": t[0][1], "label": name, "color": "#ffd400"})
        hit = trees[trees["can reach marked target"]] if "can reach marked target" in trees else trees.iloc[:0]
        for _, t in hit.iterrows():
            annotations.append({"type": "circle", "x": t["x"], "y": t["y"], "r": t["height"], "color": "#ff2a2a"})
    extra = [("How the hazard scores are spread (use this to set the class limits)", treefall.score_spread(trees["hazard"]))]
    return Section("Trees & hazard", overlays=overlays, annotations=annotations, table=table,
                   table_title="Top 20 priority trees (priority = hazard x (1 + consequence))", table_head=20,
                   extra_tables=extra,
                   text="Each crown is colored by its hazard class; dots mark tree tops. The **reasons** column "
                        "lists the factors that add most to each tree's score.")


def _coarse_raster(layers, k):
    t = layers.dtm.transform
    return io.Raster(data=None, transform=Affine(t.a * k, 0, t.c, 0, t.e * k, t.f), crs=layers.dtm.crs)


def _upsample(grid, k, shape):
    """Repeat each coarse cell k x k times and trim to the analysis grid."""
    return np.repeat(np.repeat(grid, k, axis=0), k, axis=1)[: shape[0], : shape[1]]


def _section_fall(layers, res, trees):
    """The 'Where trees could fall' part: exposure and reach maps, direction arrows, GeoTIFF downloads."""
    k = res["grid_factor"]
    shape = layers.dtm.shape
    valid = np.isfinite(layers.dtm.data)
    exposure = np.where(valid, _upsample(res["exposure"], k, shape), np.nan)
    reach = np.where(valid, _upsample(res["reach"], k, shape), np.nan)
    overlays = [
        Overlay("Fall exposure (hazard-weighted)", np.where(exposure > 0, exposure, np.nan), cmap="inferno_r", vmin=0),
        Overlay("Number of trees that could reach each spot", np.where(reach > 0, reach, np.nan), cmap="viridis", vmin=0),
    ]
    # Arrows: most likely fall direction of moderate and high hazard trees, as long as the tree is tall.
    risky = trees[trees["class"] != "low"]
    angle = np.radians(risky["fall_dir"].to_numpy())
    length = risky["height"].to_numpy()
    arrows = {"type": "arrows", "x": risky["x"].to_numpy(), "y": risky["y"].to_numpy(),
              "dx": length * np.sin(angle), "dy": length * np.cos(angle),
              "colors": [CLASS_COLORS[c] for c in risky["class"]]}
    cols = ["tree", "class", "hazard", "height", "fall_dir", "fall_dir_word"] + \
           [c for c in trees.columns if c.startswith("chance lands on")]
    table = trees[cols].rename(columns={"hazard": "hazard (0-100)", "height": "height (m)",
                                        "fall_dir": "most likely direction (°)", "fall_dir_word": "direction"})
    coarse = _coarse_raster(layers, k)
    downloads = [
        ("Fall exposure (GeoTIFF)", io.geotiff_bytes(res["exposure"], coarse), "fall_exposure.tif", "image/tiff"),
        ("Trees that could reach (GeoTIFF)", io.geotiff_bytes(res["reach"], coarse), "fall_reach_count.tif", "image/tiff"),
    ]
    text = ("**Fall exposure**: for each spot, the hazard scores of all trees that could land there, each "
            "multiplied by the chance that the tree falls that way. **Arrows** show the most likely fall direction "
            "of moderate and high hazard trees; their length is the tree's height. Each tree's direction chances "
            "mix: any direction, downhill, toward the heavy side of the crown, and with the wind.")
    return Section("Where trees could fall", overlays=overlays, annotations=[arrows], table=table,
                   table_title="Fall direction and chance of landing on each feature type", table_head=20,
                   downloads=downloads, text=text)


def _feature_overlay(masks, shape):
    """One grid with a number per feature type, plus its legend entries."""
    grid = np.full(shape, np.nan)
    classes = []
    for value, (name, label, color) in enumerate(FEATURE_STYLE, start=1):
        if name in masks and masks[name].any():
            grid[masks[name]] = value
            classes.append((value, label, color))
    return grid, classes


def _section_features(layers, res, trees):
    """The 'Features at risk' part: detected features and how many trees can reach each."""
    masks = res["masks"]
    grid, classes = _feature_overlay(masks, layers.dtm.shape)
    rows = []
    for name, mask in masks.items():
        _, pieces = ndimage.label(mask, structure=np.ones((3, 3)))
        can = trees.get(f"can reach {name}", pd.Series(False, index=trees.index))
        chance = trees.get(f"chance lands on {name}", pd.Series(0.0, index=trees.index))
        rows.append({"feature": name, "pieces": pieces, "area (m²)": mask.sum() * layers.cell_area,
                     "trees that can reach it": int(can.sum()),
                     # Total exposure = sum over trees of (hazard / 100) x (chance it lands on this feature type)
                     "total exposure score": float((trees["hazard"] / 100 * chance).sum())})
    table = pd.DataFrame(rows)
    extra = []
    if res["trail_info"]:
        t = layers.dtm.transform
        info = pd.DataFrame(res["trail_info"])
        info["x"] = t.c + (info.pop("col") + 0.5) * t.a
        info["y"] = t.f + (info.pop("row") + 0.5) * t.e
        extra.append(("Trails and roads found", info.rename(columns={"length_m": "length (m)", "width_m": "width (m)"})))
    overlays = [Overlay("Detected features", grid, classes=classes)] if classes else []
    text = ("Found automatically: **clearings** (open ground bigger than the minimum area), **trails/roads** "
            "(long, narrow open strips; paved-looking ones are roads), **buildings** (tall, smooth, not green), "
            "and **streams** (where lots of ground drains through). Marked targets are added if you clicked any.")
    return Section("Features at risk", overlays=overlays, annotations=[_tops_scatter(trees, trees["class"] == "high")],
                   table=table, table_title="What falling trees could hit", extra_tables=extra, text=text)


def _section_logs(layers, res, trees, grids, cell):
    """The 'Downed logs' part: detected logs with their axis, and logs vs random spots."""
    logs = res["logs"]
    k = res["grid_factor"]
    exposure_full = _upsample(res["exposure"], k, layers.dtm.shape)
    log_grid = np.where(res["log_labels"] > 0, 1.0, np.nan)
    overlays = [Overlay("Downed logs", log_grid, classes=[(1, "Downed log (fallen tree)", "#8c510a")]),
                Overlay("Fall exposure with logs", np.where(exposure_full > 0, exposure_full, np.nan), cmap="inferno_r", vmin=0)]
    annotations = []
    extra = []
    if len(logs):
        a = np.radians(logs["axis_deg"].to_numpy())
        half = logs["length_m"].to_numpy() / 2
        dx, dy = half * np.sin(a), half * np.cos(a)
        x, y = logs["x"].to_numpy(), logs["y"].to_numpy()
        seg = np.stack([np.column_stack([x - dx, y - dy]), np.column_stack([x + dx, y + dy])], axis=1)
        annotations.append({"type": "segments", "segments": seg, "color": "#00e5ff", "width": 3.0})
        compare = treefall.compare_logs_with_random(logs, grids, exposure_full, trees, cell)
        extra.append(("Downed logs vs random spots in the map", compare))
    table = logs.drop(columns=["row", "col"]).rename(columns={"length_m": "length (m)", "width_m": "width (m)",
                                                             "axis_deg": "axis direction (°, 0-180)"})
    if len(table):
        table["axis"] = [f"{treefall.compass_word(d)}-{treefall.compass_word(d + 180)}" for d in logs["axis_deg"]]
    text = ("Long, thin, low objects (between the log height limits above the ground) are probably trees that "
            "already fell. Which end was the root can't be told, so only the axis is given. Fences, hedges and low "
            "walls have the same shape, so check logs near houses in person. If the hazard model works, logs should "
            "be more common where slope, wetness and fall exposure are higher than at random spots.")
    return Section("Downed logs", overlays=overlays, annotations=annotations, table=table, table_title="Downed logs",
                   extra_tables=extra, text=text)


def _section_field(layers, trees):
    """The original (not blind) field sheet. The page shows the blind field check instead."""
    crs = layers.dtm.crs

    def custom(st):
        n = st.number_input("Number of top priority trees (plus the same number of random low-hazard trees)",
                            1, 200, 20, 1, key="field_n")
        sheet = treefall.field_check_export(trees, crs, int(n))
        st.download_button("Download field check sheet (CSV)", sheet.to_csv(index=False), "field_check.csv",
                           "text/csv", key="field_download")
        st.dataframe(sheet.head(10), hide_index=True)
        st.markdown("**After the field visit:** upload the filled-in sheet.")
        upload = st.file_uploader("Filled field check sheet (CSV)", type=["csv"], key="field_upload")
        if upload is not None:
            try:
                summary, table = treefall.field_check_compare(trees, pd.read_csv(upload))
            except Exception as err:
                st.error(f"Could not compare: {err}")
                return
            cols = st.columns(min(3, len(summary)))
            for i, (label, value) in enumerate(summary.items()):
                cols[i % len(cols)].markdown(f"**{label}**  \n{value}")
            st.dataframe(table, hide_index=True)

    text = ("Check the app against reality. Download the sheet (the top priority trees plus random low-hazard "
            "trees for comparison, with latitude/longitude for a phone map app), fill in the blank columns in the "
            "woods (observed_height, observed_lean_deg, dead_or_damaged y/n, root_damage y/n, notes), then upload it. "
            "Trees are matched by location, so changing sliders later doesn't break the match.")
    return Section("Field check", text=text, custom=custom)
