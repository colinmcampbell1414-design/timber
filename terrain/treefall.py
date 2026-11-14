"""Tree fall hazard: find trees, score how likely each is to fall, predict where
it would land, and find what it could hit.

All the heavy math for the "Tree fall hazard" app lives here; the app file
(terrain/apps/trees.py) only draws sliders and builds the result.

Speed rules (forest maps have ~5 million cells and thousands of trees):
- Per-tree numbers (crown area, mean slope under the crown, ...) are worked
  out for ALL trees at once with "label" sums: one pass over the grid where
  each cell adds its value to its tree's total (np.bincount).
- Anything done tree by tree only looks at a small window around that tree.

Directions are compass degrees: 0 = north, 90 = east, 180 = south, 270 = west.
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

from terrain import slope as slope_mod
from terrain import io

COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
N_BINS = 36  # fall directions in steps of 10 degrees
BIN_DEG = 360 / N_BINS
BIN_CENTERS = np.arange(N_BINS) * BIN_DEG

FACTORS = ["height", "slope", "wetness", "exposure", "slenderness", "health", "lean", "stream bank"]
DEFAULT_WEIGHTS = {"height": 0.20, "slope": 0.15, "wetness": 0.10, "exposure": 0.15, "slenderness": 0.10,
                   "health": 0.20, "lean": 0.05, "stream bank": 0.05}

# Where each factor starts counting and where it is full (starting values, adjustable
# in the app). An ordinary tree is below every start point and scores about 0.
FACTOR_POINTS = {
    "height_start": 15.0, "height_full": 30.0,     # meters tall
    "slope_start": 15.0, "slope_full": 35.0,       # degrees of hillside under the crown
    "slope_smooth": 2.0,                           # meters of smoothing before measuring slope
    "wet_start": 75.0,                             # wetness percentile (100 = wettest tree in the map)
    "gap_width": 5.0, "gap_area": 100.0,           # a real opening is this wide (m) and big (m²)
    "edge_dist": 10.0,                             # how far inside the forest an edge tree still counts (m)
    "taller_start": 3.0, "taller_full": 10.0,      # meters taller than the neighbors
    "slender_start": 3.0, "slender_full": 7.0,     # height / crown width
    "health_start": 1.0, "health_full": 3.0,       # spreads less green than the median crown
    "lean_start": 0.5, "lean_full": 1.0,           # crown middle offset / crown radius
    "bank_area": 5000.0,                           # m² draining into a channel to count as a stream bank
}

FEATURE_TYPES = ["building", "road", "trail", "stream", "clearing", "marked target"]
DEFAULT_CONSEQUENCE = {"building": 1.0, "road": 0.7, "trail": 0.5, "stream": 0.3, "clearing": 0.0,
                       "marked target": 1.0, "per other tree": 0.1, "other trees cap": 0.5}


def compass_word(deg):
    """Turn a compass angle into a word like "SE" (NaN -> "-")."""
    if deg is None or not np.isfinite(deg):
        return "-"
    return COMPASS[int(((deg % 360) + 22.5) // 45) % 8]


def _robust_z(values: np.ndarray) -> np.ndarray:
    """How unusual each value is compared with the others, in "typical spreads".

    z = (value - median) / (1.4826 x MAD), where MAD is the median distance
    from the median. Unlike the mean and standard deviation, a few extreme
    trees can't pull these numbers around. (1.4826 makes it match a normal
    standard deviation.)
    """
    med = np.nanmedian(values)
    spread = 1.4826 * np.nanmedian(np.abs(values - med))
    return (values - med) / (spread if spread > 1e-9 else 1e-9)


def _label_mean(labels_flat, values, n):
    """Average of `values` over the cells of each tree (NaN cells skipped)."""
    v = np.asarray(values, dtype=float).ravel()
    ok = np.isfinite(v)
    total = np.bincount(labels_flat[ok], weights=v[ok], minlength=n + 1)
    count = np.bincount(labels_flat[ok], minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (total / count)[1:]


# ---------------------------------------------------------------------------
# 1. Tree detection
# ---------------------------------------------------------------------------
def detect_trees(chm, cell, min_height=2.0, smooth=0.2, crown_edge=1.0, variable=True,
                 win_a=1.5, win_b=0.1, fixed_window=2.0, exclude=None):
    """Find tree tops and outline each crown.

    Tree tops are "local maxima": cells higher than everything around them.
    How far "around" is the search window:
      - variable window (default): window = a + b x height, so a 30 m tree
        searches 1.5 + 0.1 x 30 = 4.5 m wide. Tall trees have wide crowns, so
        one fixed window either splits big trees in two or merges small ones.
      - fixed window: the same width for every tree.
    Crowns: "watershed" segmentation. Picture the canopy upside down: each
    tree top is the bottom of a basin; filling the basins until they meet
    draws the borders between touching crowns.

    exclude: optional mask of cells that can't be trees (e.g. detected buildings,
    whose flat roofs would otherwise look like a tree top).

    Returns (tops, crowns): tops is a table with row, col, height for each tree
    (ordered north to south, west to east); crowns is a grid with the tree
    number (1, 2, ...) in each crown cell and 0 elsewhere.
    """
    from skimage.segmentation import watershed

    chm_s = _smooth(chm, smooth, cell)
    chm_s = np.nan_to_num(chm_s, nan=0.0)
    # Candidates: cells at least as high as their 8 neighbors.
    cand = (chm_s == ndimage.maximum_filter(chm_s, size=3, mode="nearest")) & (chm_s >= min_height)
    crown_ok = chm_s >= min(crown_edge, min_height)
    if exclude is not None:
        cand &= ~exclude
        crown_ok &= ~exclude
    rows, cols = np.nonzero(cand)
    heights = chm_s[rows, cols]
    if variable:
        radius = (win_a + win_b * heights) / 2  # half the window width, in meters
    else:
        radius = np.full(heights.shape, fixed_window / 2)
    keep = _suppress_non_maxima(rows * cell, cols * cell, heights, radius)
    rows, cols, heights = rows[keep], cols[keep], heights[keep]

    order = np.lexsort((cols, rows))  # north to south, then west to east: stable tree numbers
    rows, cols = rows[order], cols[order]
    markers = np.zeros(chm.shape, dtype=np.int32)
    markers[rows, cols] = np.arange(1, len(rows) + 1)
    crowns = watershed(-chm_s, markers, mask=crown_ok)
    return pd.DataFrame({"row": rows, "col": cols}), crowns


def _smooth(z, radius_m, cell):
    """Gaussian smoothing that ignores NaN cells (0 radius = no smoothing)."""
    if radius_m <= 0:
        return z.copy()
    valid = np.isfinite(z)
    sigma = radius_m / cell
    num = ndimage.gaussian_filter(np.where(valid, z, 0.0), sigma, mode="nearest")
    den = ndimage.gaussian_filter(valid.astype(float), sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid] = np.nan
    return out


def _suppress_non_maxima(y, x, heights, radius):
    """Keep a candidate only if no taller candidate is within its own search radius.

    Pairs of candidates closer than the biggest radius are found with a
    k-d tree (a fast way to find nearby points). Ties in height are broken
    by position so a flat-topped peak keeps exactly one top.
    """
    n = len(heights)
    if n == 0:
        return np.zeros(0, dtype=bool)
    pts = np.column_stack([x, y])
    pairs = cKDTree(pts).query_pairs(r=float(radius.max()), output_type="ndarray")
    keep = np.ones(n, dtype=bool)
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        d = np.hypot(*(pts[i] - pts[j]).T)
        j_beats_i = (heights[j] > heights[i]) | ((heights[j] == heights[i]) & (j < i))
        keep[i[j_beats_i & (d <= radius[i])]] = False
        keep[j[~j_beats_i & (d <= radius[j])]] = False
    return keep


def measure_trees(tops, crowns, chm, transform, cell):
    """Height, crown size and crown center for every tree (all at once)."""
    n = len(tops)
    lab = crowns.ravel()
    ids = np.arange(1, n + 1)
    area_cells = np.bincount(lab, minlength=n + 1)[1:].astype(float)
    area_cells = np.maximum(area_cells, 1.0)  # a tree always has at least its top cell
    height = ndimage.maximum(np.nan_to_num(chm, nan=0.0), crowns, ids) if n else np.zeros(0)
    top_h = chm[tops["row"], tops["col"]] if n else np.zeros(0)
    height = np.fmax(height, np.nan_to_num(top_h))
    r_idx, c_idx = np.indices(crowns.shape)
    cen_r = _label_mean(lab, r_idx, n)
    cen_c = _label_mean(lab, c_idx, n)
    cen_r = np.where(np.isfinite(cen_r), cen_r, tops["row"].to_numpy())
    cen_c = np.where(np.isfinite(cen_c), cen_c, tops["col"].to_numpy())
    area = area_cells * cell * cell
    t = transform
    trees = pd.DataFrame({
        "tree": ids,
        "row": tops["row"].to_numpy(), "col": tops["col"].to_numpy(),
        "x": t.c + (tops["col"].to_numpy() + 0.5) * t.a,  # map coordinates of the top (above the trunk)
        "y": t.f + (tops["row"].to_numpy() + 0.5) * t.e,
        "height": height,
        "crown_area": area,
        # Crown diameter of a circle with the same area: area = pi x (d/2)^2  ->  d = 2 x sqrt(area / pi)
        "crown_diam": 2 * np.sqrt(area / np.pi),
        "crown_x": t.c + (cen_c + 0.5) * t.a,
        "crown_y": t.f + (cen_r + 0.5) * t.e,
    })
    return trees


# ---------------------------------------------------------------------------
# 2. Hazard factors (each 0 to 1)
# ---------------------------------------------------------------------------
def ramp(x, start, full):
    """0 until the start point, rising in a straight line to 1 at the full point.

    This is how every hazard factor turns a measurement into 0-1: an ordinary
    tree (below the start point) adds nothing to the score, and only trees past
    a "concerning" level count. Example: height 15 -> 30 m gives a 20 m tree
    (20 - 15) / (30 - 15) = 0.33.
    """
    x = np.asarray(x, dtype=float)
    return np.clip((x - start) / max(full - start, 1e-9), 0.0, 1.0)


def hazard_factors(trees, crowns, grids, cell, p, open_ground=None):
    """Work out the eight hazard factors for every tree.

    grids: dict with "chm", "dtm", "twi", "acc_area", and optionally
    "vari", "exg" (all on the same grid as `crowns`).
    p: settings (see trees.py params), including the start and full point of
    every factor (see FACTOR_POINTS).
    open_ground: low, smooth ground (open_ground_mask); worked out here if not given.
    Adds factor columns (0-1) and the measurements behind them to `trees`.
    """
    n = len(trees)
    lab = crowns.ravel()
    h = trees["height"].to_numpy()
    out = trees.copy()

    # a. Height: taller trees reach farther and hit harder.
    out["f_height"] = ramp(h, p["height_start"], p["height_full"])

    # b. Slope under the crown. The ground model under thick canopy is bumpy,
    #    so the ground is smoothed first (over a few meters) to measure the
    #    real hillside, not the bumps.
    dtm_s = _smooth(grids["dtm"], p["slope_smooth"], cell)
    slope_s = slope_mod.slope_degrees(dtm_s, cell, cell)
    out["slope_deg"] = _label_mean(lab, slope_s, n)
    out["f_slope"] = ramp(out["slope_deg"].fillna(0), p["slope_start"], p["slope_full"])

    # c. Soil wetness: average TWI under the crown, ranked within the map
    #    (percentile 100 = the wettest tree spot in this map). Only the wettest
    #    trees count: 0 below the start percentile, 1 for the very wettest.
    twi = _label_mean(lab, grids["twi"], n)
    out["twi"] = twi
    wet_pct = 100 * pd.Series(twi).rank(pct=True).fillna(0).to_numpy() if n else np.zeros(0)
    out["wetness_percentile"] = wet_pct
    out["f_wetness"] = ramp(wet_pct, p["wet_start"], 100.0)

    # d. Exposure to wind: the larger of
    #    (i) edge tree: distance from the crown edge to the nearest REAL opening
    #        (smooth open ground, wide and big enough; the narrow spaces
    #        between neighboring crowns don't count). 1 if the crown touches an
    #        opening, 0 at the "edge distance" or more inside the forest.
    #    (ii) sticking out: how much taller than the median tree within 15 m
    #        (ramp from the start to the full point).
    chm = grids["chm"]
    # An opening must be low AND smooth ground: under dense canopy the ground model
    # sometimes follows the treetops, which looks like a gap but is really forest.
    if open_ground is None:
        open_ground = open_ground_mask(grids, cell, p)
    gap = (np.nan_to_num(chm, nan=99.0) < p["gap_height"]) & open_ground
    real_gap = real_openings(gap, cell, p["gap_width"], p["gap_area"])
    if real_gap.any():
        dist_to_gap = ndimage.distance_transform_edt(~real_gap, sampling=cell)
        edge_dist = ndimage.minimum(dist_to_gap, crowns, np.arange(1, n + 1)) if n else np.zeros(0)
        # A crown cell right next to the opening is one cell away: count that as touching (0 m).
        edge_dist = np.maximum(np.asarray(edge_dist, dtype=float) - cell, 0.0)
    else:
        edge_dist = np.full(n, np.inf)
    out["gap_dist_m"] = edge_dist
    f_edge = 1 - ramp(edge_dist, 0.0, p["edge_dist"])
    taller = _taller_than_neighbors(trees, radius=15.0)
    out["taller_than_neighbors_m"] = taller
    f_taller = ramp(taller, p["taller_start"], p["taller_full"])
    out["exposure_edge"], out["exposure_taller"] = f_edge, f_taller
    out["f_exposure"] = np.maximum(f_edge, f_taller)

    # e. Slenderness: height / crown diameter. Tall thin trees are less stable.
    ratio = h / np.maximum(trees["crown_diam"].to_numpy(), cell)
    out["slenderness"] = ratio
    out["f_slenderness"] = ramp(ratio, p["slender_start"], p["slender_full"])

    # f. Health from the photo: crowns that are less green than the other
    #    crowns (low VARI and low ExG). z = how unusual, in typical spreads
    #    (negative = less green than usual). 0 until z is "start" spreads below
    #    the median, 1 at "full" spreads below.
    season = p.get("season", "summer leaf-on")
    if grids.get("vari") is not None and season != "leaf-off":
        vari = _label_mean(lab, grids["vari"], n)
        exg = _label_mean(lab, grids["exg"], n)
        out["mean_vari"], out["mean_exg"] = vari, exg
        z = (np.nan_to_num(_robust_z(vari)) + np.nan_to_num(_robust_z(exg))) / 2
        out["greenness_z"] = z
        out["f_health"] = ramp(-z, p["health_start"], p["health_full"])
        out["possibly stressed"] = _robust_z(vari) < -p["stress_k"]
        # Possible dead tree ("snag"): tall enough, very low greenness, and a small crown for its height.
        if season == "autumn color":
            # In autumn healthy trees turn yellow and red, so "less green than usual" proves
            # little. Only the snag rule is used: among the darkest (least green) 5% of
            # crowns AND a small crown for its height.
            darkest = vari <= np.nanpercentile(vari, 5) if np.isfinite(vari).any() else np.zeros(n, dtype=bool)
            out["possibly dead"] = (h > 5) & darkest & (ratio > p["snag_ratio"])
        else:
            out["possibly dead"] = (h > 5) & (z < p["snag_z"]) & (ratio > p["snag_ratio"])
    else:
        # Unknown without a photo, and switched off for a leaf-off photo (bare crowns
        # all look "dead"). The other weights are rescaled (see hazard_score).
        out["f_health"] = np.nan
        out["possibly dead"] = False

    # g. Lean / lopsided crown (low confidence): the arrow from the tree top to
    #    the middle of the crown, divided by the crown radius. 0 = balanced crown,
    #    1 = the crown's middle is a whole radius off to one side. Crowns are
    #    never perfectly round, so small offsets don't count.
    dx = (trees["crown_x"] - trees["x"]).to_numpy()  # meters east
    dy = (trees["crown_y"] - trees["y"]).to_numpy()  # meters north
    offset = np.hypot(dx, dy)
    lean_ratio = offset / np.maximum(trees["crown_diam"].to_numpy() / 2, cell)
    out["lean_ratio"] = lean_ratio
    out["f_lean"] = ramp(lean_ratio, p["lean_start"], p["lean_full"])
    # Compass direction of the heavy side (undefined if the offset is under one cell).
    out["lean_dir"] = np.where(offset >= cell, np.degrees(np.arctan2(dx, dy)) % 360, np.nan)

    # h. Stream bank: within one tree height of a real stream (a channel draining
    #    at least the "stream bank" area) AND on ground steeper than 15 degrees =
    #    the bank may be undercut by water.
    channel = np.nan_to_num(grids["acc_area"], nan=0.0) >= p["bank_area"]
    if channel.any():
        dist_to_channel = ndimage.distance_transform_edt(~channel, sampling=cell)
        d = dist_to_channel[trees["row"], trees["col"]]
    else:
        d = np.full(n, np.inf)
    out["stream_dist_m"] = d
    out["f_stream bank"] = ((d <= h) & (out["slope_deg"].fillna(0).to_numpy() >= 15)).astype(float)

    # Which way is downhill under the crown: average the (smoothed) ground's east
    # and north gradients over the crown, then point the opposite way (water and
    # trees go down, gradients point up).
    ge, gn = slope_mod.surface_gradients(dtm_s, cell, cell)
    me, mn = _label_mean(lab, ge, n), _label_mean(lab, gn, n)
    out["downhill_dir"] = np.degrees(np.arctan2(-me, -mn)) % 360
    return out


def real_openings(gap, cell, min_width, min_area):
    """Keep only the parts of the canopy gaps that are real openings.

    The spaces between neighboring crowns are narrow, but they all join up into
    one big network, so an area limit alone keeps them. First an "opening" step
    keeps only places at least `min_width` wide (shrink the gaps by half the width,
    then grow what is left back by the same amount: narrow slivers vanish), then
    pieces smaller than `min_area` are dropped.
    """
    if min_width > 0:
        half = min_width / 2
        core = ndimage.distance_transform_edt(gap, sampling=cell) >= half  # spots at least half a width from canopy
        if not core.any():
            return core
        gap = ndimage.distance_transform_edt(~core, sampling=cell) <= half  # grow back by the same amount
    lab, n = ndimage.label(gap)
    area = np.bincount(lab.ravel(), minlength=n + 1) * cell * cell
    area[0] = 0  # label 0 = not a gap
    return (area >= min_area)[lab]


def _taller_than_neighbors(trees, radius):
    """How much taller each tree is than the median height of trees within `radius` m."""
    n = len(trees)
    if n == 0:
        return np.zeros(0)
    pts = trees[["x", "y"]].to_numpy()
    h = trees["height"].to_numpy()
    neighbors = cKDTree(pts).query_ball_point(pts, r=radius)
    out = np.empty(n)
    for i, nb in enumerate(neighbors):
        others = [j for j in nb if j != i]
        # A tree with no neighbors at all stands alone: fully exposed (10 m counts as 1).
        out[i] = h[i] - np.median(h[others]) if others else 10.0
    return np.maximum(out, 0.0)


def hazard_score(factors, weights):
    """Hazard score 0-100 = 100 x weighted average of the factors.

    Weights are rescaled to add up to 1. A factor that can't be worked out
    (health without a photo) is left out and the other weights are rescaled.
    """
    total = np.zeros(len(factors))
    weight_sum = np.zeros(len(factors))
    for name in FACTORS:
        f = factors[f"f_{name}"].to_numpy(dtype=float)
        known = np.isfinite(f)
        total += np.where(known, weights[name] * np.nan_to_num(f), 0.0)
        weight_sum += np.where(known, weights[name], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * np.where(weight_sum > 0, total / weight_sum, 0.0)


def assign_classes(hazard, p):
    """Low / moderate / high for each tree, in one of two ways.

    "score" (default): fixed score limits, e.g. moderate from 35, high above 60.
        The number of flagged trees depends on how risky the trees really are.
    "percent": the riskiest X% of trees are high and the next Y% moderate.
        Always flags the same share of trees, even in a very safe forest, so it
        ranks trees within this map rather than judging them.
    """
    hazard = np.asarray(hazard, dtype=float)
    n = len(hazard)
    if p.get("class_mode", "score") == "percent":
        rank = np.empty(n, dtype=int)
        rank[np.argsort(-hazard, kind="stable")] = np.arange(n)  # 0 = highest score
        n_high = int(round(n * p["high_pct"] / 100))
        n_mod = int(round(n * p["moderate_pct"] / 100))
        return np.where(rank < n_high, "high", np.where(rank < n_high + n_mod, "moderate", "low"))
    return np.where(hazard > p["high"], "high", np.where(hazard >= p["moderate"], "moderate", "low"))


def score_spread(hazard):
    """How many trees fall in each 10-point score band (helps pick class limits)."""
    counts, _ = np.histogram(np.asarray(hazard, dtype=float), bins=np.arange(0, 101, 10))
    n = max(len(hazard), 1)
    return pd.DataFrame({"hazard score": [f"{a}-{a + 10}" for a in range(0, 100, 10)],
                         "trees": counts, "% of trees": np.round(100 * counts / n, 1)})


def reasons(trees, weights, max_items=3):
    """Plain-language list of the (up to 3) factors that add most to each tree's score."""
    texts = []
    for _, t in trees.iterrows():
        contrib = []
        for name in FACTORS:
            f = t.get(f"f_{name}", np.nan)
            if np.isfinite(f) and f > 0:  # only factors past their start point
                contrib.append((weights[name] * f, name))
        contrib.sort(reverse=True)
        words = [_reason_words(t, name) for _, name in contrib[:max_items]]
        texts.append("; ".join(words) if words else "no hazard factors past their start points")
    return texts


def _reason_words(t, name):
    """A few plain words for one hazard factor of one tree, e.g. 'on steep 31° slope'."""
    if name == "height":
        return f"Tall ({t['height']:.0f} m)"
    if name == "slope":
        return f"on steep {t['slope_deg']:.0f}° slope"
    if name == "wetness":
        return f"on wet ground (wetter than {t['wetness_percentile']:.0f}% of trees)"
    if name == "exposure":
        if t["exposure_edge"] >= t["exposure_taller"]:  # say whichever part is bigger
            return "exposed edge tree" if t["gap_dist_m"] < 1 else f"near a gap ({t['gap_dist_m']:.0f} m)"
        return f"{t['taller_than_neighbors_m']:.0f} m taller than neighbors"
    if name == "slenderness":
        return f"tall and thin (height/crown {t['slenderness']:.1f})"
    if name == "health":
        return "possibly dead crown" if t.get("possibly dead", False) else "low greenness"
    if name == "lean":
        return f"lopsided crown, heavy side {compass_word(t['lean_dir'])}"
    if name == "stream bank":
        return "on a stream bank"
    return name


# ---------------------------------------------------------------------------
# 3. Fall direction
# ---------------------------------------------------------------------------
def von_mises(mu_deg, kappa):
    """A smooth "bump" over the 36 directions, highest at mu, adding up to 1.

    Shape: exp(kappa x cos(angle - mu)). kappa = how narrow the bump is
    (0 = flat, 2 = a broad hill, 10 = a narrow spike). mu can be an array
    (one bump per tree); NaN mu gives a flat (uniform) row.
    """
    mu = np.atleast_1d(np.asarray(mu_deg, dtype=float))[:, None]
    bump = np.exp(kappa * np.cos(np.radians(BIN_CENTERS[None, :] - np.nan_to_num(mu))))
    bump = bump / bump.sum(axis=1, keepdims=True)
    bump[~np.isfinite(mu[:, 0])] = 1.0 / N_BINS
    return bump


def fall_probabilities(trees, w, kappa=2.0, wind_from=270.0, wind_on=True):
    """Chance of falling in each of 36 directions (rows = trees, adds up to 1).

    p(direction) = w_uniform x (same in every direction)
                 + w_downhill x slope factor x bump toward downhill
                 + w_lean x lean factor x bump toward the heavy side
                 + w_wind x bump toward where the wind blows TO (wind from + 180)
    then divided by the total weight so it adds up to 1. On flat ground the
    slope factor is 0, so downhill adds nothing; a balanced crown adds no lean.
    """
    n = len(trees)
    uniform = np.full((n, N_BINS), 1.0 / N_BINS)
    w_down = w["downhill"] * trees["f_slope"].to_numpy()
    w_lean = w["lean"] * np.where(np.isfinite(trees["lean_dir"]), trees["f_lean"], 0.0)
    w_wind = w["wind"] if wind_on else 0.0
    probs = (w["uniform"] * uniform
             + w_down[:, None] * von_mises(trees["downhill_dir"], kappa)
             + w_lean[:, None] * von_mises(trees["lean_dir"], kappa)
             + w_wind * von_mises(np.full(n, (wind_from + 180) % 360), kappa))
    total = probs.sum(axis=1, keepdims=True)
    probs = np.where(total > 0, probs / np.where(total > 0, total, 1), uniform)
    return probs


def _cumulative(prob_row):
    """Cumulative chance from -360 to +720 degrees (the 36 bins repeated 3 times).

    Each bin's chance is spread evenly over its 10 degrees (bin k covers
    k x 10 - 5 to k x 10 + 5). With this, the chance of falling between any
    two angles a and b is cum(b) - cum(a), which works even across north.
    """
    edges = -360 - BIN_DEG / 2 + BIN_DEG * np.arange(3 * N_BINS + 1)  # -365, -355, ..., 715
    rep = np.tile(prob_row, 3)
    cum = np.concatenate([[0.0], np.cumsum(rep)])
    return edges, cum


def _chance_between(edges, cum, lo, hi):
    """Chance of falling between angles lo and hi (degrees, hi - lo <= 360)."""
    return np.interp(hi, edges, cum) - np.interp(lo, edges, cum)


def _window(row, col, reach_cells, shape):
    r0, r1 = max(0, row - reach_cells), min(shape[0], row + reach_cells + 1)
    c0, c1 = max(0, col - reach_cells), min(shape[1], col + reach_cells + 1)
    return r0, r1, c0, c1


def _landing_geometry(tree_r, tree_c, cell, r0, r1, c0, c1):
    """Distance (m) and compass direction (deg) from a trunk to every cell in a window."""
    rr, cc = np.mgrid[r0:r1, c0:c1]
    east = (cc - tree_c) * cell
    north = (tree_r - rr) * cell  # row numbers grow southward
    dist = np.hypot(east, north)
    direction = np.degrees(np.arctan2(east, north)) % 360
    return dist, direction


def _half_angle(dist, half_width):
    """Directions within this angle of a cell put the landing strip on top of it.

    A tree falling in direction theta covers a strip from the trunk out to its
    reach, as wide as its crown. A cell at distance d is inside that strip if
    its sideways distance d x sin(angle difference) is at most half the crown
    width, i.e. if the angle difference is at most arcsin(half width / d).
    Cells closer than half the width are covered for any direction up to 90 deg.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.arcsin(np.clip(half_width / np.maximum(dist, 1e-9), 0, 1)))


def exposure_maps(trees, probs, shape, cell, safety=1.0, grid_factor=1):
    """Fall exposure maps on a grid `grid_factor` times coarser than the analysis grid.

    exposure: for each cell, the sum over trees of
              (hazard score / 100) x (chance the tree's landing strip covers the cell)
    reach:    how many trees could land on the cell (in any direction)
    A tree can only reach cells within height x safety factor of its trunk.
    """
    k = max(1, int(grid_factor))
    big = cell * k
    shape_c = (int(np.ceil(shape[0] / k)), int(np.ceil(shape[1] / k)))
    exposure = np.zeros(shape_c)
    reach = np.zeros(shape_c)
    for i, t in enumerate(trees.itertuples(index=False)):
        reach_m = t.height * safety
        tr, tc = (t.row + 0.5) / k - 0.5, (t.col + 0.5) / k - 0.5  # trunk position on the coarse grid
        r0, r1, c0, c1 = _window(int(round(tr)), int(round(tc)), int(np.ceil(reach_m / big)) + 1, shape_c)
        dist, direction = _landing_geometry(tr, tc, big, r0, r1, c0, c1)
        inside = dist <= reach_m
        reach[r0:r1, c0:c1] += inside
        half = _half_angle(dist, t.crown_diam / 2)
        edges, cum = _cumulative(probs[i])
        chance = _chance_between(edges, cum, direction - half, direction + half)
        chance[dist < big / 2] = 1.0  # the trunk's own cell is always under the tree
        exposure[r0:r1, c0:c1] += np.where(inside, t.hazard / 100 * chance, 0.0)
    return exposure, reach


def landing_chances(trees, probs, masks, cell, safety=1.0):
    """For each tree and each feature mask: can it reach it, and the chance it lands on it.

    For every feature cell within reach, find the range of directions that
    would put the landing strip on it. The tree "lands on the feature" for
    any direction in the union of those ranges; its chance is the total fall
    probability over those directions (worked out on a 1-degree circle).
    Returns {name: (can_reach bool array, chance array)}.
    """
    n = len(trees)
    out = {name: (np.zeros(n, dtype=bool), np.zeros(n)) for name in masks}
    shape = next(iter(masks.values())).shape
    # Which 10-degree bin each 1-degree step belongs to (bin k covers k x 10 - 5 to k x 10 + 5).
    bin_of_degree = ((np.arange(360) + BIN_DEG / 2) // BIN_DEG).astype(int) % N_BINS
    for i, t in enumerate(trees.itertuples(index=False)):
        reach_m = t.height * safety
        r0, r1, c0, c1 = _window(t.row, t.col, int(np.ceil(reach_m / cell)) + 1, shape)
        density = probs[i][bin_of_degree] / BIN_DEG  # chance per 1-degree step
        dist = direction = half = None
        for name, mask in masks.items():
            sub = mask[r0:r1, c0:c1]
            if not sub.any():
                continue
            if dist is None:
                dist, direction = _landing_geometry(t.row, t.col, cell, r0, r1, c0, c1)
                half = _half_angle(dist, t.crown_diam / 2)
            hit_cells = sub & (dist <= reach_m)
            if not hit_cells.any():
                continue
            lo = np.floor(direction[hit_cells] - half[hit_cells]).astype(int) + 360
            hi = np.ceil(direction[hit_cells] + half[hit_cells]).astype(int) + 361
            # Mark every 1-degree step covered by at least one cell's range: +1 where a
            # range starts, -1 just after it ends, then a running total (a "difference array").
            diff = np.bincount(lo, minlength=1081)[:1081] - np.bincount(hi, minlength=1081)[:1081]
            covered = np.cumsum(diff)[:1080] > 0
            hit = covered[0:360] | covered[360:720] | covered[720:1080]  # fold -360..720 back onto 0..360
            if dist[hit_cells].min() < cell:  # the feature is right at the trunk
                hit[:] = True
            out[name][0][i] = True
            out[name][1][i] = float(density[hit].sum())
    return out


# ---------------------------------------------------------------------------
# 5. Features that falling trees could hit
# ---------------------------------------------------------------------------
def roughness(z, cell, window_m=1.0):
    """Standard deviation of height within a moving window (m): how bumpy the surface is.

    std = sqrt(average of z² - (average of z)²) over the window. Roofs and
    pavement are smooth; tree crowns are rough.
    """
    size = max(3, int(round(window_m / cell)) | 1)
    valid = np.isfinite(z)
    zf = np.where(valid, z, 0.0)
    count = ndimage.uniform_filter(valid.astype(float), size)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = ndimage.uniform_filter(zf, size) / count
        mean_sq = ndimage.uniform_filter(zf * zf, size) / count
        return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def surface_roughness(surface, cell, window_m=1.0):
    """Bumpiness of a surface with the hillside's own slope taken out.

    First subtract a smoothed copy of the surface (smoothed over the same
    window), which removes the overall slope and leaves only the small bumps;
    then measure the bumps' spread with roughness(). A lawn or trail on a
    steep hill is still smooth; a tree crown is bumpy.
    """
    valid = np.isfinite(surface)
    sigma = window_m / cell
    num = ndimage.gaussian_filter(np.where(valid, surface, 0.0), sigma, mode="nearest")
    den = ndimage.gaussian_filter(valid.astype(float), sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        bumps = surface - num / den
    return roughness(bumps, cell, window_m)


def open_ground_mask(grids, cell, p):
    """Open ground: canopy lower than the open height AND a smooth surface.

    In dense forest the drone often can't see the ground, so the "ground" model
    follows the treetops and the canopy height looks like 0; the bumpy crowns
    give those spots away. Roughness is measured over the low cells only, so a
    trail next to a wall of trees doesn't look rough just because the window
    sees the trees. Only cells the photo saw count (see detect_features).
    """
    chm = grids["chm"]
    valid = np.isfinite(chm)
    if grids.get("has_photo") is not None:
        valid &= grids["has_photo"]
    surface = grids["dtm"] + np.nan_to_num(chm)
    low = valid & (np.nan_to_num(chm, nan=99) < p["open_height"])
    smooth = np.nan_to_num(surface_roughness(np.where(low, surface, np.nan), cell, 1.0), nan=99) < p["open_roughness"]
    return low & smooth


def detect_features(grids, cell, p, open_ground=None):
    """Find clearings, trails/roads, buildings and streams automatically.

    Only cells the photo actually saw are used ("has_photo" in grids, if there
    is an orthophoto): where there is no photo, WebODM fills the surface models
    with smooth guesses that would look like open ground.

    Returns (masks, info): masks = {type: boolean grid}, info = list of dicts
    describing each detected trail/road piece.
    """
    chm = grids["chm"]
    valid = np.isfinite(chm)
    if grids.get("has_photo") is not None:
        valid &= grids["has_photo"]
    masks = {}
    info = []
    cell_area = cell * cell
    green = gray = None
    if grids.get("exg") is not None:
        # "Green" if either vegetation index says so.
        green = (np.nan_to_num(grids["exg"], nan=1) >= p["green_exg"]) | (np.nan_to_num(grids["vari"], nan=1) >= p["green_vari"])
    if grids.get("saturation") is not None:
        # "Gray-looking" (roofs, pavement): low color saturation. Fall leaves are red or
        # yellow rather than green, but they are still colorful, so this works in autumn too.
        gray = np.nan_to_num(grids["saturation"], nan=1) < p["gray_saturation"]

    # Open ground: low and smooth (see open_ground_mask).
    surface = grids["dtm"] + np.nan_to_num(chm)
    if open_ground is None:
        open_ground = open_ground_mask(grids, cell, p)
    if p["find_clearings"]:
        labels, n = ndimage.label(open_ground, structure=np.ones((3, 3)))
        sizes = np.bincount(labels.ravel(), minlength=n + 1) * cell_area
        big = sizes >= p["clearing_area"]
        big[0] = False
        masks["clearing"] = big[labels]

    # Trails and roads: open ground that is long and narrow.
    if p["find_trails"]:
        trail, road, info = _find_trails(open_ground, grids, cell, p, gray, surface)
        masks["trail"], masks["road"] = trail, road

    # Buildings: tall, roof-colored, smooth on top, not too steep, bigger than the minimum area.
    # "Smooth" ignores the roof's own slope, so sloped roof planes count as smooth, and it
    # is measured over the tall cells only (the drop to the ground at the edge doesn't count).
    # Roofs are rarely steeper than 45 degrees, while the tip of a tree is steep.
    if p["find_buildings"]:
        tall = valid & (np.nan_to_num(chm) > p["building_height"])
        if gray is not None:
            # Roof colors: gray AND bright. (Shaded crowns can look grayish too, but they are dark.)
            roof_color = gray & (np.nan_to_num(grids["brightness"], nan=0) >= p["roof_brightness"])
        elif green is not None:
            roof_color = ~green
        else:
            roof_color = np.ones(chm.shape, dtype=bool)
        top = np.where(tall, surface, np.nan)
        smooth_top = np.nan_to_num(surface_roughness(top, cell, 1.0), nan=99) < p["building_roughness"]
        pitch = slope_mod.slope_degrees(top, cell, cell)
        gentle = np.nan_to_num(pitch, nan=90) <= p["roof_pitch"]
        cand = tall & roof_color & smooth_top & gentle
        # The 1 m roughness window makes roof edges look rough; grow the smooth part
        # back out by half a window, but only over tall, roof-colored cells.
        grow = max(1, int(round(0.5 / cell)))
        cand = ndimage.binary_dilation(cand, iterations=grow) & tall & roof_color
        labels, n = ndimage.label(cand, structure=np.ones((3, 3)))
        sizes = np.bincount(labels.ravel(), minlength=n + 1) * cell_area
        big = sizes >= p["building_area"]
        big[0] = False
        masks["building"] = big[labels]

    # Streams / drainage channels: lots of ground drains through these cells.
    if p["find_streams"]:
        masks["stream"] = np.nan_to_num(grids["acc_area"], nan=0.0) >= p["stream_area"]
    return masks, info


def _find_trails(open_ground, grids, cell, p, gray, surface):
    """Long, narrow strips of open ground -> trails (or roads if they look paved).

    1. Skeletonize: thin every open area down to a 1-cell-wide center line.
    2. Width along the center line = 2 x distance to the nearest non-open cell.
    3. Keep center-line cells whose width is between the min and max trail
       width, group them into pieces, and keep pieces longer than 15 m,
       at least 5 times longer than they are wide (long and thin, not a blob),
       and fairly straight (they don't curl back on themselves).
    4. The trail is the open ground within half its width of a kept center line.
    5. Paved-looking (gray and smooth) pieces are "road/driveway".
    """
    from skimage.morphology import skeletonize

    skel = skeletonize(open_ground)
    width = 2 * ndimage.distance_transform_edt(open_ground, sampling=cell)
    ok = skel & (width >= p["trail_min_width"]) & (width <= p["trail_max_width"])
    labels, n = ndimage.label(ok, structure=np.ones((3, 3)))
    trail = np.zeros(open_ground.shape, dtype=bool)
    road = np.zeros(open_ground.shape, dtype=bool)
    info = []
    if n == 0:
        return trail, road, info
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    # Length of a center line ~ number of cells x cell size (diagonal steps are a bit longer).
    length = counts * cell * 1.1
    mean_width = np.bincount(labels.ravel(), width.ravel(), n + 1) / np.maximum(counts, 1)
    # Straightness: straight-line extent of the center line / its length along the line.
    # About 0.9 for a straight line; small for a line that curls back on itself.
    extent = shape_axes(labels, n, cell)[0]
    straight = extent / np.maximum(length, cell)
    keep = ((length >= p["trail_length"]) & (length >= p["trail_elongation"] * mean_width)
            & (straight >= p["trail_straightness"]))
    keep[0] = False
    centre = keep[labels]
    if not centre.any():
        return trail, road, info
    # Each open cell belongs to its nearest kept center-line cell (and that cell's piece).
    dist, (ir, ic) = ndimage.distance_transform_edt(~centre, sampling=cell, return_indices=True)
    near = open_ground & (dist <= width[ir, ic] / 2 + cell)
    piece = np.where(near, labels[ir, ic], 0)
    # Per-piece averages for all pieces at once (label sums, no loop over the grid).
    lab = piece.ravel()
    cells = np.bincount(lab, minlength=n + 1).astype(float)
    rows_sum = np.bincount(lab, np.indices(piece.shape)[0].ravel(), n + 1)
    cols_sum = np.bincount(lab, np.indices(piece.shape)[1].ravel(), n + 1)
    paved = np.zeros(n + 1, dtype=bool)
    if gray is not None:
        rough = np.nan_to_num(surface_roughness(surface, cell, 1.0), nan=1.0)  # bumps, not the hill's slope
        gray_sum = np.bincount(lab, gray.ravel().astype(float), n + 1)
        rough_sum = np.bincount(lab, rough.ravel(), n + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            # Paved-looking: mostly gray AND smooth (average 1 m roughness below the limit).
            paved = (gray_sum / cells > 0.7) & (rough_sum / cells < p["paved_roughness"])
    is_road = paved[piece] & (piece > 0)
    road[is_road] = True
    trail[(piece > 0) & ~is_road] = True
    for k in np.flatnonzero(keep):
        if cells[k] == 0:
            continue
        info.append({"type": "road" if paved[k] else "trail", "length_m": float(length[k]),
                     "width_m": float(mean_width[k]),
                     "row": rows_sum[k] / cells[k], "col": cols_sum[k] / cells[k]})
    return trail, road, info


# ---------------------------------------------------------------------------
# 6. Downed logs
# ---------------------------------------------------------------------------
def shape_axes(labels, n, cell):
    """Size and direction of every labeled object at once, from its "second moments".

    For each object: how spread out its cells are along its longest direction
    and across it (the same numbers skimage's regionprops uses for major/minor
    axis length). For a rectangle, length = sqrt(12 x spread²) along the long
    axis, and the same across it for the width.
    Returns (length m, width m, long-axis compass angle 0-180, mean east m, mean north m),
    each an array indexed by label (index 0 = background).
    """
    lab = labels.ravel()
    rr, cc = np.indices(labels.shape)
    east = (cc * cell).ravel().astype(float)
    north = (-rr * cell).ravel().astype(float)
    count = np.bincount(lab, minlength=n + 1).astype(float)
    count[0] = 1
    # Averages of x, y, x², y², x·y for every object at once.
    me, mn = np.bincount(lab, east, n + 1) / count, np.bincount(lab, north, n + 1) / count
    vee = np.bincount(lab, east * east, n + 1) / count - me**2
    vnn = np.bincount(lab, north * north, n + 1) / count - mn**2
    ven = np.bincount(lab, east * north, n + 1) / count - me * mn
    # Spread along the long and short axes (eigenvalues of the 2x2 spread matrix).
    # Each cell is a small square, which adds cell²/12 of spread in every direction.
    mid = (vee + vnn) / 2
    half_diff = np.sqrt(((vee - vnn) / 2) ** 2 + ven**2)
    long_var = mid + half_diff + cell**2 / 12
    short_var = np.maximum(mid - half_diff, 0) + cell**2 / 12
    # Direction of the long axis: angle from east (counterclockwise), turned into a compass angle 0-180.
    axis = np.degrees(0.5 * np.arctan2(2 * ven, vee - vnn))
    return np.sqrt(12 * long_var), np.sqrt(12 * short_var), (90 - axis) % 180, me, mn


def detect_logs(chm, transform, cell, low=0.2, high=1.5, min_length=4.0, max_width=1.0):
    """Long, thin, low objects (0.2-1.5 m above ground) = trees that already fell.

    Each low object's length, width and direction come from shape_axes().
    Which end was the root is unknown, so the direction is reported as an
    axis angle from 0 to 180 degrees.
    """
    band = np.isfinite(chm) & (np.nan_to_num(chm) >= low) & (np.nan_to_num(chm) <= high)
    labels, n = ndimage.label(band, structure=np.ones((3, 3)))
    if n == 0:
        return pd.DataFrame(columns=["log", "x", "y", "length_m", "width_m", "axis_deg"]), labels
    length, width, axis_compass, me, mn = shape_axes(labels, n, cell)
    keep = (length >= min_length) & (width <= max_width) & (length / width >= 3)
    keep[0] = False
    ids = np.flatnonzero(keep)
    t = transform
    logs = pd.DataFrame({
        "log": np.arange(1, len(ids) + 1),
        "x": t.c + (me[ids] / cell + 0.5) * t.a,
        "y": t.f + (-mn[ids] / cell + 0.5) * t.e,
        "length_m": length[ids], "width_m": width[ids], "axis_deg": axis_compass[ids],
        "row": -mn[ids] / cell, "col": me[ids] / cell,
    })
    mapping = np.zeros(n + 1, dtype=np.int32)
    mapping[ids] = logs["log"].to_numpy()
    return logs, mapping[labels]


def compare_logs_with_random(logs, grids, exposure_full, trees, cell, n_random=500, seed=42):
    """Mean slope / wetness / fall exposure / nearby tree hazard at downed logs vs random spots."""
    valid = np.isfinite(grids["chm"])
    rng = np.random.default_rng(seed)
    rows, cols = np.nonzero(valid)
    pick = rng.choice(len(rows), size=min(n_random, len(rows)), replace=False)
    spots = {"at downed logs": (logs["row"].round().astype(int).to_numpy(), logs["col"].round().astype(int).to_numpy()),
             "at random spots": (rows[pick], cols[pick])}
    tree_pts = trees[["row", "col"]].to_numpy() * cell if len(trees) else np.zeros((0, 2))
    kd = cKDTree(tree_pts) if len(trees) else None
    out = {}
    for name, (r, c) in spots.items():
        r = np.clip(r, 0, valid.shape[0] - 1)
        c = np.clip(c, 0, valid.shape[1] - 1)
        nearby = []
        if kd is not None:
            for nb in kd.query_ball_point(np.column_stack([r, c]) * cell, r=15.0):
                nearby.append(trees["hazard"].to_numpy()[nb].mean() if nb else np.nan)
        out[name] = {
            "number of spots": len(r),
            "mean slope (°)": float(np.nanmean(grids["slope"][r, c])) if len(r) else np.nan,
            "mean wetness (TWI)": float(np.nanmean(grids["twi"][r, c])) if len(r) else np.nan,
            "mean fall exposure": float(np.nanmean(exposure_full[r, c])) if len(r) else np.nan,
            "mean hazard of trees within 15 m": float(np.nanmean(nearby)) if nearby else np.nan,
        }
    return pd.DataFrame(out).reset_index().rename(columns={"index": "measure"})


# ---------------------------------------------------------------------------
# 7. Field check
# ---------------------------------------------------------------------------
FIELD_BLANKS = ["observed_height", "observed_lean_deg", "dead_or_damaged", "root_damage", "notes"]


def field_check_export(trees, crs, n_top=20, seed=42):
    """Top N priority trees plus N random low-hazard trees, ready to check in the woods."""
    top = trees.sort_values("priority", ascending=False).head(n_top).assign(group="top priority")
    low_pool = trees[(trees["class"] == "low") & ~trees["tree"].isin(top["tree"])]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(low_pool), size=min(n_top, len(low_pool)), replace=False) if len(low_pool) else []
    low = low_pool.iloc[np.sort(pick)].assign(group="random low hazard")
    sheet = pd.concat([top, low], ignore_index=True)
    lon, lat = io.map_to_latlon(sheet["x"].to_numpy(), sheet["y"].to_numpy(), crs)
    out = pd.DataFrame({
        "tree": sheet["tree"], "group": sheet["group"],
        "latitude": np.round(lat, 7), "longitude": np.round(lon, 7),
        "x": sheet["x"].round(2), "y": sheet["y"].round(2),
        "height_m": sheet["height"].round(1), "hazard": sheet["hazard"].round(0), "class": sheet["class"],
        "predicted_fall_dir_deg": sheet["fall_dir"].round(0), "predicted_fall_dir": sheet["fall_dir_word"],
        "possibly_dead": sheet["possibly dead"], "reasons": sheet["reasons"],
    })
    for col in FIELD_BLANKS:
        out[col] = ""
    return out


def _yes(value):
    return str(value).strip().lower() in ("y", "yes", "true", "1")


def field_check_compare(trees, filled: pd.DataFrame, match_distance=2.0):
    """Compare field observations with the app. Trees are matched by location
    (nearest detected tree within `match_distance` m), so renumbering trees
    after changing a slider doesn't break the match."""
    filled = filled.copy()
    filled.columns = [str(c).strip().lower() for c in filled.columns]
    if not {"x", "y"} <= set(filled.columns):
        raise ValueError("The CSV needs the x and y columns from the exported sheet.")
    kd = cKDTree(trees[["x", "y"]].to_numpy())
    dist, idx = kd.query(filled[["x", "y"]].to_numpy(float))
    matched = dist <= match_distance
    rows = []
    for (_, f), ok, i in zip(filled.iterrows(), matched, idx):
        if not ok:
            continue
        t = trees.iloc[i]
        obs_h = pd.to_numeric(f.get("observed_height"), errors="coerce")
        lean = pd.to_numeric(f.get("observed_lean_deg"), errors="coerce")
        dead = f.get("dead_or_damaged", "")
        roots = f.get("root_damage", "")
        filled_in = any(str(v).strip() not in ("", "nan") for v in (dead, roots)) or np.isfinite(obs_h) or np.isfinite(lean)
        problem = _yes(dead) or _yes(roots) or (np.isfinite(lean) and lean >= 15)
        rows.append({"tree": t["tree"], "group": f.get("group", ""), "class": t["class"], "hazard": t["hazard"],
                     "app_height": t["height"], "observed_height": obs_h,
                     "app_possibly_dead": bool(t["possibly dead"]),
                     "observed_dead": _yes(dead) if str(dead).strip() not in ("", "nan") else np.nan,
                     "observed_problem": problem if filled_in else np.nan})
    table = pd.DataFrame(rows)
    summary = {"Trees matched": f"{matched.sum()} of {len(filled)}"}
    if len(table):
        err = (table["app_height"] - table["observed_height"]).dropna()
        if len(err):
            summary["Height error (app - field)"] = f"mean {err.mean():+.1f} m, RMSE {np.sqrt(np.mean(err**2)):.1f} m (n={len(err)})"
        dead = table.dropna(subset=["observed_dead"])
        if len(dead):
            agree = (dead["app_possibly_dead"] == dead["observed_dead"].astype(bool)).mean()
            summary["'Possibly dead' matched the field"] = f"{100 * agree:.0f}% of {len(dead)} trees"
        prob = table.dropna(subset=["observed_problem"])
        if len(prob):
            high = prob[prob["class"].isin(["high", "moderate"])]["observed_problem"].astype(float)
            low = prob[prob["class"] == "low"]["observed_problem"].astype(float)
            summary["Problems seen: moderate/high-hazard trees"] = f"{100 * high.mean():.0f}% of {len(high)}" if len(high) else "-"
            summary["Problems seen: low-hazard trees"] = f"{100 * low.mean():.0f}% of {len(low)}" if len(low) else "-"
    return summary, table


# ---------------------------------------------------------------------------
# Whole analysis
# ---------------------------------------------------------------------------
def analyze(grids, transform, crs, cell, p, targets_mask=None):
    """Run every step and return a dict of tables and grids for the app."""
    chm = grids["chm"]
    # Features first, so building roofs are never mistaken for tree tops.
    open_ground = open_ground_mask(grids, cell, p)  # worked out once, used twice
    masks, trail_info = detect_features(grids, cell, p, open_ground)
    if targets_mask is not None and targets_mask.any():
        masks["marked target"] = targets_mask
    tops, crowns = detect_trees(chm, cell, p["min_height"], p["smooth"], p["crown_edge"], p["variable"],
                                p["win_a"], p["win_b"], p["window"], exclude=masks.get("building"))
    trees = measure_trees(tops, crowns, chm, transform, cell)
    if len(trees) == 0:
        return {"trees": trees, "crowns": crowns, "masks": masks}
    trees = hazard_factors(trees, crowns, grids, cell, p, open_ground)
    trees["hazard"] = hazard_score(trees, p["weights"])
    trees["class"] = assign_classes(trees["hazard"].to_numpy(), p)
    trees["reasons"] = reasons(trees, p["weights"])

    probs = fall_probabilities(trees, p["dir_weights"], p["kappa"], p["wind_from"], p["wind_on"])
    trees["fall_dir"] = BIN_CENTERS[np.argmax(probs, axis=1)]
    trees["fall_dir_word"] = [compass_word(d) for d in trees["fall_dir"]]

    k = max(1, int(round(p["exposure_cell"] / cell)))
    exposure, reach = exposure_maps(trees, probs, chm.shape, cell, p["safety"], k)

    chances = landing_chances(trees, probs, masks, cell, p["safety"]) if masks else {}
    for name, (can, chance) in chances.items():
        trees[f"can reach {name}"] = can
        trees[f"chance lands on {name}"] = chance

    # Domino effect: other tree tops within this tree's reach.
    pts = trees[["x", "y"]].to_numpy()
    nbrs = cKDTree(pts).query_ball_point(pts, r=trees["height"].to_numpy() * p["safety"])
    trees["trees it could hit"] = [len(nb) - 1 for nb in nbrs]

    # Consequence = the most serious thing the tree can reach.
    cons = np.zeros(len(trees))
    for name in masks:
        cons = np.maximum(cons, np.where(trees[f"can reach {name}"], p["consequence"].get(name, 0.0), 0.0))
    other = np.minimum(p["consequence"]["per other tree"] * trees["trees it could hit"], p["consequence"]["other trees cap"])
    trees["consequence"] = np.maximum(cons, other)
    # Priority = hazard x (1 + consequence): a likely-to-fall tree that can hit a house comes first.
    trees["priority"] = trees["hazard"] * (1 + trees["consequence"])
    order = np.lexsort((-trees["hazard"].to_numpy(), -trees["priority"].to_numpy()))  # highest priority first
    trees = trees.iloc[order].reset_index(drop=True)
    probs = probs[order]  # keep the direction chances lined up with the sorted table

    logs, log_labels = detect_logs(chm, transform, cell, p["log_low"], p["log_high"], p["log_length"], p["log_width"])
    return {"trees": trees, "crowns": crowns, "probs": probs, "exposure": exposure, "reach": reach, "grid_factor": k,
            "masks": masks, "trail_info": trail_info, "logs": logs, "log_labels": log_labels}
