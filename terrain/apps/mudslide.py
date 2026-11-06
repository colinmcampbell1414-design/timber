"""App 1: Mudslide / slope stability susceptibility."""

import numpy as np
from scipy import ndimage

from terrain import slope
from terrain.apps.base import LOW_MOD_HIGH, Overlay, Result, classify, nan_smooth, patches, ramp, rank_normalize

NAME = "Mudslide / slope stability"
DESCRIPTION = (
    "Scores how likely each spot is to slump or slide, by combining how steep it is, how wet "
    "it tends to stay, whether the ground curves inward so water collects there, and (if there is an "
    "orthophoto) whether the soil is bare. The score is relative: it compares places within this map."
)
REQUIRED_INPUTS = ["dtm"]
SHAPES = {}
TEST_AT_HOME = (
    "After heavy rain, walk the site and photograph places where soil is exposed, slumping, cracked, "
    "or washing out. Compare them with the high-score areas on the map. Do the high areas match? "
    "Are there problem spots the map missed?"
)


def params(st) -> dict:
    st.markdown("**Weights** (how much each factor counts; they are rescaled to add up to 1)")
    c1, c2, c3, c4 = st.columns(4)
    w_slope = c1.slider("Slope", 0.0, 1.0, 0.4, 0.05)
    w_wet = c2.slider("Wetness (TWI)", 0.0, 1.0, 0.3, 0.05)
    w_curv = c3.slider("Concave curvature", 0.0, 1.0, 0.2, 0.05)
    w_bare = c4.slider("Bare soil (needs photo)", 0.0, 1.0, 0.1, 0.05)
    st.markdown("**When each factor starts counting** (0 below the start, rising to 1; ordinary ground adds nothing)")
    c1, c2, c3 = st.columns(3)
    slope_start = c1.slider("Slope starts counting at (degrees)", 0.0, 40.0, 15.0, 1.0,
                            help="Gentler ground than this gets no slope score.")
    wet_start = c2.slider("Wetness counts from percentile", 0.0, 99.0, 75.0, 1.0,
                          help="75 = only the wettest quarter of the map counts.")
    curv_start = c3.slider("Concave curvature counts from percentile", 0.0, 99.0, 75.0, 1.0,
                           help="75 = only the most water-collecting quarter of the map counts.")
    c1, c2, c3 = st.columns(3)
    max_slope = c1.slider("Slope that counts as fully steep (degrees)", 10.0, 60.0, 35.0, 1.0,
                          help="Slopes this steep or steeper get the full slope score.")
    smooth = c2.slider("Smoothing for curvature (m)", 0.05, 2.0, 0.3, 0.05,
                       help="Curvature is very noisy on fine grids; smoothing shows the real shape of the land.")
    bare_exg = c3.slider("Bare soil if ExG below", -0.2, 0.3, 0.05, 0.01,
                         help="Excess Green index; below this a cell counts as bare soil.")
    c1, c2, c3 = st.columns(3)
    moderate = c1.slider("Moderate class from score", 0.0, 1.0, 0.4, 0.05)
    high = c2.slider("High class from score", 0.0, 1.0, 0.6, 0.05)
    min_area = c3.number_input("Smallest high-score zone to list (m²)", 0.0, 100.0, 0.25, 0.05)
    return dict(w_slope=w_slope, w_wet=w_wet, w_curv=w_curv, w_bare=w_bare, max_slope=max_slope,
                slope_start=slope_start, wet_start=wet_start, curv_start=curv_start,
                smooth=smooth, bare_exg=bare_exg, moderate=moderate, high=high, min_area=min_area)


def run(layers, p) -> Result:
    warnings = []
    px, py = layers.cell
    valid = np.isfinite(layers.dtm.data)

    # Each factor is 0 until its start point and rises in a straight line to 1
    # (ramp), so ordinary ground adds nothing and only concerning spots score.
    # 1. Slope factor: 0 below the start slope, 1 at "fully steep" or steeper.
    slope_factor = ramp(layers.slope_deg, p["slope_start"], p["max_slope"])
    # 2. Wetness factor: TWI ranked within the map (1 = wettest spot); only spots
    #    wetter than the start percentile count.
    wet_factor = ramp(rank_normalize(layers.twi), p["wet_start"] / 100, 1.0)
    # 3. Curvature factor: plan curvature of the smoothed ground, ranked so that the
    #    MOST CONCAVE (most negative, water-collecting) spots get 1; only spots past
    #    the start percentile count.
    smoothed = nan_smooth(layers.dtm.data, p["smooth"], px)
    curv = slope.plan_curvature(smoothed, px, py)
    curv_factor = ramp(rank_normalize(-curv), p["curv_start"] / 100, 1.0)

    weights = {"slope": p["w_slope"], "wet": p["w_wet"], "curv": p["w_curv"], "bare": p["w_bare"]}
    factors = {"slope": slope_factor, "wet": wet_factor, "curv": curv_factor}
    # 4. Bare soil factor: fraction of bare cells (low ExG) within about 1 m.
    if layers.ortho is not None:
        bare = (layers.exg < p["bare_exg"]).astype(float)
        bare[np.isnan(layers.exg)] = np.nan
        window = max(1, int(round(1.0 / px)))
        factors["bare"] = ndimage.uniform_filter(np.nan_to_num(bare), size=window, mode="nearest")
        factors["bare"][~valid] = np.nan
    else:
        weights["bare"] = 0.0
        if p["w_bare"] > 0:
            warnings.append("No orthophoto, so bare soil is left out and the other weights are rescaled.")

    total = sum(weights.values())
    if total == 0:
        return Result(warnings=["All weights are 0. Move at least one weight slider above 0."])
    # Weighted average: each factor (0-1) times its share of the total weight.
    score = sum(weights[k] / total * np.nan_to_num(factors[k], nan=0.0) for k in factors)
    score[~valid] = np.nan
    classes = classify(score, p["moderate"], p["high"])

    table, _ = patches(classes == 2, layers, p["min_area"],
                       {"max_score": (score, "max"), "mean_slope_deg": (layers.slope_deg, "mean")})
    table = table.rename(columns={"patch": "zone", "area_m2": "area (m²)", "max_score": "max score",
                                  "mean_slope_deg": "mean slope (°)"})
    cells = np.isfinite(score).sum()
    metrics = {
        "High": f"{100 * (classes == 2).sum() / cells:.1f}% of area",
        "Moderate": f"{100 * (classes == 1).sum() / cells:.1f}% of area",
        "High-score zones": str(len(table)),
        "Weights used": ", ".join(f"{k} {weights[k] / total:.2f}" for k in weights if weights[k] > 0),
    }
    warnings.append("Classes are relative within the mapped area: 'high' means high compared with the rest "
                    "of this map, not a guaranteed landslide risk.")
    return Result(
        overlays=[Overlay("Susceptibility class", classes.astype(float), classes=LOW_MOD_HIGH),
                  Overlay("Score (0-1)", score, cmap="RdYlGn_r", vmin=0, vmax=1)],
        metrics=metrics, table=table, warnings=warnings,
    )
