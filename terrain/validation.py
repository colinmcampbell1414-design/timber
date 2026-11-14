"""Statistics for checking TIMBER against reality: baselines, AUC, design criteria.

Used by the storm check (terrain/storm.py). Everything here works on plain
arrays and tables, so it can be tested with made-up numbers whose answers are
known.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats

# Simple rankings a judge would compare the full hazard score against.
BASELINES = ["baseline_height", "baseline_height_target", "baseline_random"]
TARGET_FEATURES = ["trail", "road", "building"]


# ---------------------------------------------------------------------------
# Baseline rankings
# ---------------------------------------------------------------------------
def _rank_from_score(score) -> np.ndarray:
    """Rank 1 = highest score, 2 = next... (ties broken by order in the table)."""
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    return rank


def add_baselines(trees: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add simple rankings next to the hazard score (rank 1 = "most hazardous").

    baseline_height:        tallest tree first. Nothing else.
    baseline_height_target: tallest first, but trees that can reach a trail,
                            road or building come before all the others.
    baseline_random:        a random order (same seed = same order), which is
                            what pure guessing would give.
    hazard_rank:            the full hazard score's own ranking, for comparison.
    """
    out = trees.copy()
    height = out["height"].to_numpy(dtype=float)
    out["baseline_height"] = _rank_from_score(height)
    reach = np.zeros(len(out), dtype=bool)
    for name in TARGET_FEATURES:
        if f"can reach {name}" in out:
            reach |= out[f"can reach {name}"].to_numpy(dtype=bool)
    # Adding 1000 m puts every tree that can reach a target ahead of every tree that can't.
    out["baseline_height_target"] = _rank_from_score(height + 1000.0 * reach)
    out["baseline_random"] = _rank_from_score(np.random.default_rng(seed).random(len(out)))
    out["hazard_rank"] = _rank_from_score(out["hazard"].to_numpy(dtype=float))
    return out


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------
def auc(score, outcome) -> float:
    """AUC: the chance that a damaged tree has a higher score than an undamaged one.

    1.0 = every damaged tree outranks every undamaged tree, 0.5 = no better
    than guessing, below 0.5 = worse than guessing. Ties count as half.

    Worked out with the Mann-Whitney relation: rank all scores from low to high
    (ties share the average rank), add up the ranks of the damaged trees (R),
    then   AUC = (R - n_damaged x (n_damaged + 1) / 2) / (n_damaged x n_undamaged).
    Returns NaN if there are no damaged or no undamaged trees.
    """
    score = np.asarray(score, dtype=float)
    outcome = np.asarray(outcome, dtype=bool)
    ok = np.isfinite(score)
    score, outcome = score[ok], outcome[ok]
    n_pos, n_neg = int(outcome.sum()), int((~outcome).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(score)
    return float((ranks[outcome].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auc(score, outcome, n_boot: int = 2000, seed: int = 42, level: float = 0.95):
    """AUC with a confidence interval from the bootstrap.

    Bootstrap: draw the same number of trees at random WITH replacement (some
    trees twice, some not at all), work out the AUC, and repeat 2000 times.
    The middle 95% of those AUCs is the range the true value plausibly lies in.
    A wide range means "too few damaged trees to be sure".
    Returns (auc, low, high).
    """
    score = np.asarray(score, dtype=float)
    outcome = np.asarray(outcome, dtype=bool)
    value = auc(score, outcome)
    if not np.isfinite(value):
        return value, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(score)
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        a = auc(score[pick], outcome[pick])
        if np.isfinite(a):
            values.append(a)
    if not values:
        return value, float("nan"), float("nan")
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return value, float(low), float(high)


def auc_table(trees: pd.DataFrame, outcome, score_columns: dict, n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """AUC (with bootstrap interval) for several rankings of the same trees.

    score_columns: {"label": (column name, higher_is_riskier)}. Rank columns
    (1 = most hazardous) are passed with higher_is_riskier=False.
    """
    rows = []
    for label, (column, higher) in score_columns.items():
        if column not in trees:
            continue
        values = trees[column].to_numpy(dtype=float)
        a, low, high = bootstrap_auc(values if higher else -values, outcome, n_boot, seed)
        rows.append({"ranking": label, "AUC": a, "low 95%": low, "high 95%": high})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rates, ratios and "top N" hit rates
# ---------------------------------------------------------------------------
def fisher_2x2(a_yes: int, a_no: int, b_yes: int, b_no: int) -> float:
    """Fisher's exact test: the chance of seeing a difference at least this big
    between two groups if the groups were really the same (two-sided p-value).
    It is exact, so it is safe with small counts."""
    return float(stats.fisher_exact([[a_yes, a_no], [b_yes, b_no]])[1])


def rate_by_class(classes, outcome, order=("low", "moderate", "high")) -> pd.DataFrame:
    """How many trees of each hazard class were damaged."""
    classes = np.asarray(classes)
    outcome = np.asarray(outcome, dtype=bool)
    rows = []
    for name in order:
        here = classes == name
        n, hit = int(here.sum()), int(outcome[here].sum())
        rows.append({"class": name, "trees": n, "damaged": hit, "damage rate (%)": 100 * hit / n if n else np.nan})
    return pd.DataFrame(rows)


def high_low_ratio(table: pd.DataFrame) -> dict:
    """Damage rate of high-hazard trees / damage rate of low-hazard trees, with Fisher's test.

    If there are no "high" trees, moderate and high together are compared with low.
    """
    t = table.set_index("class")
    top = ["high"] if t.loc["high", "trees"] > 0 else ["moderate", "high"]
    hi_n, hi_hit = int(t.loc[top, "trees"].sum()), int(t.loc[top, "damaged"].sum())
    lo_n, lo_hit = int(t.loc["low", "trees"]), int(t.loc["low", "damaged"])
    hi_rate = hi_hit / hi_n if hi_n else np.nan
    lo_rate = lo_hit / lo_n if lo_n else np.nan
    if lo_rate and np.isfinite(lo_rate) and np.isfinite(hi_rate):
        ratio = hi_rate / lo_rate
    elif hi_hit > 0 and lo_hit == 0:
        ratio = float("inf")  # damage only in the upper group
    else:
        ratio = float("nan")
    p = fisher_2x2(hi_hit, hi_n - hi_hit, lo_hit, lo_n - lo_hit) if hi_n and lo_n else float("nan")
    return {"groups": " + ".join(top) + " vs low", "upper_rate": hi_rate, "low_rate": lo_rate, "ratio": ratio,
            "fisher_p": p, "counts": [hi_hit, hi_n, lo_hit, lo_n]}


def top_n_hits(rank, outcome, percents=(10, 20)) -> pd.DataFrame:
    """Of the damaged trees, how many were in the top N% of the ranking made before the storm?

    A random ranking would catch N% of them on average.
    """
    rank = np.asarray(rank, dtype=float)
    outcome = np.asarray(outcome, dtype=bool)
    n, total = len(rank), int(outcome.sum())
    rows = []
    for pct in percents:
        cutoff = max(1, int(round(n * pct / 100)))
        in_top = rank <= cutoff
        hits = int((outcome & in_top).sum())
        rows.append({"top of ranking": f"top {pct}% ({cutoff} trees)", "damaged trees in it": hits,
                     "of all damaged": total, "share caught (%)": 100 * hits / total if total else np.nan,
                     "random ranking would catch (%)": float(pct)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Logistic regression (pooled storms)
# ---------------------------------------------------------------------------
def logistic_fit(X, y, max_iter: int = 50, ridge: float = 1e-6):
    """Logistic regression by Newton-Raphson.

    Model: chance of damage p = 1 / (1 + exp(-(b0 + b1 x1 + b2 x2 ...))).
    Newton-Raphson starts with all b = 0 and repeats:
        b_new = b + (X' W X)^-1 X' (y - p),   W = p x (1 - p) for each tree
    until b stops changing. (X has a first column of ones for b0. The tiny
    "ridge" keeps the step finite when the outcome is perfectly separated.)
    Returns (coefficients, standard errors).
    """
    X = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
    y = np.asarray(y, dtype=float)
    b = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        w = p * (1 - p)
        hessian = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(hessian, X.T @ (y - p) - ridge * b)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1]))
    return b, np.sqrt(np.diag(cov))


def hazard_odds_ratio(hazard, outcome, peak_gust=None) -> dict:
    """How much the odds of damage go up per 10 hazard points.

    Odds = p / (1 - p). The model is  log(odds) = b0 + b1 x hazard (+ b2 x log(peak gust)
    when storms with different gusts are pooled). Ten more hazard points multiply
    the odds by exp(10 x b1): 1.0 = no effect, 2.0 = the odds double.
    """
    hazard = np.asarray(hazard, dtype=float)
    outcome = np.asarray(outcome, dtype=bool)
    out = {"n": int(len(hazard)), "damaged": int(outcome.sum()), "odds_ratio_per_10": float("nan"),
           "low 95%": float("nan"), "high 95%": float("nan"), "uses_gust": False}
    if outcome.sum() < 2 or (~outcome).sum() < 2:
        return out
    X = hazard[:, None]
    if peak_gust is not None and len(np.unique(np.round(peak_gust, 3))) > 1:
        X = np.column_stack([hazard, np.log(np.asarray(peak_gust, dtype=float))])
        out["uses_gust"] = True
    b, se = logistic_fit(X, outcome)
    out["odds_ratio_per_10"] = float(np.exp(10 * b[1]))
    out["low 95%"] = float(np.exp(10 * (b[1] - 1.96 * se[1])))
    out["high 95%"] = float(np.exp(10 * (b[1] + 1.96 * se[1])))
    return out


# ---------------------------------------------------------------------------
# Weight sensitivity: do the results depend on the weights that were chosen?
# ---------------------------------------------------------------------------
def scores_for_weights(F: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Hazard scores of every tree for many weight sets at once.

    F: trees x factors (each 0-1; NaN = factor unknown for that tree).
    W: weight sets x factors.
    Score = 100 x (sum of weight x factor) / (sum of the weights of the known
    factors), the same rule as treefall.hazard_score. Doing it for all weight
    sets is one matrix product, so 500 sets take a fraction of a second.
    Returns trees x weight sets.
    """
    known = np.isfinite(F)
    total = np.where(known, F, 0.0) @ W.T
    weight_sum = known.astype(float) @ W.T
    return 100 * total / np.maximum(weight_sum, 1e-12)


def _top_mask(scores: np.ndarray, top: int) -> np.ndarray:
    """True for the `top` highest-scoring trees in every column (ties broken by table order)."""
    order = np.argsort(-scores, axis=0, kind="stable")[:top]
    mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=0)
    return mask


def sensitivity_one_at_a_time(F: np.ndarray, weights: dict, top: int = 20) -> pd.DataFrame:
    """Change one weight at a time (x 0.5 and x 1.5) and see how much the ranking moves.

    After each change the weights are rescaled to add up to 1. Reported:
    - Spearman rank correlation with the original scores (1 = same order,
      0 = unrelated order);
    - how many of the original top-20 trees are still in the top 20.
    """
    names = list(weights)
    w0 = np.array([weights[n] for n in names], dtype=float)
    w0 = w0 / w0.sum()
    base = scores_for_weights(F, w0[None, :])
    base_top = _top_mask(base, top)[:, 0]
    rows = []
    for i, name in enumerate(names):
        for factor in (0.5, 1.5):
            w = w0.copy()
            w[i] *= factor
            w /= w.sum()
            s = scores_for_weights(F, w[None, :])
            rho = stats.spearmanr(base[:, 0], s[:, 0]).statistic if np.ptp(base) > 0 and np.ptp(s) > 0 else 1.0
            rows.append({"factor": name, "weight multiplied by": factor, "Spearman correlation": float(rho),
                         f"original top {top} kept": int((_top_mask(s, top)[:, 0] & base_top).sum())})
    return pd.DataFrame(rows)


def sensitivity_random(F: np.ndarray, weights: dict, n_sets: int = 500, concentration: float = 20.0, top: int = 20,
                       seed: int = 42) -> dict:
    """Try many random weight sets close to the chosen one.

    The sets are drawn from a Dirichlet distribution centered on the current
    weights: every set adds up to 1, and `concentration` says how tightly they
    cluster around the current weights (20 = each weight typically moves by a
    few hundredths; bigger = closer).
    For each tree: how often it lands in the top 20 ("top-20 stability", %).
    For each weight set: how many of the original top 20 it keeps.
    weight_stability_top20 = the mean share of the original top 20 that is kept.
    """
    names = list(weights)
    w0 = np.array([weights[n] for n in names], dtype=float)
    w0 = w0 / w0.sum()
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.maximum(concentration * w0, 1e-6), size=n_sets)
    base_top = _top_mask(scores_for_weights(F, w0[None, :]), top)[:, 0]
    tops = _top_mask(scores_for_weights(F, W), top)
    overlap = (tops & base_top[:, None]).sum(axis=0)
    n_top = int(base_top.sum())
    typical = int(np.median(overlap))
    return {"overlap": overlap, "stability_percent": 100 * tops.mean(axis=1), "base_top": base_top,
            "weight_stability_top20": float(overlap.mean() / n_top) if n_top else float("nan"),
            "summary": f"{typical} of the top {n_top} trees stay in the top {n_top} under most reasonable weight "
                       f"choices (median of {n_sets} random weight sets)."}


def factor_matrix(trees: pd.DataFrame, names) -> np.ndarray:
    """The trees x factors table of 0-1 factor values (columns f_<name>)."""
    return trees[[f"f_{n}" for n in names]].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Blind field check
# ---------------------------------------------------------------------------
DEFECT_COLUMNS = ["dead_or_dying_crown", "broken_or_dead_top", "trunk_crack_or_cavity", "root_damage_or_heaving",
                  "lean_over_15deg", "hanging_dead_branches", "undercut_bank"]
FIELD_BLANKS = ["status", "observed_height_m"] + DEFECT_COLUMNS + ["notes"]
SHEET_COLUMNS = ["code", "latitude", "longitude", "trail_distance_m", "trail_bearing_deg"] + FIELD_BLANKS


def nearest_on_mask(x, y, mask: np.ndarray, transform):
    """Distance from each point to the nearest True cell of a mask, and the compass
    bearing FROM that cell TO the point (so "35 m at 120 deg from the trail" can be walked).
    Returns (distance m, bearing deg); NaN if the mask is empty."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if mask is None or not mask.any():
        return np.full(len(x), np.nan), np.full(len(x), np.nan)
    cell = abs(transform.a)
    _, (ir, ic) = ndimage.distance_transform_edt(~mask, sampling=cell, return_indices=True)
    r = np.clip(((y - transform.f) / transform.e).astype(int), 0, mask.shape[0] - 1)
    c = np.clip(((x - transform.c) / transform.a).astype(int), 0, mask.shape[1] - 1)
    near_x = transform.c + (ic[r, c] + 0.5) * transform.a
    near_y = transform.f + (ir[r, c] + 0.5) * transform.e
    east, north = x - near_x, y - near_y
    return np.hypot(east, north), np.degrees(np.arctan2(east, north)) % 360


def blind_field_sample(trees: pd.DataFrame, n: int = 30, min_height: float = 10.0, max_trail_m: float = None,
                       seed: int = 42):
    """Choose the trees for a BLIND field check.

    high group: the N highest hazard scores.
    low group:  N random trees from the lowest third of scores.
    Both groups only use trees taller than `min_height` (so tall and short
    trees aren't being compared), and, if `max_trail_m` is given, only trees
    within that distance of a trail or road (reachable on foot).
    They are shuffled together and each gets a random 4-digit code.

    Returns (sheet, key). The SHEET goes to the woods: code, position, distance
    and bearing from the nearest trail point, and blank columns. It has NO
    score, group, height, reasons or tree number, so whoever looks at the tree
    can't be influenced. The KEY stays home and maps each code to the tree.
    """
    rng = np.random.default_rng(seed)
    pool = trees[trees["height"] > min_height]
    if max_trail_m is not None and "trail_dist_m" in pool:
        pool = pool[pool["trail_dist_m"] <= max_trail_m]
    pool = pool.sort_values("hazard", ascending=False, kind="stable")
    high = pool.head(min(n, len(pool) // 2))
    rest = pool.drop(high.index)
    lowest_third = rest[rest["hazard"] <= pool["hazard"].quantile(1 / 3)] if len(rest) else rest
    if len(lowest_third) < len(high):  # not enough: fill up from the lowest scores
        lowest_third = rest.tail(max(len(high), len(lowest_third)))
    low = lowest_third.iloc[np.sort(rng.choice(len(lowest_third), min(len(high), len(lowest_third)), replace=False))]
    both = pd.concat([high.assign(group="high"), low.assign(group="low")])
    both = both.iloc[rng.permutation(len(both))].reset_index(drop=True)
    both["code"] = rng.choice(np.arange(1000, 10000), size=len(both), replace=False)

    sheet = pd.DataFrame({"code": both["code"]})
    for col, source in [("latitude", "latitude"), ("longitude", "longitude"), ("trail_distance_m", "trail_dist_m"),
                        ("trail_bearing_deg", "trail_bearing_deg")]:
        sheet[col] = both[source].round(7 if "itude" in col else 0) if source in both else np.nan
    for col in FIELD_BLANKS:
        sheet[col] = ""
    key_cols = (["code", "tree", "group", "hazard", "hazard_rank"] + BASELINES + ["height", "fall_dir", "reasons",
                "possibly dead", "gap_dist_m", "x", "y"] + [c for c in both.columns if c.startswith("f_")])
    return sheet[SHEET_COLUMNS], both[[c for c in key_cols if c in both]].copy()


def _yes(value) -> bool:
    return str(value).strip().lower() in ("y", "yes", "true", "1")


def height_errors(app, field) -> dict:
    """App height vs field height: bias = mean(app - field), MAE = mean size of the
    error, RMSE = square root of the mean squared error (big misses count more)."""
    err = (np.asarray(app, dtype=float) - np.asarray(field, dtype=float))
    err = err[np.isfinite(err)]
    if not len(err):
        return {"n": 0, "bias_m": float("nan"), "mae_m": float("nan"), "rmse_m": float("nan")}
    return {"n": int(len(err)), "bias_m": float(err.mean()), "mae_m": float(np.abs(err).mean()),
            "rmse_m": float(np.sqrt(np.mean(err ** 2)))}


def field_import(key: pd.DataFrame, filled: pd.DataFrame, edge_m: float = 5.0, n_boot: int = 500, seed: int = 42) -> dict:
    """Score the hazard model against a filled-in blind field sheet.

    Only trees with status = found count. any_defect = yes if any defect column is y.
    - Defect rate in the high and low groups, their ratio, and Fisher's exact test.
    - AUC (chance a defective tree outranks a sound one) for the full score,
      each baseline, and each single factor.
    - Height check: bias, MAE, RMSE, for all trees and split into open/edge
      trees (crown within `edge_m` of an opening) and interior trees.
    - "Possibly dead" flag vs the observed dead_or_dying_crown: a 2x2 table.
    The "criteria" block feeds the Design criteria table.
    """
    filled = filled.copy()
    filled.columns = [str(c).strip().lower() for c in filled.columns]
    t = key.merge(filled, on="code", how="inner")
    t = t[t["status"].astype(str).str.strip().str.lower() == "found"].reset_index(drop=True)
    t["any_defect"] = np.any([t[c].map(_yes) for c in DEFECT_COLUMNS if c in t], axis=0) if len(t) else False
    out = {"checked": int(len(t)), "trees": t}
    high, low = t[t["group"] == "high"], t[t["group"] == "low"]
    hi_n, hi_hit, lo_n, lo_hit = len(high), int(high["any_defect"].sum()), len(low), int(low["any_defect"].sum())
    hi_rate, lo_rate = (hi_hit / hi_n if hi_n else np.nan), (lo_hit / lo_n if lo_n else np.nan)
    if lo_hit == 0 and hi_hit > 0:
        ratio, ratio_text = float("inf"), ("all high, none low" if hi_hit == hi_n else "defects only in the high group")
    elif lo_hit == 0:
        ratio, ratio_text = float("nan"), "no defects found in either group"
    else:
        ratio = hi_rate / lo_rate
        ratio_text = f"{ratio:.1f}x"
    fisher = fisher_2x2(hi_hit, hi_n - hi_hit, lo_hit, lo_n - lo_hit) if hi_n and lo_n else float("nan")
    out["defects"] = {"high": [hi_hit, hi_n], "low": [lo_hit, lo_n], "high_rate": hi_rate, "low_rate": lo_rate,
                      "ratio": ratio, "ratio_text": ratio_text, "fisher_p": fisher,
                      "table_2x2": pd.DataFrame({"defect": [hi_hit, lo_hit], "sound": [hi_n - hi_hit, lo_n - lo_hit]},
                                                index=["high hazard", "low hazard"])}

    columns = {"full hazard score": ("hazard", True)}
    columns.update({f"baseline: {b.replace('baseline_', '').replace('_', ' + ')}": (b, False) for b in BASELINES if b in t})
    columns.update({f"factor: {c[2:]}": (c, True) for c in t.columns if c.startswith("f_") and t[c].notna().any()})
    aucs = auc_table(t, t["any_defect"].to_numpy(dtype=bool), columns, n_boot, seed) if len(t) else pd.DataFrame()
    out["auc"] = aucs
    beats = None
    if len(aucs) and np.isfinite(aucs["AUC"].iloc[0]):
        base = aucs[aucs["ranking"].str.startswith("baseline")]["AUC"]
        beats = bool((aucs["AUC"].iloc[0] > base).all()) if len(base) else None

    field_h = pd.to_numeric(t.get("observed_height_m"), errors="coerce") if len(t) else pd.Series(dtype=float)
    edge = t["gap_dist_m"] <= edge_m if "gap_dist_m" in t else pd.Series(False, index=t.index)
    out["heights"] = pd.DataFrame([{"trees": name, **height_errors(t.loc[m, "height"], field_h[m])} for name, m in
                                   [("all", pd.Series(True, index=t.index)), ("open / edge", edge), ("interior", ~edge)]])
    out["height_pairs"] = pd.DataFrame({"app_height_m": t["height"], "field_height_m": field_h}).dropna() if len(t) else pd.DataFrame()

    if "possibly dead" in t and "dead_or_dying_crown" in t:
        flagged, dead = t["possibly dead"].astype(bool), t["dead_or_dying_crown"].map(_yes)
        out["snag_2x2"] = pd.DataFrame({"observed dead or dying": [int((flagged & dead).sum()), int((~flagged & dead).sum())],
                                        "observed healthy": [int((flagged & ~dead).sum()), int((~flagged & ~dead).sum())]},
                                       index=["app: possibly dead", "app: not flagged"])
    rmse = out["heights"]["rmse_m"].iloc[0]
    out["criteria"] = {"height_rmse_m": rmse if np.isfinite(rmse) else None,
                       "defect_rate_ratio": ("inf" if ratio == float("inf") else ratio if np.isfinite(ratio) else None),
                       "fisher_p": fisher if np.isfinite(fisher) else None, "beats_baseline": beats}
    full_auc = aucs["AUC"].iloc[0] if len(aucs) else float("nan")
    out["sentences"] = [
        f"Blind field check: {len(t)} trees found. Defects in {hi_hit} of {hi_n} high-hazard trees and {lo_hit} of "
        f"{lo_n} low-hazard trees (ratio: {ratio_text}; Fisher's exact test p = {fisher:.3g}).",
        f"The hazard score's AUC for defects was {full_auc:.2f} (0.5 = guessing); better than every simple baseline: {beats}.",
        f"Tree heights: RMSE {rmse:.2f} m over {out['heights']['n'].iloc[0]} trees (bias {out['heights']['bias_m'].iloc[0]:+.2f} m).",
    ]
    return out


# ---------------------------------------------------------------------------
# Detection check plots
# ---------------------------------------------------------------------------
def plot_sheet(plots: pd.DataFrame) -> pd.DataFrame:
    """The sheet for counting trees in circular plots (blank columns to fill in)."""
    sheet = plots[["plot_id", "latitude", "longitude", "radius_m"]].copy()
    sheet["field_count_over_10m"], sheet["notes"] = "", ""
    return sheet


def detection_rates(plots: pd.DataFrame, trees: pd.DataFrame, min_height: float = 10.0) -> dict:
    """Detection rate per plot = app trees over 10 m with their top inside the circle / trees counted in the field.

    plots: plot_id, x, y (map coordinates), radius_m, field_count_over_10m.
    Overall rate = sum of app counts / sum of field counts. This compares
    COUNTS, so a missed tree and an extra (false) tree cancel out.
    """
    tall = trees[trees["height"] > min_height]
    rows = []
    for _, p in plots.iterrows():
        field = pd.to_numeric(p.get("field_count_over_10m"), errors="coerce")
        app = int((np.hypot(tall["x"] - p["x"], tall["y"] - p["y"]) <= p["radius_m"]).sum())
        rows.append({"plot": p["plot_id"], "app count": app, "field count": field,
                     "detection rate": app / field if field and np.isfinite(field) else np.nan})
    table = pd.DataFrame(rows)
    done = table.dropna(subset=["field count"])
    overall = float(done["app count"].sum() / done["field count"].sum()) if len(done) and done["field count"].sum() else float("nan")
    return {"table": table, "overall": overall, "criteria": {"detection_rate": overall if np.isfinite(overall) else None},
            "sentences": [f"Detection check: the app found {int(done['app count'].sum())} trees over {min_height:g} m in "
                          f"{len(done)} plots where {int(done['field count'].sum())} were counted in the field "
                          f"(rate {overall:.2f}; count-based, so misses and extras can cancel)."] if len(done) else []}


# ---------------------------------------------------------------------------
# Downed log validation
# ---------------------------------------------------------------------------
def rayleigh_axial(axis_deg, reference_deg) -> dict:
    """Do log directions line up with a reference direction (e.g. downhill) more than chance?

    A log only has an AXIS: 30 and 210 degrees are the same log. So the angle
    between log and reference, d, is DOUBLED: then d and d + 180 land on the
    same point of the circle. Each log becomes an arrow of length 1 at angle 2d:
        mean resultant length R = length of the average arrow
    R = 1 means every log has the same offset from the reference; R near 0
    means the offsets point every which way. The Rayleigh test turns R and the
    number of logs n into the chance of seeing an R this big by luck:
        p = exp( sqrt(1 + 4n + 4(n² - (nR)²)) - (1 + 2n) )
    The mean offset says HOW they line up: near 0 = along the reference
    (up/downhill), near 90 = across it.
    """
    d = np.radians(np.asarray(axis_deg, dtype=float) - np.asarray(reference_deg, dtype=float))
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {"n": 0, "mean_resultant_length": float("nan"), "p": float("nan"), "mean_offset_deg": float("nan")}
    c, s = np.cos(2 * d).mean(), np.sin(2 * d).mean()
    r = float(np.hypot(c, s))
    big_r = n * r
    p = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n * n - big_r * big_r)) - (1 + 2 * n)))
    offset = float((np.degrees(np.arctan2(s, c)) / 2) % 180)  # halve the doubled angle again
    return {"n": int(n), "mean_resultant_length": r, "p": min(p, 1.0), "mean_offset_deg": min(offset, 180 - offset)}


def _points_along(logs: pd.DataFrame, k: int = 5):
    """k points evenly spaced along every log (x and y arrays, logs x k)."""
    a = np.radians(logs["axis_deg"].to_numpy(dtype=float))
    half = logs["length_m"].to_numpy(dtype=float) / 2
    steps = np.linspace(-1, 1, k)[None, :]
    return (logs["x"].to_numpy(dtype=float)[:, None] + steps * (half * np.sin(a))[:, None],
            logs["y"].to_numpy(dtype=float)[:, None] + steps * (half * np.cos(a))[:, None])


def _cells(x, y, transform, shape):
    r = np.clip(((np.asarray(y) - transform.f) / transform.e).astype(int), 0, shape[0] - 1)
    c = np.clip(((np.asarray(x) - transform.c) / transform.a).astype(int), 0, shape[1] - 1)
    return r, c


def logs_away_from(logs: pd.DataFrame, mask: np.ndarray, transform, distance_m: float = 5.0) -> pd.DataFrame:
    """Drop logs on or within `distance_m` of a trail or road: trail crews cut and
    move those, so where they lie says little about where trees fall."""
    if mask is None or not mask.any() or logs.empty:
        return logs
    near = ndimage.distance_transform_edt(~mask, sampling=abs(transform.a)) <= distance_m
    x, y = _points_along(logs)
    r, c = _cells(x, y, transform, mask.shape)
    return logs[~near[r, c].any(axis=1)].reset_index(drop=True)


def local_mean(grid: np.ndarray, cell: float, radius_m: float = 3.0) -> np.ndarray:
    """Average of a grid within about `radius_m` of every cell (NaN cells ignored)."""
    size = max(3, int(round(2 * radius_m / cell)) | 1)
    valid = np.isfinite(grid)
    total = ndimage.uniform_filter(np.where(valid, grid, 0.0), size, mode="nearest")
    count = ndimage.uniform_filter(valid.astype(float), size, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def log_factor_test(logs: pd.DataFrame, factors: dict, forest: np.ndarray, transform, cell: float, n_random: int = 1000,
                    seed: int = 42) -> dict:
    """Do downed logs lie on more hazardous ground than random forest spots?

    factors: {"slope (deg)": grid, "wetness (TWI)": grid, ...}. Each grid is
    averaged over a 3 m neighborhood; the value for a log is the mean along it.
    forest: True where a random comparison point may fall (forest that is not a
    log, trail or building).
    For each factor a Mann-Whitney U test asks whether logs have HIGHER values
    than the random points (p = chance of a difference this big by luck).
    log_factor_test_p = the smallest p x the number of factors tested
    (Bonferroni: testing several factors gives more chances of a fluke).
    """
    rng = np.random.default_rng(seed)
    spots = np.argwhere(forest)
    pick = spots[rng.choice(len(spots), size=min(n_random, len(spots)), replace=False)] if len(spots) else spots
    x, y = _points_along(logs) if len(logs) else (np.zeros((0, 5)), np.zeros((0, 5)))
    r, c = _cells(x, y, transform, forest.shape)
    rows, values = [], {}
    for name, grid in factors.items():
        smooth = local_mean(grid, cell)
        at_logs = np.nanmean(smooth[r, c], axis=1) if len(logs) else np.zeros(0)
        at_random = smooth[pick[:, 0], pick[:, 1]] if len(pick) else np.zeros(0)
        at_logs, at_random = at_logs[np.isfinite(at_logs)], at_random[np.isfinite(at_random)]
        p = float("nan")
        if len(at_logs) >= 3 and len(at_random) >= 3:
            p = float(stats.mannwhitneyu(at_logs, at_random, alternative="greater").pvalue)
        rows.append({"factor": name, "logs": len(at_logs), "median at logs": float(np.median(at_logs)) if len(at_logs) else np.nan,
                     "median at random points": float(np.median(at_random)) if len(at_random) else np.nan,
                     "Mann-Whitney p (logs higher)": p})
        values[name] = (at_logs, at_random)
    table = pd.DataFrame(rows)
    ps = table["Mann-Whitney p (logs higher)"].dropna()
    criterion = float(min(1.0, ps.min() * len(ps))) if len(ps) else None
    best = table.loc[ps.idxmin(), "factor"] if len(ps) else "-"
    return {"table": table, "values": values, "criteria": {"log_factor_test_p": criterion},
            "sentences": [f"Downed logs ({len(logs)}) vs {len(pick)} random forest points: the strongest difference is in "
                          f"{best} (corrected p = {criterion:.3g})."] if criterion is not None else []}


def match_logs(manual: pd.DataFrame, detected: pd.DataFrame, distance_m: float = 3.0) -> int:
    """How many field-measured logs the detector also found (a detected log's center within 3 m)."""
    if manual.empty or detected.empty:
        return 0
    d = np.hypot(manual["x"].to_numpy()[:, None] - detected["x"].to_numpy()[None, :],
                 manual["y"].to_numpy()[:, None] - detected["y"].to_numpy()[None, :])
    return int((d.min(axis=1) <= distance_m).sum())


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def hectares_per_hour(area_ha: float, person_hours: float, compute_seconds: float):
    """Survey speed = area / (person time for flying, processing and field work + computer time)."""
    hours = float(person_hours) + float(compute_seconds) / 3600
    return float(area_ha) / hours if hours > 0 and person_hours > 0 else None


# ---------------------------------------------------------------------------
# Results files, design criteria and the summary
# ---------------------------------------------------------------------------
GOAL = ("Design an automated system that uses only a consumer drone and free software (WebODM + this app) to find "
        "which trees in a forest pose the greatest fall risk to trails, roads and buildings, accurately enough to "
        "direct a field inspection and faster than a manual walk-through survey.")
DISCLAIMER = "Screening tool for a student science project. Not a substitute for a professional arborist assessment."

# name: (description, compare, target, results file)
CRITERIA = {
    "detection_rate": ("Trees taller than 10 m found in check plots (app count / field count)", ">=", 0.80, "results/detection_plots.json"),
    "height_rmse_m": ("App tree height vs clinometer height, RMSE (m)", "<=", 1.5, "results/field_check_results.json"),
    "defect_rate_ratio": ("Defect rate of high-hazard / low-hazard trees (blind field check)", ">=", 2.0, "results/field_check_results.json"),
    "fisher_p": ("High vs low hazard defect difference, Fisher's exact test p", "<=", 0.05, "results/field_check_results.json"),
    "beats_baseline": ("Field check: full score AUC is higher than every baseline AUC", "true", True, "results/field_check_results.json"),
    "log_factor_test_p": ("Downed logs lie on higher-hazard ground than random forest points (p)", "<=", 0.05, "results/log_validation.json"),
    "hectares_per_hour": ("Hectares surveyed per hour (person time + compute time)", ">=", 10, "results/timing.json"),
    "storm_auc_severe": ("Storm check: AUC of the frozen hazard score for fallen or snapped trees (pooled)", ">=", 0.70, "results/storms/pooled_results.json"),
    "storm_beats_baseline": ("Storm check: full score AUC is higher than the 'tallest first' baseline", "true", True, "results/storms/pooled_results.json"),
    "fall_direction_mae_deg": ("Storm check: median error of the predicted fall direction, actual-wind model (degrees)", "<=", 45, "results/storms/pooled_results.json"),
    "calm_false_change_rate": ("Calm pairs: share of trees falsely flagged as damaged", "<=", 0.01, "results/storms/pooled_results.json"),
    "weight_stability_top20": ("Sensitivity: mean share of the top 20 trees kept across random weight sets", ">=", 0.75, "results/sensitivity.json"),
}
CRITERIA_HEADER = """\
# TIMBER design criteria: the targets the system is judged against.
# Edit the TARGETS here. "source" is the results file that holds the measured value.
# compare: ">=" (measured must be at least the target), "<=" (at most), or "true".
"""


def _default_criteria() -> dict:
    return {name: {"description": d, "compare": c, "target": t, "source": s} for name, (d, c, t, s) in CRITERIA.items()}


def load_criteria(config_dir) -> dict:
    """Read config/design_criteria.yaml.

    The file is created with the default targets the first time. If it is
    missing some of the 12 criteria (an older file), they are added; targets
    you edited are kept.
    """
    import yaml

    path = Path(config_dir) / "design_criteria.yaml"
    saved = {}
    if path.exists():
        saved = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged = _default_criteria()
    for name, entry in saved.items():
        if name in merged and isinstance(entry, dict) and "target" in entry:
            merged[name]["target"] = entry["target"]
    if merged != saved:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CRITERIA_HEADER + "\n" + yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return merged


def _jsonable(obj):
    """Plain values that JSON can store (tables become lists of rows, NaN becomes null)."""
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.reset_index().to_dict(orient="records")) if obj.index.name or not isinstance(obj.index, pd.RangeIndex) \
            else _jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else ("inf" if obj == float("inf") else None)
    return obj


def save_result(work_dir, source: str, criteria: dict, sentences: list = (), details: dict = None) -> Path:
    """Write one results file: the measured criteria, plain sentences for the
    summary, and any details. `source` is the path listed in CRITERIA."""
    path = Path(work_dir) / source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable({"criteria": criteria, "sentences": list(sentences), **(details or {})}), indent=2),
                    encoding="utf-8")
    return path


def read_result(work_dir, source: str) -> dict | None:
    """Read a results file (None if it is missing or damaged)."""
    path = Path(work_dir) / source
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def measured_value(work_dir, name: str, source: str):
    """THE one place a criterion's measured value is read: results file -> "criteria" -> name."""
    result = read_result(work_dir, source)
    value = (result or {}).get("criteria", {}).get(name)
    if value == "inf":
        return float("inf")
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def evaluate_criteria(criteria: dict, work_dir) -> pd.DataFrame:
    """Table of criterion / target / measured / pass, fail or "no data yet".

    A missing results file or value is "no data yet" (never a crash).
    """
    rows = []
    for name, c in criteria.items():
        source = c.get("source", "")
        measured = measured_value(work_dir, name, source) if source else None
        compare, target = c.get("compare", ">="), c.get("target")
        if measured is None:
            status = "no data yet"
        elif compare == "true":
            status = "pass" if bool(measured) == bool(target) else "fail"
        elif compare == "<=":
            status = "pass" if measured <= target else "fail"
        else:
            status = "pass" if measured >= target else "fail"
        shown = "-" if measured is None else (str(bool(measured)).lower() if compare == "true" else f"{measured:.3g}")
        rows.append({"criterion": name, "what it means": c.get("description", ""),
                     "target": f"{compare} {target}" if compare != "true" else str(target).lower(),
                     "measured": shown, "status": status, "from file": source})
    return pd.DataFrame(rows)


def write_summary(work_dir, config_dir=None) -> str:
    """results/summary.md: the goal, the criteria table, and the plain-sentence
    results that each analysis saved with its numbers."""
    work_dir = Path(work_dir)
    table = evaluate_criteria(load_criteria(config_dir or work_dir / "config"), work_dir)
    lines = ["# TIMBER results summary", "", "## Engineering goal", "", GOAL, "", "## Design criteria", "",
             "| Criterion | Target | Measured | Status | From |", "|---|---|---|---|---|"]
    lines += [f"| {r['what it means']} | {r['target']} | {r['measured']} | {r['status'].upper()} | {r['from file']} |"
              for _, r in table.iterrows()]
    sections = [("Weight sensitivity", "results/sensitivity.json"), ("Blind field check", "results/field_check_results.json"),
                ("Detection check plots", "results/detection_plots.json"), ("Downed logs", "results/log_validation.json"),
                ("Leaf-on / leaf-off ground models", "results/dtm_comparison.json"), ("Survey speed", "results/timing.json"),
                ("Storm check (all storms pooled)", "results/storms/pooled_results.json")]
    for title, source in sections:
        result = read_result(work_dir, source)
        lines += ["", f"## {title}", ""]
        lines += [f"- {s}" for s in (result or {}).get("sentences", [])] or ["No data yet."]
    storms = sorted((work_dir / "results" / "storms").glob("*/storm_results.json")) if (work_dir / "results" / "storms").exists() else []
    for path in storms:
        result = json.loads(path.read_text(encoding="utf-8"))
        lines += ["", f"### {'Calm pair' if result['storm'].get('is_calm_control') else 'Storm'}: {result['storm']['name']}", ""]
        lines += [f"- {s}" for s in result.get("sentences", [])]
    background = read_result(work_dir, "results/accuracy.json")
    lines += ["", "## Background", "",
              ("- " + background["sentences"][0]) if background and background.get("sentences") else
              "- Best-case open-ground vertical error: no data yet (run the Accuracy checker on the yard survey)."]
    lines += ["", f"_{DISCLAIMER}_", ""]
    text = "\n".join(lines)
    (work_dir / "results").mkdir(parents=True, exist_ok=True)
    (work_dir / "results" / "summary.md").write_text(text, encoding="utf-8")
    return text
