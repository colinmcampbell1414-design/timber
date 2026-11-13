"""Statistics for checking TIMBER against reality: baselines, AUC, design criteria.

Used by the storm check (terrain/storm.py). Everything here works on plain
arrays and tables, so it can be tested with made-up numbers whose answers are
known.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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


def bootstrap_auc(score, outcome, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
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


def auc_table(trees: pd.DataFrame, outcome, score_columns: dict, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
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
# Design criteria
# ---------------------------------------------------------------------------
DEFAULT_CRITERIA = """\
# TIMBER design criteria: the targets the system is judged against.
# Edit the targets here. "source" says which results file holds the measured value.
# compare: ">=" (measured must be at least the target), "<=" (at most), or "true".

storm_auc_severe:
  description: Storm check - AUC of the frozen hazard score for severe damage (fallen or snapped)
  compare: ">="
  target: 0.70
  source: results/storms/pooled_results.json

storm_beats_baseline:
  description: Storm check - the full score's AUC is higher than the "tallest tree first" baseline
  compare: "true"
  target: true
  source: results/storms/pooled_results.json

fall_direction_mae_deg:
  description: Storm check - median error of the predicted fall direction (degrees)
  compare: "<="
  target: 45
  source: results/storms/pooled_results.json

calm_false_change_rate:
  description: Calm control - share of trees falsely flagged as damaged with no storm
  compare: "<="
  target: 0.01
  source: results/storms/pooled_results.json
"""


def load_criteria(config_dir) -> dict:
    """Read config/design_criteria.yaml (it is created with the default targets the first time)."""
    import yaml

    path = Path(config_dir) / "design_criteria.yaml"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CRITERIA, encoding="utf-8")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def evaluate_criteria(criteria: dict, work_dir) -> pd.DataFrame:
    """Table of criterion / target / measured / pass, fail or "no data yet".

    The measured value is looked up by the criterion's name in its source file
    (a JSON file under the work folder). A missing file or value is "no data yet".
    """
    rows = []
    for name, c in criteria.items():
        source = c.get("source", "")
        measured = None
        path = Path(work_dir) / source
        if source and path.exists():
            try:
                measured = json.loads(path.read_text(encoding="utf-8")).get("criteria", {}).get(name)
            except (OSError, ValueError):
                measured = None
        if isinstance(measured, float) and not np.isfinite(measured):
            measured = None
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
