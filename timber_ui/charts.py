"""Charts for the validation results, and "save every chart as a 300 dpi PNG".

Each chart is drawn from a saved results file, so the board figures can be
re-made at any time without re-running an analysis.
"""

import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def png(fig, dpi=300) -> bytes:
    out = BytesIO()
    fig.savefig(out, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out.getvalue()


def auc_figure(auc: pd.DataFrame, title: str):
    """AUC of every ranking with its 95% range. The dashed line (0.5) is pure guessing."""
    auc = auc.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(auc) + 1.4))
    colors = ["#2e7d32" if r == "full hazard score" else "#7f8c8d" if r.startswith("baseline") else "#90a4ae"
              for r in auc["ranking"]]
    value = pd.to_numeric(auc["AUC"], errors="coerce")
    low = (value - pd.to_numeric(auc["low 95%"], errors="coerce")).clip(lower=0).fillna(0)
    high = (pd.to_numeric(auc["high 95%"], errors="coerce") - value).clip(lower=0).fillna(0)
    ax.barh(auc["ranking"], value.fillna(0), xerr=[low, high], color=colors, ecolor="black", capsize=3)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC (0.5 = guessing, 1.0 = perfect); lines = 95% range")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def rose_figure(directions_deg, wind_from_deg=None, title="Fall directions of storm-felled trees", axial=False):
    """Rose diagram: how many point toward each direction (north up, clockwise).
    axial=True draws each log axis in both directions (a log has no "front")."""
    d = np.asarray(directions_deg, dtype=float) % 360
    if axial:
        d = np.concatenate([d, (d + 180) % 360])
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    counts, edges = np.histogram(d, bins=np.arange(0, 361, 20))
    ax.bar(np.radians(edges[:-1] + 10), counts, width=np.radians(20), color="#c0392b", edgecolor="white", alpha=0.85)
    if wind_from_deg is not None and np.isfinite(wind_from_deg):
        top = max(counts.max(), 1)
        ax.annotate("", xy=(np.radians((wind_from_deg + 180) % 360), top), xytext=(np.radians(wind_from_deg), top),
                    arrowprops=dict(arrowstyle="-|>", color="#1565c0", lw=2.5))
        title += f"\n(blue arrow: wind from {wind_from_deg:.0f}°)"
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    return fig


def overlap_histogram(overlap, top=20):
    """How many of the original top 20 each random weight set kept."""
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(overlap, bins=np.arange(-0.5, top + 1.5, 1), color="#2e7d32", edgecolor="white")
    ax.set_xlabel(f"Original top-{top} trees still in the top {top}")
    ax.set_ylabel("Number of random weight sets")
    ax.set_title("Top-20 overlap across random weight sets")
    fig.tight_layout()
    return fig


def height_scatter(pairs: pd.DataFrame):
    """App height vs field height with the 1:1 line (points on the line = perfect)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pairs["field_height_m"], pairs["app_height_m"], s=22, color="#2e7d32", edgecolor="black", linewidth=0.4)
    top = float(max(pairs.max().max(), 1)) * 1.05
    ax.plot([0, top], [0, top], color="black", linestyle="--", linewidth=1, label="1:1 line")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Field height (m)")
    ax.set_ylabel("App height (m)")
    ax.set_title("Tree height: app vs field")
    ax.legend()
    fig.tight_layout()
    return fig


def log_boxplots(values: dict):
    """For each factor: values at downed logs next to values at random forest points."""
    names = list(values)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.8), squeeze=False)
    for ax, name in zip(axes[0], names):
        at_logs, at_random = values[name]
        ax.boxplot([np.asarray(at_logs, dtype=float), np.asarray(at_random, dtype=float)], showfliers=False)
        ax.set_xticks([1, 2], ["logs", "random"])
        ax.set_title(name, fontsize=10)
    fig.tight_layout()
    return fig


def export_figures(work_dir) -> list:
    """Save every chart that has data as a 300 dpi PNG in results/figures/. Returns the file names."""
    work_dir = Path(work_dir)
    results, figures = work_dir / "results", work_dir / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    saved = []

    def load(name):
        path = results / name
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except ValueError:
            return None

    def save(name, fig):
        (figures / name).write_bytes(png(fig))
        saved.append(name)

    r = load("sensitivity.json")
    if r and r.get("overlap"):
        save("sensitivity_top20_overlap.png", overlap_histogram(r["overlap"]))
    r = load("field_check_results.json")
    if r:
        if r.get("auc"):
            save("field_check_auc.png", auc_figure(pd.DataFrame(r["auc"]), "Blind field check: which ranking finds defective trees?"))
        if r.get("height_pairs"):
            save("field_check_heights.png", height_scatter(pd.DataFrame(r["height_pairs"])))
    r = load("log_validation.json")
    if r:
        if r.get("values"):
            save("downed_logs_factors.png", log_boxplots(r["values"]))
        if r.get("axes_deg"):
            save("downed_logs_rose.png", rose_figure(r["axes_deg"], title="Downed log directions", axial=True))
    for path in sorted((results / "storms").glob("*/storm_results.json")) if (results / "storms").exists() else []:
        storm_result = json.loads(path.read_text(encoding="utf-8"))
        slug = path.parent.name
        auc = storm_result["prediction"]["outcomes"]["severe"]["auc"]
        if auc and any(a["AUC"] is not None for a in auc):
            save(f"storm_{slug}_auc.png", auc_figure(pd.DataFrame(auc), f"{storm_result['storm']['name']}: severe damage"))
        directions = path.parent / "fall_directions.csv"
        if directions.exists():
            d = pd.read_csv(directions)
            save(f"storm_{slug}_rose.png", rose_figure(d["observed_fall_deg"], storm_result["storm"].get("wind_from_deg")))
    return saved
