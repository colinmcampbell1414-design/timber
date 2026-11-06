"""Shared pieces for the Hazards & Applications page.

Every application module (mudslide.py, trees.py, ...) has the same parts:
    NAME            short name shown in the dropdown
    DESCRIPTION     2-3 plain-language sentences
    REQUIRED_INPUTS e.g. ["dtm"], ["dtm", "dsm"], ["dtm", "ortho"]
    SHAPES          things the user marks on the map, e.g. {"building": ShapeSpec(...)}
    TEST_AT_HOME    how to check the results in real life
    params(st)      draws the module's sliders and returns their values (a dict)
    run(layers, params) -> Result

This file holds the data types they share (Layers, Result, Overlay,
ShapeSpec) and small helper functions (geometry, normalizing, grouping
cells into patches).
"""

import hashlib
import pickle
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as PolygonPath
from scipy import ndimage

from terrain import hydrology, slope, vegetation
from terrain.treefall import ramp  # noqa: F401 (shared: 0 until a start point, 1 at a full point)
from terrain.io import Raster

M3_TO_FT3 = 35.3147  # cubic feet in one cubic meter
M3_TO_YD3 = 1.30795  # cubic yards in one cubic meter
M_TO_FT = 3.28084
M_TO_IN = 39.3701
M2_TO_FT2 = 10.7639

RESULT_CACHE_VERSION = 1  # bump when an analysis changes, so old saved results are ignored
KEEP_RESULTS = 12  # saved results kept per analysis; the oldest are deleted

INPUT_NAMES = {"dtm": "DTM (terrain model)", "dsm": "DSM (surface model)", "ortho": "orthophoto"}

# Class colors used by several apps (low / moderate / high).
LOW_MOD_HIGH = [(0, "Low", "#1a9850"), (1, "Moderate", "#fdae61"), (2, "High", "#d73027")]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass
class ShapeSpec:
    """Something the user marks on the map (by clicking or with a CSV).

    kind: "points"   - each click is one point (e.g. targets)
          "polygons" - clicks are corners of an outline (e.g. a building)
          "lines"    - clicks are points along a path (e.g. a walkway)
    """

    kind: str
    label: str
    help: str = ""
    required: bool = False


@dataclass
class Layers:
    """All the map layers an app can use, on ONE shared grid of cells.

    dtm:   ground heights (Raster). Always present.
    dsm:   surface heights incl. trees/buildings, same grid (Raster) or None.
    ortho: photo colors, rows x cols x 3 from 0 to 1, or None.
    shapes: what the user marked: {"building": [[(x, y), ...], ...], ...}
            in map coordinates (meters).
    Derived layers (slope, flow, ...) are worked out the first time they are
    used and then remembered.
    """

    dtm: Raster
    dsm: Raster | None = None
    ortho: np.ndarray | None = None
    shapes: dict = field(default_factory=dict)
    flow_result: dict | None = None  # can be passed in if already computed (saves time)
    # Optional disk cache for slow results: a folder, plus a code identifying this
    # survey and its settings (see terrain/survey.py). Both None = don't remember.
    cache_dir: object = None
    cache_key: str | None = None

    def remember(self, name: str, settings, compute):
        """Run compute() once per (survey, settings) and keep the answer on disk.

        The next time the same survey is analyzed with the same settings (even
        after restarting TIMBER) the saved answer is loaded instead, which takes
        about a second instead of the full analysis time.
        """
        if self.cache_dir is None or self.cache_key is None:
            return compute()
        code = hashlib.md5(repr((name, RESULT_CACHE_VERSION, self.cache_key, settings)).encode()).hexdigest()[:16]
        path = Path(self.cache_dir) / f"{name}_{code}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    result = pickle.load(f)  # only ever files TIMBER saved itself
                path.touch()  # "recently used"
                return result
            except Exception:
                path.unlink(missing_ok=True)  # damaged: work it out again
        result = compute()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(path)
            saved = sorted(Path(self.cache_dir).glob(f"{name}_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in saved[KEEP_RESULTS:]:  # keep only the most recent results
                old.unlink(missing_ok=True)
        except OSError:
            pass  # can't save: still fine, just not remembered
        return result

    def has(self, name: str) -> bool:
        return {"dtm": self.dtm is not None, "dsm": self.dsm is not None, "ortho": self.ortho is not None}[name]

    @property
    def cell(self):
        return self.dtm.pixel_size  # (cell width, cell height) in meters

    @property
    def cell_area(self):
        px, py = self.cell
        return px * py

    @cached_property
    def slope_deg(self):
        return slope.slope_degrees(self.dtm.data, *self.cell)

    @cached_property
    def aspect_deg(self):
        return slope.aspect_degrees(self.dtm.data, *self.cell)

    @cached_property
    def flow(self):
        if self.flow_result is not None:
            return self.flow_result
        return hydrology.analyze_flow(self.dtm.data, *self.cell)

    @cached_property
    def twi(self):
        return hydrology.topographic_wetness_index(self.flow["acc_area_m2"], self.slope_deg, self.cell[0])

    @cached_property
    def exg(self):
        return vegetation.exg(self.ortho) if self.ortho is not None else None

    @cached_property
    def vari(self):
        return vegetation.vari(self.ortho) if self.ortho is not None else None

    @cached_property
    def xy(self):
        """Map coordinates (x, y) of every cell center."""
        return cell_centers(self.dtm)


def shape_list(layers: Layers, key: str) -> list:
    """The user's shapes for `key` as lists of (x, y) corners.

    Shapes are stored as {"label": "house", "xy": [(x, y), ...]}; plain
    lists of corners are accepted too (handy in tests).
    """
    return [s["xy"] if isinstance(s, dict) else s for s in layers.shapes.get(key, [])]


def shape_labels(layers: Layers, key: str) -> list:
    """The label of each shape (e.g. "house"), or "" if it has none."""
    return [s.get("label", "") if isinstance(s, dict) else "" for s in layers.shapes.get(key, [])]


def missing_inputs(app, layers: Layers) -> str | None:
    """A message naming the files an app needs but doesn't have, or None if all are there."""
    missing = [name for name in app.REQUIRED_INPUTS if not layers.has(name)]
    if not missing:
        return None
    names = " and ".join(INPUT_NAMES[m] for m in missing)
    files = ", ".join(f"*{m}*.tif" if m != "ortho" else "*orthophoto*.tif" for m in missing)
    return (f"'{app.NAME}' needs the {names}. Export it from WebODM (Download Assets) and put "
            f"it in the data folder (a file name containing: {files}).")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
@dataclass
class Overlay:
    """One map layer to draw on top of the background.

    Either classes (whole numbers, each with a label and color), or a
    continuous value shown with a color scale.
    """

    name: str
    data: np.ndarray
    classes: list | None = None  # [(value, label, color), ...]
    cmap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    units: str = ""


@dataclass
class Section:
    """One tab of a bigger app (e.g. the tree app's "Downed logs" tab).

    Each tab can have its own map (overlays + annotations), a main table
    (showing the first `table_head` rows, with the rest in an expander),
    extra tables, short text, download buttons, and a `custom(st)` function
    for anything else (e.g. a file upload that compares results).
    """

    title: str
    overlays: list = field(default_factory=list)
    annotations: list = field(default_factory=list)
    table: pd.DataFrame | None = None
    table_title: str = ""
    table_head: int | None = None
    extra_tables: list = field(default_factory=list)  # [(title, DataFrame), ...]
    text: str = ""
    downloads: list = field(default_factory=list)  # [(label, data, file name, mime type), ...]
    custom: object = None  # function(st) or None


@dataclass
class Result:
    """What every app's run() returns."""

    overlays: list = field(default_factory=list)  # [Overlay, ...], first one is shown by default
    metrics: dict = field(default_factory=dict)  # {"label": "value with units"}
    table: pd.DataFrame = field(default_factory=pd.DataFrame)  # flagged features
    warnings: list = field(default_factory=list)  # plain-language warnings
    # Extra things to draw, in map coordinates. Each is a dict with a "type":
    #   circle(x, y, r, color, label), arrow(x, y, dx, dy, color),
    #   polygon(xy, color), line(xy, color), point(x, y, label, color),
    #   segments(segments, color, width),
    #   scatter(x, y arrays, colors, size)          many points at once
    #   arrows(x, y, dx, dy arrays, color)           many arrows at once
    annotations: list = field(default_factory=list)
    # Optional tabs. If given, the page shows the metrics, then one tab per section
    # (each with its own map, tables and downloads) instead of the single map/table.
    sections: list = field(default_factory=list)

    @property
    def legend(self):
        """Legend entries of the first overlay: [(label, color), ...]."""
        if self.overlays and self.overlays[0].classes:
            return [(label, color) for _, label, color in self.overlays[0].classes]
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cell_centers(raster: Raster):
    """Map x, y of the center of every cell (two grids)."""
    t = raster.transform
    rows, cols = raster.shape
    c, r = np.meshgrid(np.arange(cols) + 0.5, np.arange(rows) + 0.5)
    return t.c + c * t.a, t.f + r * t.e


def polygon_mask(raster: Raster, polygon) -> np.ndarray:
    """True for cells whose center is inside the polygon [(x, y), ...]."""
    if len(polygon) < 3:
        return np.zeros(raster.shape, dtype=bool)
    x, y = cell_centers(raster)
    inside = PolygonPath(np.asarray(polygon, float)).contains_points(np.column_stack([x.ravel(), y.ravel()]))
    return inside.reshape(raster.shape)


def shapes_mask(raster: Raster, polygons) -> np.ndarray:
    """True for cells inside ANY of the polygons."""
    mask = np.zeros(raster.shape, dtype=bool)
    for poly in polygons:
        mask |= polygon_mask(raster, poly)
    return mask


def _segment_distance(px, py, x1, y1, x2, y2):
    """Distance from points (px, py) to the line segment (x1, y1)-(x2, y2)."""
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return np.hypot(px - x1, py - y1)
    # How far along the segment the closest point is (0 = start, 1 = end).
    t = np.clip(((px - x1) * dx + (py - y1) * dy) / length_sq, 0, 1)
    return np.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def distance_to_shape(px, py, shape, closed=True) -> np.ndarray:
    """Distance (m) from points to a shape: a point, a line, or a polygon.

    Points inside a closed polygon (3+ corners) have distance 0.
    """
    single = np.ndim(px) == 0  # one point given as plain numbers
    px, py = np.atleast_1d(np.asarray(px, float)), np.atleast_1d(np.asarray(py, float))
    pts = [tuple(map(float, v)) for v in shape]
    if len(pts) == 1:
        dist = np.hypot(px - pts[0][0], py - pts[0][1])
    else:
        edges = list(zip(pts[:-1], pts[1:]))
        if closed and len(pts) >= 3:
            edges.append((pts[-1], pts[0]))
        dist = np.full(px.shape, np.inf)
        for (x1, y1), (x2, y2) in edges:
            dist = np.minimum(dist, _segment_distance(px, py, x1, y1, x2, y2))
        if closed and len(pts) >= 3:
            inside = PolygonPath(np.asarray(pts)).contains_points(np.column_stack([px.ravel(), py.ravel()]))
            dist[inside.reshape(px.shape)] = 0.0
    return float(dist[0]) if single else dist


def distance_to_shapes(px, py, shapes, closed=True) -> np.ndarray:
    """Distance from points to the nearest of several shapes (inf if there are none)."""
    dist = np.full(np.shape(px), np.inf)
    for shape in shapes:
        dist = np.minimum(dist, distance_to_shape(px, py, shape, closed))
    return dist


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Turn values into 0-1 by rank: 0 = lowest in the map, 1 = highest.

    So 0.9 means "higher than 90% of the mapped area". This makes very
    different measurements (wetness, curvature, ...) comparable, but the
    result is RELATIVE to this map only.
    """
    out = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    n = int(valid.sum())
    if n == 0:
        return out
    ranks = np.argsort(np.argsort(values[valid], kind="stable"), kind="stable")
    out[valid] = ranks / max(n - 1, 1)
    return out


def nan_smooth(z: np.ndarray, radius_m: float, cell: float) -> np.ndarray:
    """Gaussian-smoothed copy of z that ignores NaN cells.

    Each cell becomes a weighted average of its neighborhood (closer cells
    count more). Dividing by the smoothed "valid" mask means missing cells
    don't pull the average toward 0.
    """
    sigma = max(radius_m / cell, 0.01)
    valid = np.isfinite(z)
    filled = np.where(valid, z, 0.0)
    num = ndimage.gaussian_filter(filled, sigma, mode="nearest")
    den = ndimage.gaussian_filter(valid.astype(float), sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid] = np.nan
    return out


def classify(score: np.ndarray, moderate: float, high: float) -> np.ndarray:
    """0 = low, 1 = moderate, 2 = high; -1 where there is no data."""
    classes = np.where(score >= high, 2, np.where(score >= moderate, 1, 0))
    classes[~np.isfinite(score)] = -1
    return classes


def patches(mask: np.ndarray, layers: Layers, min_area: float = 0.0, values: dict | None = None) -> pd.DataFrame:
    """Group touching flagged cells into numbered patches and describe each one.

    values: {"column name": (grid, "max" or "mean")} extra summaries per patch.
    Returns a table with: patch, area_m2, x, y (center), plus the extras,
    biggest first. Also returns the label grid (0 = not in a patch).
    """
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if count == 0:
        return pd.DataFrame(columns=["patch", "area_m2", "x", "y", *(values or {})]), labels
    ids = np.arange(1, count + 1)
    area = ndimage.sum(mask, labels, ids) * layers.cell_area
    x, y = layers.xy
    table = pd.DataFrame({
        "label": ids,
        "area_m2": area,
        "x": ndimage.mean(x, labels, ids),
        "y": ndimage.mean(y, labels, ids),
    })
    for name, (grid, how) in (values or {}).items():
        g = np.nan_to_num(grid, nan=-np.inf if how == "max" else 0.0)
        table[name] = ndimage.maximum(g, labels, ids) if how == "max" else ndimage.mean(g, labels, ids)
    table = table[table["area_m2"] >= min_area].sort_values("area_m2", ascending=False)
    # Renumber the kept patches 1, 2, 3 ... (biggest first); dropped ones become 0.
    # A lookup table does it in one step: new_id_of[old label] = new label.
    # (Looping over patches with `labels == id` would scan the whole map once per
    # patch, which takes minutes on a big map with thousands of patches.)
    new_id_of = np.zeros(int(labels.max()) + 1, dtype=labels.dtype)
    new_id_of[table["label"].to_numpy()] = np.arange(1, len(table) + 1)
    new_labels = new_id_of[labels]
    table.insert(0, "patch", np.arange(1, len(table) + 1))
    return table.drop(columns="label").reset_index(drop=True), new_labels


def outline_annotations(shapes, color="#ffffff", closed=True):
    """Annotations that draw the user's shapes on the map."""
    out = []
    for shape in shapes:
        if len(shape) == 1:
            out.append({"type": "point", "x": shape[0][0], "y": shape[0][1], "color": color, "label": ""})
        else:
            out.append({"type": "polygon" if closed and len(shape) >= 3 else "line", "xy": shape, "color": color})
    return out
