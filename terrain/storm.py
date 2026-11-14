"""Storm check: did the trees TIMBER called hazardous actually fail in a real wind storm?

The same forest is flown before and after a storm. The steps, in order:

1. FREEZE the predictions made from the BEFORE survey (tree table, settings and
   a fingerprint of the files), so they can't be changed after the damage is known.
2. ALIGN the after survey to the before survey (sideways shift, height offset
   and tilt), because two drone flights never line up exactly.
3. DETECT what changed for every tree (fallen / snapped top / crown damage /
   no change / uncertain) and find NEW downed logs.
4. NULL TEST: run the same detection on a "calm pair" (two flights with no
   storm between them) to measure how much "damage" is just noise.
5. SCORE the frozen predictions against what happened: hazard score vs damage,
   predicted vs observed fall direction, landing strips and the exposure map.
6. FIELD CONFIRMATION: a small blind check sheet for the woods.

All of the math is here; the "Storm check" screens are in timber_ui/storm_panel.py.
Directions are compass degrees (0 = north, 90 = east). Wind direction is where
the wind comes FROM, so a tree pushed by a west wind (270) falls toward 90.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.spatial import cKDTree

from terrain import io, treefall, validation
from terrain import slope as slope_mod

DAMAGE_CLASSES = ["fallen", "snapped top", "crown damage", "no change", "uncertain"]
SEVERE = ["fallen", "snapped top"]
ANY_DAMAGE = ["fallen", "snapped top", "crown damage"]
CLASS_COLORS = {"fallen": "#d73027", "snapped top": "#fc8d59", "crown damage": "#fee08b", "no change": "#1a9850",
                "uncertain": "#9e9e9e"}

# Starting values for the change detection (all adjustable in the app).
CHANGE_DEFAULTS = {
    "change_threshold": None,  # m; None = 3 x the leftover noise on stable ground (at least 0.5 m)
    "fallen_top": 0.60,        # fallen: the top dropped by more than this share of the tree's height...
    "fallen_crown": 0.70,      # ...AND more than this share of the crown is gone
    "snap_min_m": 2.0,         # snapped top: the top dropped by more than this many meters...
    "snap_frac": 0.20,         # ...and by more than this share of the height (whichever is bigger)
    "crown_frac": 0.25,        # crown damage: more than this share of the crown is gone
    "top_radius": 1.5,         # m around the old tree top searched for the new top
    "buffer_m": 1.0,           # m added around each crown when looking for missing data
    "edge_m": 5.0,             # trees closer than this to the edge of the map are "uncertain"
    "top_only": False,         # leaf-drop mode: use only the top drop (ignore crown thinning)
    "log_match_m": 2.0,        # a log in both surveys: centers within this distance...
    "log_match_deg": 20.0,     # ...and directions within this angle
    "link_factor": 1.5,        # a new log belongs to a fallen tree if one end is within this x crown radius
    "gap_area": 10.0,          # m²: smallest new canopy gap (not explained by a tree) to list
}
RMS_WARNING_M = 0.15    # leftover height noise on stable ground above this = poor alignment
SHIFT_WARNING_M = 0.5   # leftover sideways shift above this = poor alignment


# ---------------------------------------------------------------------------
# 1a. Storm list (config/storms.yaml) and wind data
# ---------------------------------------------------------------------------
STORM_TEMPLATE = {
    "name": "", "date_start": "", "date_end": "", "before_survey": "", "after_survey": "",
    "weather_station": {"id": "", "name": ""}, "peak_gust_mps": None, "mean_wind_mps": None,
    "wind_from_deg": None, "is_calm_control": False, "notes": "",
}
STORMS_HEADER = """\
# One entry per storm (or calm pair) that was flown before and after.
#   before_survey / after_survey: folder names inside the data folder.
#   wind_from_deg: where the wind came FROM during the strongest winds (270 = west).
#   is_calm_control: true for two flights with NO storm between them (the null test).
# Wind numbers come from the nearest airport weather station (see the README).
"""
SPEED_TO_MPS = {"m/s": 1.0, "knots": 0.514444, "mph": 0.44704, "km/h": 0.277778}


def load_storms(config_dir) -> list:
    """Read config/storms.yaml (an empty list if the file doesn't exist yet)."""
    import yaml

    path = Path(config_dir) / "storms.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    storms = data.get("storms", []) if isinstance(data, dict) else data
    return [{**STORM_TEMPLATE, **s} for s in storms or []]


def save_storms(config_dir, storms: list):
    """Write the storm list to config/storms.yaml."""
    import yaml

    path = Path(config_dir) / "storms.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump({"storms": [_plain(s) for s in storms]}, sort_keys=False, allow_unicode=True)
    path.write_text(STORMS_HEADER + text, encoding="utf-8")


def storm_slug(name: str) -> str:
    """A safe folder name for a storm, e.g. "Nov 4 gale" -> "nov_4_gale"."""
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "storm"


def read_station_csv(source, speed_unit: str = "m/s") -> pd.DataFrame:
    """Read a weather-station CSV into columns time, speed_mps, gust_mps, direction.

    Column names are matched loosely, so files from NOAA / ASOS downloads (for
    example the Iowa Environmental Mesonet: valid, sknt, gust, drct) work as
    they are. `speed_unit` is the unit used in the file.
    """
    df = pd.read_csv(source, comment="#", skipinitialspace=True)
    names = {str(c).strip().lower(): c for c in df.columns}

    def find(*options):
        for o in options:
            if o in names:
                return df[names[o]]
        for key, col in names.items():  # then anything that contains the word
            if any(o in key for o in options):
                return df[col]
        return None

    time = find("time", "valid", "datetime", "timestamp", "date")
    speed = find("wind speed", "wind_speed", "speed", "sknt", "sped")
    gust = find("gust")
    direction = find("direction", "drct", "wind_dir", "dir")
    if time is None or speed is None or direction is None:
        raise ValueError("The CSV needs columns for time, wind speed and wind direction (gust is optional).")
    k = SPEED_TO_MPS[speed_unit]
    out = pd.DataFrame({
        "time": pd.to_datetime(time, errors="coerce"),
        "speed_mps": pd.to_numeric(speed, errors="coerce") * k,
        "gust_mps": pd.to_numeric(gust, errors="coerce") * k if gust is not None else np.nan,
        "direction": pd.to_numeric(direction, errors="coerce"),
    }).dropna(subset=["time"])
    # A reading with no gust reported: the gust is at least the steady wind speed.
    out["gust_mps"] = out["gust_mps"].fillna(out["speed_mps"])
    return out.sort_values("time").reset_index(drop=True)


def weighted_wind_direction(direction_deg, gust) -> float:
    """Average wind direction, counting strong gusts much more than light wind.

    Directions can't be averaged like ordinary numbers: the average of 350 and
    10 degrees should be 0 (north), not 180. So each reading is turned into an
    arrow (sin, cos of its direction) whose length is its weight, the arrows are
    added up, and the direction of the total arrow is the answer:
        mean direction = atan2( sum(w x sin(d)), sum(w x cos(d)) ),   w = gust²
    Gust squared, because the push of wind on a tree grows with speed squared.
    """
    d = np.radians(np.asarray(direction_deg, dtype=float))
    w = np.asarray(gust, dtype=float) ** 2
    ok = np.isfinite(d) & np.isfinite(w)
    if not ok.any() or w[ok].sum() == 0:
        return float("nan")
    return float(np.degrees(np.arctan2((w[ok] * np.sin(d[ok])).sum(), (w[ok] * np.cos(d[ok])).sum())) % 360)


def wind_stats(readings: pd.DataFrame, start, end) -> dict:
    """Peak gust, mean wind and gust-weighted direction between `start` and `end`."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end == end.normalize():  # an end date without a time means "to the end of that day"
        end = end + pd.Timedelta(days=1)
    r = readings[(readings["time"] >= start) & (readings["time"] <= end)]
    if r.empty:
        raise ValueError("No readings between the storm's start and end dates.")
    return {"peak_gust_mps": round(float(r["gust_mps"].max()), 1),
            "mean_wind_mps": round(float(r["speed_mps"].mean()), 1),
            "wind_from_deg": round(weighted_wind_direction(r["direction"], r["gust_mps"]), 0),
            "readings": int(len(r))}


# ---------------------------------------------------------------------------
# 1b. Frozen predictions
# ---------------------------------------------------------------------------
PREVAILING, ACTUAL = "prevailing-wind prediction", "actual-wind prediction"
FROZEN_FILES = {PREVAILING: ("predictions_frozen.csv", "settings.json"),
                ACTUAL: ("predictions_frozen_actual_wind.csv", "settings_actual_wind.json")}
GRIDS_FILE = "frozen_grids.npz"
LOG_FILE = "freeze_log.jsonl"
PROB_COLUMNS = [f"p_{int(d):03d}" for d in treefall.BIN_CENTERS]  # chance of falling toward 0, 10, ... 350 degrees


class FrozenError(Exception):
    """The frozen predictions are missing or were changed after they were frozen."""


def _plain(obj):
    """Turn numpy numbers, paths and dates into plain values that JSON / YAML can store."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (Path, datetime, pd.Timestamp)) or hasattr(obj, "isoformat"):
        return str(obj)
    return obj


def sha256_file(path) -> str:
    """SHA-256 fingerprint of a file: 64 letters and digits that change completely
    if even one character of the file changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_version(project_dir=None) -> str:
    """The git commit of the code, if the project is a git repository ("unknown" otherwise)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_dir or Path(__file__).parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _hashes(out_dir: Path, files) -> dict:
    return {name: sha256_file(out_dir / name) for name in files if (out_dir / name).exists()}


def _combined(hashes: dict) -> str:
    """One fingerprint for a set of files (the fingerprint of their fingerprints)."""
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def freeze_predictions(out_dir, trees: pd.DataFrame, probs: np.ndarray, settings: dict, crs=None, grids: dict = None,
                       label: str = PREVAILING, note: str = "", now: datetime = None) -> dict:
    """Save the predictions so later analysis can only READ them.

    Writes, in results/storms/<storm>/:
      predictions_frozen.csv  every tree: position, height, crown, every factor, hazard
                              score, priority, baseline ranks, predicted fall direction,
                              chance of landing on each feature, and the chance of
                              falling in each of the 36 directions (p_000 ... p_350)
      settings.json           every slider value and the code version
      frozen_grids.npz        the crown outlines, exposure map and feature masks
      freeze_log.jsonl        one line per freeze: the time and the SHA-256
                              fingerprints of the files above
    Returns the log entry.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_name, settings_name = FROZEN_FILES[label]
    now = now or datetime.now(timezone.utc)

    table = trees.copy()
    if crs is not None and "latitude" not in table:
        lon, lat = io.map_to_latlon(table["x"].to_numpy(), table["y"].to_numpy(), crs)
        table["latitude"], table["longitude"] = np.round(lat, 7), np.round(lon, 7)
    for i, col in enumerate(PROB_COLUMNS):
        table[col] = np.round(probs[:, i], 6)
    table.to_csv(out_dir / csv_name, index=False)

    info = {"label": label, "frozen_at_utc": now.isoformat(timespec="seconds"), "code_version": code_version(),
            "trees": int(len(table)), "note": note, "settings": _plain(settings)}
    (out_dir / settings_name).write_text(json.dumps(info, indent=2), encoding="utf-8")

    files = [csv_name, settings_name]
    if grids is not None:
        np.savez_compressed(out_dir / GRIDS_FILE, **grids)
    if (out_dir / GRIDS_FILE).exists():
        files.append(GRIDS_FILE)
    hashes = _hashes(out_dir, files)
    entry = {"label": label, "frozen_at_utc": info["frozen_at_utc"], "sha256": hashes, "combined": _combined(hashes),
             "note": note}
    with open(out_dir / LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def freeze_log(out_dir) -> list:
    """Every freeze so far (one entry per line of freeze_log.jsonl), oldest first."""
    path = Path(out_dir) / LOG_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_frozen(out_dir, label: str = PREVAILING) -> dict:
    """Check the frozen files against the fingerprints written when they were frozen.

    Returns {"exists", "ok", "frozen_at_utc", "combined", "message"}.
    """
    out_dir = Path(out_dir)
    entries = [e for e in freeze_log(out_dir) if e["label"] == label]
    csv_name, _ = FROZEN_FILES[label]
    if not entries or not (out_dir / csv_name).exists():
        return {"exists": False, "ok": False, "message": "Not frozen yet."}
    last = entries[-1]
    now = _hashes(out_dir, last["sha256"])
    changed = [name for name, h in last["sha256"].items() if now.get(name) != h]
    ok = not changed
    message = ("Unchanged since it was frozen." if ok else
               "CHANGED since it was frozen: " + ", ".join(changed) + ". These predictions can't be trusted.")
    return {"exists": True, "ok": ok, "frozen_at_utc": last["frozen_at_utc"], "combined": last["combined"],
            "message": message, "times_frozen": len(entries)}


def load_frozen(out_dir, label: str = PREVAILING):
    """Read the frozen predictions. Raises FrozenError if they are missing or were changed.

    Returns (trees, probs, settings): the tree table, the 36 direction chances
    for every tree, and the saved settings.
    """
    out_dir = Path(out_dir)
    status = verify_frozen(out_dir, label)
    if not status["exists"]:
        raise FrozenError(f"No frozen predictions ({label}) in {out_dir}. Freeze them before the storm.")
    if not status["ok"]:
        raise FrozenError(status["message"])
    csv_name, settings_name = FROZEN_FILES[label]
    trees = pd.read_csv(out_dir / csv_name)
    probs = trees[PROB_COLUMNS].to_numpy(dtype=float)
    info = json.loads((out_dir / settings_name).read_text(encoding="utf-8"))
    return trees.drop(columns=PROB_COLUMNS), probs, info


def load_frozen_grids(out_dir) -> dict:
    """The frozen crown outlines, exposure map and feature masks."""
    with np.load(Path(out_dir) / GRIDS_FILE, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def freeze_actual_wind(out_dir, wind_from_deg: float, cell: float, now: datetime = None) -> dict:
    """A second frozen copy that uses the storm's ACTUAL wind direction.

    Only the fall-direction part changes (the hazard scores are copied as they
    are). It is worked out from the first frozen copy and its settings, never
    from the after survey, and is labeled "actual-wind prediction" so the two
    are never confused: the first is a true forecast, this one answers "would
    the direction model have been right if the wind had been known?"
    """
    trees, _, info = load_frozen(out_dir, PREVAILING)
    s = info["settings"]
    probs = treefall.fall_probabilities(trees, s["dir_weights"], s["kappa"], float(wind_from_deg), True)
    trees["fall_dir"] = treefall.BIN_CENTERS[np.argmax(probs, axis=1)]
    trees["fall_dir_word"] = [treefall.compass_word(d) for d in trees["fall_dir"]]
    grids = load_frozen_grids(out_dir) if (Path(out_dir) / GRIDS_FILE).exists() else {}
    masks = {k[5:]: v.astype(bool) for k, v in grids.items() if k.startswith("mask_")}
    if masks:
        for name, (can, chance) in treefall.landing_chances(trees, probs, masks, cell, s.get("safety", 1.0)).items():
            trees[f"can reach {name}"], trees[f"chance lands on {name}"] = can, chance
    settings = {**s, "wind_from": float(wind_from_deg), "wind_on": True}
    return freeze_predictions(out_dir, trees, probs, settings, label=ACTUAL, now=now,
                              note=f"fall directions recomputed with the storm's wind from {wind_from_deg:.0f} deg")


# ---------------------------------------------------------------------------
# 2. Aligning the before and after surveys
# ---------------------------------------------------------------------------
def texture(surface: np.ndarray, cell: float, smooth_m: float = 2.0) -> np.ndarray:
    """The small-scale shape of a surface: the surface minus a smoothed copy of itself.

    This removes the overall height and tilt (which differ between flights) and
    keeps edges of crowns and roofs, which is what the two surveys are matched
    on. Lighting and colors don't matter because only heights are used.
    """
    valid = np.isfinite(surface)
    z = np.where(valid, surface, 0.0)
    sigma = smooth_m / cell
    num = ndimage.gaussian_filter(z, sigma, mode="nearest")
    den = np.maximum(ndimage.gaussian_filter(valid.astype(float), sigma, mode="nearest"), 1e-6)
    out = np.clip(z - num / den, -3.0, 3.0)  # clip so one tall object can't dominate
    out[~valid] = 0.0
    return out


def horizontal_shift(before: np.ndarray, after: np.ndarray, cell: float, mask: np.ndarray = None):
    """How far the after survey is displaced sideways from the before survey.

    Uses phase correlation: the two pictures are compared at every possible
    shift at once (with a Fourier transform), and the shift where they agree
    best is the answer, refined to 1/20 of a cell.
    Returns (east_m, north_m): the after survey shows everything this far east
    and north of where the before survey shows it.
    """
    from skimage.registration import phase_cross_correlation

    a, b = texture(before, cell), texture(after, cell)
    both = np.isfinite(before) & np.isfinite(after)
    if mask is not None:
        both &= mask
    a, b = a * both, b * both
    shift, _, _ = phase_cross_correlation(a, b, upsample_factor=20, normalization=None)
    # `shift` (rows, cols) is how far to move the AFTER picture to fit the before one,
    # so the after picture is displaced by the opposite. Rows grow southward.
    return float(-shift[1] * cell), float(shift[0] * cell)


def shift_grid(grid: np.ndarray, east_m: float, north_m: float, cell: float) -> np.ndarray:
    """Move a grid's contents by (-east_m, -north_m): undo a displacement found by horizontal_shift."""
    rows, cols = north_m / cell, -east_m / cell
    if grid.ndim == 3:
        return np.dstack([shift_grid(grid[..., i], east_m, north_m, cell) for i in range(grid.shape[2])])
    return ndimage.shift(grid.astype(float), (rows, cols), order=1, mode="constant", cval=np.nan)


def local_shifts(before: np.ndarray, after: np.ndarray, cell: float, block_m: float = 100.0) -> pd.DataFrame:
    """The sideways shift measured separately in blocks of the map.

    Two drone models can also differ slightly in scale or rotation, so the shift
    is not the same everywhere (for example +5 m at the west edge and -1 m at
    the east edge). Returns one row per block with enough data in both surveys:
    its center (meters east / north of the map's top-left corner) and its shift.
    """
    size = max(8, int(round(block_m / cell)))
    rows = []
    for r0 in range(0, before.shape[0] - size // 2, size):
        for c0 in range(0, before.shape[1] - size // 2, size):
            sl = (slice(r0, r0 + size), slice(c0, c0 + size))
            both = np.isfinite(before[sl]) & np.isfinite(after[sl])
            if both.mean() < 0.5 or both.shape[0] < size // 2 or both.shape[1] < size // 2:
                continue
            east, north = horizontal_shift(before[sl], after[sl], cell)
            rows.append({"x": (c0 + both.shape[1] / 2) * cell, "y": -(r0 + both.shape[0] / 2) * cell,
                         "east": east, "north": north})
    return pd.DataFrame(rows, columns=["x", "y", "east", "north"])


def fit_shift_model(blocks: pd.DataFrame, max_residual: float = 1.0):
    """Fit how the shift changes across the map:  shift = a + b x east + c x north.

    This "affine" model covers a constant shift (a), and stretch, rotation and
    shear (b, c). It is fitted to the block shifts by least squares; blocks that
    disagree with the fit by more than `max_residual` meters (a bad match, e.g.
    at the ragged edge of a survey) are dropped and the fit is repeated.
    Returns (3 x 2 coefficients for east and north shift, number of blocks used),
    or (None, 0) if fewer than 6 blocks are usable.
    """
    if len(blocks) < 6:
        return None, 0
    A = np.column_stack([np.ones(len(blocks)), blocks["x"], blocks["y"]])
    target = blocks[["east", "north"]].to_numpy()
    keep = np.ones(len(blocks), dtype=bool)
    coef = None
    for _ in range(4):
        if keep.sum() < 6:
            return None, 0
        coef, *_ = np.linalg.lstsq(A[keep], target[keep], rcond=None)
        resid = np.hypot(*(target - A @ coef).T)
        limit = max(max_residual, 3 * np.median(resid[keep]))
        new_keep = resid <= limit
        if (new_keep == keep).all():
            break
        keep = new_keep
    return coef, int(keep.sum())


def warp_grid(grid: np.ndarray, coef: np.ndarray, cell: float) -> np.ndarray:
    """Undo a shift that changes across the map (see fit_shift_model).

    For every cell, work out the shift there, and read the after survey at the
    shifted position (a straight-line blend of the 4 nearest cells).
    """
    if grid.ndim == 3:
        return np.dstack([warp_grid(grid[..., i], coef, cell) for i in range(grid.shape[2])])
    r, c = np.indices(grid.shape, dtype=float)
    x, y = c * cell, -r * cell
    east = coef[0, 0] + coef[1, 0] * x + coef[2, 0] * y
    north = coef[0, 1] + coef[1, 1] * x + coef[2, 1] * y
    # Content that the before survey shows at (r, c) is `east` meters east and
    # `north` meters north of there in the after survey (north = smaller row).
    return ndimage.map_coordinates(grid.astype(float), [r - north / cell, c + east / cell], order=1,
                                   mode="constant", cval=np.nan)


def stable_ground(chm_before, chm_after, slope_deg, building=None, max_chm: float = 0.5, max_slope: float = 10.0,
                  surfaces=None, cell: float = None, max_roughness: float = 0.15):
    """Open ground that should not change in a storm: low canopy in BOTH surveys,
    gentle slope (a small sideways error matters little there) and not a building.

    surfaces: (before DSM, after DSM). If given, the ground must also be SMOOTH in
    both (bumps under 0.15 m). Under thick canopy the ground model often follows
    the treetops, so the canopy height looks like 0 there, but the surface is
    bumpy; lawns, fields and roads are smooth.
    """
    low_b = np.nan_to_num(chm_before, nan=99) < max_chm
    low_a = np.nan_to_num(chm_after, nan=99) < max_chm
    stable = low_b & low_a & (np.nan_to_num(slope_deg, nan=99) < max_slope)
    if building is not None:
        stable &= ~building
    if surfaces is not None:
        for surface, low in zip(surfaces, (low_b, low_a)):
            rough = treefall.surface_roughness(np.where(low, surface, np.nan), cell, 1.0)
            stable &= np.nan_to_num(rough, nan=99) < max_roughness
    return stable


def fit_plane(diff: np.ndarray, mask: np.ndarray, cell: float):
    """Fit a tilted plane to a height difference on the stable cells.

    Two drone models differ by an unknown height offset and a slight tilt. The
    plane  diff = c + a x east + b x north  that is closest to the stable cells
    (least squares) is that difference. Cells more than 3 "typical spreads"
    from the first fit are left out of a second fit, so a parked car or a
    regrown bush doesn't pull the plane.
    Returns (plane grid to subtract, report): the report has the offset at the
    middle of the map, the tilt in degrees and the leftover RMS on stable ground.
    """
    r, c = np.indices(diff.shape)
    east = (c - diff.shape[1] / 2) * cell
    north = (diff.shape[0] / 2 - r) * cell
    ok = mask & np.isfinite(diff)
    if ok.sum() < 10:
        return np.zeros(diff.shape), {"cells": int(ok.sum()), "offset_m": float("nan"), "tilt_deg": float("nan"),
                                      "rms_m": float("nan")}
    coef = None
    for _ in range(2):
        A = np.column_stack([np.ones(ok.sum()), east[ok], north[ok]])
        coef, *_ = np.linalg.lstsq(A, diff[ok], rcond=None)
        resid = diff - (coef[0] + coef[1] * east + coef[2] * north)
        spread = 1.4826 * np.median(np.abs(resid[ok] - np.median(resid[ok])))
        ok = ok & (np.abs(resid) <= max(3 * spread, 0.02))
    plane = coef[0] + coef[1] * east + coef[2] * north
    report = {"cells": int(ok.sum()), "offset_m": float(coef[0]),
              "tilt_deg": float(np.degrees(np.arctan(np.hypot(coef[1], coef[2])))),
              "rms_m": _stable_rms(diff - plane, mask)}
    return plane, report


def _chm(dsm, dtm):
    chm = dsm - dtm
    return np.where(np.isfinite(chm), np.maximum(chm, 0.0), np.nan)


def _stable_rms(diff, stable) -> float:
    """Leftover height noise on stable ground: RMS of the middle 98% of the differences."""
    values = diff[stable & np.isfinite(diff)]
    if len(values) < 10:
        return float("nan")
    lo, hi = np.percentile(values, [1, 99])
    core = values[(values >= lo) & (values <= hi)]
    return float(np.sqrt(np.mean(core ** 2)))


def align_surveys(before: dict, after: dict, cell: float, building: np.ndarray = None, warp_window_m: float = 40.0) -> tuple:
    """Line the after survey up with the before survey.

    before / after: {"dsm": grid, "dtm": grid, "ortho": rows x cols x 3 or None},
    both already on the before survey's grid.
    1. Sideways: find the shift between the two surface models and undo it.
    2. Height: on stable open ground, fit and subtract a plane (offset + tilt),
       separately for the surface model (DSM) and the ground model (DTM).
    3. Warp: drone models made without ground control points also bend slightly
       ("doming"), by different amounts in each flight, so a plane is not enough
       over a whole forest. The bend is the TYPICAL height difference nearby: the
       median of (after - before) within `warp_window_m` meters. A median ignores
       the minority of cells that really changed, so a fallen tree is not
       smoothed away. (A blowdown wider than about half the window would be:
       then make the window bigger. 0 = skip this step.)
    Returns (aligned after dict, report). The report lists the shift, the
    offsets, tilts, warp, leftover RMS, and any warnings.
    """
    east, north = horizontal_shift(before["dsm"], after["dsm"], cell)
    moved = {k: (shift_grid(v, east, north, cell) if v is not None else None) for k, v in after.items()}
    # On big maps the shift can change across the map (scale / rotation): measure what is
    # left in 100 m blocks and, if there are enough blocks, undo it with an affine model.
    blocks = local_shifts(before["dsm"], moved["dsm"], cell)
    coef, used = fit_shift_model(blocks)
    variation = 0.0
    if coef is not None:
        variation = float(np.hypot(blocks["east"], blocks["north"]).quantile(0.9))
        moved = {k: (warp_grid(v, coef, cell) if v is not None else None) for k, v in moved.items()}
        left = local_shifts(before["dsm"], moved["dsm"], cell)
        # Leftover shift = the typical (median) block shift after the correction.
        res_e, res_n = (float(np.median(np.hypot(left["east"], left["north"]))), 0.0) if len(left) else (0.0, 0.0)
    else:
        # How much sideways shift is left after the correction (should be close to 0).
        res_e, res_n = horizontal_shift(before["dsm"], moved["dsm"], cell)

    slope_deg = slope_mod.slope_degrees(before["dtm"], cell, cell)
    stable = stable_ground(_chm(before["dsm"], before["dtm"]), _chm(moved["dsm"], moved["dtm"]), slope_deg, building,
                           surfaces=(before["dsm"], moved["dsm"]), cell=cell)
    report = {"shift_east_m": east, "shift_north_m": north, "shift_m": float(np.hypot(east, north)),
              "residual_shift_m": float(np.hypot(res_e, res_n)), "shift_blocks_used": used,
              "shift_variation_m": variation, "stable_cells": int(stable.sum()),
              "stable_share": float(stable.mean())}
    for name in ("dsm", "dtm"):
        plane, rep = fit_plane(moved[name] - before[name], stable, cell)
        moved[name] = moved[name] - plane
        report[name] = rep
    report["warp_m"] = 0.0
    if warp_window_m > 0:
        diff = moved["dsm"] - before["dsm"]
        warp = io.local_ground(diff, cell, window_m=warp_window_m)
        moved["dsm"], moved["dtm"] = moved["dsm"] - warp, moved["dtm"] - warp
        # Size of the bend that was removed: 95% of the map was bent by less than this.
        report["warp_m"] = float(np.nanpercentile(np.abs(warp[np.isfinite(diff)]), 95)) if np.isfinite(diff).any() else 0.0
        report["dsm"]["rms_plane_only_m"] = report["dsm"]["rms_m"]
        report["dsm"]["rms_m"] = _stable_rms(moved["dsm"] - before["dsm"], stable)
    rms = report["dsm"]["rms_m"]
    warnings = []
    if not np.isfinite(rms):
        warnings.append("No stable open ground was found in both surveys, so heights could not be aligned.")
    elif rms > RMS_WARNING_M:
        warnings.append(f"Leftover height noise on stable ground is {rms:.2f} m (above {RMS_WARNING_M} m): "
                        "alignment may be too poor to detect small damage.")
    if report["residual_shift_m"] > SHIFT_WARNING_M:
        warnings.append(f"A sideways shift of {report['residual_shift_m']:.2f} m is left after alignment "
                        f"(above {SHIFT_WARNING_M} m): alignment may be too poor to detect small damage.")
    report["warnings"] = warnings
    report["ok"] = not warnings
    return moved, report


# ---------------------------------------------------------------------------
# 3. What changed, tree by tree
# ---------------------------------------------------------------------------
def default_change_threshold(rms_m: float) -> float:
    """A height drop counts as real if it is over 3 x the noise on stable ground (never under 0.5 m)."""
    return max(0.5, 3.0 * rms_m) if np.isfinite(rms_m) else 0.5


def _disk(radius_cells: int) -> np.ndarray:
    r = max(1, int(radius_cells))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy ** 2 + xx ** 2) <= r * r


def _per_tree(labels_flat, values, n):
    """Sum of `values` over the cells of each tree (index = tree number)."""
    return np.bincount(labels_flat, weights=np.asarray(values, dtype=float).ravel(), minlength=n + 1)


def detect_changes(trees: pd.DataFrame, crowns: np.ndarray, chm_before: np.ndarray, chm_after: np.ndarray, cell: float,
                   p: dict = None, alignment_ok: bool = True, color_before=None, color_after=None):
    """Measure the change inside every tree's BEFORE crown and classify it.

    For each tree:
      top_drop_m       = old tree-top height - highest after-storm canopy within 1.5 m of the old top
      crown_loss_frac  = share of crown cells whose canopy dropped by more than the change threshold
      volume_loss_frac = sum of the canopy drops / sum of the before canopy heights in the crown
    (The after canopy at each cell is taken as the highest of the cell and its 8
    neighbors, so being one cell off sideways is not read as a drop.)

    Classes, checked in this order:
      uncertain    : near the map edge, missing data in the after survey, or poor alignment
      fallen       : top_drop > 60% of height AND crown_loss > 70%
      snapped top  : top_drop > max(2 m, 20% of height)
      crown damage : crown_loss > 25%
      no change    : everything else
    Returns (table with one row per tree, info dict incl. the leaf-drop check).
    """
    p = {**CHANGE_DEFAULTS, **(p or {})}
    thr = p["change_threshold"] if p["change_threshold"] is not None else 0.5
    n = int(max(crowns.max(), trees["tree"].max()))
    ids = trees["tree"].to_numpy(dtype=int)
    lab = crowns.ravel()
    after_ok = np.isfinite(chm_after)
    before0, after0 = np.nan_to_num(chm_before), np.nan_to_num(chm_after)

    # Highest after-storm canopy near each old top.
    near_top = ndimage.maximum_filter(after0, footprint=_disk(round(p["top_radius"] / cell)), mode="nearest")
    rows, cols = trees["row"].to_numpy(dtype=int), trees["col"].to_numpy(dtype=int)
    height = trees["height"].to_numpy(dtype=float)
    top_drop = height - near_top[rows, cols]

    # Drop at every cell (never negative: growth is not damage).
    after_near = ndimage.maximum_filter(after0, size=3, mode="nearest")
    drop = np.where(after_ok, np.maximum(before0 - after_near, 0.0), 0.0)
    cells = np.bincount(lab, minlength=n + 1).astype(float)
    seen = _per_tree(lab, after_ok, n)
    lost = _per_tree(lab, (drop > thr) & after_ok, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        crown_loss = (lost / seen)[ids]
        volume_loss = (_per_tree(lab, drop, n) / _per_tree(lab, np.where(after_ok, before0, 0.0), n))[ids]

    # Missing after-survey data in or just around the crown (crown grown by the buffer).
    from skimage.segmentation import expand_labels

    grown = expand_labels(crowns, distance=max(1, round(p["buffer_m"] / cell))).ravel()
    grown_cells = np.bincount(grown, minlength=n + 1).astype(float)
    missing = 1 - _per_tree(grown, after_ok, n) / np.maximum(grown_cells, 1)
    edge = p["edge_m"] / cell
    near_edge = ((rows < edge) | (cols < edge) | (rows >= crowns.shape[0] - edge) | (cols >= crowns.shape[1] - edge))
    uncertain = near_edge | (missing[ids] > 0.10) | (cells[ids] == 0) | ~np.isfinite(crown_loss) | (not alignment_ok)

    crown_loss_c = np.nan_to_num(crown_loss)
    fallen = (top_drop > p["fallen_top"] * height) & ((crown_loss_c > p["fallen_crown"]) | p["top_only"])
    snapped = top_drop > np.maximum(p["snap_min_m"], p["snap_frac"] * height)
    crown = (crown_loss_c > p["crown_frac"]) & (not p["top_only"])
    cls = np.where(uncertain, "uncertain", np.where(fallen, "fallen", np.where(snapped, "snapped top",
                   np.where(crown, "crown damage", "no change"))))

    out = pd.DataFrame({"tree": ids, "damage": cls, "top_drop_m": top_drop, "crown_loss_frac": crown_loss,
                        "volume_loss_frac": volume_loss, "after_missing_frac": missing[ids], "near_edge": near_edge})
    if color_before is not None and color_after is not None:
        # Drop in greenness (VARI, ExG) inside the crown. Reported only: seasons change it too.
        for key in color_before:
            mean_b = _per_tree(lab, np.nan_to_num(color_before[key]), n) / np.maximum(cells, 1)
            mean_a = _per_tree(lab, np.nan_to_num(color_after[key]), n) / np.maximum(cells, 1)
            out[f"{key}_drop"] = (mean_b - mean_a)[ids]

    # Leaf-drop check: a storm damages a minority of trees. If the TYPICAL tree lost
    # over 10% of its crown, the whole canopy thinned, which is what leaf fall looks like.
    judged = ~uncertain
    median_loss = float(np.nanmedian(crown_loss[judged])) if judged.any() else float("nan")
    info = {"change_threshold_m": float(thr), "median_crown_loss": median_loss,
            "leaf_drop_suspected": bool(np.isfinite(median_loss) and median_loss > 0.10), "top_only": bool(p["top_only"])}
    return out, info


def log_ends(logs: pd.DataFrame) -> np.ndarray:
    """Both ends of every log as an array [log, end (0 or 1), (x, y)]."""
    a = np.radians(logs["axis_deg"].to_numpy(dtype=float))
    half = logs["length_m"].to_numpy(dtype=float) / 2
    dx, dy = half * np.sin(a), half * np.cos(a)
    x, y = logs["x"].to_numpy(dtype=float), logs["y"].to_numpy(dtype=float)
    return np.stack([np.column_stack([x - dx, y - dy]), np.column_stack([x + dx, y + dy])], axis=1)


def axis_difference(a, b):
    """Angle between two axes (0-90): an axis at 10 degrees and one at 190 are the same."""
    d = np.abs((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) % 180)
    return np.minimum(d, 180 - d)


def find_new_logs(logs_before: pd.DataFrame, logs_after: pd.DataFrame, p: dict = None) -> pd.DataFrame:
    """Logs in the after survey that were not there before.

    A log is "already there" if a before-log has its center within 2 m and its
    direction within 20 degrees. Everything else is a new log.
    """
    p = {**CHANGE_DEFAULTS, **(p or {})}
    new = logs_after.copy()
    if len(logs_before) and len(new):
        dist, idx = cKDTree(logs_before[["x", "y"]].to_numpy()).query(new[["x", "y"]].to_numpy())
        same_axis = axis_difference(new["axis_deg"], logs_before["axis_deg"].to_numpy()[idx]) <= p["log_match_deg"]
        new = new[~((dist <= p["log_match_m"]) & same_axis)]
    return new.reset_index(drop=True)


def link_logs_to_trees(changes: pd.DataFrame, trees: pd.DataFrame, new_logs: pd.DataFrame, p: dict = None):
    """Match new logs to fallen trees and work out which way each tree fell.

    A new log belongs to a fallen tree if one of its ends lies within 1.5 x the
    crown radius of the tree's old trunk. That end is the ROOT end, so the fall
    direction is fully known: the compass direction from the root end to the
    other end. (A log on its own only has an axis: 120 or 300 degrees.)
    If a tree has several such logs, the longest is used.
    Returns (changes with observed_fall_deg / observed_log_m / log columns, new_logs with a "tree" column).
    """
    p = {**CHANGE_DEFAULTS, **(p or {})}
    changes = changes.copy()
    changes["observed_fall_deg"] = np.nan
    changes["observed_log_m"] = np.nan
    changes["log"] = np.nan
    new_logs = new_logs.copy()
    new_logs["tree"] = np.nan
    new_logs["root_x"], new_logs["root_y"], new_logs["fall_deg"] = np.nan, np.nan, np.nan
    fallen = changes.index[changes["damage"] == "fallen"]
    if not len(fallen) or not len(new_logs):
        return changes, new_logs
    info = trees.set_index("tree")
    ends = log_ends(new_logs)
    best = {}
    for i in fallen:
        tid = changes.at[i, "tree"]
        trunk = info.loc[tid, ["x", "y"]].to_numpy(dtype=float)
        reach = p["link_factor"] * info.at[tid, "crown_diam"] / 2
        d = np.hypot(*(ends - trunk).transpose(2, 0, 1))  # distance of both ends of every log to this trunk
        for j in np.flatnonzero(d.min(axis=1) <= reach):
            # Each log goes to the fallen tree whose trunk its end is closest to.
            if j not in best or d[j].min() < best[j][1]:
                best[j] = (i, d[j].min(), int(np.argmin(d[j])))
    for j, (i, _, root) in best.items():
        length = new_logs.at[j, "length_m"]
        if np.isfinite(changes.at[i, "observed_log_m"]) and changes.at[i, "observed_log_m"] >= length:
            continue
        far = ends[j, 1 - root] - ends[j, root]
        azimuth = float(np.degrees(np.arctan2(far[0], far[1])) % 360)  # compass: atan2(east, north)
        changes.at[i, "observed_fall_deg"], changes.at[i, "observed_log_m"] = azimuth, length
        changes.at[i, "log"] = new_logs.at[j, "log"]
        new_logs.at[j, "tree"] = changes.at[i, "tree"]
        new_logs.at[j, "root_x"], new_logs.at[j, "root_y"] = ends[j, root]
        new_logs.at[j, "fall_deg"] = azimuth
    return changes, new_logs


def unexplained_gaps(crowns, chm_before, chm_after, transform, cell, threshold, min_height, min_area) -> pd.DataFrame:
    """New canopy gaps that no tree in the before table explains.

    Canopy that was tree-high before, dropped by more than the threshold, and
    is outside every before crown (for example a tree that came down from
    outside the mapped trees). Pieces smaller than `min_area` are ignored.
    """
    drop = np.nan_to_num(chm_before) - np.nan_to_num(chm_after)
    gap = (crowns == 0) & (np.nan_to_num(chm_before) >= min_height) & (drop > threshold) & np.isfinite(chm_after)
    labels, n = ndimage.label(gap, structure=np.ones((3, 3)))
    if n == 0:
        return pd.DataFrame(columns=["gap", "x", "y", "area_m2", "mean_drop_m"])
    idx = np.arange(1, n + 1)
    area = ndimage.sum(gap, labels, idx) * cell * cell
    keep = area >= min_area
    centers = np.array(ndimage.center_of_mass(gap, labels, idx))[keep]
    t = transform
    return pd.DataFrame({"gap": np.arange(1, keep.sum() + 1),
                         "x": t.c + (centers[:, 1] + 0.5) * t.a if keep.any() else [],
                         "y": t.f + (centers[:, 0] + 0.5) * t.e if keep.any() else [],
                         "area_m2": area[keep], "mean_drop_m": ndimage.mean(drop, labels, idx)[keep]})


def compare_surveys(before: dict, after: dict, trees: pd.DataFrame, crowns: np.ndarray, transform, cell: float,
                    settings: dict, p: dict = None, building: np.ndarray = None, seasons=(None, None),
                    trust_alignment: bool = False) -> dict:
    """Everything in steps 2 and 3 for one pair of surveys.

    before / after: {"dsm", "dtm", "ortho"} on the before grid. `trees`,
    `crowns` and `settings` come from the FROZEN predictions.
    Both canopy heights use the BEFORE ground model: the ground doesn't move in
    a storm, and under trees each flight's ground model has its own errors,
    which would otherwise look like canopy change.
    """
    from terrain import vegetation

    p = {**CHANGE_DEFAULTS, **(p or {})}
    aligned, report = align_surveys(before, after, cell, building)
    chm_b = _chm(before["dsm"], before["dtm"])
    chm_a = _chm(aligned["dsm"], before["dtm"])
    if p["change_threshold"] is None:
        p["change_threshold"] = default_change_threshold(report["dsm"]["rms_m"])

    colors = (None, None)
    if before.get("ortho") is not None and aligned.get("ortho") is not None:
        colors = tuple({"vari": vegetation.vari(o), "exg": vegetation.exg(o)} for o in (before["ortho"], aligned["ortho"]))
    changes, info = detect_changes(trees, crowns, chm_b, chm_a, cell, p, report["ok"] or trust_alignment, *colors)

    log_p = [settings.get(k, d) for k, d in (("log_low", 0.2), ("log_high", 1.5), ("log_length", 4.0), ("log_width", 1.0))]
    logs_b, _ = treefall.detect_logs(chm_b, transform, cell, *log_p)
    logs_a, _ = treefall.detect_logs(chm_a, transform, cell, *log_p)
    new_logs = find_new_logs(logs_b, logs_a, p)
    changes, new_logs = link_logs_to_trees(changes, trees, new_logs, p)
    gaps = unexplained_gaps(crowns, chm_b, chm_a, transform, cell, p["change_threshold"],
                            settings.get("min_height", 5.0), p["gap_area"])

    info["seasons"] = list(seasons)
    info["seasons_differ"] = bool(seasons[0] and seasons[1] and seasons[0] != seasons[1])
    warnings = list(report["warnings"])
    if info["seasons_differ"] or info["leaf_drop_suspected"]:
        warnings.append("Many crowns thinned everywhere; this looks like leaf drop, not storm damage. "
                        "Consider classifying with the top drop and new logs only.")
    return {"alignment": report, "changes": changes, "info": info, "new_logs": new_logs, "logs_before": logs_b,
            "gaps": gaps, "chm_before": chm_b, "chm_after": chm_a, "after_ortho": aligned.get("ortho"),
            "thresholds": p, "warnings": warnings}


# ---------------------------------------------------------------------------
# 4. Null test (calm pair)
# ---------------------------------------------------------------------------
def change_counts(changes: pd.DataFrame) -> dict:
    """How many trees are in each damage class (uncertain trees are not "compared")."""
    counts = {c: int((changes["damage"] == c).sum()) for c in DAMAGE_CLASSES}
    counts["compared"] = int(len(changes) - counts["uncertain"])
    counts["any_damage"] = sum(counts[c] for c in ANY_DAMAGE)
    counts["severe"] = sum(counts[c] for c in SEVERE)
    return counts


def calm_comparison(storm_counts: dict, calm_counts: dict) -> pd.DataFrame:
    """The storm's change rate next to the calm pair's false change rate.

    With no storm, every "changed" tree is a false alarm of the method. Fisher's
    exact test says how likely the storm's higher rate would be if the storm had
    really done nothing (a small p = the storm damage is real, not noise).
    """
    rows = []
    for label, key in [("fallen", "fallen"), ("snapped top", "snapped top"), ("crown damage", "crown damage"),
                       ("severe (fallen or snapped)", "severe"), ("any damage", "any_damage")]:
        s, sn = storm_counts[key], storm_counts["compared"]
        c, cn = calm_counts[key], calm_counts["compared"]
        rows.append({"class": label, "storm": f"{s} of {sn}", "storm rate (%)": 100 * s / sn if sn else np.nan,
                     "calm pair (false changes)": f"{c} of {cn}", "calm rate (%)": 100 * c / cn if cn else np.nan,
                     "Fisher p": validation.fisher_2x2(s, sn - s, c, cn - c) if sn and cn else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Did the system predict it?
# ---------------------------------------------------------------------------
def join_outcomes(frozen: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """The frozen tree table with the observed damage added (uncertain trees kept but marked)."""
    table = frozen.merge(changes, on="tree", how="inner")
    table["judged"] = table["damage"] != "uncertain"
    table["any_damage"] = table["damage"].isin(ANY_DAMAGE)
    table["severe"] = table["damage"].isin(SEVERE)
    return table


def score_columns(table: pd.DataFrame) -> dict:
    """Every ranking that is tested: the full score, the baselines, and each factor alone."""
    cols = {"full hazard score": ("hazard", True)}
    for name, label in [("baseline_height", "baseline: tallest first"),
                        ("baseline_height_target", "baseline: tallest that can reach a target"),
                        ("baseline_random", "baseline: random order")]:
        if name in table:
            cols[label] = (name, False)  # rank columns: 1 = most hazardous
    for name in treefall.FACTORS:
        if f"f_{name}" in table and table[f"f_{name}"].notna().any():
            cols[f"factor: {name}"] = (f"f_{name}", True)
    return cols


def prediction_metrics(table: pd.DataFrame, peak_gust=None, n_boot: int = 2000, seed: int = 42) -> dict:
    """Hazard score vs damage, using only frozen scores. Uncertain trees are left out."""
    t = table[table["judged"]].reset_index(drop=True)
    out = {"trees": int(len(t)), "outcomes": {}}
    if "hazard_rank" not in t:
        t["hazard_rank"] = validation._rank_from_score(t["hazard"])
    gust = None
    if peak_gust is not None:
        gust = t[peak_gust].to_numpy(dtype=float) if isinstance(peak_gust, str) else np.full(len(t), float(peak_gust))
    for key in ("severe", "any_damage"):
        y = t[key].to_numpy(dtype=bool)
        aucs = validation.auc_table(t, y, score_columns(t), n_boot, seed)
        rates = validation.rate_by_class(t["class"], y)
        out["outcomes"][key] = {
            "damaged": int(y.sum()),
            # Fewer than 10 damaged trees: percentages and p-values mean little; show the counts.
            "small_sample": bool(y.sum() < 10),
            "auc": aucs, "rates": rates, "ratio": validation.high_low_ratio(rates),
            "top_n": validation.top_n_hits(t["hazard_rank"], y),
            "odds": validation.hazard_odds_ratio(t["hazard"], y, gust),
        }
    return out


def angle_error(observed, predicted):
    """Smallest angle between two compass directions (0-180 degrees)."""
    d = np.abs((np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)) % 360)
    return np.minimum(d, 360 - d)


def direction_log_likelihood(observed_deg, probs: np.ndarray) -> tuple:
    """How well the 36-direction forecast fits the directions trees really fell.

    Each tree's forecast gives a chance to each 10-degree direction. The
    log-likelihood adds up ln(chance given to the direction the tree actually
    fell). Guessing (every direction 1/36) scores n x ln(1/36). A forecast that
    puts more chance on the true directions scores HIGHER (closer to 0).
    Returns (model log-likelihood, uniform log-likelihood).
    """
    observed = np.asarray(observed_deg, dtype=float)
    bins = ((observed + treefall.BIN_DEG / 2) // treefall.BIN_DEG).astype(int) % treefall.N_BINS
    chance = np.maximum(probs[np.arange(len(observed)), bins], 1e-9)
    return float(np.log(chance).sum()), float(len(observed) * np.log(1 / treefall.N_BINS))


def direction_metrics(table: pd.DataFrame, probs_by_label: dict, wind_from_deg=None) -> dict:
    """Predicted vs observed fall direction for fallen trees with a linked log.

    table: joined table (frozen + outcomes); it must be in the same row order
    as the probability arrays in probs_by_label {label: (frozen trees, probs)}.
    Models compared: both frozen forecasts, and "downhill only", "lean only",
    "wind only" (the storm's actual wind) and a uniform guess (which is 90
    degrees off on average).
    """
    known = table[table["observed_fall_deg"].notna() & table["judged"]]
    out = {"trees": int(len(known)), "table": pd.DataFrame(), "per_tree": pd.DataFrame(), "likelihood": pd.DataFrame()}
    if known.empty:
        return out
    obs = known["observed_fall_deg"].to_numpy(dtype=float)
    per_tree = pd.DataFrame({"tree": known["tree"].to_numpy(), "observed_fall_deg": obs})
    rows, like = [], []
    for label, (frozen, probs) in probs_by_label.items():
        pos = pd.Series(np.arange(len(frozen)), index=frozen["tree"].to_numpy())[known["tree"].to_numpy()].to_numpy()
        predicted = treefall.BIN_CENTERS[np.argmax(probs[pos], axis=1)]
        err = angle_error(obs, predicted)
        per_tree[f"predicted ({label})"], per_tree[f"error ({label})"] = predicted, err
        rows.append({"model": label, "trees": len(err), "median error (°)": float(np.median(err)),
                     "within 45° (%)": float(100 * np.mean(err <= 45))})
        ll, ll_uniform = direction_log_likelihood(obs, probs[pos])
        like.append({"model": label, "log-likelihood": ll, "uniform guess": ll_uniform,
                     "better than guessing": bool(ll > ll_uniform)})
    simple = {"downhill only": known.get("downhill_dir"), "lean only": known.get("lean_dir")}
    if wind_from_deg is not None and np.isfinite(wind_from_deg):
        simple["wind only (actual wind)"] = pd.Series((float(wind_from_deg) + 180) % 360, index=known.index)
    for label, predicted in simple.items():
        if predicted is None:
            continue
        ok = predicted.notna().to_numpy()
        if ok.any():
            err = angle_error(obs[ok], predicted.to_numpy(dtype=float)[ok])
            rows.append({"model": label, "trees": int(ok.sum()), "median error (°)": float(np.median(err)),
                         "within 45° (%)": float(100 * np.mean(err <= 45))})
    rows.append({"model": "uniform guess (expected)", "trees": len(obs), "median error (°)": 90.0, "within 45° (%)": 25.0})
    out.update(table=pd.DataFrame(rows), per_tree=per_tree, likelihood=pd.DataFrame(like))
    return out


def landing_metrics(table: pd.DataFrame, new_logs: pd.DataFrame, grids: dict, chm_before: np.ndarray, transform,
                    cell: float, settings: dict, seed: int = 42) -> dict:
    """Landing strips, the exposure map and features, against the new logs.

    - For each fallen tree with a log: did the log's far end stay within the
      tree's reach (height x safety factor)? Did it lie inside the landing strip
      of the single most likely direction (within half a crown width of that line)?
    - Is the FROZEN fall exposure higher under new logs than at random forest
      points (Mann-Whitney U test)?
    - Did a new log land on a detected trail, road, stream or building, and had
      the tree it came from been flagged as able to reach that feature?
    """
    out = {"strip": {}, "exposure": {}, "features": pd.DataFrame()}
    known = table[table["observed_fall_deg"].notna()]
    if len(known):
        reach = known["height"].to_numpy(dtype=float) * settings.get("safety", 1.0)
        length = known["observed_log_m"].to_numpy(dtype=float)
        off = np.radians(angle_error(known["observed_fall_deg"], known["fall_dir"]))
        # Sideways distance of the log's far end from the predicted line = length x sin(angle off).
        in_strip = (length <= reach) & (off < np.pi / 2) & (length * np.sin(off) <= known["crown_diam"].to_numpy() / 2)
        out["strip"] = {"trees": int(len(known)), "within reach (any direction)": int((length <= reach).sum()),
                        "inside the most likely strip": int(in_strip.sum())}

    def cell_of(x, y):
        r = np.clip(((np.asarray(y) - transform.f) / transform.e).astype(int), 0, chm_before.shape[0] - 1)
        c = np.clip(((np.asarray(x) - transform.c) / transform.a).astype(int), 0, chm_before.shape[1] - 1)
        return r, c

    if len(new_logs) and "exposure" in grids:
        k = int(grids.get("grid_factor", 1))
        r, c = cell_of(new_logs["x"], new_logs["y"])
        exposure = grids["exposure"]
        at_logs = exposure[np.clip(r // k, 0, exposure.shape[0] - 1), np.clip(c // k, 0, exposure.shape[1] - 1)]
        forest = np.argwhere(np.nan_to_num(chm_before) >= settings.get("min_height", 5.0))
        rng = np.random.default_rng(seed)
        pick = forest[rng.choice(len(forest), size=min(1000, len(forest)), replace=False)] if len(forest) else forest
        at_random = exposure[np.clip(pick[:, 0] // k, 0, exposure.shape[0] - 1),
                             np.clip(pick[:, 1] // k, 0, exposure.shape[1] - 1)] if len(pick) else np.zeros(0)
        p_value = float("nan")
        if len(at_logs) >= 3 and len(at_random) >= 3:
            p_value = float(stats.mannwhitneyu(at_logs, at_random, alternative="greater").pvalue)
        out["exposure"] = {"new logs": int(len(at_logs)), "mean exposure under new logs": float(np.mean(at_logs)),
                           "mean exposure at random forest points": float(np.mean(at_random)) if len(at_random) else np.nan,
                           "Mann-Whitney p (logs higher)": p_value}

    masks = {k[5:]: v.astype(bool) for k, v in grids.items() if k.startswith("mask_")}
    rows = []
    if len(new_logs) and masks:
        ends = log_ends(new_logs)
        frozen = table.set_index("tree")
        for j in range(len(new_logs)):
            steps = np.linspace(0, 1, 21)[:, None]
            r, c = cell_of(*(ends[j, 0] + steps * (ends[j, 1] - ends[j, 0])).T)  # 21 points along the log
            tree = new_logs["tree"].iloc[j] if "tree" in new_logs else np.nan
            for name, mask in masks.items():
                if name in ("clearing",) or not mask[r, c].any():
                    continue
                flagged = None
                if np.isfinite(tree) and f"can reach {name}" in frozen.columns:
                    flagged = bool(frozen.at[tree, f"can reach {name}"])
                rows.append({"log": new_logs["log"].iloc[j], "landed on": name, "from tree": tree,
                             "tree had been flagged for it": flagged})
    out["features"] = pd.DataFrame(rows, columns=["log", "landed on", "from tree", "tree had been flagged for it"])
    return out


# ---------------------------------------------------------------------------
# 5d. Saving results (one storm, and all storms pooled)
# ---------------------------------------------------------------------------
def _records(df: pd.DataFrame) -> list:
    return _plain(df.replace({np.nan: None}).to_dict(orient="records")) if df is not None and len(df) else []


def save_storm_results(out_dir, storm: dict, comparison: dict, table: pd.DataFrame, prediction: dict, direction: dict,
                       landing: dict, frozen_status: dict) -> dict:
    """Write storm_results.json, storm_trees.csv and new_logs.csv for one storm."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = change_counts(comparison["changes"])
    keep = [c for c in ["tree", "x", "y", "latitude", "longitude", "height", "crown_diam", "hazard", "hazard_rank", "class",
                        "reasons", "baseline_height", "baseline_height_target", "baseline_random", "fall_dir",
                        "downhill_dir", "lean_dir", "damage", "judged", "severe", "any_damage", "top_drop_m",
                        "crown_loss_frac", "volume_loss_frac", "observed_fall_deg", "observed_log_m"]
            + [f"f_{n}" for n in treefall.FACTORS] if c in table]
    table[keep].to_csv(out_dir / "storm_trees.csv", index=False)
    comparison["new_logs"].to_csv(out_dir / "new_logs.csv", index=False)
    if len(direction.get("per_tree", [])):
        direction["per_tree"].to_csv(out_dir / "fall_directions.csv", index=False)

    outcomes = {}
    for key, o in prediction["outcomes"].items():
        outcomes[key] = {"damaged": o["damaged"], "small_sample": o["small_sample"], "auc": _records(o["auc"]),
                         "rates": _records(o["rates"]), "ratio": _plain(o["ratio"]), "top_n": _records(o["top_n"]),
                         "odds": _plain(o["odds"])}
    results = {
        "storm": _plain(storm), "analyzed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen": _plain(frozen_status), "alignment": _plain(comparison["alignment"]), "info": _plain(comparison["info"]),
        "thresholds": _plain(comparison["thresholds"]), "counts": counts, "new_logs": int(len(comparison["new_logs"])),
        "unexplained_gaps": int(len(comparison["gaps"])), "warnings": comparison["warnings"],
        "prediction": {"trees": prediction["trees"], "outcomes": outcomes},
        "direction": {"trees": direction["trees"], "table": _records(direction["table"]),
                      "likelihood": _records(direction["likelihood"])},
        "landing": {"strip": _plain(landing["strip"]), "exposure": _plain(landing["exposure"]),
                    "features": _records(landing["features"])},
    }
    results["sentences"] = storm_sentences(results)
    (out_dir / "storm_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def storm_sentences(r: dict) -> list:
    """Plain sentences about one storm for results/summary.md."""
    c, a = r["counts"], r["alignment"]
    out = [f"Predictions frozen {r['frozen'].get('frozen_at_utc', '?')} UTC (fingerprint "
           f"{str(r['frozen'].get('combined', ''))[:16]}...); frozen before the storm: {r['frozen'].get('frozen_before_storm')}.",
           f"Alignment: sideways shift {a['shift_m']:.2f} m, leftover height noise on stable ground {a['dsm']['rms_m']:.2f} m.",
           f"Of {c['compared']} trees compared: {c['fallen']} fallen, {c['snapped top']} snapped top, "
           f"{c['crown damage']} crown damage; {r['new_logs']} new logs."]
    if r["storm"].get("is_calm_control"):
        rate = c["any_damage"] / c["compared"] if c["compared"] else float("nan")
        out.append(f"Calm pair: {c['any_damage']} of {c['compared']} trees falsely flagged ({100 * rate:.2f}%).")
        return out
    severe = r["prediction"]["outcomes"]["severe"]
    full = next((x for x in severe["auc"] if x["ranking"] == "full hazard score"), None)
    if full and full["AUC"] is not None:
        note = " (fewer than 10 damaged trees: a count, not a statistic)" if severe["small_sample"] else ""
        out.append(f"Frozen hazard score vs severe damage: AUC {full['AUC']:.2f} (95% range {full['low 95%']:.2f} to "
                   f"{full['high 95%']:.2f}){note}.")
    for row in r["direction"]["table"][:2]:
        out.append(f"Fall direction, {row['model']}: median error {row['median error (°)']:.0f}° over {row['trees']} trees.")
    strip, exposure = r["landing"]["strip"], r["landing"]["exposure"]
    if strip:
        out.append(f"Landing zone: {strip['within reach (any direction)']} of {strip['trees']} logs ended within the "
                   f"predicted reach; {strip['inside the most likely strip']} inside the most likely strip.")
    if exposure:
        out.append(f"Frozen fall exposure under new logs: {exposure['mean exposure under new logs']:.3f} vs "
                   f"{exposure['mean exposure at random forest points']:.3f} at random forest points.")
    return out


def pool_results(storms_dir, storms: list, n_boot: int = 2000, seed: int = 42) -> dict:
    """Combine every analyzed storm into results/storms/pooled_results.json.

    Storms are pooled by stacking their tree tables (each tree with its FROZEN
    score and its outcome in that storm). Calm pairs are pooled separately to
    give the false change rate. The "criteria" block feeds the Design criteria table.
    """
    storms_dir = Path(storms_dir)
    tables, calm, errors, used = [], {"changed": 0, "compared": 0}, [], []
    for s in storms:
        folder = storms_dir / storm_slug(s["name"])
        if not (folder / "storm_results.json").exists():
            continue
        results = json.loads((folder / "storm_results.json").read_text(encoding="utf-8"))
        if s.get("is_calm_control"):
            calm["changed"] += results["counts"]["any_damage"]
            calm["compared"] += results["counts"]["compared"]
            continue
        t = pd.read_csv(folder / "storm_trees.csv")
        t["storm"], t["peak_gust_mps"] = s["name"], s.get("peak_gust_mps") or np.nan
        tables.append(t)
        used.append(s["name"])
        if (folder / "fall_directions.csv").exists():
            d = pd.read_csv(folder / "fall_directions.csv")
            # Prefer the forecast that used the real wind, if it was made.
            col = next((c for c in (f"error ({ACTUAL})", f"error ({PREVAILING})") if c in d), None)
            if col:
                errors += d[col].dropna().tolist()

    criteria = {"storm_auc_severe": None, "storm_beats_baseline": None, "fall_direction_mae_deg": None,
                "calm_false_change_rate": calm["changed"] / calm["compared"] if calm["compared"] else None}
    pooled = {"storms": used, "calm": calm, "criteria": criteria, "prediction": None, "direction_errors": len(errors)}
    if tables:
        t = pd.concat(tables, ignore_index=True)
        t = t[t["judged"]].reset_index(drop=True)
        gust = "peak_gust_mps" if t["peak_gust_mps"].notna().all() else None
        prediction = prediction_metrics(t.assign(hazard_rank=validation._rank_from_score(t["hazard"])), gust, n_boot, seed)
        severe = prediction["outcomes"]["severe"]
        aucs = severe["auc"].set_index("ranking")["AUC"]
        full, base = aucs.get("full hazard score", np.nan), aucs.get("baseline: tallest first", np.nan)
        if np.isfinite(full):
            criteria["storm_auc_severe"] = float(full)
            criteria["storm_beats_baseline"] = bool(full > base) if np.isfinite(base) else None
        pooled["prediction"] = {"trees": prediction["trees"], "outcomes": {
            k: {"damaged": o["damaged"], "small_sample": o["small_sample"], "auc": _records(o["auc"]),
                "rates": _records(o["rates"]), "ratio": _plain(o["ratio"]), "top_n": _records(o["top_n"]),
                "odds": _plain(o["odds"])} for k, o in prediction["outcomes"].items()}}
    if errors:
        criteria["fall_direction_mae_deg"] = float(np.median(errors))
    sentences = []
    if criteria["storm_auc_severe"] is not None:
        sentences.append(f"{len(used)} storm(s) pooled: AUC of the frozen hazard score for severe damage "
                         f"{criteria['storm_auc_severe']:.2f}; beats 'tallest first': {criteria['storm_beats_baseline']}.")
    if criteria["fall_direction_mae_deg"] is not None:
        sentences.append(f"Median fall direction error {criteria['fall_direction_mae_deg']:.0f}° over {len(errors)} trees.")
    if criteria["calm_false_change_rate"] is not None:
        sentences.append(f"Calm pairs: {calm['changed']} of {calm['compared']} trees falsely flagged "
                         f"({100 * criteria['calm_false_change_rate']:.2f}%).")
    pooled["sentences"] = sentences
    storms_dir.mkdir(parents=True, exist_ok=True)
    (storms_dir / "pooled_results.json").write_text(json.dumps(_plain(pooled), indent=2), encoding="utf-8")
    return pooled


# ---------------------------------------------------------------------------
# 6. Field confirmation (small and blind)
# ---------------------------------------------------------------------------
FIELD_COLUMNS = ["status", "storm_damage", "fall_azimuth_deg", "notes"]
FIELD_DAMAGE = {"none": "no change", "branches": "crown damage", "top broken": "snapped top", "fallen": "fallen",
                "uprooted": "fallen"}


def field_sheet(table: pd.DataFrame, n_max: int = 30, seed: int = 42):
    """A blind check sheet: every fallen or snapped tree (up to 30) mixed with the
    same number of random "no change" trees, shuffled, each with a random 4-digit code.

    Returns (sheet, key). The SHEET goes to the woods and says nothing about
    what the app found. The KEY stays home and maps each code back to the tree.
    """
    rng = np.random.default_rng(seed)
    damaged = table[table["damage"].isin(SEVERE)]
    if len(damaged) > n_max:
        damaged = damaged.iloc[np.sort(rng.choice(len(damaged), n_max, replace=False))]
    pool = table[table["damage"] == "no change"]
    control = pool.iloc[np.sort(rng.choice(len(pool), min(len(damaged), len(pool)), replace=False))] if len(pool) else pool
    both = pd.concat([damaged, control]).iloc[rng.permutation(len(damaged) + len(control))].reset_index(drop=True)
    both["code"] = rng.choice(np.arange(1000, 10000), size=len(both), replace=False)
    sheet = both[["code"] + [c for c in ("latitude", "longitude") if c in both]].copy()
    for col in FIELD_COLUMNS:
        sheet[col] = ""
    key_cols = ["code", "tree", "damage", "hazard", "height", "x", "y", "fall_dir", "observed_fall_deg"]
    return sheet, both[[c for c in key_cols if c in both]].copy()


def field_compare(key: pd.DataFrame, filled: pd.DataFrame) -> dict:
    """Compare the filled-in sheet with the app's change detection.

    storm_damage is one of: none / branches / top broken / fallen / uprooted.
    For each class the app assigned:
      precision = of the trees the app put in the class, the share the field agreed with
      recall    = of the trees the field put in the class, the share the app had found
    Field fall directions (phone compass, root -> top) are kept next to the app's.
    """
    filled = filled.copy()
    filled.columns = [str(c).strip().lower() for c in filled.columns]
    t = key.merge(filled, on="code", how="inner")
    t = t[t["status"].astype(str).str.strip().str.lower() == "found"].copy()
    t["field_damage"] = t["storm_damage"].astype(str).str.strip().str.lower().map(FIELD_DAMAGE)
    t = t.dropna(subset=["field_damage"])
    rows = []
    for cls in ["fallen", "snapped top", "no change"]:
        app, field = t["damage"] == cls, t["field_damage"] == cls
        both = int((app & field).sum())
        rows.append({"class": cls, "app says": int(app.sum()), "field says": int(field.sum()), "both agree": both,
                     "precision (%)": 100 * both / app.sum() if app.sum() else np.nan,
                     "recall (%)": 100 * both / field.sum() if field.sum() else np.nan})
    t["field_fall_deg"] = pd.to_numeric(t.get("fall_azimuth_deg"), errors="coerce")
    # Where the field gives a direction it replaces the app's; both are kept.
    t["best_fall_deg"] = t["field_fall_deg"].fillna(t.get("observed_fall_deg"))
    return {"checked": int(len(t)), "table": pd.DataFrame(rows), "trees": t}
