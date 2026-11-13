"""Prepare a survey (DTM + DSM + photo + water flow on one grid) and remember it on disk.

Preparing a big forest survey takes 10-30 seconds: reading the files, putting
them on one grid, repairing pits and working out where water flows. The result
only depends on the files and a few settings, so it is saved in a cache folder.
The next time the same survey is opened with the same settings (even after
restarting TIMBER) it loads in about a second.

The saved copy is recognized by a "fingerprint": the names, sizes and change
times of the survey's files plus the settings. If a file is replaced or a
setting changes, the fingerprint changes and the survey is prepared again.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from terrain import hydrology, io

CACHE_VERSION = 1  # bump when the preparation steps change, so old saved copies are ignored
KEEP_CACHED = 8  # keep this many prepared surveys; the oldest ones are deleted


@dataclass
class Survey:
    """One survey, ready for analysis. All grids share the DTM's cells."""

    name: str
    dtm: io.Raster  # ground heights on the analysis grid (pits repaired, tilt removed if asked)
    dsm: io.Raster | None  # surface heights (ground + trees + buildings), or None
    ortho: np.ndarray | None  # photo colors, rows x cols x 3 from 0 to 1 (NaN = no photo), or None
    flow: dict  # water flow results (see hydrology.analyze_flow)
    files: dict  # {"dtm": Path, "dsm": Path or None, "ortho": Path or None}
    repaired: float = 0.0  # share of the map changed by the pit repair (0-1)
    tilt: dict | None = None  # the tilt that was removed, if "remove best-fit tilt" was on
    from_cache: bool = False
    info: dict = field(default_factory=dict)


# Notes about one flight, kept in survey.json inside the survey's folder.
SEASONS = ["summer leaf-on", "autumn color", "leaf-off"]
INFO_TEMPLATE = {"date": "", "time": "", "season": SEASONS[0], "flight_altitude_m": None, "overlap_percent": None,
                 "notes": ""}


def read_info(folder) -> dict:
    """The flight notes in <survey folder>/survey.json (the blank template if there is none).

    The date is guessed from a folder name like forest_2026-11-01 when it isn't filled in.
    """
    path = Path(folder) / "survey.json"
    info = dict(INFO_TEMPLATE)
    if path.exists():
        try:
            info.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    tail = Path(folder).name[-10:]
    if not info["date"] and len(tail) == 10 and tail[4] == "-" and tail[7] == "-" and tail.replace("-", "").isdigit():
        info["date"] = tail
    return info


def write_info(folder, info: dict):
    path = Path(folder) / "survey.json"
    path.write_text(json.dumps({**INFO_TEMPLATE, **info}, indent=2), encoding="utf-8")


def find_files(folder: Path) -> dict:
    """The survey's DTM, DSM and orthophoto, found by name (WebODM names them ...-dtm.tif etc.)."""
    tifs = sorted(folder.glob("*.tif*"))

    def first(word):
        return next((p for p in tifs if word in p.name.lower()), None)

    return {"dtm": first("dtm") or (tifs[0] if tifs else None), "dsm": first("dsm"), "ortho": first("ortho")}


def fingerprint(files: dict, cell_size, pit_depth, detrend) -> str:
    """A short code that changes whenever a file or a setting changes."""
    parts = [CACHE_VERSION, cell_size, pit_depth, bool(detrend)]
    for kind in ("dtm", "dsm", "ortho"):
        p = files.get(kind)
        if p is not None:
            stat = Path(p).stat()
            parts += [kind, Path(p).name, stat.st_size, stat.st_mtime_ns]
    return hashlib.md5(json.dumps(parts, default=str).encode()).hexdigest()[:16]


def prepare(folder, cell_size=None, pit_depth=10.0, detrend=False, cache_dir=None) -> Survey:
    """Load and prepare a survey folder, using the saved copy in `cache_dir` if there is one.

    cell_size: analysis cell in meters (None = the DTM's own cells).
    pit_depth: repair ground-model pits deeper than this (m); None = no repair.
    detrend: remove the best-fit tilt of the whole map first.
    """
    folder = Path(folder)
    files = find_files(folder)
    if files["dtm"] is None:
        raise FileNotFoundError(f"No GeoTIFF files in {folder}")
    key = fingerprint(files, cell_size, pit_depth, detrend)
    cache_file = Path(cache_dir) / f"{folder.name}_{key}.npz" if cache_dir else None
    if cache_file is not None and cache_file.exists():
        try:
            survey = _load(cache_file, folder.name, files)
            cache_file.touch()  # "recently used", so it is kept when old copies are cleaned up
            return survey
        except Exception:
            cache_file.unlink(missing_ok=True)  # a damaged copy: prepare again

    survey = _prepare_fresh(folder.name, files, cell_size, pit_depth, detrend)
    if cache_file is not None:
        try:
            _save(cache_file, survey)
            _clean_cache(Path(cache_dir))
        except OSError:
            pass  # no space or no permission: TIMBER still works, it just can't remember
    return survey


def _prepare_fresh(name, files, cell_size, pit_depth, detrend) -> Survey:
    """The slow way: read every file and work everything out."""
    native = io.load_raster(files["dtm"])
    dtm = io.resample(native, cell_size) if cell_size else native
    repaired = 0.0
    if pit_depth is not None:
        dtm, pit = io.repair_pits(dtm, max_drop=pit_depth)
        valid = np.isfinite(dtm.data).sum()
        repaired = float(pit.sum() / valid) if valid else 0.0
    tilt = None
    if detrend:
        flat, tilt = hydrology.remove_tilt(dtm.data, *dtm.pixel_size)
        dtm = io.Raster(data=flat, transform=dtm.transform, crs=dtm.crs)
    dsm = None
    if files["dsm"] is not None:
        dsm = io.resample_to_grid(io.load_raster(files["dsm"]), dtm)
        if tilt is not None:  # remove the same tilt so DSM - DTM stays correct
            dsm = io.Raster(data=dsm.data - hydrology.tilt_plane(dsm.shape, *dsm.pixel_size, tilt),
                            transform=dsm.transform, crs=dsm.crs)
    ortho = io.read_rgb_on_grid(files["ortho"], dtm) if files["ortho"] is not None else None
    flow = hydrology.analyze_flow(dtm.data, *dtm.pixel_size)
    native_cell = native.pixel_size[0]
    return Survey(name=name, dtm=dtm, dsm=dsm, ortho=ortho, flow=flow, files=files, repaired=repaired, tilt=tilt,
                  info={"native_cell_m": native_cell, "native_shape": list(native.shape)})


def _save(path: Path, s: Survey):
    """Write a prepared survey as one .npz file (a zip of numpy arrays, not compressed so it loads fast)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "dtm": s.dtm.data,
        # Only what can't be worked out again cheaply: filled = ground + depth, and
        # cells draining = area draining / area of one cell.
        "flow_depth": s.flow["depth"].astype("float32"),
        "flow_acc_area": s.flow["acc_area_m2"].astype("float32"),
        "flow_receiver": s.flow["receiver"].astype("int32"),
    }
    if s.dsm is not None:
        arrays["dsm"] = s.dsm.data
    if s.ortho is not None:
        arrays["ortho"] = s.ortho.astype("float32")
    meta = {"transform": list(s.dtm.transform)[:6], "crs": s.dtm.crs.to_wkt() if s.dtm.crs else None,
            "repaired": s.repaired, "tilt": s.tilt, "info": s.info}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, meta=np.array(json.dumps(meta, default=float)), **arrays)
    tmp.replace(path)  # only a complete file ever gets the real name


def _load(path: Path, name: str, files: dict) -> Survey:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        transform = Affine(*meta["transform"])
        crs = CRS.from_wkt(meta["crs"]) if meta["crs"] else None
        dtm = io.Raster(data=z["dtm"], transform=transform, crs=crs)
        dsm = io.Raster(data=z["dsm"], transform=transform, crs=crs) if "dsm" in z else None
        ortho = z["ortho"].astype("float64") if "ortho" in z else None
        depth = z["flow_depth"].astype("float64")
        acc_area = z["flow_acc_area"].astype("float64")
        receiver = z["flow_receiver"].astype("int64")
    px, py = dtm.pixel_size
    flow = {"z": dtm.data, "filled": dtm.data + depth, "depth": depth, "receiver": receiver,
            "acc_cells": acc_area / (px * py), "acc_area_m2": acc_area, "tilt": meta["tilt"]}
    return Survey(name=name, dtm=dtm, dsm=dsm, ortho=ortho, flow=flow, files=files, repaired=meta["repaired"],
                  tilt=meta["tilt"], from_cache=True, info=meta["info"])


def _clean_cache(cache_dir: Path):
    """Delete the oldest prepared surveys so the cache folder doesn't grow forever."""
    saved = sorted(cache_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in saved[KEEP_CACHED:]:
        old.unlink(missing_ok=True)
