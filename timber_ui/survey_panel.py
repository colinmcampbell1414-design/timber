"""Sidebar: choose a survey and how it is prepared. Every page uses the result."""

import dataclasses
import time

import numpy as np

import rasterio
import streamlit as st

from terrain import fusion, hydrology, io, survey as survey_mod
from timber_ui.paths import CACHE_DIR, DATA_ROOT, survey_folders


def default_cell_cm(path) -> float:
    """Starting cell size: 5 cm for small areas, coarser for big ones.

    A 500 m wide forest at 5 cm would be over 100 million cells, which is very
    slow, so larger maps start coarser (it can still be changed).
    """
    try:
        with rasterio.open(path) as src:
            width_m = max(src.bounds.right - src.bounds.left, src.bounds.top - src.bounds.bottom)
    except Exception:
        return 5.0
    if width_m > 600:
        return 50.0  # e.g. the ~900 m forest: 50 cm keeps it near 3 million cells
    if width_m > 300:
        return 25.0
    if width_m > 100:
        return 10.0
    return 5.0


@st.cache_resource(show_spinner=False, max_entries=3)
def load_survey(folder: str, cell_size: float, pit_depth: float | None, detrend: bool, _version: str):
    """Prepare a survey once per settings and keep it in memory while TIMBER runs.

    (It is also saved on disk by terrain/survey.py, so after a restart it loads
    in about a second.) `_version` is the files' fingerprint: when a file
    changes, the survey is prepared again.
    """
    start = time.time()
    s = survey_mod.prepare(folder, cell_size, pit_depth, detrend, cache_dir=CACHE_DIR)
    s.info["load_seconds"] = time.time() - start
    return s


@st.cache_resource(show_spinner="Loading the full-detail DTM...", max_entries=2)
def load_native(path: str, mtime: float):
    """The original DTM at full detail (only the accuracy checker needs it)."""
    return io.load_raster(path)


def sidebar() -> survey_mod.Survey | None:
    """Draw the survey controls and return the prepared survey (None if there is no data)."""
    folders = survey_folders()
    st.sidebar.markdown("### Survey")
    if not folders:
        st.sidebar.info(f"No surveys yet. Put each WebODM survey in its own folder inside\n\n`{DATA_ROOT}`")
        return None
    names = [p.name for p in folders]
    name = st.sidebar.selectbox("Survey", names, label_visibility="collapsed",
                                help="Each folder in the data folder is one survey (DTM, DSM and orthophoto "
                                     "from one WebODM task).")
    folder = DATA_ROOT / name
    files = survey_mod.find_files(folder)

    with st.sidebar.expander("Survey settings"):
        cell_cm = st.number_input(
            "Analysis cell size (cm)", min_value=1.0, max_value=100.0, value=default_cell_cm(files["dtm"]), step=1.0,
            key=f"cell_cm_{name}",  # each survey remembers its own setting
            help="Bigger cells are faster and smooth out noise. Big maps start coarser so they load quickly.")
        repair = st.checkbox(
            "Repair ground-model pits", value=True,
            help="Under thick canopy WebODM's ground model sometimes drops into deep pits far below the real "
                 "ground, which makes trees look 60 m tall. Pit cells are replaced by the typical ground height "
                 "within 30 m. The accuracy checker always uses the original file.")
        pit_depth = st.slider("Pit = deeper than this below the surroundings (m)", 2.0, 30.0, 10.0, 0.5,
                              disabled=not repair)
        detrend = st.checkbox("Remove best-fit tilt", value=False,
                              help="Subtracts the flat plane that best fits the whole map first. Only useful to "
                                   "see how much results depend on the model's overall tilt.")

    version = survey_mod.fingerprint(files, cell_cm / 100, pit_depth if repair else None, detrend)
    with st.spinner(f"Preparing {name}... The first time takes 10-30 seconds; after that it opens in about a second."):
        try:
            s = load_survey(str(folder), cell_cm / 100, pit_depth if repair else None, detrend, version)
        except Exception as err:
            st.sidebar.error(f"Could not load this survey: {err}")
            return None
    note = "opened from saved copy" if s.from_cache else "prepared and saved for next time"
    st.sidebar.caption(f"{s.dtm.shape[1]} x {s.dtm.shape[0]} cells at {cell_cm:g} cm · {note} "
                       f"({s.info.get('load_seconds', 0):.1f} s)" +
                       (f" · {s.repaired * 100:.1f}% pit-repaired" if repair and s.repaired else ""))
    s.key = version
    s.folder = folder
    # How this survey was prepared (the storm check prepares the same grid again later).
    s.settings = {"cell_size": cell_cm / 100, "pit_depth": pit_depth if repair else None, "detrend": detrend}
    s.own_dtm = s.dtm  # this survey's own ground model, kept for the DTM comparison
    source = dtm_source_picker(folder)
    if source is not None and s.dsm is not None:
        label, path, units = source
        try:
            with st.spinner(f"Fitting the ground model from {label} to this survey..."):
                fused = load_fused(version, str(path), path.stat().st_mtime, units, label, s)
            fused.key, fused.folder, fused.settings, fused.own_dtm = f"{version}_{label}_{units}", folder, s.settings, s.dtm
            r = fused.info["dtm_report"]
            st.sidebar.caption(f"Ground model from **{label}**: offset {r['offset_before_m']:+.2f} m and tilt "
                               f"{r['tilt_deg']:.2f}° removed; {r['rms_after_m']:.2f} m of noise left on stable ground.")
            s = fused
        except Exception as err:
            st.sidebar.error(f"Could not use that ground model: {err}")
    flight_notes(folder)
    return s


def dtm_sources(folder) -> dict:
    """Other ground models that could be used with this survey: the DTM of every other
    survey folder, and any GeoTIFF in data/outside_dtm (e.g. public lidar)."""
    out = {}
    for other in survey_folders():
        files = survey_mod.find_files(other)
        if other != folder and files["dtm"] is not None and other.name != "outside_dtm":
            out[f"survey: {other.name}"] = files["dtm"]
    outside = DATA_ROOT / "outside_dtm"
    for path in sorted(outside.glob("*.tif*")) if outside.exists() else []:
        out[f"outside: {path.name}"] = path
    return out


def dtm_source_picker(folder):
    """Sidebar box: use this survey's own ground model, or one from another survey / outside file."""
    sources = dtm_sources(folder)
    with st.sidebar.expander("Ground model (DTM) source"):
        st.caption("Leaf-on drone surveys can't see the ground under trees. A leaf-off survey of the same place, or "
                   "public lidar (put the GeoTIFF in data/outside_dtm), gives a better ground model. It is lined up "
                   "with this survey on stable open ground.")
        choice = st.selectbox("Ground model from", ["this survey"] + list(sources), key=f"dtm_source_{folder.name}")
        units = st.selectbox("Height units of that file", fusion.UNIT_CHOICES, key=f"dtm_units_{folder.name}",
                             help="Auto reads the file's coordinate system. Lidar from US agencies is often in feet.")
    if choice == "this survey":
        return None
    return choice, sources[choice], "auto" if units.startswith("auto") else units


@st.cache_resource(show_spinner=False, max_entries=2)
def load_fused(version: str, path: str, mtime: float, units: str, label: str, _survey):
    """The survey with its ground model replaced by another one (aligned), and the water flow redone."""
    s = _survey
    cell = s.dtm.pixel_size[0]
    other = fusion.to_grid(io.load_raster(path), s.dtm, units)
    aligned, report = fusion.align_dtm(other.data, s.dtm.data, s.dsm.data, cell)
    # Where the other model has no data, this survey's own ground is kept.
    dtm = io.Raster(data=np.where(np.isfinite(aligned), aligned, s.dtm.data), transform=s.dtm.transform, crs=s.dtm.crs)
    flow = hydrology.analyze_flow(dtm.data, *dtm.pixel_size)
    return dataclasses.replace(s, dtm=dtm, flow=flow, info={**s.info, "dtm_source": label, "dtm_report": report})


def flight_notes(folder):
    """Sidebar box for the flight notes kept in the survey's folder (survey.json)."""
    info = survey_mod.read_info(folder)
    seasons = survey_mod.SEASONS
    with st.sidebar.expander("Flight notes"):
        with st.form(f"notes_{folder.name}", border=False):
            date = st.text_input("Flight date (YYYY-MM-DD)", info["date"])
            time_ = st.text_input("Time of day (HH:MM)", info["time"])
            season = st.selectbox("Leaves", seasons, index=seasons.index(info["season"]) if info["season"] in seasons else 0,
                                  help="Leaf condition matters: leaf drop between two flights looks like crown damage.")
            altitude = st.number_input("Flight altitude (m)", 0.0, 500.0, float(info["flight_altitude_m"] or 0.0), 5.0)
            overlap = st.number_input("Photo overlap (%)", 0.0, 100.0, float(info["overlap_percent"] or 0.0), 5.0)
            notes = st.text_area("Notes", info["notes"], height=80)
            if st.form_submit_button("Save flight notes"):
                survey_mod.write_info(folder, {"date": date.strip(), "time": time_.strip(), "season": season,
                                               "flight_altitude_m": altitude or None, "overlap_percent": overlap or None,
                                               "notes": notes})
                st.success("Saved to survey.json.")
