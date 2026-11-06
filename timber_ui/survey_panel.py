"""Sidebar: choose a survey and how it is prepared. Every page uses the result."""

import time

import rasterio
import streamlit as st

from terrain import io, survey as survey_mod
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
    return s
