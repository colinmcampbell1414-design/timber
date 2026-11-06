"""Home page: what TIMBER does, the current survey at a glance, and how to start."""

import base64
import os

import numpy as np
import streamlit as st

from timber_ui.paths import ASSETS, CACHE_DIR, DATA_ROOT
from timber_ui.style import DISCLAIMER


def _logo_html() -> str:
    logo = ASSETS / "timber.png"
    if not logo.exists():
        return ""
    data = base64.b64encode(logo.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}">'


def home_page(survey):
    st.markdown(
        f'<div class="timber-hero">{_logo_html()}<div><h1>TIMBER</h1>'
        "<p><b>T</b>ree <b>I</b>nstability <b>M</b>apping <b>B</b>y <b>E</b>levation &amp; <b>R</b>eflectance</p>"
        "</div></div>", unsafe_allow_html=True)
    st.write("Finds the trees in a forest that could fall on trails, roads and buildings, using only a consumer "
             "drone and free software. **Elevation** (the drone's surface and ground models) gives each tree's "
             "height, the slope and where water collects; **reflectance** (the photo's colors) is the health check.")

    if survey is None:
        st.warning("No survey loaded yet.")
        _how_to_add()
        return

    st.subheader(f"Survey: {survey.name}")
    dtm = survey.dtm
    left, bottom, right, top = dtm.bounds
    valid = np.isfinite(dtm.data)
    area_ha = valid.sum() * dtm.pixel_size[0] * dtm.pixel_size[1] / 10000
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Area mapped", f"{area_ha:.1f} ha")
    c2.metric("Map size", f"{right - left:.0f} x {top - bottom:.0f} m")
    c3.metric("Analysis cell", f"{dtm.pixel_size[0] * 100:.0f} cm")
    c4.metric("Surface model (DSM)", "yes" if survey.dsm is not None else "missing")
    c5.metric("Photo", "yes" if survey.ortho is not None else "missing")
    if survey.dsm is None:
        st.error("This survey has no DSM, so trees can't be found. Export the DSM from WebODM too.")
    if survey.repaired > 0.005:
        st.info(f"{survey.repaired * 100:.1f}% of the ground model was pit-repaired (spots where WebODM's "
                "ground dropped far below its surroundings, usually under thick canopy). See **Ground check**.")

    st.subheader("How to use TIMBER")
    steps = st.columns(4)
    steps[0].markdown("**1. Tree hazard**  \nEvery tree's hazard score, where it would fall, and what it could hit. "
                      "Click a tree for details.")
    steps[1].markdown("**2. Field check**  \nIn the Tree hazard page's *Field check* tab: download a sheet, visit "
                      "the trees, upload what you found.")
    steps[2].markdown("**3. Ground check**  \nMake sure the ground model is believable (pits, fake cliffs, fake "
                      "ponds under leaves).")
    steps[3].markdown("**4. Slope stability**  \nSteep, wet, water-collecting ground that could slump.")

    with st.expander("Where TIMBER keeps its files"):
        _how_to_add()
        st.markdown(f"Saved results (safe to delete; they are worked out again when needed): `{CACHE_DIR}`")
    st.caption(DISCLAIMER)


def _how_to_add():
    st.markdown(
        f"Surveys go in **`{DATA_ROOT}`**, one folder per survey. In WebODM use **Download Assets** and put the "
        "DTM, DSM and orthophoto GeoTIFFs (`...-dtm.tif`, `...-dsm.tif`, `...-orthophoto.tif`) in a new folder "
        "there, e.g. `forest`. Then pick it in the sidebar.")
    if hasattr(os, "startfile") and st.button("Open the data folder"):
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        os.startfile(DATA_ROOT)  # opens it in File Explorer (Windows)
