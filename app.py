"""TIMBER: Tree Instability Mapping By Elevation & Reflectance.

Run it in its own window:   python timber.py   (or double-click TIMBER.exe)
Or in a browser tab:        streamlit run app.py

This file only sets up the pages. The pages are in timber_ui/; all the math
is in terrain/.
"""

import streamlit as st

from terrain.apps import mudslide, trees
from timber_ui import style, survey_panel
from timber_ui.accuracy_page import accuracy_page
from timber_ui.analysis_page import analysis_page
from timber_ui.ground import ground_page
from timber_ui.home import home_page
from timber_ui.storm_panel import storm_check
from timber_ui.style import DISCLAIMER_GENERAL
from timber_ui.validation_panel import design_criteria, field_check, log_validation, sensitivity_box

st.set_page_config(page_title="TIMBER", page_icon="🌲", layout="wide", initial_sidebar_state="expanded")
style.apply()

# The survey is chosen in the sidebar and shared by every page.
survey = survey_panel.sidebar()


def page_home():
    home_page(survey)


def page_trees():
    analysis_page(trees, survey, "Tree hazard",
                  "Which trees could fall, which way, and what they could hit.",
                  extra_parts={"Field check": field_check, "Design criteria": design_criteria,
                               "Storm check": storm_check},
                  part_extras={"Trees & hazard": sensitivity_box, "Downed logs": log_validation})


def page_slope():
    analysis_page(mudslide, survey, "Slope stability",
                  "Ground that is steep, wet and collects water, where soil could slump or slide.",
                  banner=DISCLAIMER_GENERAL)


def page_ground():
    ground_page(survey)


def page_accuracy():
    accuracy_page(survey)


pages = {
    "TIMBER": [
        st.Page(page_home, title="Home", icon=":material/home:", default=True),
        st.Page(page_trees, title="Tree hazard", icon=":material/forest:", url_path="trees"),
        st.Page(page_slope, title="Slope stability", icon=":material/landslide:", url_path="slope"),
    ],
    "Data quality": [
        st.Page(page_ground, title="Ground check", icon=":material/terrain:", url_path="ground"),
        st.Page(page_accuracy, title="Accuracy checker", icon=":material/straighten:", url_path="accuracy"),
    ],
}
st.navigation(pages, position="sidebar").run()
