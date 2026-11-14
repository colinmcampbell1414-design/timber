"""TIMBER's look: a few style rules on top of the theme in .streamlit/config.toml."""

import streamlit as st

from timber_ui.paths import ASSETS

GREEN = "#4cbf7f"
CLASS_BADGE = {"low": "#1a9850", "moderate": "#e08a2c", "high": "#d73027"}

CSS = """
<style>
/* Page width and spacing */
.block-container { padding-top: 1.6rem; max-width: 1500px; }
/* Summary numbers as cards */
div[data-testid="stMetric"] {
    background: #17211b; border: 1px solid #253229; border-radius: 10px; padding: 10px 14px;
}
[data-testid="stMetricLabel"] p { color: #9fb3a6; font-size: 0.85rem; }
[data-testid="stMetricLabel"] * { white-space: normal !important; overflow: visible !important;
                                     text-overflow: clip !important; }
/* Let long numbers wrap instead of being cut off with "..." in narrow windows */
div[data-testid="stMetricValue"] { font-size: 1.55rem; }
div[data-testid="stMetricValue"] div { white-space: normal; overflow: visible; text-overflow: clip; }
/* Tabs: a bit bigger and greener when selected */
button[data-baseweb="tab"] p { font-size: 0.98rem; }
/* Header block */
.timber-hero { display: flex; align-items: center; gap: 18px; margin-bottom: 0.4rem; }
.timber-hero img { width: 64px; height: 64px; }
.timber-hero h1 { margin: 0; padding: 0; font-size: 2.3rem; letter-spacing: 0.08em; }
.timber-hero p { margin: 0; color: #9fb3a6; }
.timber-badge { display: inline-block; padding: 2px 10px; border-radius: 999px; color: white;
                font-weight: 600; font-size: 0.85rem; }
.timber-card { background: #17211b; border: 1px solid #253229; border-radius: 12px; padding: 14px 18px; }
.timber-muted { color: #9fb3a6; }
</style>
"""

DISCLAIMER = ("Screening tool for a student science project. Not a substitute for a professional "
              "arborist assessment.")  # the tree pages
DISCLAIMER_GENERAL = "Screening tool for a student science project. Not a substitute for a professional assessment."


def apply():
    """Add the style rules and the sidebar logo (call once per page view)."""
    st.markdown(CSS, unsafe_allow_html=True)
    logo = ASSETS / "timber.png"
    if logo.exists():
        st.logo(str(logo), size="large")


def page_header(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def badge(text: str, color: str) -> str:
    """A small colored label, e.g. for a tree's hazard class (HTML for st.markdown)."""
    return f'<span class="timber-badge" style="background:{color}">{text}</span>'
