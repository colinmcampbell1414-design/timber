"""The page for one analysis (tree fall hazard or slope stability).

Order on the page: short description, settings (in a form: nothing re-runs until
you press "Run analysis"), summary cards, then one tab per part of the result,
each with an interactive map, tables and downloads.
"""

import hashlib
import inspect
import time

import streamlit as st

from terrain import io
from terrain.apps import base as app_base
from timber_ui import maps
from timber_ui.paths import CACHE_DIR
from timber_ui.style import DISCLAIMER
from timber_ui.widgets import shape_editor, show_metrics, show_table, tree_card


def layers_for(survey) -> app_base.Layers:
    """The survey's grids in the form every analysis uses, with the disk cache switched on."""
    return app_base.Layers(dtm=survey.dtm, dsm=survey.dsm, ortho=survey.ortho, flow_result=survey.flow,
                           shapes=st.session_state.setdefault("shapes", {}), cache_dir=CACHE_DIR,
                           cache_key=survey.key)


@st.cache_resource(show_spinner=False, max_entries=6)
def run_analysis(app_name: str, survey_key: str, run_key: str, _app, _layers, _p):
    """Run an analysis once per (analysis, survey, settings, marked shapes) and keep it in memory.

    Arguments starting with "_" are not used to recognize a repeat; the keys
    before them are. Switching tabs or layers is then instant.
    """
    start = time.time()
    result = _app.run(_layers, _p)
    result.metrics = dict(result.metrics)
    return result, time.time() - start


def analysis_page(app, survey, title, subtitle):
    st.title(title)
    st.caption(subtitle)
    if survey is None:
        st.info("Choose a survey in the sidebar first (see the Home page for how to add one).")
        return
    layers = layers_for(survey)
    st.write(app.DESCRIPTION)
    message = app_base.missing_inputs(app, layers)
    if message:
        st.error(message)
        return

    slug = app.__name__.split(".")[-1]
    view = st.session_state.setdefault("view", {"photo": survey.ortho is not None, "opacity": 0.7})

    # ---- Settings (a form: change as many as you like, then press Run) --------
    with st.expander("⚙️  Settings", expanded=False):
        with st.form(f"settings_{slug}", border=False):
            st.caption("All numbers are **starting values — adjustable**. Change them, then press **Run analysis**.")
            if len(inspect.signature(app.params).parameters) > 1:  # apps that need the map size (forest defaults)
                left, bottom, right, top = survey.dtm.bounds
                p = app.params(st, {"width_m": max(right - left, top - bottom)})
            else:
                p = app.params(st)
            st.form_submit_button("Run analysis", type="primary")
    for key, spec in app.SHAPES.items():
        shape_editor(key, spec, survey, view["photo"])

    # ---- Run (or reuse) ------------------------------------------------------
    run_key = hashlib.md5(repr((app.__name__, sorted(p.items(), key=str),
                                sorted(layers.shapes.items(), key=str))).encode()).hexdigest()
    with st.spinner(f"Running {app.NAME.lower()}... (the first run on a survey can take up to a minute; "
                    "results are saved, so next time it is instant)"):
        try:
            result, seconds = run_analysis(app.__name__, survey.key, run_key, _app=app, _layers=layers, _p=p)
        except Exception as err:  # show the problem instead of crashing the whole app
            st.error(f"Something went wrong running '{app.NAME}': {err}")
            return
    st.caption(f"Analysis ready in {seconds:.1f} s.")
    for w in result.warnings:
        st.warning(w, icon="⚠️")

    # ---- Map display options --------------------------------------------------
    c1, c2, _ = st.columns([1, 1, 2])
    if survey.ortho is not None:
        view["photo"] = c1.toggle("Photo", value=view["photo"], key=f"photo_{slug}",
                                  help="Off = shaded relief of the ground model.")
    view["opacity"] = c2.slider("Overlay strength", 0.1, 1.0, view["opacity"], 0.05, key=f"opacity_{slug}")

    if result.sections:
        if result.metrics:
            show_metrics(result.metrics)
        # One part at a time (tabs would draw every part's map on each click, which is slow).
        titles = [s.title for s in result.sections]
        choice = st.segmented_control("Show", titles, default=titles[0], key=f"part_{slug}",
                                      label_visibility="collapsed") or titles[0]
        show_section(slug, result.sections[titles.index(choice)], survey, view)
    else:
        sec = app_base.Section("Result", overlays=result.overlays, annotations=result.annotations,
                               table=result.table, table_title="Flagged areas")
        if result.metrics:
            show_metrics(result.metrics)
        show_section(slug, sec, survey, view)

    with st.expander("How to test this at home"):
        st.write(app.TEST_AT_HOME)
    st.caption(DISCLAIMER)


def show_section(slug, sec, survey, view):
    """One tab: text, interactive map (with layer choice), downloads, tables."""
    key = f"{slug}_{sec.title}".replace(" ", "_").replace("&", "and")
    if sec.text:
        st.markdown(sec.text)
    picked_ids = []
    overlay = None
    if sec.overlays:
        names = [o.name for o in sec.overlays]
        chosen = st.radio("Map layer", names, horizontal=True, key=f"layer_{key}") if len(names) > 1 else names[0]
        overlay = sec.overlays[names.index(chosen)]
        fig = maps.interactive_map(survey, overlay, sec.annotations, view["photo"], view["opacity"], view_key=key)
        clickable = any(a.get("ids") is not None for a in sec.annotations if a["type"] == "scatter")
        picked_ids = maps.show_interactive(fig, f"map_{key}", selectable=clickable)
        if clickable:
            st.caption("Scroll or drag a box to zoom in, double-click to see the whole map (the hand button in "
                       "the map's corner switches to moving it). Point at a tree for a summary; click it for "
                       "all its details below.")
        _downloads_for_map(key, survey, overlay, sec, view)
    for i, (label, data, name, mime) in enumerate(sec.downloads):
        st.download_button(label, data, name, mime, key=f"dl_{key}_{i}")

    is_tree_table = sec.table is not None and "tree" in sec.table and "reasons" in sec.table
    picked_row = show_table(sec.table, sec.table_title, sec.table_head, key=key, selectable=is_tree_table)
    if is_tree_table:
        row = None
        if picked_row is not None:
            row = picked_row
        elif picked_ids:
            match = sec.table[sec.table["tree"] == picked_ids[0]]
            row = match.iloc[0] if len(match) else None
        if row is not None:
            with st.container(border=True):
                tree_card(row, survey.dtm.crs)
        else:
            st.caption("Click a tree on the map or a row in the table to see its details.")
    for i, (title, extra) in enumerate(sec.extra_tables):
        show_table(extra, title, key=f"{key}_extra{i}")
    if sec.custom is not None:
        sec.custom(st)


def _downloads_for_map(key, survey, overlay, sec, view):
    """GeoTIFF and PNG downloads. They are made only when asked for, because
    making them for every tab on every click would slow the page down."""
    store = st.session_state.setdefault("downloads", {})
    c1, c2 = st.columns(2)
    tif_key, png_key = (key, overlay.name, "tif"), (key, overlay.name, view["photo"], view["opacity"], "png")
    if tif_key in store:
        c1.download_button("⬇ Save overlay as GeoTIFF", store[tif_key], f"{key}_overlay.tif", "image/tiff",
                           key=f"tif_{key}")
    elif c1.button("Prepare overlay GeoTIFF", key=f"mk_tif_{key}"):
        store[tif_key] = io.geotiff_bytes(overlay.data, survey.dtm)
        st.rerun()
    if png_key in store:
        c2.download_button("⬇ Save map as PNG (200 dpi)", store[png_key], f"{key}_map.png", "image/png",
                           key=f"png_{key}")
    elif c2.button("Prepare map PNG", key=f"mk_png_{key}"):
        with st.spinner("Drawing the full-detail map..."):
            store[png_key] = maps.render_png(survey, overlay, sec.annotations, view["photo"], view["opacity"])
        st.rerun()
