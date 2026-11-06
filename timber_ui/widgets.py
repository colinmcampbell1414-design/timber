"""Small building blocks used by several pages: tables, summary cards, the tree
detail card, and the editor for marking points/outlines on the map."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

from terrain import treefall
from timber_ui import maps
from timber_ui.style import CLASS_BADGE, badge


def show_metrics(metrics: dict):
    """Summary numbers as a row of cards."""
    items = list(metrics.items())
    for start in range(0, len(items), 5):
        cols = st.columns(5)
        for col, (label, value) in zip(cols, items[start:start + 5]):
            col.metric(label, value)


def show_table(table, title, head=None, key="", selectable=False):
    """A table (first `head` rows, the rest in an expander) with a CSV download.
    With selectable=True, clicking a row returns that row (else None)."""
    if table is None:
        return None
    if title:
        st.markdown(f"**{title}**")
    if not len(table):
        st.write("Nothing found with the current settings.")
        return None
    shown = table.head(head) if head else table
    picked = None
    if selectable:
        event = st.dataframe(shown, hide_index=True, width="stretch", on_select="rerun",
                             selection_mode="single-row", key=f"table_{key}",
                             column_config=_number_formats(shown))
        rows = event.selection.rows if event is not None else []
        picked = shown.iloc[rows[0]] if rows else None
    else:
        st.dataframe(shown, hide_index=True, width="stretch", column_config=_number_formats(shown))
    if head and len(table) > head:
        with st.expander(f"Show all {len(table)} rows"):
            st.dataframe(table, hide_index=True, width="stretch", column_config=_number_formats(table))
    st.download_button("Table as CSV", table.to_csv(index=False), f"{key or 'table'}.csv", "text/csv",
                       key=f"csv_{key}")
    return picked


def _number_formats(table):
    """Show decimals neatly (2 places) without changing the numbers in the CSV."""
    return {c: st.column_config.NumberColumn(format="%.2f") for c in table.columns
            if pd.api.types.is_float_dtype(table[c])}


# ---------------------------------------------------------------------------
# Tree detail card
# ---------------------------------------------------------------------------
def tree_card(row: pd.Series, crs=None):
    """Everything about one tree: class, score, why, where it would fall, and its factors."""
    cls = str(row["class"])
    st.markdown(f"#### Tree {int(row['tree'])} &nbsp; {badge(cls.upper(), CLASS_BADGE.get(cls, '#555'))}",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hazard score", f"{row['hazard (0-100)']:.0f} / 100")
    c2.metric("Height", f"{row['height (m)']:.1f} m ({row['height (ft)']:.0f} ft)")
    c3.metric("Likely fall direction", f"{row['likely fall direction']} ({row['fall direction (°)']:.0f}°)")
    c4.metric("Priority", f"{row['priority']:.0f}")
    st.markdown(f"**Why:** {row['reasons']}")
    reach = [c.replace("can reach ", "") for c in row.index if c.startswith("can reach ") and bool(row[c])]
    st.markdown(f"**Could reach:** {', '.join(reach) if reach else 'no detected features'} · "
                f"**could hit** {int(row['trees it could hit'])} other trees")
    if crs is not None:
        from terrain import io
        lon, lat = io.map_to_latlon([row["x"]], [row["y"]], crs)
        st.caption(f"Location: {float(lat[0]):.6f}, {float(lon[0]):.6f} (latitude, longitude) · "
                   f"map x {row['x']:.1f}, y {row['y']:.1f}")
    names = [n for n in treefall.FACTORS if f"{n} factor" in row.index]
    values = [float(np.nan_to_num(row[f"{n} factor"])) for n in names]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color="#4cbf7f",
                           text=[f"{v:.2f}" for v in values], textposition="outside"))
    fig.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=30, t=30, b=10),
                      title=dict(text="Hazard factors (0 = no concern, 1 = full)", font=dict(size=13)),
                      xaxis=dict(range=[0, 1.15]), paper_bgcolor="#17211b", plot_bgcolor="#17211b")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"factors_{int(row['tree'])}")


# ---------------------------------------------------------------------------
# Marking things on the map (click or CSV)
# ---------------------------------------------------------------------------
CLICK_MAP_WIDTH = 800  # pixels


def _map_to_pixel(raster, x, y, scale):
    t = raster.transform
    return (x - t.c) / t.a * scale, (y - t.f) / t.e * scale


def _pixel_to_map(raster, col, row, scale):
    t = raster.transform
    return t.c + col / scale * t.a, t.f + row / scale * t.e


def _draw_shapes(img, raster, scale, shapes, drawing, kind):
    """Draw finished shapes (yellow) and the one being drawn (red) on the click map."""
    draw = ImageDraw.Draw(img)

    def px(pt):
        return _map_to_pixel(raster, pt[0], pt[1], scale)

    for shape in shapes:
        pts = [px(v) for v in shape["xy"]]
        if len(pts) == 1 or kind == "points":
            x, y = pts[0]
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline="#ffd400", width=3)
            if shape.get("label"):
                draw.text((x + 7, y - 7), shape["label"], fill="#ffd400")
        elif kind == "polygons":
            draw.polygon(pts, outline="#ffd400", width=3)
        else:
            draw.line(pts, fill="#ffd400", width=3)
    if drawing:
        pts = [px(v) for v in drawing]
        if len(pts) > 1:
            draw.line(pts, fill="#ff2a2a", width=2)
        for x, y in pts:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill="#ff2a2a")
    return img


def shapes_to_csv(shapes) -> str:
    rows = [{"shape": i, "label": s.get("label", ""), "x": x, "y": y}
            for i, s in enumerate(shapes, start=1) for x, y in s["xy"]]
    return pd.DataFrame(rows, columns=["shape", "label", "x", "y"]).to_csv(index=False)


def shapes_from_csv(file, kind):
    """Read shapes from a CSV with columns x, y and optional shape (number) and label."""
    df = pd.read_csv(file)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"x", "y"} <= set(df.columns):
        raise ValueError("The CSV needs columns x and y (map coordinates). Optional: shape, label.")
    if "shape" not in df.columns:
        # Points: every row is its own point. Outlines/paths: all rows form one shape.
        df["shape"] = np.arange(len(df)) if kind == "points" else 1
    if "label" not in df.columns:
        df["label"] = ""
    shapes = []
    for _, group in df.groupby("shape", sort=False):
        label = group["label"].iloc[0]
        shapes.append({"label": "" if pd.isna(label) else str(label),
                       "xy": [(float(x), float(y)) for x, y in zip(group["x"], group["y"])]})
    return shapes


def shape_editor(key, spec, survey, use_photo):
    """Let the user mark points/outlines/paths by clicking the map, or by uploading a CSV."""
    state = st.session_state.setdefault("shapes", {})
    shapes = state.setdefault(key, [])
    drawing = st.session_state.setdefault("drawing", {}).setdefault(key, [])
    raster = survey.dtm
    title = f"Mark on map: {spec.label}" + (" (required)" if spec.required else "")
    with st.expander(f"{title} — {len(shapes)} marked", expanded=spec.required and not shapes):
        st.caption(spec.help)
        if spec.kind == "points":
            st.caption("Each click adds a point.")
            label = st.text_input("Label for new points", value="house", key=f"label_{key}")
        else:
            what = "corner of the outline" if spec.kind == "polygons" else "point along the path"
            st.caption(f"Each click adds a {what}. Click **Finish shape** when done, then start the next one.")
            label = ""

        rgb = (maps.background_rgb(survey.key, use_photo, survey) * 255).astype("uint8")
        scale = CLICK_MAP_WIDTH / rgb.shape[1]
        img = Image.fromarray(rgb).resize((CLICK_MAP_WIDTH, max(1, int(rgb.shape[0] * scale))), Image.NEAREST)
        img = _draw_shapes(img, raster, scale, shapes, drawing, spec.kind)
        click = streamlit_image_coordinates(img, key=f"click_{key}")
        last = st.session_state.get(f"last_click_{key}")
        if click is not None and click != last:
            st.session_state[f"last_click_{key}"] = click
            # The image may be shown smaller than its real size; scale the click back.
            shown_w = click.get("width") or img.width
            shown_h = click.get("height") or img.height
            col, row = click["x"] * img.width / shown_w, click["y"] * img.height / shown_h
            point = _pixel_to_map(raster, col, row, scale)
            if spec.kind == "points":
                shapes.append({"label": label, "xy": [point]})
            else:
                drawing.append(point)
            st.rerun()

        needed = 3 if spec.kind == "polygons" else 2
        c1, c2, c3 = st.columns(3)
        if spec.kind != "points" and c1.button("Finish shape", key=f"finish_{key}", disabled=len(drawing) < needed):
            shapes.append({"label": "", "xy": list(drawing)})
            drawing.clear()
            st.rerun()
        if c2.button("Undo last click", key=f"undo_{key}", disabled=not (drawing or shapes)):
            if drawing:
                drawing.pop()
            elif shapes:
                shapes.pop()
            st.rerun()
        if c3.button("Clear all", key=f"clear_{key}", disabled=not (drawing or shapes)):
            shapes.clear()
            drawing.clear()
            st.rerun()

        c1, c2 = st.columns(2)
        upload = c1.file_uploader("...or upload a CSV (columns x, y; optional shape, label)", type=["csv"],
                                  key=f"upload_{key}")
        if upload is not None and st.session_state.get(f"uploaded_{key}") != upload.file_id:
            try:
                state[key] = shapes_from_csv(upload, spec.kind)
                st.session_state[f"uploaded_{key}"] = upload.file_id
                st.rerun()
            except Exception as err:
                st.error(f"Could not read the CSV: {err}")
        c2.download_button("Download these as CSV", shapes_to_csv(shapes), f"{key}.csv", "text/csv",
                           key=f"download_{key}", disabled=not shapes)
    return state[key]
