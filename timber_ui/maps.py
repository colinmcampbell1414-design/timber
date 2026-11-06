"""Maps: interactive (zoom, pan, hover, click a tree) and static (PNG for posters/reports).

Interactive maps use Plotly. A forest map has millions of cells, far more than a
screen can show, so the background picture (photo or hillshade, with the colored
overlay blended in) is shrunk to at most MAX_PIXELS across before it is sent to
the window. Tree tops, arrows and lines are drawn on top as real points and
lines, so they stay sharp when you zoom in.
"""

from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Circle, Patch, Polygon
from PIL import Image

from terrain import slope

MAX_PIXELS = 1600  # biggest side of the background picture, in pixels
MAP_HEIGHT = 720  # height of an interactive map on screen, in pixels


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def local_extent(raster):
    """Plot extent in meters measured from the raster's lower-left corner.

    Small numbers like "12 m" are easier to read than full UTM coordinates.
    """
    left, bottom, right, top = raster.bounds
    return (0.0, right - left, 0.0, top - bottom)


def new_map_axes():
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlabel("Meters east of left edge")
    ax.set_ylabel("Meters north of bottom edge")
    ax.set_aspect("equal")
    return fig, ax


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


@st.cache_resource(show_spinner=False, max_entries=6)
def background_rgb(survey_key: str, use_photo: bool, _survey) -> np.ndarray:
    """The full-size background (rows x cols x 3, 0-1): the photo, or the hillshade if there is none."""
    s = _survey
    if use_photo and s.ortho is not None:
        return np.nan_to_num(s.ortho, nan=0.12)
    shade = np.nan_to_num(slope.hillshade(s.dtm.data, *s.dtm.pixel_size, z_factor=2.0), nan=0.12)
    return np.dstack([shade] * 3)


def _step(shape) -> int:
    """Keep every k-th cell so the picture is at most MAX_PIXELS across."""
    return max(1, int(np.ceil(max(shape) / MAX_PIXELS)))


def overlay_rgba(overlay, k):
    """The overlay's colors (rows x cols x 3) and where it has a value (mask), every k-th cell."""
    data = np.asarray(overlay.data, dtype=float)[::k, ::k]
    if overlay.classes:
        rgb = np.zeros((*data.shape, 3))
        mask = np.zeros(data.shape, dtype=bool)
        for value, _, color in overlay.classes:
            here = data == value
            rgb[here] = matplotlib.colors.to_rgb(color)
            mask |= here
        return rgb, mask
    mask = np.isfinite(data)
    vmin = overlay.vmin if overlay.vmin is not None else (np.nanpercentile(data, 2) if mask.any() else 0)
    vmax = overlay.vmax if overlay.vmax is not None else (np.nanpercentile(data, 98) if mask.any() else 1)
    norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-6)
    rgb = matplotlib.colormaps[overlay.cmap](norm(np.nan_to_num(data, nan=vmin)))[..., :3]
    return rgb, mask


def map_picture(bg, overlay, opacity) -> Image.Image:
    """Background with the overlay blended in: color = (1 - a) x background + a x overlay color."""
    k = _step(bg.shape[:2])
    out = bg[::k, ::k].copy()
    if overlay is not None:
        rgb, mask = overlay_rgba(overlay, k)
        out[mask] = (1 - opacity) * out[mask] + opacity * rgb[mask]
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype("uint8"))


# ---------------------------------------------------------------------------
# Interactive map
# ---------------------------------------------------------------------------
def interactive_map(survey, overlay, annotations, use_photo=True, opacity=0.7, height=MAP_HEIGHT, view_key="map"):
    """A Plotly figure: picture + overlay legend + annotations (all in meters from the lower-left corner)."""
    dtm = survey.dtm
    left, bottom, right, top = dtm.bounds
    width_m, height_m = right - left, top - bottom
    bg = background_rgb(survey.key, use_photo, survey)
    fig = go.Figure()
    fig.add_layout_image(dict(source=map_picture(bg, overlay, opacity), xref="x", yref="y", x=0, y=height_m,
                              sizex=width_m, sizey=height_m, sizing="stretch", layer="below"))
    # Two invisible corner points make the axes cover the whole map.
    fig.add_trace(go.Scatter(x=[0, width_m], y=[0, height_m], mode="markers", marker=dict(opacity=0),
                             hoverinfo="skip", showlegend=False))
    if overlay is not None:
        _legend(fig, overlay)
    for a in annotations:
        _add_annotation(fig, a, left, bottom)
    # Equal meters on both axes; the map hugs the top-left so there is no gap under the legend.
    fig.update_xaxes(range=[0, width_m], title="Meters east of left edge", showgrid=False, zeroline=False,
                     constrain="domain", constraintoward="left")
    fig.update_yaxes(range=[0, height_m], title="Meters north of bottom edge", showgrid=False, zeroline=False,
                     scaleanchor="x", scaleratio=1, constrain="domain", constraintoward="top")
    # Height to suit the map's shape, assuming a map about 1000 pixels wide (between 450 and `height`).
    height = int(np.clip(1000 * height_m / max(width_m, 1e-9) + 70, 450, height))
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=36, b=10),
                      paper_bgcolor="#0e1411", plot_bgcolor="#0e1411",
                      dragmode="zoom",  # drag = zoom box (clicking a tree only works in this mode)
                      title=dict(text=overlay.name if overlay is not None else "", x=0.01, font=dict(size=15)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                                  bgcolor="rgba(0,0,0,0)"),
                      uirevision=view_key,  # keeps your zoom when the page refreshes
                      hoverlabel=dict(bgcolor="#17211b", font_size=12))
    return fig


def show_interactive(fig, key, selectable=False):
    """Draw an interactive map. Scroll or drag a box to zoom, double-click to zoom out.
    With selectable=True, returns the ids of clicked points (e.g. tree numbers)."""
    config = {"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]}
    if not selectable:
        st.plotly_chart(fig, width="stretch", config=config, key=key)
        return []
    event = st.plotly_chart(fig, width="stretch", config=config, key=key, on_select="rerun",
                            selection_mode=("points",))
    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        points = []
    ids = []
    for p in points:
        data = p.get("customdata")
        if data is None:  # fast point layers may not send their ids back: look them up by position
            trace = fig.data[p.get("curve_number", -1)] if 0 <= p.get("curve_number", -1) < len(fig.data) else None
            index = p.get("point_index", p.get("point_number"))
            if trace is not None and trace.customdata is not None and index is not None:
                data = trace.customdata[index]
        if isinstance(data, dict):  # Streamlit sends [id] as {"0": id}
            data = next(iter(data.values()), None)
        if data is not None:
            ids.append(data[0] if isinstance(data, (list, tuple, np.ndarray)) else data)
    return ids


def _legend(fig, overlay):
    if overlay.classes:
        for _, label, color in overlay.classes:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                     marker=dict(size=12, symbol="square", color=color)))
    else:
        data = np.asarray(overlay.data, dtype=float)
        finite = data[np.isfinite(data)]
        vmin = overlay.vmin if overlay.vmin is not None else (float(np.percentile(finite, 2)) if finite.size else 0)
        vmax = overlay.vmax if overlay.vmax is not None else (float(np.percentile(finite, 98)) if finite.size else 1)
        cmap = matplotlib.colormaps[overlay.cmap]
        scale = [[t, matplotlib.colors.to_hex(cmap(t))] for t in np.linspace(0, 1, 11)]
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(colorscale=scale, cmin=vmin, cmax=vmax, color=[vmin], showscale=True,
                                             colorbar=dict(title=overlay.units or "", thickness=14, len=0.6))))


def _lines_trace(xs, ys, color, width, name=None):
    return go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color=color, width=width), hoverinfo="skip",
                        name=name, showlegend=name is not None)


def _with_gaps(starts, ends):
    """Many separate line pieces as one trace: x1, x2, None, x1, x2, None, ..."""
    n = len(starts)
    out = np.full(n * 3, np.nan)
    out[0::3], out[1::3] = starts, ends
    return out


def _add_annotation(fig, a, left, bottom):
    t = a["type"]
    if t == "scatter" and len(a["x"]):  # many points at once (e.g. thousands of tree tops)
        ids = a.get("ids")
        fig.add_trace(go.Scattergl(
            x=np.asarray(a["x"]) - left, y=np.asarray(a["y"]) - bottom, mode="markers", name=a.get("name", "points"),
            marker=dict(size=a.get("sizes", max(5, a.get("size", 8) * 0.7)), color=a["colors"],
                        line=dict(width=0.5, color="rgba(0,0,0,0.6)")),
            customdata=None if ids is None else np.asarray(ids)[:, None],
            hovertext=a.get("hover"), hoverinfo="text" if a.get("hover") is not None else "skip"))
    elif t == "arrows" and len(a["x"]):  # many arrows, drawn to scale in meters
        x, y = np.asarray(a["x"]) - left, np.asarray(a["y"]) - bottom
        dx, dy = np.asarray(a["dx"]), np.asarray(a["dy"])
        colors = np.asarray(a["colors"])
        for color in np.unique(colors):
            m = colors == color
            tx, ty = x[m] + dx[m], y[m] + dy[m]  # arrow tips
            length = np.hypot(dx[m], dy[m])
            head = np.minimum(0.25 * length, 3.0)  # arrowhead size: a quarter of the arrow, at most 3 m
            ang = np.arctan2(dy[m], dx[m])
            xs, ys = [_with_gaps(x[m], tx)], [_with_gaps(y[m], ty)]
            for side in (+1, -1):  # two short lines 25 degrees back from the tip
                back = ang + np.pi + side * np.radians(25)
                xs.append(_with_gaps(tx, tx + head * np.cos(back)))
                ys.append(_with_gaps(ty, ty + head * np.sin(back)))
            fig.add_trace(_lines_trace(np.concatenate(xs), np.concatenate(ys), color, 1.6))
    elif t == "segments" and len(a["segments"]):
        seg = np.asarray(a["segments"]) - np.array([left, bottom])
        fig.add_trace(_lines_trace(_with_gaps(seg[:, 0, 0], seg[:, 1, 0]), _with_gaps(seg[:, 0, 1], seg[:, 1, 1]),
                                   a["color"], a.get("width", 2.0), name=a.get("name")))
    elif t == "circle":
        ang = np.linspace(0, 2 * np.pi, 49)
        fig.add_trace(go.Scatter(x=a["x"] - left + a["r"] * np.cos(ang), y=a["y"] - bottom + a["r"] * np.sin(ang),
                                 mode="lines", line=dict(color=a["color"], width=1.5, dash="dash"),
                                 hoverinfo="skip", showlegend=False))
    elif t in ("polygon", "line"):
        xy = np.array(a["xy"], dtype=float) - np.array([left, bottom])
        if t == "polygon":
            xy = np.vstack([xy, xy[:1]])
        fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="lines", line=dict(color=a["color"], width=2),
                                 hoverinfo="skip", showlegend=False))
    elif t == "point":
        fig.add_trace(go.Scatter(x=[a["x"] - left], y=[a["y"] - bottom], mode="markers+text", text=[a.get("label", "")],
                                 textposition="top right", marker=dict(size=9, color=a["color"]),
                                 textfont=dict(color="white"), hoverinfo="skip", showlegend=False))


# ---------------------------------------------------------------------------
# Static map (PNG download), full detail
# ---------------------------------------------------------------------------
def render_png(survey, overlay, annotations, use_photo=True, opacity=0.7, dpi=200) -> bytes:
    """The same map as a picture file for reports and the science-fair board."""
    dtm = survey.dtm
    left, bottom = dtm.bounds[0], dtm.bounds[1]
    extent = local_extent(dtm)
    fig, ax = new_map_axes()
    ax.imshow(background_rgb(survey.key, use_photo, survey), extent=extent, interpolation="bilinear")
    handles = []
    if overlay is not None:
        data = np.asarray(overlay.data, dtype=float)
        if overlay.classes:
            values = [v for v, _, _ in overlay.classes]
            cmap = ListedColormap([c for _, _, c in overlay.classes])
            norm = BoundaryNorm([v - 0.5 for v in values] + [values[-1] + 0.5], cmap.N)
            ax.imshow(np.ma.masked_where(~np.isin(data, values), data), cmap=cmap, norm=norm, extent=extent,
                      alpha=opacity, interpolation="nearest")
            handles = [Patch(color=c, label=l) for _, l, c in overlay.classes]
        else:
            im = ax.imshow(np.ma.masked_invalid(data), cmap=overlay.cmap, vmin=overlay.vmin, vmax=overlay.vmax,
                           extent=extent, alpha=opacity, interpolation="nearest")
            fig.colorbar(im, ax=ax, shrink=0.8, label=overlay.name)

    def loc(x, y):
        return x - left, y - bottom

    for a in annotations:
        t = a["type"]
        if t == "circle":
            ax.add_patch(Circle(loc(a["x"], a["y"]), a["r"], fill=False, edgecolor=a["color"], linewidth=1.5,
                                linestyle="--"))
        elif t == "polygon":
            ax.add_patch(Polygon([loc(*v) for v in a["xy"]], closed=True, fill=False, edgecolor=a["color"], linewidth=2))
        elif t == "line":
            xy = np.array([loc(*v) for v in a["xy"]])
            ax.plot(xy[:, 0], xy[:, 1], color=a["color"], linewidth=2)
        elif t == "point":
            x, y = loc(a["x"], a["y"])
            ax.plot(x, y, "o", markersize=5, color=a["color"], markeredgecolor="black", markeredgewidth=0.5)
        elif t == "segments" and len(a["segments"]):
            ax.add_collection(LineCollection(np.asarray(a["segments"]) - np.array([left, bottom]), colors=a["color"],
                                             linewidths=a.get("width", 1.2)))
        elif t == "scatter" and len(a["x"]):
            ax.scatter(np.asarray(a["x"]) - left, np.asarray(a["y"]) - bottom, s=a.get("size", 8), c=a["colors"],
                       edgecolors="black", linewidths=0.3, zorder=3)
        elif t == "arrows" and len(a["x"]):
            ax.quiver(np.asarray(a["x"]) - left, np.asarray(a["y"]) - bottom, a["dx"], a["dy"], color=a["colors"],
                      angles="xy", scale_units="xy", scale=1, width=0.002, headwidth=4, zorder=4)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(overlay.name if overlay is not None else "")
    out = BytesIO()
    fig.savefig(out, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out.getvalue()
