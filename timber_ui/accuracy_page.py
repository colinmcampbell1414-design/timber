"""Accuracy checker: compares the ground model with heights measured by hand at check points.

Background for TIMBER: it gives the best-case vertical error of the ground model on open ground.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import linregress

from terrain import accuracy, io, slope
from timber_ui.maps import local_extent, new_map_axes, show_fig
from timber_ui.survey_panel import load_native


def csv_points_input(folder):
    """Choose a check point CSV from data/ or upload one. Returns a DataFrame or None."""
    csv_files = sorted(folder.glob("*.csv")) if folder.exists() else []
    options = [p.name for p in csv_files] + ["Upload a file..."]
    choice = st.selectbox("Check point CSV (columns: id, x, y, measured_z)", options)
    if choice == "Upload a file...":
        upload = st.file_uploader("Check point CSV", type=["csv"])
        if upload is None:
            return None
        source = upload
    else:
        source = folder / choice
    try:
        return accuracy.read_checkpoints(source)
    except Exception as err:
        st.error(f"Could not read the CSV: {err}")
        return None


@st.cache_data(show_spinner="Loading photo...")
def cached_image(path: str, mtime: float, bounds: tuple | None):
    return io.load_image(path, bounds)


def draw_background(ax, dtm, photo, view_bounds=None):
    """Draw the orthophoto behind a map if there is one, otherwise the hillshade.

    view_bounds: (left, bottom, right, top) in map coordinates of the area being
    shown, so only that part of the photo is loaded (sharper when zoomed in).
    Returns True if the photo was used.
    """
    left, bottom = dtm.bounds[0], dtm.bounds[1]
    if photo is not None:
        key = tuple(round(v, 1) for v in view_bounds) if view_bounds else None
        try:
            img = cached_image(str(photo), photo.stat().st_mtime, key)
        except Exception:
            img = None
        if img is not None and img[2] == dtm.crs:
            rgba, (l, b, r, t), _ = img
            ax.imshow(rgba, extent=(l - left, r - left, b - bottom, t - bottom), interpolation="bilinear")
            return True
    ax.imshow(slope.hillshade(dtm.data, *dtm.pixel_size, z_factor=2.0), cmap="gray",
              extent=local_extent(dtm), vmin=0, vmax=1)
    return False


def corner_inputs(native):
    """Inputs for the two corner markers. Returns map coordinates (x1, y1, x2, y2)."""
    left, bottom, right, top = native.bounds
    # Starting values inside the map, just so the preview shows something.
    default_map = (left + 0.1 * (right - left), bottom + 0.2 * (top - bottom),
                   left + 0.9 * (right - left), bottom + 0.2 * (top - bottom))

    fmt = st.radio(
        "WebODM showed the coordinates as...",
        ["Map x / y (meters, e.g. 523456.12)", "Latitude / longitude (degrees, e.g. 40.71234)"],
        horizontal=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    if fmt.startswith("Map"):
        x1 = c1.number_input("First marker x (easting)", value=round(default_map[0], 3), step=0.01, format="%.3f")
        y1 = c2.number_input("First marker y (northing)", value=round(default_map[1], 3), step=0.01, format="%.3f")
        x2 = c3.number_input("End of row 1 x (easting)", value=round(default_map[2], 3), step=0.01, format="%.3f")
        y2 = c4.number_input("End of row 1 y (northing)", value=round(default_map[3], 3), step=0.01, format="%.3f")
        if max(abs(x1), abs(y1)) < 180:
            st.warning("These look like latitude/longitude, not map meters. Switch the option above.")
        return x1, y1, x2, y2

    lons, lats = io.map_to_latlon([default_map[0], default_map[2]], [default_map[1], default_map[3]], native.crs)
    lat1 = c1.number_input("First marker latitude", value=round(float(lats[0]), 7), step=1e-6, format="%.7f")
    lon1 = c2.number_input("First marker longitude", value=round(float(lons[0]), 7), step=1e-6, format="%.7f")
    lat2 = c3.number_input("End of row 1 latitude", value=round(float(lats[1]), 7), step=1e-6, format="%.7f")
    lon2 = c4.number_input("End of row 1 longitude", value=round(float(lons[1]), 7), step=1e-6, format="%.7f")
    xs, ys = io.latlon_to_map([lon1, lon2], [lat1, lat2], native.crs)
    st.caption(f"Converted to map coordinates: first ({xs[0]:.3f}, {ys[0]:.3f}), "
               f"end of row ({xs[1]:.3f}, {ys[1]:.3f}).")
    return float(xs[0]), float(ys[0]), float(xs[1]), float(ys[1])


def grid_points_input(native, folder, photo):
    """Grid helper: work out x, y for every marker from 2 corners, then add the hand-measured heights."""
    st.markdown(
        "**A. Where is the grid?** In WebODM, read the coordinates of just two markers: "
        "the **first marker** (column 1, row 1) and the **last marker in that same row**. "
        "The app works out where all the others are."
    )
    x1, y1, x2, y2 = corner_inputs(native)

    c1, c2, c3 = st.columns(3)
    cols = c1.number_input("Columns (markers per row)", min_value=2, value=17, step=1)
    rows = c2.number_input("Rows", min_value=1, value=6, step=1)
    spacing = c3.number_input("Spacing between rows (m)", min_value=0.01, value=0.5, step=0.05, format="%.3f")

    c1, c2 = st.columns(2)
    side = c1.radio(
        "Standing on the first marker, looking along row 1, the other rows are on your...",
        ["left", "right"], horizontal=True,
    )
    order_label = c2.radio(
        "How are the markers numbered in your notes?",
        ["Row by row (every row starts at column 1)", "Zig-zag (every other row runs backwards)"],
    )
    order = "rows" if order_label.startswith("Row") else "snake"

    try:
        grid = accuracy.grid_points(x1, y1, x2, y2, int(cols), int(rows), spacing, side, order)
    except ValueError as err:
        st.error(str(err))
        return None

    # Column spacing comes from the two corners; check it agrees with the spacing entered.
    col_step = float(np.hypot(x2 - x1, y2 - y1)) / (int(cols) - 1)
    st.caption(f"Spacing along each row, worked out from the two corners: {col_step:.3f} m.")
    if abs(col_step - spacing) > 0.05 * spacing:
        st.warning(
            f"The corners are {col_step:.3f} m apart per column, but row spacing is {spacing:.3f} m. "
            "If your grid is square, check the corner coordinates."
        )

    # Preview: zoom in on the grid so the markers are visible on the photo.
    margin = max(0.5, 2 * spacing)
    view = (grid["x"].min() - margin, grid["y"].min() - margin, grid["x"].max() + margin, grid["y"].max() + margin)
    left, bottom = native.bounds[0], native.bounds[1]
    fig, ax = new_map_axes()
    has_photo = draw_background(ax, native, photo, view)
    gx, gy = grid["x"] - left, grid["y"] - bottom
    ax.scatter(gx, gy, s=70, facecolors="none", edgecolors="#ff2a2a", linewidths=1.2)
    show_all = st.checkbox("Show every marker number on the preview")
    label_rows = range(len(grid)) if show_all else (0, int(cols) - 1, len(grid) - 1)
    for i in label_rows:
        ax.annotate(str(grid["id"].iloc[i]), (gx.iloc[i], gy.iloc[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=7 if show_all else 10, color="#ff2a2a", weight="bold")
    ax.set_xlim(view[0] - left, view[2] - left)
    ax.set_ylim(view[1] - bottom, view[3] - bottom)
    ax.set_title("Grid preview: each red circle should sit on a golf ball / rock")
    show_fig(fig)
    if not has_photo:
        st.info("Tip: put the orthophoto from WebODM (Download Assets → Orthophoto GeoTIFF) in the "
                "`data` folder. The preview will then show the photo, so you can see the golf balls.")

    # ---- B. Hand-measured heights ---------------------------------------
    st.markdown(
        f"**B. Your hand-measured heights.** Type or paste all {len(grid)} numbers **in marker "
        "order** (1, 2, 3, ...), separated by spaces, commas or new lines. For a marker you "
        "couldn't measure, type `x` so the others stay in order."
    )
    c1, c2 = st.columns(2)
    unit = c1.selectbox("Units you wrote them in", list(accuracy.UNIT_TO_METERS), index=1)
    downward = c2.checkbox(
        "These are distances measured DOWN from a level line / reference "
        "(bigger number = lower ground)",
        help="For example, measuring from a level string down to the ground. The app flips "
             "these so that higher ground has a bigger height, like the drone model.",
    )
    text = st.text_area("Measured heights", height=180, key="measured_heights_text",
                        placeholder="12.5  13.1  12.9  ...")
    if not text.strip():
        return None
    try:
        values = accuracy.parse_height_list(text)
    except ValueError as err:
        st.error(str(err))
        return None
    if len(values) != len(grid):
        st.error(f"You entered {len(values)} values, but the grid has {len(grid)} markers. "
                 "They must match so each height goes with the right marker.")
        return None

    table = grid[["id", "x", "y"]].copy()
    table["measured_z"] = accuracy.measured_to_meters(values, unit, downward)
    table["measured_raw"] = values  # the numbers exactly as written in the notes
    table["raw_unit"] = unit
    skipped = table.loc[table["measured_z"].isna(), "id"].tolist()
    if skipped:
        st.caption(f"Skipped markers: {', '.join(map(str, skipped))}")

    # ---- C. Save so nothing needs to be typed again ------------------------
    st.markdown("**C. Save** these check points as a CSV so you can reload them with **Load a CSV** next time.")
    c1, c2 = st.columns([2, 1])
    name = c1.text_input("File name (saved in the data folder)", value="checkpoints.csv")
    target = folder / Path(name).name
    exists = target.exists()
    overwrite = c2.checkbox("Overwrite existing file", value=False) if exists else True
    if c2.button("Save to data folder", disabled=not name.lower().endswith(".csv") or not overwrite):
        folder.mkdir(exist_ok=True)
        table.to_csv(target, index=False)
        st.success(f"Saved {target.name}.")
    st.download_button("...or download it", table.to_csv(index=False), Path(name).name or "checkpoints.csv", "text/csv")

    return table.dropna(subset=["measured_z"]).reset_index(drop=True)


def errors_map(raster, table, err_col, title, photo, highlight=None, labels=False):
    """Map of the check points colored by error, zoomed to the points."""
    left, bottom, _, _ = raster.bounds
    x, y = table["x"] - left, table["y"] - bottom
    err_cm = table[err_col] * 100
    limit = float(np.nanmax(np.abs(err_cm))) if np.isfinite(err_cm).any() else 1.0
    limit = max(limit, 1e-6)
    margin = 1.0
    view = (table["x"].min() - margin, table["y"].min() - margin, table["x"].max() + margin, table["y"].max() + margin)

    fig, ax = new_map_axes()
    draw_background(ax, raster, photo, view)
    # Red = model too high, blue = model too low (after removing the mean offset).
    if highlight is None:
        edge, width = "black", 0.5
    else:
        edge, width = np.where(highlight, "black", "none"), 1.5
    sc = ax.scatter(x, y, c=err_cm, cmap="RdBu_r", vmin=-limit, vmax=limit, s=90, marker="s",
                    edgecolors=edge, linewidths=width)
    fig.colorbar(sc, ax=ax, label="Error (cm): red = model too high, blue = too low")
    if labels:
        for xi, yi, pid in zip(x, y, table["id"]):
            ax.annotate(str(pid), (xi, yi), fontsize=6, ha="center", va="center")
    ax.set_xlim(view[0] - left, view[2] - left)
    ax.set_ylim(view[1] - bottom, view[3] - bottom)
    ax.set_title(title)
    return fig


def format_stats(df):
    """Show meters with 3 decimals (millimeters) and inches with 2."""
    fmt = {c: "{:.3f}" for c in df.columns if c.endswith("(m)")}
    fmt.update({c: "{:.2f}" for c in df.columns if c.endswith("(in)")})
    return df.style.format(fmt)


def accuracy_page(survey):
    st.title("Accuracy checker")
    st.caption("Ground model accuracy against check points you measured by hand: the best-case vertical "
               "error on open ground.")
    if survey is None:
        st.info("Choose a survey in the sidebar first.")
        return
    folder, photo, raster = survey.folder, survey.files.get("ortho"), survey.dtm
    dtm_path = survey.files["dtm"]
    native_raster = load_native(str(dtm_path), dtm_path.stat().st_mtime)
    st.write(
        "Compares the heights in the drone model with heights you measured by hand at check "
        "points. The model's absolute height is shifted by an unknown amount (no ground control "
        "points), so the app first removes the **average difference** (mean offset). The errors "
        "that are left show how well the model gets the *shape* of the ground right."
    )
    st.caption("Heights are read from the original, full-detail DTM (not the resampled one) "
               "using bilinear interpolation: a weighted average of the 4 nearest cells.")

    # ---- Step 1: check points --------------------------------------------
    st.subheader("Step 1: Check points")
    mode = st.radio("How do you want to enter check points?", ["Load a CSV", "Grid helper"], horizontal=True)
    points = csv_points_input(folder) if mode == "Load a CSV" else grid_points_input(native_raster, folder, photo)
    if points is None or points.empty:
        return
    st.success(f"{len(points)} check points loaded.")

    # ---- Step 2: models ---------------------------------------------------
    st.subheader("Step 2: Models to check")
    models = {"DTM": native_raster}
    tif_files = sorted(folder.glob("*.tif*")) if folder.exists() else []
    dsm_options = ["None"] + [p.name for p in tif_files]
    dsm_default = next((i + 1 for i, p in enumerate(tif_files) if "dsm" in p.name.lower()), 0)
    dsm_choice = st.selectbox("Also check a DSM (surface model, includes grass)?", dsm_options, index=dsm_default)
    if dsm_choice != "None":
        p = folder / dsm_choice
        try:
            models["DSM"] = load_native(str(p), p.stat().st_mtime)
        except Exception as err:
            st.error(f"Could not load the DSM: {err}")

    # Slope at each point comes from the working (resampled) DTM, which is less noisy.
    slope_deg = slope.slope_degrees(raster.data, *raster.pixel_size)
    slope_raster = io.Raster(data=slope_deg, transform=raster.transform, crs=raster.crs)
    table, offsets = accuracy.compare_points(points, models, slope_raster)

    for name in models:
        missed = table.loc[table[f"{name}_z"].isna(), "id"].tolist()
        if missed:
            st.warning(f"{len(missed)} point(s) are outside the {name} or on missing data and were "
                       f"left out: {', '.join(map(str, missed))}")

    # ---- Step 3: results --------------------------------------------------
    st.subheader("Step 3: Results")
    cols = st.columns(len(models))
    for col, name in zip(cols, models):
        col.metric(f"{name} mean offset", f"{offsets[name]:+.3f} m")
    st.caption(
        "The mean offset is the average of (model height − measured height). It includes the "
        "height of your reference point and the drone's GPS error, so it is removed and is "
        "**not** an accuracy measure."
    )

    full_stats = {name: accuracy.error_stats(table[f"{name}_error"]) for name in models}
    st.dataframe(format_stats(accuracy.stats_table(full_stats)), hide_index=True, width="stretch")
    with st.expander("What do these statistics mean?"):
        st.markdown(
            "- **Mean error**: average error after removing the offset. It is 0 by design (a check).\n"
            "- **Standard deviation**: how spread out the errors are.\n"
            "- **RMSE** (root mean square error): square each error, average them, take the square "
            "root. The most common single accuracy number.\n"
            "- **Max absolute error**: the single worst point.\n"
            "- **LE90**: 90% of points have an error smaller than this.\n"
        )

    name = st.radio("Show charts for", list(models), horizontal=True) if len(models) > 1 else "DTM"
    err_col = f"{name}_error"
    valid = table.dropna(subset=[err_col]).reset_index(drop=True)
    if valid.empty:
        st.error("None of the check points could be sampled. Check that x, y use the DTM's coordinate system.")
        return

    labels = st.checkbox("Show point numbers on the map")
    left, right = st.columns(2)
    with left:
        show_fig(errors_map(native_raster, valid, err_col, f"{name} error at each check point", photo, labels=labels))
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(valid[err_col] * 100, bins=min(20, max(5, len(valid) // 4)), color="#4575b4", edgecolor="white")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Error (cm)")
        ax.set_ylabel("Number of points")
        ax.set_title(f"{name} error histogram")
        show_fig(fig)

        use_abs = st.radio("Error vs slope: plot", ["Absolute error (size only)", "Signed error"],
                           horizontal=True).startswith("Absolute")
        sub = valid.dropna(subset=["slope_deg"])
        yvals = (sub[err_col].abs() if use_abs else sub[err_col]) * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(sub["slope_deg"], yvals, s=18, color="#4575b4")
        ax.set_xlabel(f"Local slope (degrees, from {raster.pixel_size[0] * 100:.0f} cm grid)")
        ax.set_ylabel("Absolute error (cm)" if use_abs else "Error (cm)")
        if len(sub) >= 3 and sub["slope_deg"].nunique() > 1:
            # Trend line: the straight line that best fits the points (least squares).
            fit = linregress(sub["slope_deg"], yvals)
            xs = np.linspace(sub["slope_deg"].min(), sub["slope_deg"].max(), 50)
            ax.plot(xs, fit.intercept + fit.slope * xs, color="#d73027",
                    label=f"trend: {fit.slope:+.2f} cm per degree, R² = {fit.rvalue ** 2:.2f}")
            ax.legend(fontsize=8)
        ax.set_title(f"{name} error vs slope")
        show_fig(fig)
    st.caption("R² is how well the trend line fits, from 0 (no pattern) to 1 (perfect line). "
               "A small R² means slope doesn't explain much of the error.")

    # ---- Subset comparison -------------------------------------------------
    st.subheader("Subset comparison")
    st.write(
        "Would fewer check points give the same answer? The app picks an evenly spread subset "
        "(start with one point, then repeatedly add the point farthest from those already picked) "
        "and computes the same statistics, including its own mean offset."
    )
    if len(valid) < 3:
        st.info("Need at least 3 sampled points for a subset comparison.")
    else:
        k = st.slider("Subset size", min_value=2, max_value=len(valid), value=min(30, len(valid)))
        idx = accuracy.evenly_spread_subset(valid["x"], valid["y"], k)
        in_subset = np.zeros(len(valid), dtype=bool)
        in_subset[idx] = True
        sub_errors, _ = accuracy.align_errors(valid.loc[in_subset, f"{name}_z"], valid.loc[in_subset, "measured_z"])
        compare = {"All points": full_stats[name], f"Subset of {k}": accuracy.error_stats(sub_errors)}
        left, right = st.columns(2)
        left.dataframe(format_stats(accuracy.stats_table(compare)), hide_index=True, width="stretch")
        with right:
            show_fig(errors_map(native_raster, valid, err_col, "Subset points (black outline)", photo, highlight=in_subset))

    # ---- Export -----------------------------------------------------------
    st.subheader("Export")
    export = table.copy()
    for m in models:
        export[f"{m}_error_in"] = export[f"{m}_error"] / accuracy.METERS_PER_INCH
    if len(valid) >= 3:
        export[f"in_subset_{name}"] = export["id"].isin(valid.loc[in_subset, "id"])
    st.download_button("Download results CSV", export.to_csv(index=False), "accuracy_results.csv", "text/csv")
    with st.expander("Show results table"):
        st.dataframe(export, hide_index=True)
