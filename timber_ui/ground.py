"""Ground check: is the survey's ground model believable? Overview, slope and water flow."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from terrain import fusion, hydrology, io, slope, validation
from timber_ui.paths import RESULTS_DIR, WORK_DIR
from timber_ui.survey_panel import dtm_sources
from timber_ui.maps import local_extent, new_map_axes, show_fig
from timber_ui.accuracy_page import draw_background

COMPASS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]


def compass_name(degrees: float) -> str:
    return COMPASS[int((degrees % 360 + 22.5) // 45) % 8]


def ground_page(survey):
    st.title("Ground check")
    st.caption("Look at the ground model before trusting the tree results: under thick leaves WebODM sometimes "
               "can't see the ground.")
    if survey is None:
        st.info("Choose a survey in the sidebar first.")
        return
    # One view at a time (tabs would draw all three every time, which is slow on big maps).
    views = {"Overview": overview_tab, "Slope": slope_tab, "Water flow": water_tab, "DTM comparison": dtm_tab}
    choice = st.segmented_control("View", list(views), default="Overview", key="ground_view",
                                  label_visibility="collapsed") or "Overview"
    views[choice](survey)


@st.cache_resource(show_spinner=False, max_entries=6)
def _aligned_dtm(survey_key: str, path: str, mtime: float, units: str, _survey):
    """Another ground model on this survey's grid, lined up with the survey's own DTM."""
    other = fusion.to_grid(io.load_raster(path), _survey.own_dtm, units)
    return fusion.align_dtm(other.data, _survey.own_dtm.data, _survey.dsm.data, _survey.own_dtm.pixel_size[0])


def dtm_tab(survey):
    """Compare ground models: this survey's own, other surveys', and outside files."""
    st.write("Photogrammetry can't see the ground under leaves, so a leaf-on ground model rides up into the canopy "
             "and trees look too short. This compares every available ground model for this area. To USE one for "
             "the tree analysis, pick it under **Ground model (DTM) source** in the sidebar.")
    if survey.dsm is None:
        st.info("This survey has no DSM, so canopy heights can't be compared.")
        return
    available = dtm_sources(survey.folder)
    if not available:
        st.info("No other ground model yet. Add a leaf-off survey of the same area as its own folder in the data "
                "folder, or put a lidar GeoTIFF in data/outside_dtm.")
        return
    units = st.selectbox("Height units of the outside files", fusion.UNIT_CHOICES, key="dtm_compare_units")
    units = "auto" if units.startswith("auto") else units
    own_name = f"survey: {survey.name}"
    sources, reports = {own_name: survey.own_dtm.data}, {}
    for label, path in available.items():
        try:
            with st.spinner(f"Lining up {label}..."):
                aligned, report = _aligned_dtm(survey.key.split("_")[0], str(path), path.stat().st_mtime, units, survey)
            if np.isfinite(aligned).mean() > 0.05:  # skip models that don't cover this area
                sources[label], reports[label] = aligned, report
        except Exception as err:
            st.warning(f"{label} could not be used: {err}")
    if len(sources) < 2:
        st.info("None of the other ground models covers this survey's area.")
        return
    reference = st.selectbox("Reference ground model", list(sources), index=len(sources) - 1,
                             help="Usually the leaf-off survey or the lidar: the one you trust most.")
    cell = survey.own_dtm.pixel_size[0]
    table = fusion.compare_dtms(sources, reference, survey.dsm.data, cell)
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="%.2f") for c in table.columns if c != "DTM source"})
    align = pd.DataFrame([{"DTM source": k, "offset removed (m)": r["offset_before_m"], "tilt removed (°)": r["tilt_deg"],
                           "noise left on stable ground (m)": r["rms_after_m"]} for k, r in reports.items()])
    st.markdown("**Vertical alignment on stable open ground**")
    st.dataframe(align, hide_index=True, width="stretch")

    sentences = [f"{row['DTM source']}: under canopy {row['under canopy: mean difference (m)']:+.2f} m from the reference "
                 f"on average (RMS {row['under canopy: RMS (m)']:.2f} m), open ground RMS {row['open ground: RMS (m)']:.2f} m; "
                 f"median canopy height {row['median canopy height (m)']:.1f} m." for _, row in table.iterrows()]
    field = RESULTS_DIR / "field_check_results.csv"
    details = {"table": table, "reference": reference}
    if field.exists():
        checked = pd.read_csv(field)
        checked["observed_height_m"] = pd.to_numeric(checked.get("observed_height_m"), errors="coerce")
        checked = checked.dropna(subset=["observed_height_m"])
        if len(checked):
            rows = []
            for label, dtm in sources.items():
                app = fusion.heights_at(checked["x"], checked["y"], survey.dsm.data, dtm, survey.own_dtm.transform)
                rows.append({"DTM source": label, **validation.height_errors(app, checked["observed_height_m"])})
            heights = pd.DataFrame(rows)
            st.markdown("**Tree height error against the field heights, for every ground model** (the before / after result)")
            st.dataframe(heights, hide_index=True, width="stretch")
            details["field_heights"] = heights
            sentences += [f"Field heights with {r['DTM source']}: RMSE {r['rmse_m']:.2f} m (n = {r['n']})." for _, r in heights.iterrows()]
    else:
        st.caption("After a blind field check with heights, this page also shows the height RMSE for every ground model.")
    validation.save_result(WORK_DIR, "results/dtm_comparison.json", {}, sentences, details)

    pick = st.multiselect("Hillshades side by side", list(sources), default=list(sources)[:2], max_selections=3)
    cols = st.columns(max(len(pick), 1))
    extent = local_extent(survey.own_dtm)
    for col, label in zip(cols, pick):
        with col:
            fig, ax = new_map_axes()
            ax.imshow(slope.hillshade(sources[label], cell, cell, z_factor=2.0), cmap="gray", extent=extent, vmin=0, vmax=1)
            ax.set_title(label)
            show_fig(fig)


def overview_tab(survey):
    raster = survey.dtm
    st.write(
        "The DTM (Digital Terrain Model) is a grid of bare-ground heights made by WebODM "
        "from the drone photos. Below is a shaded-relief view and a map of heights."
    )

    info = io.summarize(raster)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original cell size", f"{survey.info.get('native_cell_m', info['pixel_x_m']) * 100:.2f} cm")
    c2.metric("Working cell size", f"{info['pixel_x_m'] * 100:.2f} cm")
    c3.metric("Grid size", f"{info['cols']} x {info['rows']}")
    c4.metric("Height range (relief)", f"{info['relief_m']:.2f} m")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Map width", f"{info['width_m']:.1f} m")
    c2.metric("Map height", f"{info['height_m']:.1f} m")
    c3.metric("Area with data", f"{info['area_m2']:.0f} m²")
    c4.metric("Cells with data", f"{info['valid_fraction'] * 100:.0f} %")
    st.caption(f"Coordinate system: {info['crs']}. Lower-left corner at map x = "
               f"{raster.bounds[0]:.2f}, y = {raster.bounds[1]:.2f}.")


    st.markdown("**Shading**")
    c1, c2, c3 = st.columns(3)
    azimuth = c1.slider("Sun direction (degrees from north)", 0, 360, 315, 5)
    altitude = c2.slider("Sun height above horizon (degrees)", 5, 90, 45, 5)
    z_factor = c3.slider("Vertical exaggeration", 1.0, 10.0, 2.0, 0.5,
                         help="Multiplies heights for the shading only, to make small bumps visible.")

    px, py = raster.pixel_size
    shade = slope.hillshade(raster.data, px, py, azimuth, altitude, z_factor)
    rel = io.relative_heights(raster.data)
    extent = local_extent(raster)

    left, right = st.columns(2)
    with left:
        fig, ax = new_map_axes()
        ax.imshow(shade, cmap="gray", extent=extent, vmin=0, vmax=1)
        ax.set_title("Hillshade (shaded relief)")
        show_fig(fig)
    with right:
        fig, ax = new_map_axes()
        ax.imshow(shade, cmap="gray", extent=extent, vmin=0, vmax=1)
        im = ax.imshow(rel, cmap="terrain", extent=extent, alpha=0.6)
        fig.colorbar(im, ax=ax, label="Height above lowest point (m)")
        ax.set_title("Relative height")
        show_fig(fig)

    st.caption(
        "Heights are shown relative to the lowest point in the map. Absolute elevation "
        "(height above sea level) is not reliable in this project because no ground "
        "control points were used and drone GPS is off by about 1 m, but height "
        "differences within the map are meaningful."
    )


def slope_tab(survey):
    raster = survey.dtm
    st.write(
        "Slope is how steep the ground is, in degrees: 0° is flat and 90° is a vertical "
        "wall. Steep slopes are more likely to have mudslides and erosion. "
        "Adjust the class boundaries below."
    )

    st.subheader("Slope class boundaries (degrees)")
    names = slope.SLOPE_CLASS_NAMES
    breaks = []
    cols = st.columns(len(slope.DEFAULT_SLOPE_BREAKS))
    lower = 0.0
    for i, (col, default) in enumerate(zip(cols, slope.DEFAULT_SLOPE_BREAKS)):
        # Each boundary must be bigger than the previous one.
        value = col.slider(
            f"{names[i]} / {names[i + 1]}",
            min_value=lower + 0.5,
            max_value=85.0 + i,
            value=max(default, lower + 0.5),
            step=0.5,
        )
        breaks.append(value)
        lower = value

    px, py = raster.pixel_size
    slope_deg = slope.slope_degrees(raster.data, px, py)
    classes = slope.classify_slope(slope_deg, breaks)
    extent = local_extent(raster)
    shade = slope.hillshade(raster.data, px, py, z_factor=2.0)

    colors = ["#1a9850", "#a6d96a", "#fee08b", "#f46d43", "#a50026"]
    left, right = st.columns(2)
    with left:
        fig, ax = new_map_axes()
        ax.imshow(shade, cmap="gray", extent=extent, vmin=0, vmax=1)
        masked = np.ma.masked_less(classes, 0)  # hide no-data cells
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), cmap.N)
        ax.imshow(masked, cmap=cmap, norm=norm, extent=extent, alpha=0.7, interpolation="nearest")
        edges = [0.0, *breaks, 90.0]
        handles = [Patch(color=colors[k], label=f"{names[k]} ({edges[k]:g}–{edges[k + 1]:g}°)")
                   for k in range(len(colors))]
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_title("Slope classes")
        show_fig(fig)
    with right:
        vmax = st.slider("Color scale maximum (degrees)", 5, 90, 45, 5)
        fig, ax = new_map_axes()
        im = ax.imshow(slope_deg, cmap="magma_r", extent=extent, vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, label="Slope (degrees)")
        ax.set_title("Slope (continuous)")
        show_fig(fig)

    st.subheader("How much ground is in each class")
    table = slope.slope_class_table(slope_deg, breaks, cell_area_m2=px * py)
    left, right = st.columns(2)
    left.dataframe(table.style.format({"Area (m²)": "{:.1f}", "Percent": "{:.1f}"}),
                   hide_index=True, width="stretch")
    with right:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        valid = slope_deg[np.isfinite(slope_deg)]
        ax.hist(valid, bins=60, range=(0, max(vmax, float(np.nanmax(valid)) if valid.size else 1)),
                color="#555555")
        for b in breaks:
            ax.axvline(b, color="#d73027", linestyle="--", linewidth=1)
        ax.set_xlabel("Slope (degrees)")
        ax.set_ylabel("Number of cells")
        ax.set_title("Slope histogram (red lines = class boundaries)")
        show_fig(fig)

    st.caption(
        f"Slope is measured over neighboring cells, so it depends on cell size "
        f"(currently {px * 100:.1f} cm). Smaller cells show more small-scale bumps from "
        "grass and noise; larger cells show the overall shape of the land."
    )


def water_tab(survey):
    raster = survey.dtm
    st.write(
        "Where does rain go, and where does it collect? The app fills every dip in the ground "
        "up to the level where water would spill out (**ponding**), then sends water from each "
        "cell to its steepest downhill neighbor (**flow direction**) and counts how much ground "
        "drains through each cell (**flow accumulation**)."
    )
    px, py = raster.pixel_size
    cell_area = px * py
    st.caption(f"Working cell size: {px * 100:.1f} cm (change it in the sidebar's survey settings). "
               "Smaller cells show more detail but also more tiny dips from grass and noise.")

    # ---- Settings ---------------------------------------------------------
    photo = survey.files.get("ortho")
    background = st.radio("Background", ["Orthophoto", "Hillshade"] if photo else ["Hillshade"], horizontal=True,
                          key="water_background")

    c1, c2, c3 = st.columns(3)
    min_depth_cm = c1.slider("Minimum ponding depth (cm)", 0.1, 20.0, 1.0, 0.1,
                             help="Cells where water would be deeper than this count as ponding.")
    min_area = c2.number_input("Ignore ponds smaller than (m²)", min_value=0.0, value=0.05, step=0.01, format="%.3f",
                               help=f"One cell is {cell_area:.4f} m². Tiny ponds are usually noise.")
    result = survey.flow
    path_options = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0,
                    5000.0, 10000.0]
    threshold = c3.select_slider(
        "Flow path threshold: area draining through a cell (m²)",
        options=path_options,
        # Starting value: the smallest one that keeps flow paths to about 2% of the map.
        value=hydrology.suggest_path_threshold(result["acc_area_m2"], path_options),
        help="Cells with at least this much uphill ground draining through them are drawn as flow paths. "
             "The starting value keeps the paths to about 2% of the map, so big maps aren't one big tangle.",
    )
    if survey.tilt is not None:
        tilt = survey.tilt
        st.info(
            f"Removed a tilt of **{tilt['tilt_deg']:.1f}°**, sloping down toward the "
            f"**{compass_name(tilt['downhill_toward_deg'])}** ({tilt['downhill_toward_deg']:.0f}°). "
            "This removes the model's overall tilt, whether it is a model error or the real slope "
            "of the land. If the land really slopes, the results with tilt removed will show "
            "ponds that would not really form, so compare both, and don't trust just this one."
        )

    labels, ponds = hydrology.find_ponds(result["depth"], min_depth_cm / 100.0, cell_area, min_area)
    acc_area = result["acc_area_m2"]
    stream = np.nan_to_num(acc_area, nan=0.0) >= threshold
    segments = hydrology.flow_segments(result["receiver"], stream, px, py)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ponds", f"{len(ponds)}")
    c2.metric("Total pond area", f"{ponds['area_m2'].sum():.2f} m²")
    c3.metric("Total water held", f"{ponds['volume_m3'].sum() * 1000:.0f} liters")
    c4.metric("Deepest pond", f"{ponds['max_depth_m'].max() * 100:.1f} cm" if len(ponds) else "–")

    # ---- Maps ---------------------------------------------------------------
    left, right = st.columns(2)
    extent = local_extent(raster)
    with left:
        fig, ax = new_map_axes()
        if background == "Orthophoto":
            draw_background(ax, raster, survey.files.get("ortho"), raster.bounds)
        else:
            ax.imshow(slope.hillshade(raster.data, px, py, z_factor=2.0), cmap="gray", extent=extent, vmin=0, vmax=1)
        # Ponds in blue, darker = deeper. The color scale starts at a clearly visible
        # blue so even the shallowest ponds stand out against the background.
        pond_depth_cm = np.ma.masked_where(labels == 0, result["depth"] * 100)
        if len(ponds):
            pond_cmap = ListedColormap(plt.cm.Blues(np.linspace(0.45, 1.0, 256)))
            im = ax.imshow(pond_depth_cm, cmap=pond_cmap, extent=extent, interpolation="nearest",
                           vmin=min_depth_cm, vmax=max(float(pond_depth_cm.max()), min_depth_cm + 0.1), alpha=0.9)
            fig.colorbar(im, ax=ax, label="Ponding depth (cm)", shrink=0.8)
        # Flow paths as lines: thicker = more water.
        if len(segments):
            seg_acc = acc_area.ravel()[np.flatnonzero(stream.ravel())]
            seg_acc = seg_acc[result["receiver"][np.flatnonzero(stream.ravel())] >= 0]
            widths = 0.6 + 2.4 * np.log10(seg_acc / threshold + 1) / np.log10(np.nanmax(acc_area) / threshold + 1)
            ax.add_collection(LineCollection(segments, colors="#00d5ff", linewidths=widths))
        # Number the biggest ponds so they can be found in the table.
        rows = raster.shape[0]
        for _, p in ponds.head(20).iterrows():
            # The label sits a little up and to the right, so it doesn't hide small ponds.
            ax.annotate(str(int(p["pond"])), ((p["col"] + 0.5) * px, (rows - p["row"] - 0.5) * py),
                        xytext=(9, 9), textcoords="offset points",
                        color="white", fontsize=8, weight="bold", ha="center", va="center",
                        bbox=dict(boxstyle="circle,pad=0.15", fc="#08306b", ec="none", alpha=0.85),
                        arrowprops=dict(arrowstyle="-", color="#08306b", lw=0.8))
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.legend(handles=[Patch(color="#2171b5", label="Ponding"),
                           Line2D([], [], color="#00d5ff", lw=2, label=f"Flow path (≥ {threshold:g} m² draining)")],
                  loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_title("Predicted ponding and main flow paths")
        show_fig(fig)
    with right:
        fig, ax = new_map_axes()
        ax.imshow(slope.hillshade(raster.data, px, py, z_factor=2.0), cmap="gray", extent=extent, vmin=0, vmax=1)
        log_min, log_max = np.log10(cell_area), np.log10(np.nanmax(acc_area))
        im = ax.imshow(np.log10(acc_area), cmap="viridis", extent=extent, alpha=0.75, interpolation="nearest",
                       vmin=log_min, vmax=max(log_max, log_min + 0.1))
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        ticks = [t for t in range(-4, 6) if log_min <= t <= log_max]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{10.0 ** t:g}" for t in ticks])
        cbar.set_label("Area draining through cell (m², log scale)")
        ax.set_title("Flow accumulation")
        show_fig(fig)

    # ---- Pond table ---------------------------------------------------------
    st.subheader("Ponds")
    if ponds.empty:
        st.write("No ponds deeper than the minimum depth and larger than the minimum area.")
    else:
        table = pd.DataFrame({
            "Pond": ponds["pond"],
            "Area (m²)": ponds["area_m2"],
            "Max depth (cm)": ponds["max_depth_m"] * 100,
            "Average depth (cm)": ponds["mean_depth_m"] * 100,
            "Water held (liters)": ponds["volume_m3"] * 1000,
            "Meters east": (ponds["col"] + 0.5) * px,
            "Meters north": (raster.shape[0] - ponds["row"] - 0.5) * py,
        })
        st.dataframe(table.style.format({
            "Area (m²)": "{:.3f}", "Max depth (cm)": "{:.1f}", "Average depth (cm)": "{:.1f}",
            "Water held (liters)": "{:.1f}", "Meters east": "{:.1f}", "Meters north": "{:.1f}",
        }), hide_index=True, width="stretch")
        st.caption("Location is the pond's center, in meters from the map's lower-left corner "
                   "(the same as the map axes). Ponds are numbered from most to least water.")
        st.download_button("Download pond table (CSV)", table.to_csv(index=False), "ponds.csv", "text/csv")

    with st.expander("How this works, and its limits"):
        st.markdown(
            "1. **Fill depressions**: imagine water rising from the edges of the map. The lowest "
            "unvisited cell is always handled next, and any neighbor lower than it is raised to its "
            "level. Dips are filled up to where they would overflow.\n"
            "2. **Ponding depth** = filled surface − original ground.\n"
            "3. **Flow direction (D8)**: each cell sends its water to whichever of its 8 neighbors "
            "is steepest downhill. On the flat tops of filled ponds, water heads to the spill point.\n"
            "4. **Flow accumulation**: every cell gets 1 cell of rain and passes everything it has "
            "collected downhill. The total is shown as area (m²) of ground draining through each cell.\n\n"
            "**Limits:** water reaching the map edge or a no-data area is assumed to flow away. "
            "Grass, leaves and model noise make small fake dips, so tiny ponds are usually not real. "
            "Soil soaking up water is ignored, so these are worst-case puddles in heavy rain."
        )
