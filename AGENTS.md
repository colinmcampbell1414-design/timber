# TIMBER — Project Instructions

**TIMBER** = **T**ree **I**nstability **M**apping **B**y **E**levation & **R**eflectance
(elevation = the DSM/DTM; reflectance = the VARI/ExG health check). Use this name
in the UI and docs. The project is focused on the tree fall hazard tool.

## What this project is
A science fair project on drone photogrammetry for environmental and safety uses.
Drone photos are processed in WebODM into elevation models. This app loads those
outputs and runs screening algorithms to flag terrain hazards:
- slope / mudslide susceptibility
- water flow, ponding, and simple irrigation planning
- tree fall risk (later phase)
- an accuracy checker that compares WebODM elevations to hand-measured check points

This is a **screening tool for a student project**, not an engineering product.
The UI should say so. All thresholds must be adjustable (sliders), and every
formula should have a short comment explaining it in plain language, because the
student must be able to explain the code to judges.

## TIMBER UI and speed rules
- UI code goes in `timber_ui/` (app.py only sets up pages). Math stays in `terrain/`.
- Slow results must be cached: surveys via `terrain/survey.py` (disk), analyses via
  `Layers.remember` (disk) plus `st.cache_resource` (memory). Bump `CACHE_VERSION` /
  `RESULT_CACHE_VERSION` when a preparation or analysis step changes its output.
- Settings live in an `st.form` (nothing re-runs until "Run analysis"); show only one
  result part at a time; make big downloads (GeoTIFF, PNG) only when asked.
- Interactive maps use `timber_ui/maps.py` (Plotly, picture shrunk to <= 1600 px).
- Keep TIMBER.exe working: new libraries or data files must be added to TIMBER.spec,
  and files may only be written to the data/cache folders from timber_ui/paths.py
  (an installed copy can't write to its own Program Files folder).

## Tech stack
- Python 3.11+
- Streamlit for the UI. Normally started through `timber.py`, which shows it in
  its own desktop window; `streamlit run app.py` (browser) also still works.
  The desktop build must keep working: if you add a library or a data file the
  app needs, add it to TIMBER.spec too.
- rasterio (read GeoTIFFs), numpy, scipy
- pandas (check point CSVs)
- matplotlib or plotly for maps and charts
- pysheds for depression filling, flow direction, and flow accumulation
  (if it fails to install, write a simple numpy D8 implementation instead)
- pytest for tests

Keep dependencies in `requirements.txt`. Avoid heavy GIS installs (no GDAL CLI,
no QGIS dependency).

## Input data (from WebODM "Download Assets")
Put files in `data/` (git-ignored if large):
- `dtm.tif` — Digital Terrain Model (bare ground). Main input.
- `dsm.tif` — Digital Surface Model (ground + grass/objects/trees). Optional.
- `orthophoto.tif` — map image. Optional, used as a background.

Facts about this data:
- Rasters are in a projected CRS in **meters** (UTM). Always get pixel size from
  the raster transform; never assume it.
- Handle nodata values (mask them; never treat them as elevation 0).
- Absolute elevation is NOT trustworthy (no ground control points; GPS error
  about 1 m). Only elevation *differences* are meaningful. Never display raw
  elevations as "height above sea level".
- Native resolution is very fine. For flow and hazard analysis, allow resampling
  to a user-chosen cell size (default 5 cm) to reduce noise and speed things up.

## App structure
```
timber.py              # desktop launcher: TIMBER in its own window (pywebview + hidden Streamlit);
                       # chooses data/cache/log folders (project, portable, or installed)
TIMBER.spec, build_exe.ps1, requirements-build.txt  # build dist\TIMBER\TIMBER.exe (PyInstaller)
installer/TIMBER.iss, build_installer.ps1           # Windows installer (Inno Setup 6)
app.py                 # sets up the pages (st.navigation); the page code is in timber_ui/
timber_ui/             # UI only: home, analysis_page (trees + slope), ground, accuracy_page,
                       # survey_panel (sidebar), maps (Plotly + PNG), widgets, style, paths
.streamlit/config.toml # TIMBER's theme
terrain/
  survey.py            # prepare a survey (DTM + DSM + photo + flow on one grid) and cache it on disk
  io.py                # load rasters, mask nodata, resample, read pixel size
  slope.py             # slope (degrees), aspect, curvature, hillshade
  hydrology.py         # fill depressions, flow direction, flow accumulation,
                       # ponding depth, topographic wetness index (TWI)
  accuracy.py          # check point comparison and statistics
  vegetation.py        # VARI, ExG and color saturation from the orthophoto
  treefall.py          # tree fall hazard math (the trees app is only the wrapper)
  apps/                # Phase 5 applications, one module each (see below)
    __init__.py        # registry: APPS list that builds the dropdown
    base.py            # shared Layers / Result / Overlay / ShapeSpec + helpers
    trees.py mudslide.py   # tree fall hazard (the focus) and slope stability
tests/                 # pytest tests using synthetic DEMs
data/                  # user's WebODM exports (not committed)
```

## Build phases (do one phase at a time; stop and let the student test)
1. **Load and view**: upload or pick a DTM, show hillshade and elevation map,
   show resolution, size, and elevation range (as relative heights).
2. **Slope**: slope map in degrees with adjustable class thresholds.
3. **Accuracy checker** (see below).
4. **Water flow**: fill depressions, flow accumulation map, predicted ponding
   areas (filled DEM minus original DEM > adjustable depth threshold).
   Show flow paths over the hillshade or orthophoto.
5. **Hazards & Applications**: one page with a dropdown of applications:
   tree fall hazard and slope stability (mudslide) only. The other Phase 5
   apps (erosion, drainage, irrigation, rain garden, ice, trip hazards, volume)
   stay in the original project (../terrain-hazard-app), which must not be
   changed by TIMBER work. Do not add new hazard apps to TIMBER.
   Follow the structure in "Phase 5 structure" below.

## Phase 5 structure (follow this for any new application)
- Each application is its own module `terrain/apps/<name>.py` exposing:
  `NAME`, `DESCRIPTION` (2-3 plain sentences), `REQUIRED_INPUTS`
  (subset of `["dtm", "dsm", "ortho"]`), `SHAPES` (dict of `ShapeSpec` for
  things the user marks: kind `points` / `polygons` / `lines`), `TEST_AT_HOME`,
  `params(st) -> dict` (draws its own sliders), and
  `run(layers, params) -> Result`.
- `Result` (in `terrain/apps/base.py`) holds `overlays` (list of `Overlay`:
  classes with labels/colors, or a continuous value with a color scale),
  `metrics` (dict), `table` (DataFrame of flagged features), `warnings`
  (list of str) and `annotations` (circles, arrows, outlines, points, line
  segments in map coordinates).
- `Layers` gives every app the DTM, DSM and orthophoto on ONE grid plus
  derived layers (slope, aspect, flow, TWI, ExG, VARI) computed on first use.
  Reuse `terrain/slope.py`, `terrain/hydrology.py` and `terrain/vegetation.py`;
  never duplicate their math in an app.
- Register the module in `terrain/apps/__init__.py` (`APPS`). Apps whose inputs
  are missing still appear in the dropdown with a message naming the file needed.
- The page shows, in order: description, sliders, map (overlay + legend),
  summary metrics, flagged-features table, exports (overlay GeoTIFF, map PNG,
  table CSV), and an expander "How to test this at home".
- Shared sidebar controls: analysis cell size (default 5 cm), "remove best-fit
  tilt" (off by default), background (orthophoto / hillshade), overlay opacity.
- Shapes are marked by clicking the map (streamlit-image-coordinates) or by
  uploading a CSV with columns `x, y` (map CRS) and optional `shape, label`.
  Shapes are shared between apps by key.
- Banner: "Screening tool for a student science project. Not a substitute for
  a professional assessment." Default thresholds are labeled "starting values —
  adjustable".
- Tests: every app gets a synthetic-data test with a known answer in
  `tests/test_apps.py`; `tests/conftest.py` has `FakeStreamlit` so `params()`
  can be called without a browser (returns each widget's default).

## Accuracy checker requirements
The student's experiment uses **check points only** (no GCPs in processing).
- Input: CSV with columns `id, x, y, measured_z`. `x, y` are map coordinates
  (same CRS as the DTM) of each marker center. `measured_z` is the hand-measured
  height relative to a reference point, in meters.
- Also support a "grid helper": user enters map coordinates of two grid corners,
  grid size (columns × rows), and spacing; the app computes x, y for every point.
  (The student's current grid is 17 × 6 = 102 points.)
- Sample the DTM (and DSM if present) at each point with bilinear interpolation.
- Because absolute height is arbitrary, align by the **mean offset**:
  error_i = (model_z_i − measured_z_i) − mean(model_z − measured_z).
  Report the mean offset separately.
- Statistics: mean error, standard deviation, RMSE, max absolute error,
  LE90 (90th percentile of absolute error), number of points. Show in meters
  and inches.
- Charts: error heat map on the grid; histogram of errors; error vs local slope
  scatter plot (slope from the DTM at each point) with a trend line.
- Subset comparison: compute the same statistics on an evenly spread subset
  (user picks size, default 30) to compare with the full set.
- Export results to CSV.

## Testing
Write pytest tests with synthetic DEMs whose answers are known:
- tilted plane → constant slope equal to the known angle
- flat plane → slope 0, no flow accumulation direction bias crashes
- bowl/pit → ponding at the center with the known depth
- V-shaped valley → flow accumulates along the valley line
- accuracy: synthetic DEM + points with known errors → exact expected RMSE,
  and a constant offset is fully removed by mean alignment
Run `pytest` after each phase; all tests must pass.

## Style
- Clear, beginner-readable code with docstrings and comments.
- Small functions; keep math in `terrain/`, UI in `app.py`.
- Units always in meters internally; label units in the UI.
- Add a short README section for each phase: what it does and how to run it.
