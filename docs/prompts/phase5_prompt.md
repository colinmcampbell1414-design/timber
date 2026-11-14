Read AGENTS.md first. Build Phase 5: a single "Hazards & Applications" page
with a dropdown (st.selectbox) that switches between the applications below.
Also add a short Phase 5 section to AGENTS.md and the README describing the
structure, so future work follows it.

=====================================================================
SHARED STRUCTURE (build this first)
=====================================================================
- Put each application in its own module: terrain/apps/<name>.py.
- Every module exposes the same interface so the page can loop over them:
    NAME, DESCRIPTION (2-3 plain-language sentences),
    REQUIRED_INPUTS (e.g. ["dtm"], ["dtm","dsm"], ["dtm","ortho"]),
    params(st) -> dict          # draws its own sliders, returns values
    run(layers, params) -> Result
  where Result holds: overlay raster(s), a legend, summary metrics (dict),
  a table of flagged features (pandas DataFrame), and warnings (list of str).
- A registry (terrain/apps/__init__.py) lists all apps; the dropdown is
  built from it. Apps whose required inputs are missing appear in the
  dropdown but show a clear message saying which file is needed.
- Shared sidebar controls for this page: analysis cell size (default 5 cm),
  "remove best-fit tilt" checkbox (off by default, reuse Phase 4 code),
  background (orthophoto / hillshade), overlay opacity.
- Every app page shows, in this order: description, sliders, map with
  overlay + legend, summary metrics, flagged-features table, and an
  expander titled "How to test this at home" with the text given below.
- Export buttons on every app: overlay as GeoTIFF, map as PNG, table as CSV.
- Banner on the page: "Screening tool for a student science project. Not a
  substitute for a professional assessment." Default thresholds are
  labeled "starting values — adjustable".
- Reuse existing code from terrain/slope.py and terrain/hydrology.py
  (slope, aspect, curvature, hillshade, fill, flow accumulation, ponding,
  TWI). Do not duplicate it.
- Map clicks: use streamlit-image-coordinates (or plotly click events) so
  the user can click points or draw simple polygons on the map. Always also
  allow uploading the same points/polygons as a CSV (x, y in map CRS).
- New shared helper: terrain/vegetation.py with RGB vegetation indices from
  the orthophoto: VARI = (G - R) / (G + R - B) and ExG = 2G - R - B, with
  safe division. Used by several apps.

=====================================================================
APPLICATIONS (build in this order; run pytest after each one)
=====================================================================

1. Mudslide / slope stability susceptibility
   Inputs: DTM (+ orthophoto optional).
   Method: weighted score from normalized slope, TWI (wetness), plan
   curvature (concave = collects water), and bare-soil fraction from the
   orthophoto (low ExG) if available. Adjustable weights that sum to 1.
   Classes: low / moderate / high. Note in the UI that classes are
   relative within the mapped area.
   Test at home: compare high-score areas with places where soil is
   exposed, slumping, or washing out after heavy rain.

2. Tree fall hazard
   Inputs: DTM + DSM (+ orthophoto optional).
   Method: canopy height model CHM = DSM - DTM. Find tree tops as local
   maxima above a minimum height (default 2 m) with an adjustable search
   window; segment crowns with watershed (scikit-image). For each tree:
   height, crown area, crown center. Fall zone = circle with radius
   = height x safety factor (default 1.0, slider 1.0-1.5). User marks
   "targets" (house, shed, driveway, fence, play area) by clicking or CSV.
   Flag trees whose fall zone reaches a target; rank by height and
   distance. If orthophoto exists, report mean VARI per crown and flag
   unusually low (possibly stressed or dead) crowns relative to the others.
   Test at home: measure 3-5 trees' heights by hand (clinometer phone app
   or the shadow/stick method) and compare with the app; check that flagged
   trees really are within reach of the targets.

3. Erosion
   Mode A — susceptibility (one DTM): stream power index
   SPI = ln(flow_accumulation_area x tan(slope)); flag high-SPI cells on
   bare or sparse ground (low ExG) as likely erosion paths.
   Mode B — change detection (two DTMs from different dates): user uploads
   a second DTM; resample both to the same grid; before differencing, align
   them by removing the plane fitted to the difference over "stable areas"
   the user clicks (patio, driveway). DEM of Difference = new - old. Ignore
   changes smaller than a level-of-detection threshold (default 3 cm,
   adjustable; explain it comes from the accuracy test). Report area and
   volume lost (erosion) and gained (deposition) in m3 and cubic feet.
   Test at home: fly the same area before and after a big rain, or after
   digging a small test hole / piling a known amount of soil or mulch.

4. Flooding & foundation drainage
   Inputs: DTM.
   Method: reuse Phase 4 ponding and flow paths. User outlines a building
   (click polygon or CSV). Sample the ground along lines running outward
   from the outline for an adjustable distance (default 3 m / 10 ft).
   Flag wall segments where the ground does not fall away by at least an
   adjustable grade (default 5%, i.e. about 6 in over 10 ft), and segments
   where flow paths point toward the wall. Show arrows.
   Test at home: during or after rain, check whether the flagged sides of
   the house are where water actually collects or downspouts dump.

5. Irrigation & lawn health
   Inputs: DTM + orthophoto.
   Method: wet zones = high flow accumulation or ponding; dry zones =
   convex high ground and steep south/southwest-facing slopes (aspect);
   overlay VARI to show stressed (brown/yellow) turf. Produce 3 zones:
   needs-more-water / normal / needs-less-water, with area of each.
   Test at home: compare dry zones with brown patches in late summer, and
   wet zones with soggy spots after rain.

6. Rain garden / stormwater siting
   Inputs: DTM (+ building outline from app 4 if available).
   Method: candidate cells where flow accumulation is high (water arrives
   there), slope is gentle (default under 12%, adjustable), distance from
   the building is at least an adjustable minimum (default 3 m / 10 ft),
   and not already a deep ponding spot. Group candidates into sites; rank
   by contributing area. Estimate a garden size as an adjustable
   percentage (default 20%) of the contributing area.
   Test at home: check whether ranked sites match where runoff flows from
   the driveway, downspouts, or roof.

7. Winter ice hazard on walkways and driveways
   Inputs: DTM (+ user-marked paved areas, click polygon or CSV).
   Method: compute a hillshade for the winter-solstice noon sun:
   sun elevation = 90 - latitude - 23.44 degrees (latitude input,
   default 39.2), sun azimuth 180. Shaded + north-facing + sloped paved
   cells are flagged as "stays icy longest"; steep paved cells (default
   over 8%) are flagged as slip risk. Combine into low/moderate/high.
   Note that trees and buildings cast shadows too: if a DSM exists, use it
   for the shading instead of the DTM.
   Test at home: after a snow or ice event, photograph which parts of the
   driveway/walkway melt last and compare.

8. Trip hazards & uneven ground
   Inputs: DTM (or DSM for hard surfaces).
   Method: local relief = max - min height in a small moving window
   (default 15 cm); flag abrupt steps above a threshold (default 4 cm,
   adjustable) and holes/bumps from the difference between the DTM and a
   smoothed DTM. Mark user paths (click polyline) to only check along them.
   Test at home: compare with a known step, curb, paver lip, or root,
   measured with a ruler.

9. Volume measurement (stockpiles, holes)
   Inputs: DTM or DSM.
   Method: user outlines a pile or hole; base surface = plane fitted to
   the heights along the outline; volume = sum of (height - base) x cell
   area, positive for piles, negative for holes. Report m3, cubic feet,
   and cubic yards.
   Test at home: measure a mulch or soil pile made from a known number of
   bags (volume printed on the bag) and compare.

=====================================================================
TESTS (pytest, synthetic data with known answers)
=====================================================================
- Registry: every app has the required interface; missing inputs give a
  message, not a crash.
- Mudslide: steeper synthetic slope scores higher than a flat one.
- Trees: two synthetic cone "trees" of known heights (e.g. 5 m and 8 m)
  are detected with heights within 5%; a target 6 m away is flagged for
  the 8 m tree only.
- Erosion: second DTM with a known 0.5 m3 pit -> reported loss within 5%;
  a pure tilt between the two DTMs is removed by stable-area alignment.
- Drainage: ground sloping toward a wall is flagged; sloping away is not.
- Irrigation: a bowl gives a wet zone at the center.
- Rain garden: candidates are never closer than the minimum distance.
- Ice: a north-facing slope is flagged more than a south-facing one at
  latitude 39.2.
- Trip hazard: a 5 cm step is flagged; a smooth 3 degree ramp is not.
- Volume: a synthetic box pile of known size returns the exact volume
  within 2%.

Keep code beginner-readable with plain-language comments on every formula.
Stop after all tests pass and give me the run command and a 1-line summary
of each app.
