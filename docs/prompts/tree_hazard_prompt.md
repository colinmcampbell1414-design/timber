Read AGENTS.md first. Upgrade the "Tree fall hazard" app in terrain/apps/trees.py
so it works automatically on any dataset (especially data/forest), with no
clicking required. Keep the existing tree detection, crown watershed, VARI
stress check, and optional user-marked targets, but add everything below.
Put heavy math in a new module terrain/treefall.py and keep trees.py as the
app wrapper (params / run). Plain-language comments on every formula.

=====================================================================
GOAL
=====================================================================
1. Give EVERY tree a general fall-hazard score (how likely it is to fall),
   whether or not anything is nearby.
2. Automatically predict WHERE each tree would most likely land.
3. Automatically find things in the map that falling trees could hit or
   block (trails, roads, clearings, buildings, streams, other trees).
4. Produce a map of "fall exposure": how many trees, weighted by hazard,
   could land on each spot.

=====================================================================
1. BETTER TREE DETECTION FOR FORESTS
=====================================================================
- Add a "variable window" option (on by default): the tree-top search
  window grows with height, window_m = a + b x height (defaults a = 1.5 m,
  b = 0.1, sliders). Tall trees have wide crowns; one fixed window either
  splits big trees or merges small ones.
- Defaults when the map is wider than 100 m: minimum tree height 5 m,
  canopy smoothing 0.5 m. Keep current defaults for small maps.
- Must stay fast on data/forest at 25 cm cells (~5 million cells, possibly
  thousands of trees). Use bounding windows per tree, never full-grid
  loops per tree.

=====================================================================
2. PER-TREE HAZARD FACTORS (each normalized 0-1, shown in the table)
=====================================================================
For each tree compute, at the tree top / crown:
a. Height: taller trees reach farther and hit harder. Normalize against
   the tallest tree (or a fixed 30 m, whichever is larger).
b. Slope at the base: mean DTM slope under the crown. Trees on steep
   ground uproot more easily. 0 at 0 deg, 1 at 35 deg or more.
c. Soil wetness: TWI from terrain/hydrology.py averaged under the crown.
   Saturated soil holds roots poorly. Normalize by percentile within map.
d. Exposure / wind: (i) distance from the crown edge to the nearest canopy
   gap (CHM below 2 m) - edge trees catch more wind; (ii) how much taller
   the tree is than the median tree within 15 m. Combine as the max.
e. Slenderness: height / crown diameter (crown diameter = 2 x sqrt(area/pi)).
   Tall thin trees are less stable. 0 at ratio 2, 1 at ratio 6 or more.
f. Health: from the orthophoto. Low VARI and low ExG compared with other
   crowns (robust z-score using median and MAD, like the existing stress
   check). Also flag "possible dead tree (snag)": height above 5 m, very
   low greenness, small crown for its height.
g. Lean / crown asymmetry (estimated, low confidence): vector from the tree
   top to the crown's centroid, divided by crown radius. Large offset =
   lopsided crown, which often means a lean or a heavy side. Store its
   direction in degrees.
h. Stream bank: within 1 x height of a channel (flow accumulation above a
   threshold) AND base slope above 15 deg = undercut-bank risk.

Hazard score = 100 x weighted sum of the factors. Sliders for each weight
(defaults: height 0.20, slope 0.15, wetness 0.10, exposure 0.15,
slenderness 0.10, health 0.20, lean 0.05, stream bank 0.05), normalized to
sum to 1. Classes: low < 35, moderate 35-60, high > 60 (sliders).
Add a "reasons" text column that lists the top 2-3 factors in words, e.g.
"Tall (24 m); on steep 28 deg slope; possibly dead crown".
Label the UI clearly: this is a relative screening score inside this map,
not a certified arborist assessment.

=====================================================================
3. FALL DIRECTION AND LANDING ZONE
=====================================================================
Model where each tree would most likely land as a direction probability
around the trunk (36 bins of 10 deg):
  p(direction) = w_uniform x uniform
               + w_downhill x bump centered on the downhill (aspect) direction
               + w_lean x bump centered on the crown-asymmetry direction
               + w_wind x bump centered on (prevailing wind direction + 180)
Bumps are von Mises shapes (kappa slider, default 2). Default weights:
uniform 0.4, downhill 0.3, lean 0.15, wind 0.15. Prevailing wind direction
input in degrees "wind comes FROM" (default 270, west), with a checkbox to
turn wind off. Scale the downhill weight by the slope factor (flat ground
-> no downhill preference).
Landing zone: a tree landing in direction theta covers a strip from the
trunk out to height x safety factor, as wide as the crown diameter.
Output per tree: most likely fall direction (deg and compass word, e.g.
"SE"), and the probability it lands inside each detected feature.
Draw an arrow on the map for the most likely direction of each moderate or
high hazard tree, length = height.

=====================================================================
4. FALL EXPOSURE MAP
=====================================================================
Raster where each cell = sum over trees of
  hazard_score/100 x probability that the tree's landing strip covers it.
Also a second raster "number of trees that could reach this spot"
(unweighted, any direction). Both as overlays with legends.
Export both as GeoTIFF.

=====================================================================
5. AUTO-DETECTED FEATURES AT RISK (no clicking needed)
=====================================================================
Detect automatically, each toggleable, all thresholds as sliders:
- Open ground / clearings: CHM < 0.5 m, area > 20 m2.
- Trails and roads: open ground that is long and narrow: skeletonize the
  open mask (skimage.morphology.skeletonize), keep pieces longer than
  15 m whose width (2 x distance transform on the skeleton) is 1-8 m.
  Mark paved-looking ones (low ExG, low roughness) as "road/driveway".
- Buildings: CHM > 2.5 m, not green (low ExG/VARI), low surface roughness
  (std of DSM in a 1 m window below 0.15 m), area > 15 m2.
- Streams / drainage channels: flow accumulation above a threshold
  (default contributing area 500 m2). Trees falling into channels can
  block water and cause local flooding.
- Other trees (domino effect): count how many other crowns each tree's
  landing zone covers.
User-marked targets (existing feature) still work and are added to these.
For each feature type report: number of trees whose landing zone reaches
it, and its total exposure score. In the tree table add columns: "can reach
trail/road", "can reach building", "can reach stream", "trees it could hit".

Final priority for each tree = hazard score x (1 + consequence), where
consequence = 1.0 building, 0.7 road, 0.5 trail, 0.3 stream, 0.1 per other
tree hit (cap 0.5), 0 otherwise (weights adjustable). Sort the table by
priority. Show "Top 20 priority trees" first.

=====================================================================
6. VALIDATION TOOLS (for the science fair)
=====================================================================
- Downed logs detector: elongated objects 0.2-1.5 m above ground, longer
  than 4 m and narrower than 1 m (label + region properties: major axis
  length, minor axis length). These are trees that already fell. Report
  their direction (from the root end to the top is unknown, so report the
  axis angle) and compare their locations with the fall-exposure map and
  with the hazard factors of nearby standing trees. Show the comparison as
  a small table: mean slope / wetness / exposure at downed logs vs at
  random forest locations.
- Field check export: CSV of the top N priority trees (default 20) plus N
  random low-hazard trees, with latitude and longitude (use
  io.map_to_latlon), height, predicted fall direction, reasons, and blank
  columns: observed_height, observed_lean_deg, dead_or_damaged (y/n),
  root_damage (y/n), notes. The user fills this in on a phone in the woods.
- Field check import: upload the filled CSV and report agreement:
  height error (mean, RMSE), how often "possibly dead" matched, and
  whether high-hazard trees had more observed problems than low-hazard ones.

=====================================================================
UI
=====================================================================
Keep everything inside the existing "Tree fall hazard" entry of the
Hazards & Applications dropdown. Use st.tabs inside it:
"Trees & hazard", "Where trees could fall", "Features at risk",
"Downed logs", "Field check". Sliders grouped in expanders
("Tree detection", "Hazard weights", "Fall direction", "Features").
Metrics row: trees found, high-hazard trees, trees that can reach a
road/trail, trees that can reach a building, downed logs found.
"How to test this at home" text: measure 8-10 trees (clinometer app),
include some deep in the canopy and some at clearings; check flagged
dead/leaning trees in person; compare downed logs found by the app with
real logs.

=====================================================================
TESTS (pytest, synthetic data)
=====================================================================
- Two cone trees of 10 m and 20 m: detected, heights within 5%, the
  20 m tree scores higher on the height factor.
- Same tree on flat vs 30 deg slope: slope factor higher on the slope and
  the most likely fall direction is within 20 deg of downhill.
- Crown shifted east of its top: lean direction within 20 deg of east.
- Fall exposure: a single tree on flat ground with wind off -> exposure
  forms a ring/disk of radius = height (within one cell), zero beyond.
- Trail detection: a 2 m wide, 40 m long clearing strip in a synthetic
  forest is detected as a trail; a round clearing is not.
- Building detection: a flat 10 x 8 m box, 4 m tall, not green, is a
  building; a cone tree of the same height is not.
- Downed log: a 6 m x 0.6 m bar 0.5 m tall is detected with the right
  axis angle (within 10 deg).
- Priority: identical trees, one able to reach a building -> it ranks first.
- Performance: a 2000 x 2000 synthetic CHM with 1000 trees runs in under
  60 seconds.

Run pytest; all tests must pass. Update the README Phase 5 section for the
tree app. Stop when done and give me the run command and what to try first
on the forest dataset.
