Read AGENTS.md first. If there is no CLAUDE.md in the repo root, create one that contains the single line "@AGENTS.md" so you (Claude Code) load it every session.
This is Phase 6: turn the project into ONE focused engineering project built around the tree fall hazard app, and add the tools needed to prove it works. Do not add any new hazard apps. Do not break the other Phase 5 apps.
Before coding, look through the repo and make a short checklist of the steps below. Then work through them in order, running pytest after each step. Only stop early if something is truly blocked; if so, tell me what and why. Keep code beginner-readable with plain-language comments on every formula.
===================================================================== 0. CHECK WHAT EXISTS FIRST

* The tree hazard upgrade is specified in docs/prompts/tree_hazard_prompt.md (per-tree hazard factors, fall direction, exposure map, auto-detected features, downed logs, field check). Check whether terrain/treefall.py exists and the tree app has those tabs.
* If that upgrade is missing or only partly built, implement it from that file FIRST, run its tests, then continue with Phase 6.
* Everything below extends terrain/treefall.py and terrain/apps/trees.py. Put new validation math in a new module terrain/validation.py.

===================================================================== PROJECT GOAL (write this into AGENTS.md and the README)
Engineering goal: design an automated system that uses only a consumer drone and free software (WebODM + this app) to find which trees in a forest pose the greatest fall risk to trails, roads and buildings, accurately enough to direct a field inspection and faster than a manual walk-through survey.
Design criteria (targets live in config/design_criteria.yaml so I can edit them; create that file with these defaults): detection_rate >= 0.80 trees taller than 10 m found in check plots height_rmse_m <= 1.5 app height vs clinometer height defect_rate_ratio >= 2.0 defect rate of high-hazard / low-hazard trees fisher_p <= 0.05 high vs low hazard defect difference beats_baseline true full score AUC > every baseline AUC log_factor_test_p <= 0.05 downed logs sit on higher-hazard ground than random forest points hectares_per_hour >= 10 person-time to survey (entered by me) + compute time
The yard accuracy work (Phases 1-4) is now background only. Do not extend it. It is reported as one line: "best-case open-ground vertical error".
=====================================================================

1. BASELINE RANKINGS ===================================================================== A judge will ask "is your multi-factor score better than something simple?" Compute these rankings next to the full hazard score for every tree, stored as columns in the tree table:

* baseline_height: rank by height only.
* baseline_height_target: rank by height, only counting trees whose landing zone (any direction) can reach a trail, road or building; others ranked after them by height.
* baseline_random: a random ranking with a fixed seed (slider, default 42). These are used by the field check import (step 3) and the design criteria tab (step 8).

===================================================================== 2. WEIGHT SENSITIVITY
Show how much the results depend on the hazard weights I chose.

* One-at-a-time: for each factor, multiply its weight by 0.5 and by 1.5 (renormalize to sum 1). Report the Spearman rank correlation between the new and original scores, and how many of the original top 20 trees stay in the top 20.
* Random weights: draw 500 weight sets from a Dirichlet distribution centered on the current weights (concentration slider, default 20). For each tree report how often it lands in the top 20 ("top-20 stability", 0-100%). Show a histogram of top-20 overlap.
* New expander "Sensitivity" in the "Trees & hazard" tab with both tables and a plain-language summary line, e.g. "17 of the top 20 trees stay in the top 20 under most reasonable weight choices."
* Must run in under 30 s on data/forest (vectorize: the factor table is trees x factors, so scores for all weight sets = one matrix product).

===================================================================== 3. BLIND FIELD CHECK (replaces the old field check export/import)
The field check must be BLIND: when I look at a tree in the woods I must not know whether the app called it high or low hazard.
Export (button in the "Field check" tab):

* Sample N high-hazard trees and N low-hazard trees (N slider, default 30; "low" = lowest third of scores, and only trees taller than 10 m so the comparison is fair). Option to restrict both groups to trees within X m of a trail (default 30 m) so they are reachable on foot.
* Shuffle them together and give each a random 4-digit code.
* field_sheet.csv (what I take into the woods) has ONLY: code, latitude, longitude (io.map_to_latlon), distance and compass bearing from the nearest trail point, and blank columns: status (found / not_found / inaccessible), observed_height_m, dead_or_dying_crown, broken_or_dead_top, trunk_crack_or_cavity, root_damage_or_heaving, lean_over_15deg, hanging_dead_branches, undercut_bank, notes (defect columns are y/n). No height, score, group or reasons.
* field_key.csv (stays home) maps code -> tree id, group, score, all baseline ranks, app height, predicted fall direction, reasons.
* field_booklet.html: one card per code with a 40 x 40 m orthophoto crop centered on the tree, the tree's crown outlined, a north arrow and a 10 m scale bar, plus the lat/lon and bearing text. No hazard info. This is how I find the right tree when phone GPS is off by several meters.

Import (upload filled field_sheet.csv; field_key.csv is read from the export folder or uploaded):

* any_defect = yes if any defect column is y. Only count status = found.
* Defect rate per group, ratio high/low, Fisher's exact test (scipy.stats.fisher_exact) with the 2x2 table shown.
* AUC (probability a defective tree outranks a sound one; use the Mann-Whitney U relation) for: full score, each baseline, and each single factor. Show as a bar chart. This answers "does the score beat simple rules?" and "which factors actually matter?"
* Height check: mean error (bias), MAE, RMSE, and a scatter plot of app vs field height with a 1:1 line. Split results into "open/edge" and "interior" trees using the existing edge-distance factor.
* "Possibly dead" flag vs observed dead_or_dying_crown: 2x2 table.
* Write everything to results/field_check_results.json and a CSV.

===================================================================== 4. DETECTION CHECK PLOTS

* I choose 3-5 circular plots. Input: CSV of plot_id, latitude, longitude, radius_m (default 12 m) or click on the map.
* Export plot_sheet.csv + a booklet card per plot (orthophoto crop with the circle drawn), blank columns: field_count_over_10m, notes.
* Import: detection rate per plot = app trees over 10 m with tops inside the circle / field count. Report per plot and overall (sum / sum). Note in the UI that this is count-based, so missed trees and extra trees can cancel; also show app count and field count side by side.

===================================================================== 5. DOWNED LOG VALIDATION (upgrade the existing detector)

* Logs on or within 5 m of trails/roads are often cut by trail crews, so exclude them by default (checkbox) and say so in the UI.
* Factor test: compare slope, wetness (TWI), edge exposure and canopy height around each log (mean in a 3 m buffer) with 1000 random points in forest that are not on logs, trails or buildings. Mann-Whitney U test per factor, table of medians and p-values, box plots.
* Direction test: logs only have an axis (root end unknown), so use axial statistics. For each log on ground steeper than 5 deg, angle difference d = log axis - downhill aspect; double it (2d) so 0 and 180 deg mean the same thing; report the mean resultant length and a Rayleigh test on the doubled angles (implement the formula; comment it). Also a rose diagram of log axes. Plain-language result, e.g. "logs line up with the slope more than chance (p = 0.01)".
* Manual logs: allow importing logs I measured in the field (CSV of latitude, longitude, axis_azimuth_deg, length_m) because the detector may miss logs under canopy. Run the same two tests on them, and report how many manual logs the detector also found (within 3 m).

===================================================================== 6. LEAF-ON / LEAF-OFF FUSION AND OUTSIDE DTMs
Photogrammetry can't see the ground under leaves, so the October (leaf-on) DTM is probably poor under canopy. I will fly a leaf-off survey in December. This step is the main design iteration of the project.

* Let the dataset picker choose the DTM and the DSM/orthophoto from DIFFERENT survey folders (e.g. DTM from data/forest_leafoff, DSM and orthophoto from data/forest).
* Also allow an outside DTM GeoTIFF (e.g. public lidar from Maryland iMAP / USGS 3DEP). Reproject and resample it to the DSM grid with rasterio (bilinear), handling different CRS and units (feet vs meters: read the CRS units; also a manual override dropdown).
* Vertical alignment: drone surveys have different absolute heights. Find "stable open ground" automatically (CHM below 0.5 m in both surveys, slope below 10 deg, not a building), fit a plane to DTM_a - DTM_b there, and subtract it. Reuse the stable-area plane fit from the erosion app. Show the before/after offset and the leftover RMS on stable ground.
* New "DTM comparison" expander: for each available DTM source, show the difference from the reference DTM (choose which is reference) split into open ground vs under canopy, hillshades side by side, and the CHM each one produces. If field heights exist (step 3), show height RMSE for every DTM source in one table. That table is the before/after result for the board.

===================================================================== 7. SEASON SETTING FOR THE HEALTH FACTOR
Autumn color makes healthy trees look "stressed" in VARI/ExG.

* Add a dropdown "Orthophoto season": summer leaf-on / autumn color / leaf-off. Store it per survey in a small survey.json in the folder, default "summer leaf-on"; set data/forest to "autumn color".
* autumn color: health weight default 0.05 (others renormalized), warning banner, and "possibly dead" uses only the snag rule (height, small crown, very low greenness relative to the darkest 5%), not the z-score alone.
* leaf-off: health factor and "possibly dead" are turned off with a message, unless a leaf-on orthophoto is chosen via step 6.

===================================================================== 8. "DESIGN CRITERIA" TAB AND PROJECT RESULTS

* New tab "Design criteria" in the tree app: table with criterion, target (from config/design_criteria.yaml), measured value, pass / fail / "no data yet", and which file the number came from.
* Timing: record compute time for the full tree pipeline per run and the map area in hectares. I type in my flight + processing + field time in hours (number inputs saved to results/timing.json). hectares_per_hour = area / (my hours + compute hours).
* Button "Export results for board": writes results/summary.md with the goal, criteria table, key numbers from steps 2-6 in plain sentences, and saves every chart as PNG at 300 dpi to results/figures/.
* Keep the existing banner: "Screening tool for a student science project. Not a substitute for a professional arborist assessment."

===================================================================== TESTS (pytest, synthetic data with known answers)

* Baselines: baseline_height ranks a 25 m tree above a 15 m tree; baseline_random is identical for the same seed and different otherwise.
* Sensitivity: if all factors but height are zero for every tree, every weight set gives the same ranking (Spearman = 1, top-20 overlap = 20).
* Blind export: field_sheet.csv contains none of the columns score, group, height, reasons, tree id; every code appears once in field_key.csv; groups are each exactly N when enough trees exist.
* Field import: a made-up sheet where all high trees have defects and no low trees do gives ratio = inf handled cleanly (show "all high, none low"), Fisher p < 0.001, full-score AUC = 1.0; all-equal defects give AUC 0.5.
* AUC helper matches sklearn.metrics.roc_auc_score on random data (if sklearn is installed; otherwise against a hand-worked example).
* Detection plots: a synthetic plot with 10 cone trees and field count 10 gives rate 1.0; removing 2 trees gives 0.8.
* Axial statistics: log axes at 0 and 180 deg on a slope facing 0 deg give mean resultant length 1; uniformly spread axes give p > 0.05; tightly aligned axes (n = 30, +-10 deg) give p < 0.01.
* Log factor test: logs placed only on steep cells give a lower p for slope than logs placed randomly.
* Fusion: a DTM shifted by +3.2 m and tilted 0.5 deg is aligned to within 2 cm RMS on stable ground; a DTM in US survey feet is converted to meters.
* Season: autumn color lowers the health weight; leaf-off disables it.
* Design criteria: missing results files show "no data yet", not a crash.
* Performance: sensitivity with 500 weight sets on 5000 trees under 10 s.

===================================================================== DONE WHEN
All tests pass (old and new). README and AGENTS.md updated with the goal, Phase 6 structure and the field workflow. Then stop and give me:

1. The run command.
2. A short checklist of the field data I need to collect, with the exact file each one goes into (field sheet, plot sheet, manual logs, timing).
3. What to look at first on data/forest, and anything about my current forest DTM under canopy that looks suspicious.
