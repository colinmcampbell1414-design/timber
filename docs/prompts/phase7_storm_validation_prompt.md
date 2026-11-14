Read AGENTS.md first. This is Phase 7: validate the tree fall hazard system against REAL wind storms. I will fly the same forest before and after wind storms. The app must find which trees were actually damaged or fell, and test whether the hazard score, fall direction model and exposure map predicted them. Do not add new hazard apps. Do not break Phases 5-6.
Before coding, look through the repo (especially terrain/treefall.py, terrain/validation.py, the erosion app's stable-area plane fit, the downed log detector and the Design criteria tab) and make a short checklist of the steps below. Work through them in order, running pytest after each step. Only stop early if something is truly blocked. Keep code beginner-readable with plain-language comments on every formula. New math goes in a new module terrain/storm.py; UI goes in a new tab "Storm check" in the tree app.
===================================================================== WHY THIS MATTERS (write a short version into AGENTS.md and the README)
The field check (Phase 6) tests whether high-hazard trees LOOK defective. The storm check tests whether they actually FAIL. Because the scores are frozen before the storm, this is a true prediction, not a fit after the fact. It is the strongest evidence in the project.
=====================================================================

1. SURVEYS, STORMS AND FROZEN PREDICTIONS =====================================================================

* Each flight is its own folder: data/forest_YYYY-MM-DD (same structure as data/forest). survey.json gets: date, time, season, flight altitude, overlap, and a free-text notes field.
* New file config/storms.yaml, one entry per storm: name, date_start, date_end, before_survey, after_survey, weather_station (id + name), peak_gust_mps, mean_wind_mps, wind_from_deg (direction during strongest winds), notes. I fill in the wind numbers from the nearest airport station (e.g. NOAA / ASOS); add an optional helper that reads a station CSV I download (columns: time, wind speed, gust, direction) and computes peak gust, mean wind during the storm window and the gust-weighted mean direction (circular mean weighted by gust speed squared; comment the formula).
* "Freeze predictions" button: runs the full tree pipeline on the BEFORE survey with the current settings and saves results/storms/<storm>/predictions_frozen.csv (every tree: id, x, y, lat/lon, height, crown area, all factors, hazard score, priority, baseline ranks, predicted fall direction, landing-zone probabilities) plus settings.json (all slider values, code git hash if available) and a SHA-256 hash of both files written to a log with a timestamp. Later analysis must READ this frozen file, never recompute scores, so I can show judges the predictions existed before the damage was known.
* Also save a second frozen copy using the storm's ACTUAL wind direction (once known) for the direction tests in step 5. Keep both clearly labeled: "prevailing-wind prediction" vs "actual-wind prediction".

===================================================================== 2. ALIGNING BEFORE AND AFTER SURVEYS
Changes of a few tenths of a meter must not be mistaken for damage.

* Horizontal: estimate the shift between the two orthophotos and DSMs on stable areas (open ground, buildings, roads) using phase correlation (skimage.registration.phase_cross_correlation, upsampled to sub-cell) on hillshades or gradients, not raw colors (lighting changes). Report the shift in meters; apply it to the after survey.
* Vertical: reuse the stable-area plane fit (CHM < 0.5 m in both, slope < 10 deg, not a building) on DTM and on DSM. Report offset, tilt and leftover RMS.
* Show a warning if leftover RMS on stable ground is above 0.15 m or the horizontal shift residual is above 0.5 m: "alignment may be too poor to detect small damage."

===================================================================== 3. DETECTING WHAT CHANGED (per tree)
For every tree in the frozen BEFORE table, measure inside its BEFORE crown polygon (and a 1 m buffer):

* top_drop_m: before tree-top height - after max CHM within 1.5 m of the old top.
* crown_loss_frac: fraction of crown cells whose CHM dropped by more than the change threshold.
* volume_loss_frac: sum of positive CHM drops / sum of before CHM in the crown.
* color change: drop in mean VARI/ExG (only reported, not used to classify if the seasons differ). Classify each tree: fallen (windthrow) : top_drop > 60% of height AND crown_loss > 70% snapped top : top_drop > max(2 m, 20% of height), not fallen crown damage : crown_loss > 25% (large branches lost), not above no change : otherwise uncertain : tree near the map edge, in a gap in the after survey, or alignment warning active All thresholds are sliders. The change threshold defaults to 3 x the stable-ground RMS from step 2 (minimum 0.5 m).
* New downed logs: run the downed log detector on both surveys; logs in the after survey with no matching log (within 2 m, axis within 20 deg) in the before survey are "new logs".
* Link new logs to fallen trees: a new log whose one end lies within 1.5 x crown radius of a fallen tree's old trunk belongs to that tree. That end is the ROOT end, so for storm-felled trees the fall direction is fully known (root -> top), not just an axis. Store observed fall azimuth and observed log length.
* New canopy gaps not explained by any tree (e.g. trees that came down from outside the map) are listed separately.
* Leaf drop guard: if the two surveys have different seasons in survey.json, or if the median crown_loss_frac of "no change" candidates across the whole map is above 10%, show a warning banner: "Many crowns thinned everywhere; this looks like leaf drop, not storm damage." Offer an option to classify using top_drop and new logs only.

===================================================================== 4. NULL TEST: HOW MUCH "DAMAGE" IS JUST NOISE?
I will sometimes fly the same area twice with NO storm in between (same day or a calm few days apart). The app must support a "calm pair" entry in storms.yaml (peak_gust_mps can be low, add is_calm_control: true).

* Run the exact same change detection on calm pairs and report the false change rate per class (e.g. "2 of 1,450 trees falsely flagged as snapped top").
* In the storm results, show the storm's change rate next to the calm false change rate, with a Fisher's exact test, so I can say storm damage was real and not processing noise.

===================================================================== 5. DID THE SYSTEM PREDICT IT?
Using ONLY the frozen predictions, for each storm and for all storms pooled: a. Hazard score vs damage:

* Outcomes: any_damage (fallen, snapped or crown damage) and severe (fallen or snapped). Exclude "uncertain" trees.
* AUC (Mann-Whitney relation, reuse validation.py) for the full score, each baseline, and each single factor, with 95% bootstrap CIs (2000 resamples, fixed seed). Bar chart.
* Damage rate by hazard class (low / moderate / high) with counts, and the high/low ratio with Fisher's exact test.
* "Hit rate in the top N": of the trees that were damaged, how many were in the top 10%, 20% of the frozen ranking, vs what random ranking would give.
* Pooled across storms: logistic regression of severe damage on hazard score + log(peak gust) (statsmodels if installed, otherwise a small Newton-Raphson implementation). Report the odds ratio per 10 hazard points. Plain-language summary line.
* Small-sample warning: if fewer than 10 damaged trees, say so clearly and show exact counts instead of leaning on p-values. b. Fall direction (fallen trees with a linked log):
* Angular error between observed fall azimuth and the most likely predicted direction, for BOTH frozen versions (prevailing wind vs actual wind) and for "downhill only", "lean only", "wind only" and a uniform guess. Report median absolute error and a table.
* Log-likelihood of the observed directions under the predicted 36-bin probability vs under uniform (higher = model is better than guessing). Comment the formula.
* Rose diagram: observed fall azimuths with the storm wind direction marked. c. Landing zone and exposure map:
* For each fallen tree: did its new log lie inside the predicted landing strip (any direction, and the most likely direction)?
* Mean frozen exposure value under new logs vs under random forest points (Mann-Whitney U). Did any new log land on a detected trail, road, stream or building, and had that feature been flagged? d. Write everything to results/storms/<storm>/storm_results.json + CSV, and a pooled results/storms/pooled_results.json.

===================================================================== 6. FIELD CONFIRMATION (small, blind)
Remote change detection can be wrong. Export a blind check sheet like Phase 6: every tree classified fallen or snapped (up to 30) mixed with an equal number of random "no change" trees, shuffled, 4-digit codes, booklet cards showing the BEFORE orthophoto crop (so I can find the spot). Columns: status, storm_damage (none / branches / top broken / fallen / uprooted), fall_azimuth_deg (if fallen, measured with phone compass root -> top), notes. Import reports change-detection precision and recall per class, and replaces app fall azimuths with field ones where given (keep both).
===================================================================== 7. UI: "Storm check" TAB
Storm picker (from storms.yaml), sub-sections in expanders: "Freeze predictions", "Alignment", "Change map", "Did we predict it?", "Fall direction", "Calm control", "Field confirmation".

* Change map: before/after orthophoto slider (or side by side), CHM difference overlay (red = loss), tree outlines colored by damage class, new logs drawn as arrows root -> top, predicted direction arrows in a different color for the same trees.
* Metrics row: trees compared, fallen, snapped, crown damage, new logs, calm false change rate, storm AUC (severe).
* Results table of damaged trees: id, class, frozen hazard score and rank, reasons (frozen), predicted vs observed direction, error.
* Add to the Design criteria tab and config/design_criteria.yaml: storm_auc_severe >= 0.70 storm_beats_baseline true (full score AUC > baseline_height AUC) fall_direction_mae_deg <= 45 calm_false_change_rate <= 0.01 "no data yet" until a storm has been analyzed.
* "Export results for board" also writes a storm section in results/summary.md and the change map, AUC chart and rose diagram as 300 dpi PNGs.

===================================================================== 8. FLIGHT CONSISTENCY CHECKLIST (show in the tab as text, and in README)

* Same flight plan, altitude, overlap, camera angle and exposure settings for every survey of a site; save the plan file in the folder.
* Fly before storms as close to the storm as safely possible (ideally within 1-3 days) and after as soon as winds are within the drone's limits and it is safe. Never fly in or near storm conditions.
* Similar time of day (shadows change the orthophoto and the DSM).
* Do at least one calm-pair repeat flight per season as the null test.
* Note leaf condition (on / turning / off) in survey.json every flight.
* Check that the after survey fully covers the before area; damaged areas at the map edge will be "uncertain".

===================================================================== TESTS (pytest, synthetic data with known answers)

* Alignment: an after DSM shifted 1.3 m east, 0.4 m north and +2.1 m up with a 0.3 deg tilt is recovered to within 0.1 m horizontal and 3 cm RMS vertical on stable ground.
* Change classes: from a synthetic forest, remove one cone tree entirely and add a 0.6 m-tall bar log from its base pointing 120 deg -> fallen, observed azimuth within 10 deg of 120 (root end correct, not 300). Cut the top 40% off another -> snapped top. Remove one side of a crown -> crown damage. Untouched trees with 0.1 m random noise -> no change.
* Calm pair: two copies of the same forest with independent 0.1 m noise give a false change rate of 0 at default thresholds.
* Leaf drop guard: lowering every crown by 30% of its height margin triggers the leaf-drop warning.
* Frozen predictions: the storm analysis reads predictions_frozen.csv even after the hazard weights are changed; the stored hash matches; changing the frozen file is detected.
* Prediction metrics: if damaged trees are exactly the top-scoring ones, AUC = 1.0; random damage gives AUC near 0.5 (within 0.1 over many seeds); the bootstrap CI contains the true value.
* Direction: fall directions all equal to the predicted direction give error 0 and log-likelihood above uniform; uniform random directions give log-likelihood near uniform.
* Gust-weighted wind direction: readings at 350 and 10 deg average to 0 deg, not 180.
* Design criteria: storm criteria show "no data yet" with no storms.

===================================================================== DONE WHEN
All tests pass (old and new). README and AGENTS.md describe the storm workflow. Then stop and give me:

1. The run command.
2. A step-by-step checklist for the next storm: what to fly and when, what to click to freeze predictions, what wind data to download and where it goes, and what to do after the after-flight.
3. Your honest estimate of how many damaged trees I'd need (pooled over storms) for the AUC result to be meaningful, given the number of trees in data/forest.
