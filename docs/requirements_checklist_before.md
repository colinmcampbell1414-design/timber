# TIMBER requirements checklist

Status BEFORE the audit (written before any code was changed). The final version replaces this file.

| ID | requirement (short) | status | evidence | notes |
|---|---|---|---|---|
| G1 | AGENTS.md (goal, phases, field workflow); CLAUDE.md with @AGENTS.md | PARTIAL | AGENTS.md | No CLAUDE.md; goal sentence not word-for-word; no test |
| G2 | README: install, run, a section per phase, goal, field/storm workflows | PARTIAL | README.md | Goal sentence and Phase 6 section missing; no test |
| G3 | requirements.txt lists every imported package; optional packages optional | PARTIAL | requirements.txt | No test that every import is listed |
| G4 | App registry; every app has the shared interface | DONE | terrain/apps/__init__.py: APPS; tests/test_apps.py::test_every_app_has_the_shared_interface |  |
| G5 | Missing inputs / results files give a clear message, never a traceback | DONE | terrain/apps/base.py: missing_inputs; validation.py: evaluate_criteria; tests/test_apps.py::test_missing_inputs_give_a_message_not_a_crash; tests/test_storm.py::test_storm_results_files_and_design_criteria |  |
| G6 | Shared helpers reused, not duplicated | PARTIAL | terrain/slope.py, hydrology.py, vegetation.py, validation.py: auc | Duplicates found: NaN-aware smoothing (treefall._smooth vs base.nan_smooth), two plane fits (hydrology.remove_tilt vs storm.fit_plane), two compass-word helpers |
| G7 | GeoTIFF / PNG / CSV export and the screening banner in every app | PARTIAL | timber_ui/analysis_page.py: _downloads_for_map, show_table | Banner text not tested |
| G8 | All random processes take a seed (default 42), same seed = same output | PARTIAL | validation.py: add_baselines (42) | Other seeds default to 0 (bootstrap, field sheets, random points) |
| G9 | Units shown everywhere; feet inputs and US survey foot CRSs converted | PARTIAL | terrain/io.py: unit_warnings; accuracy.py: UNIT_TO_METERS | Rasters in feet only give a warning, they are not converted |
| G10 | Every formula has a plain-language comment | PARTIAL | all of terrain/ | No automatic check |
| P1 | Mudslide / slope stability app | DONE | terrain/apps/mudslide.py; tests/test_apps.py::test_mudslide_steep_slope_scores_higher_than_flat |  |
| P2 | Tree fall app (upgraded) | DONE | terrain/apps/trees.py; tests/test_treefall.py |  |
| P3 | Erosion app | BLOCKED |  | Removed from TIMBER on purpose (owner's decision, Nov 2026); lives in ../terrain-hazard-app |
| P4 | Flooding & foundation drainage app | BLOCKED |  | Removed from TIMBER on purpose; lives in ../terrain-hazard-app |
| P5 | Irrigation & lawn health app | BLOCKED |  | Removed from TIMBER on purpose; lives in ../terrain-hazard-app |
| P6 | Rain garden siting app | BLOCKED |  | Removed from TIMBER on purpose; lives in ../terrain-hazard-app |
| P7 | Winter ice hazard app | BLOCKED |  | Removed from TIMBER on purpose; lives in ../terrain-hazard-app |
| P8 | Trip hazards app | BLOCKED |  | Removed from TIMBER on purpose; lives in ../terrain-hazard-app |
| P9 | Volume measurement app | BLOCKED |  | Removed from TIMBER on purpose; lives in ../terrain-hazard-app |
| T1 | Tree detection (CHM, variable window, watershed, big-map defaults, fast) | DONE | treefall.py: detect_trees; tests/test_treefall.py::test_two_cone_trees_heights_and_height_factor, ::test_performance_2000_by_2000_with_1000_trees |  |
| T2 | Eight hazard factors 0-1 and the snag flag | DONE | treefall.py: hazard_factors; tests/test_treefall.py (slope, lean, exposure, ordinary-tree tests) |  |
| T3 | Weighted score 0-100, classes, reasons | DONE | treefall.py: hazard_score, assign_classes, reasons; tests/test_treefall.py::test_percent_classes_give_exact_counts |  |
| T4 | Fall direction: 36-bin von Mises mixture, wind toggle, compass word, arrows | DONE | treefall.py: fall_probabilities; tests/test_treefall.py::test_slope_factor_and_fall_direction_downhill |  |
| T5 | Landing strip, exposure and reach rasters, GeoTIFF export | DONE | treefall.py: exposure_maps; tests/test_treefall.py::test_fall_exposure_is_a_disk_of_radius_height |  |
| T6 | Auto features, user targets, can-reach columns, priority | DONE | treefall.py: detect_features, landing_chances; tests/test_treefall.py::test_trail_strip..., ::test_building_found..., ::test_tree_that_can_reach_building_ranks_first |  |
| T7 | Downed log detector | DONE | treefall.py: detect_logs; tests/test_treefall.py::test_downed_log_axis_angle |  |
| T8 | Seven tabs and the metrics row | PARTIAL | trees.py: run (5 sections); timber_ui/storm_panel.py (Storm check, Design criteria) | Test only checks five tab names |
| T9 | Season setting with health-weight and snag-rule changes | MISSING | survey.py: SEASONS (stored only) | Season is saved but changes nothing |
| T10 | Leaf-on/leaf-off fusion, outside DTM, vertical alignment, DTM comparison | MISSING |  | Not built |
| V1 | Baselines as table columns | DONE | validation.py: add_baselines; tests/test_storm.py::test_baselines |  |
| V2 | Weight sensitivity (one-at-a-time and 500 Dirichlet sets) | MISSING |  | Not built |
| V3 | Blind field export (sheet, key, booklet) | MISSING | treefall.py: field_check_export (not blind) | Old sheet shows the hazard and class |
| V4 | Field import (defects, Fisher, AUC, heights, snag table, results files) | MISSING | treefall.py: field_check_compare (basic) | Not built |
| V5 | Detection check plots | MISSING |  | Not built |
| V6 | Downed log validation (factor test, axial test, rose, manual logs) | MISSING | treefall.py: compare_logs_with_random (means only) | Not built |
| S1 | storms.yaml with calm control; station CSV; gust-weighted direction | DONE | storm.py: load_storms, read_station_csv, weighted_wind_direction; tests/test_storm.py::test_gust_weighted_direction_wraps_around_north, ::test_wind_stats_from_station_csv, ::test_storms_yaml_round_trip |  |
| S2 | Frozen predictions (both versions), settings, hash log, tamper detection | DONE | storm.py: freeze_predictions, load_frozen; tests/test_storm.py::test_frozen_predictions_are_read_back_unchanged_and_tampering_is_detected, ::test_actual_wind_copy_changes_only_the_direction |  |
| S3 | Alignment: phase correlation, plane fit, warnings | DONE | storm.py: align_surveys; tests/test_storm.py::test_alignment_recovers_shift_offset_and_tilt, ::test_poor_alignment_gives_a_warning |  |
| S4 | Per-tree change measures and classes; threshold from stable RMS | DONE | storm.py: detect_changes; tests/test_storm.py::test_fallen_snapped_crown_damage_and_no_change |  |
| S5 | New logs linked to fallen trees, observed azimuth, unexplained gaps | PARTIAL | storm.py: link_logs_to_trees, unexplained_gaps; tests/test_storm.py::test_fallen_snapped_crown_damage_and_no_change | unexplained_gaps has no test |
| S6 | Leaf-drop guard and top-drop-only mode | PARTIAL | storm.py: detect_changes; tests/test_storm.py::test_leaf_drop_guard | top_only mode has no test |
| S7 | Calm-pair false change rate vs storm rate (Fisher) | DONE | storm.py: calm_comparison; tests/test_storm.py::test_calm_pair_has_no_false_changes |  |
| S8 | Prediction tests from frozen scores | DONE | storm.py: prediction_metrics; tests/test_storm.py::test_auc_is_one..., ::test_random_damage..., ::test_few_damaged_trees... |  |
| S9 | Fall direction tests and rose diagram | DONE | storm.py: direction_metrics; tests/test_storm.py::test_direction_perfect_prediction, ::test_direction_random_falls_are_no_better_than_guessing |  |
| S10 | Landing-zone hit rate and exposure under new logs | PARTIAL | storm.py: landing_metrics; tests/test_storm.py::test_storm_results_files_and_design_criteria | Exposure comparison has no known-answer test |
| S11 | Blind storm field confirmation | DONE | storm.py: field_sheet, field_compare; tests/test_storm.py::test_blind_field_sheet_and_comparison |  |
| S12 | Storm check tab, summary.md storm section, figures | PARTIAL | timber_ui/storm_panel.py | No test of the summary or figure files |
| D1 | All 12 criteria in the yaml and tab, each with PASS / FAIL / NO DATA tests | MISSING | validation.py: DEFAULT_CRITERIA (4 storm criteria only) | 8 criteria missing; no per-criterion tests |
| D2 | Each measured value computed by exactly one function, shared with its tab | PARTIAL | storm.py: pool_results | Only the storm criteria exist |
| D3 | Export for board: summary.md (goal, criteria, V2-V6, S7-S10) + 300 dpi figures | PARTIAL | timber_ui/storm_panel.py: write_summary | Storm section only; goal not word-for-word |
| A1 | scripts/audit.py | MISSING |  | Not built |
| A2 | tests/test_end_to_end.py | MISSING |  | Not built |
| A3 | This checklist with final statuses and evidence | PARTIAL | docs/requirements_checklist.md | This is the BEFORE version |

Counts: DONE 20, PARTIAL 16, MISSING 10, BROKEN 0, BLOCKED 7
