"""Re-check the whole TIMBER project.   Run:  python scripts/audit.py

It checks every requirement in the list below and prints one table per
section, then "AUDIT PASSED" or "AUDIT FAILED: <requirement IDs>". It also
rewrites docs/requirements_checklist.md with the status and evidence of every
requirement. The exit code is 0 only if nothing failed (BLOCKED items are
listed but do not fail the audit).

A requirement is DONE when every test named as its evidence passes and its
extra check (if any) is true; BROKEN when a test fails or the check is false;
MISSING when a named test doesn't exist.
"""

import importlib
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REMOVED = "Removed from TIMBER on purpose (owner's decision, Nov 2026); the app lives in ../terrain-hazard-app."
T, S, V, E, A = "tests/test_treefall.py", "tests/test_storm.py", "tests/test_validation.py", "tests/test_end_to_end.py", "tests/test_apps.py"

# (ID, short requirement, where it lives, [tests that prove it], name of an extra check or None, note)
REQUIREMENTS = [
    ("G1", "AGENTS.md (goal, phases, field workflow); CLAUDE.md with @AGENTS.md", "AGENTS.md, CLAUDE.md", [f"{E}::test_project_documents_exist_and_state_the_goal"], "files", ""),
    ("G2", "README: install, run, a section per phase, goal, field/storm workflows", "README.md", [f"{E}::test_project_documents_exist_and_state_the_goal"], None, ""),
    ("G3", "requirements.txt lists every imported package; optional packages optional", "requirements.txt", [f"{E}::test_every_imported_package_is_in_the_requirements", f"{E}::test_optional_packages_are_not_required"], "imports", "sklearn / statsmodels are not used at all: AUC and logistic regression are written out"),
    ("G4", "App registry; every app has the shared interface", "terrain/apps/__init__.py: APPS", [f"{A}::test_every_app_has_the_shared_interface"], "interface", ""),
    ("G5", "Missing inputs / results files give a message, never a traceback", "base.py: missing_inputs; validation.py: evaluate_criteria", [f"{A}::test_missing_inputs_give_a_message_not_a_crash", f"{E}::test_each_criterion_passes_fails_and_shows_no_data"], None, ""),
    ("G6", "Shared helpers reused, not duplicated", "slope.py, hydrology.py, vegetation.py, validation.py: auc, storm.py: fit_plane", [f"{V}::test_other_dtm_is_aligned_on_stable_ground", "tests/test_shared_helpers.py"], None, "Merged: NaN-aware smoothing (one function), stable-ground RMS. Kept apart on purpose: hydrology.remove_tilt (whole-map tilt for water flow) vs storm.fit_plane (robust fit on stable cells); compass words (N/NE in tables vs 'north' in sentences)"),
    ("G7", "GeoTIFF / PNG / CSV export and the screening banner", "timber_ui/analysis_page.py, style.py", [f"{E}::test_banners"], None, ""),
    ("G8", "Random processes take a seed (default 42) and repeat exactly", "validation.py, storm.py", [f"{E}::test_random_steps_default_to_seed_42", f"{E}::test_running_twice_gives_the_same_numbers", f"{V}::test_sensitivity_changes_with_real_factors_and_is_seeded"], None, ""),
    ("G9", "Units shown; feet and US survey foot rasters converted", "fusion.py: vertical_factor, to_grid; io.py: unit_warnings", [f"{V}::test_us_survey_feet_are_converted_to_meters"], None, "Outside DTMs in feet are converted. A whole SURVEY in feet is not converted: it gets a warning (WebODM exports meters)"),
    ("G10", "Every formula has a plain-language comment", "terrain/", [f"{E}::test_every_function_in_the_math_modules_is_explained"], None, "Checked automatically at function level (every function has a docstring); comments on single lines are by review"),
    ("P1", "Mudslide / slope stability app", "terrain/apps/mudslide.py", [f"{A}::test_mudslide_steep_slope_scores_higher_than_flat"], None, ""),
    ("P2", "Tree fall app (upgraded)", "terrain/apps/trees.py", [f"{A}::test_trees_heights_and_fall_zone_targets", f"{T}::test_app_runs_and_has_five_tabs"], None, ""),
    ("P3", "Erosion app", "", [], "blocked", REMOVED),
    ("P4", "Flooding & foundation drainage app", "", [], "blocked", REMOVED),
    ("P5", "Irrigation & lawn health app", "", [], "blocked", REMOVED),
    ("P6", "Rain garden siting app", "", [], "blocked", REMOVED),
    ("P7", "Winter ice hazard app", "", [], "blocked", REMOVED),
    ("P8", "Trip hazards app", "", [], "blocked", REMOVED),
    ("P9", "Volume measurement app", "", [], "blocked", REMOVED),
    ("T1", "Tree detection (CHM, variable window, watershed, big-map defaults, fast)", "treefall.py: detect_trees", [f"{T}::test_two_cone_trees_heights_and_height_factor", f"{T}::test_performance_2000_by_2000_with_1000_trees"], None, ""),
    ("T2", "Eight hazard factors 0-1 and the snag flag", "treefall.py: hazard_factors", [f"{T}::test_slope_factor_and_fall_direction_downhill", f"{T}::test_lopsided_crown_leans_east", f"{T}::test_ordinary_interior_tree_scores_low_and_small_holes_are_not_openings", f"{T}::test_large_clearing_makes_its_neighbors_edge_trees", f"{V}::test_snag_flag_table"], None, ""),
    ("T3", "Weighted score 0-100, classes, reasons", "treefall.py: hazard_score, assign_classes, reasons", [f"{T}::test_percent_classes_give_exact_counts", f"{V}::test_scores_for_weights_match_the_hazard_score"], None, ""),
    ("T4", "Fall direction: 36-bin von Mises mixture, wind toggle, compass word", "treefall.py: fall_probabilities", [f"{T}::test_slope_factor_and_fall_direction_downhill", f"{S}::test_actual_wind_copy_changes_only_the_direction"], None, ""),
    ("T5", "Landing strip, exposure and reach rasters, GeoTIFF export", "treefall.py: exposure_maps", [f"{T}::test_fall_exposure_is_a_disk_of_radius_height"], None, ""),
    ("T6", "Auto features, user targets, can-reach columns, priority", "treefall.py: detect_features, landing_chances", [f"{T}::test_trail_strip_found_but_round_clearing_is_not", f"{T}::test_building_found_but_cone_tree_of_same_height_is_not", f"{T}::test_tree_that_can_reach_building_ranks_first"], None, ""),
    ("T7", "Downed log detector", "treefall.py: detect_logs", [f"{T}::test_downed_log_axis_angle"], None, ""),
    ("T8", "Seven tabs and the metrics row", "trees.py: TREE_TABS; app.py", [f"{E}::test_tree_page_has_the_seven_parts", f"{T}::test_app_runs_and_has_five_tabs"], "tabs", ""),
    ("T9", "Season setting with health-weight and snag-rule changes", "trees.py: default_weights; treefall.py: hazard_factors; survey.py: read_info", [f"{V}::test_season_changes_the_health_settings"], None, ""),
    ("T10", "Leaf-on/leaf-off fusion, outside DTM, vertical alignment, comparison", "fusion.py; timber_ui/survey_panel.py: load_fused; ground.py: dtm_tab", [f"{V}::test_other_dtm_is_aligned_on_stable_ground", f"{V}::test_us_survey_feet_are_converted_to_meters", f"{V}::test_heights_at_positions"], None, "Only the DTM can come from another survey; the DSM and photo always come from the chosen survey"),
    ("V1", "Baselines as table columns", "validation.py: add_baselines", [f"{S}::test_baselines"], None, ""),
    ("V2", "Weight sensitivity (one-at-a-time and 500 Dirichlet sets), fast", "validation.py: sensitivity_one_at_a_time, sensitivity_random", [f"{V}::test_sensitivity_when_only_height_matters", f"{V}::test_sensitivity_changes_with_real_factors_and_is_seeded", f"{V}::test_sensitivity_is_fast"], "forest_speed", ""),
    ("V3", "Blind field export (sheet, key, booklet)", "validation.py: blind_field_sample; validation_panel.py: _tree_booklet", [f"{V}::test_blind_sheet_hides_everything_about_the_prediction"], None, "The booklet (photo cards) is drawn by the screen code and has no automatic test"),
    ("V4", "Field import (defects, Fisher, AUC, heights, snag table, results files)", "validation.py: field_import", [f"{V}::test_field_import_all_high_and_no_low_defects", f"{V}::test_field_import_equal_defects_and_not_found_trees", f"{V}::test_snag_flag_table", f"{S}::test_auc_matches_a_hand_worked_example"], None, ""),
    ("V5", "Detection check plots", "validation.py: plot_sheet, detection_rates", [f"{V}::test_detection_rate_per_plot_and_overall"], None, ""),
    ("V6", "Downed log validation (trail exclusion, factor test, axial test, manual logs)", "validation.py: log_factor_test, rayleigh_axial, logs_away_from, match_logs", [f"{V}::test_axial_statistics", f"{V}::test_log_factor_test_finds_logs_on_steep_ground", f"{V}::test_logs_near_trails_are_excluded_and_manual_logs_matched"], None, ""),
    ("S1", "storms.yaml with calm control; station CSV; gust-weighted direction", "storm.py: load_storms, read_station_csv, weighted_wind_direction", [f"{S}::test_gust_weighted_direction_wraps_around_north", f"{S}::test_wind_stats_from_station_csv", f"{S}::test_storms_yaml_round_trip"], None, ""),
    ("S2", "Frozen predictions (both versions), hash log, tamper detection", "storm.py: freeze_predictions, load_frozen", [f"{S}::test_frozen_predictions_are_read_back_unchanged_and_tampering_is_detected", f"{S}::test_actual_wind_copy_changes_only_the_direction", f"{S}::test_nothing_frozen_gives_a_clear_error"], None, ""),
    ("S3", "Alignment: phase correlation, plane fit, warnings", "storm.py: align_surveys", [f"{S}::test_alignment_recovers_shift_offset_and_tilt", f"{S}::test_poor_alignment_gives_a_warning", f"{S}::test_shift_that_changes_across_the_map_is_fitted"], None, ""),
    ("S4", "Per-tree change measures and classes; threshold from stable RMS", "storm.py: detect_changes", [f"{S}::test_fallen_snapped_crown_damage_and_no_change", f"{S}::test_trees_at_the_map_edge_are_uncertain"], None, ""),
    ("S5", "New logs linked to fallen trees, observed azimuth, unexplained gaps", "storm.py: link_logs_to_trees, unexplained_gaps", [f"{S}::test_fallen_snapped_crown_damage_and_no_change", f"{S}::test_log_already_there_before_is_not_new", f"{V}::test_unexplained_gap_is_listed"], None, ""),
    ("S6", "Leaf-drop guard and top-drop-only mode", "storm.py: detect_changes", [f"{S}::test_leaf_drop_guard", f"{V}::test_top_only_mode_ignores_crown_thinning"], None, ""),
    ("S7", "Calm-pair false change rate vs storm rate (Fisher)", "storm.py: calm_comparison", [f"{S}::test_calm_pair_has_no_false_changes"], None, ""),
    ("S8", "Prediction tests from frozen scores", "storm.py: prediction_metrics", [f"{S}::test_auc_is_one_when_the_top_scores_are_the_damaged_trees", f"{S}::test_random_damage_gives_auc_near_half_and_the_interval_covers_it", f"{S}::test_few_damaged_trees_gives_a_small_sample_flag"], None, ""),
    ("S9", "Fall direction tests and rose diagram", "storm.py: direction_metrics; charts.py: rose_figure", [f"{S}::test_direction_perfect_prediction", f"{S}::test_direction_random_falls_are_no_better_than_guessing"], None, ""),
    ("S10", "Landing-zone hit rate and exposure under new logs", "storm.py: landing_metrics", [f"{S}::test_storm_results_files_and_design_criteria", f"{V}::test_exposure_under_new_logs_is_compared_with_random_points"], None, ""),
    ("S11", "Blind storm field confirmation", "storm.py: field_sheet, field_compare", [f"{S}::test_blind_field_sheet_and_comparison"], None, ""),
    ("S12", "Storm check tab, summary.md storm section, figures", "timber_ui/storm_panel.py; validation.py: write_summary; charts.py: export_figures", [f"{E}::test_summary_has_goal_criteria_and_sentences"], None, "The screens themselves are checked by hand in the browser, not by a test"),
    ("D1", "All 12 criteria in the yaml and tab, each with PASS / FAIL / NO DATA tests", "validation.py: CRITERIA, evaluate_criteria", [f"{E}::test_the_yaml_has_exactly_the_twelve_criteria", f"{E}::test_each_criterion_passes_fails_and_shows_no_data"], "criteria", ""),
    ("D2", "Each measured value computed by one function, shared with its tab", "validation.py: save_result, measured_value", [f"{E}::test_edited_targets_are_kept_and_missing_criteria_are_added", f"{E}::test_inf_ratio_round_trip", f"{E}::test_hectares_per_hour"], None, "Screens call the same validation/storm function and save its 'criteria' block; the table only reads that block"),
    ("D3", "Export for board: summary.md + 300 dpi figures", "validation.py: write_summary; charts.py: export_figures", [f"{E}::test_summary_has_goal_criteria_and_sentences"], "figures", ""),
    ("A1", "scripts/audit.py", "scripts/audit.py", [], "self", ""),
    ("A2", "tests/test_end_to_end.py", "terrain/selfcheck.py", [f"{E}::test_whole_pipeline_produces_every_output"], "end_to_end", ""),
    ("A3", "This checklist with final statuses and evidence", "docs/requirements_checklist.md", [], "self", "Rewritten by every audit run"),
]

REQUIRED_FILES = ["AGENTS.md", "CLAUDE.md", "README.md", "requirements.txt", "config/design_criteria.yaml", "config/storms.yaml",
                  "docs/prompts/phase5_prompt.md", "docs/prompts/tree_hazard_prompt.md",
                  "docs/prompts/phase6_validation_prompt.md", "docs/prompts/phase7_storm_validation_prompt.md"]
MODULES = ["terrain.io", "terrain.slope", "terrain.hydrology", "terrain.vegetation", "terrain.accuracy", "terrain.treefall",
           "terrain.survey", "terrain.validation", "terrain.storm", "terrain.fusion", "terrain.selfcheck", "terrain.apps",
           "terrain.apps.base", "terrain.apps.trees", "terrain.apps.mudslide"]


def run_pytest() -> dict:
    """Run the whole test suite once; returns {"file::test": "passed" / "failed"} and the counts."""
    report = Path(tempfile.mkdtemp()) / "pytest.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={report}"], cwd=ROOT, capture_output=True, text=True)
    outcomes = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        name = case.get("name").split("[")[0]  # parametrized tests count as one
        key = f"tests/{case.get('classname').split('.')[-1]}.py::{name}"
        bad = case.find("failure") is not None or case.find("error") is not None
        outcomes[key] = "failed" if bad or outcomes.get(key) == "failed" else "passed"
    return outcomes


def extra_checks() -> dict:
    """The checks that are not a single test: files, imports, interface, criteria, tabs, end to end, speed."""
    out, notes = {}, {}
    missing = [f for f in REQUIRED_FILES if not (ROOT / f).exists()]
    out["files"], notes["files"] = not missing, f"missing: {missing}" if missing else ""

    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as err:
            failed.append(f"{name} ({err})")
    out["imports"], notes["imports"] = not failed, "; ".join(failed)

    from terrain import apps, selfcheck, validation
    from terrain.apps import trees

    bad = [f"{app.__name__}.{attr}" for app in apps.APPS for attr in apps.REQUIRED_ATTRIBUTES if not hasattr(app, attr)]
    out["interface"], notes["interface"] = not bad, ", ".join(bad)

    try:
        import yaml

        saved = yaml.safe_load((ROOT / "config" / "design_criteria.yaml").read_text(encoding="utf-8")) or {}
        wrong = [n for n in validation.CRITERIA if n not in saved or saved[n].get("target") is None
                 or not isinstance(saved[n]["target"], (bool, int, float))]
        extra = [n for n in saved if n not in validation.CRITERIA]
        out["criteria"], notes["criteria"] = not wrong and not extra and len(saved) == 12, f"problems: {wrong + extra}" if wrong or extra else ""
    except Exception as err:
        out["criteria"], notes["criteria"] = False, str(err)

    source = (ROOT / "terrain" / "apps" / "trees.py").read_text(encoding="utf-8") + (ROOT / "app.py").read_text(encoding="utf-8")
    absent = [t for t in trees.TREE_TABS if f'"{t}"' not in source]
    out["tabs"], notes["tabs"] = not absent and len(trees.TREE_TABS) == 7, f"missing: {absent}" if absent else ""

    work = Path(tempfile.mkdtemp())
    try:
        result = selfcheck.run(work, n_boot=50)
        lost = [step for step, path in result["steps"].items() if path is None]
        out["end_to_end"], notes["end_to_end"] = not lost, f"no output from: {lost}" if lost else f"{len(result['steps'])} steps"
        from timber_ui import charts

        figures = charts.export_figures(work)
        out["figures"], notes["figures"] = len(figures) >= 3, f"{len(figures)} figures written"
    except Exception as err:
        out["end_to_end"], notes["end_to_end"] = False, f"crashed: {err}"
        out["figures"], notes["figures"] = False, "end-to-end run crashed"

    out["forest_speed"], notes["forest_speed"] = forest_speed()
    out["self"], notes["self"] = True, ""
    return out, notes


def forest_speed():
    """If data/forest exists: time the tree analysis and the sensitivity run on it."""
    folder = ROOT / "data" / "forest"
    if not folder.exists():
        return True, "data/forest not present: speed not measured"
    from terrain import survey, validation
    from terrain.apps import base, trees
    from terrain.selfcheck import DefaultWidgets

    start = time.time()
    s = survey.prepare(folder, 0.25, 10.0, False, cache_dir=ROOT / ".timber_cache")
    load = time.time() - start
    layers = base.Layers(dtm=s.dtm, dsm=s.dsm, ortho=s.ortho, flow_result=s.flow)  # no result cache: a real run
    p = trees.params(DefaultWidgets(), {"width_m": 555.0, "season": survey.read_info(folder)["season"]})
    start = time.time()
    table = trees.run(layers, p).raw["trees"]
    pipeline = time.time() - start
    start = time.time()
    F = validation.factor_matrix(table, list(p["weights"]))
    validation.sensitivity_one_at_a_time(F, p["weights"])
    validation.sensitivity_random(F, p["weights"])
    sens = time.time() - start
    ok = sens < 30 and pipeline < 120
    return ok, (f"data/forest: {len(table)} trees; survey load {load:.1f} s, tree pipeline {pipeline:.1f} s (limit 120), "
                f"sensitivity {sens:.2f} s (limit 30)")


def main() -> int:
    print("Running pytest...")
    outcomes = run_pytest()
    passed, failed = sum(v == "passed" for v in outcomes.values()), sum(v == "failed" for v in outcomes.values())
    print(f"pytest: {passed} passed, {failed} failed (parametrized tests counted once)")
    print("Running the extra checks (files, imports, end to end, speed)...")
    checks, check_notes = extra_checks()

    rows = []
    for rid, text, where, tests, check, note in REQUIREMENTS:
        if check == "blocked":
            rows.append((rid, text, "BLOCKED", "", note))
            continue
        results = []
        for t in tests:
            if "::" in t:
                results.append(outcomes.get(t, "missing"))
            else:  # a whole test file
                in_file = [v for k, v in outcomes.items() if k.startswith(t + "::")]
                results.append("missing" if not in_file else "failed" if "failed" in in_file else "passed")
        check_ok = checks.get(check, True) if check else True
        if "failed" in results or not check_ok:
            status = "BROKEN"
        elif "missing" in results:
            status = "MISSING"
        else:
            status = "DONE"
        detail = "; ".join(x for x in (note, check_notes.get(check, "") if check else "") if x)
        rows.append((rid, text, status, "; ".join(x for x in [where] + tests if x), detail))

    print()
    print(f"{'section':<28}{'passing / total':<18}not passing")
    names = {"G": "G  Global / structure", "P": "P  Phase 5 apps", "T": "T  Tree fall core", "V": "V  Validation",
             "S": "S  Storm validation", "D": "D  Design criteria", "A": "A  Audit tooling"}
    for letter, name in names.items():
        mine = [r for r in rows if r[0][0] == letter]
        good = sum(r[2] == "DONE" for r in mine)
        others = ", ".join(f"{r[0]} {r[2]}" for r in mine if r[2] != "DONE")
        print(f"{name:<28}{f'{good} / {len(mine)}':<18}{others}")
    for key in ("end_to_end", "figures", "forest_speed"):
        print(f"  {key}: {'ok' if checks[key] else 'FAILED'}  {check_notes[key]}")

    counts = Counter(r[2] for r in rows)
    lines = ["# TIMBER requirements checklist", "",
             f"Written by `python scripts/audit.py` on {time.strftime('%Y-%m-%d %H:%M')}. "
             f"pytest: {passed} passed, {failed} failed.", "",
             "Status: DONE = every test named as evidence passes; BROKEN = a test or check fails; MISSING = a named test "
             "doesn't exist; BLOCKED = can't be met, with the reason. The status before the audit is in "
             "`requirements_checklist_before.md`.", "",
             "| ID | requirement (short) | status | evidence | notes |", "|---|---|---|---|---|"]
    lines += [f"| {i} | {t} | {s} | {e} | {n} |" for i, t, s, e, n in rows]
    lines += ["", "Counts: " + ", ".join(f"{k} {counts.get(k, 0)}" for k in ("DONE", "PARTIAL", "MISSING", "BROKEN", "BLOCKED")), ""]
    (ROOT / "docs" / "requirements_checklist.md").write_text("\n".join(lines), encoding="utf-8")

    bad = [r[0] for r in rows if r[2] in ("BROKEN", "MISSING")]
    blocked = [r[0] for r in rows if r[2] == "BLOCKED"]
    print()
    print("Counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if blocked:
        print(f"BLOCKED (explained in docs/requirements_checklist.md): {', '.join(blocked)}")
    if bad or failed:
        print("AUDIT FAILED: " + (", ".join(bad) if bad else f"{failed} failing tests"))
        return 1
    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
