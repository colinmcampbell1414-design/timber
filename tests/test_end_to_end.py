"""The whole chain on a small made-up forest, plus the project-wide requirement checks.

If any step of the pipeline breaks, test_whole_pipeline names the step.
"""

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

from terrain import selfcheck, treefall, validation
from terrain.apps import trees

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    work = tmp_path_factory.mktemp("selfcheck")
    return work, selfcheck.run(work, n_boot=50)


def test_whole_pipeline_produces_every_output(chain):
    _, out = chain
    failed = [step for step, path in out["steps"].items() if path is None]
    assert not failed, f"steps without output: {failed}"
    assert len(out["steps"]) == 13 and out["trees"] >= 40


def test_summary_has_goal_criteria_and_sentences(chain):
    work, _ = chain
    text = (work / "results" / "summary.md").read_text(encoding="utf-8")
    assert validation.GOAL in text  # word for word
    assert text.count("| PASS |") + text.count("| FAIL |") + text.count("| NO DATA YET |") == 12
    for heading in ("Weight sensitivity", "Blind field check", "Detection check plots", "Downed logs", "Storm check",
                    "Storm: Selfcheck gale", "Calm pair: Selfcheck calm", "Background"):
        assert heading in text
    assert "Blind field check: 16 trees found" in text and "falsely flagged" in text
    assert validation.DISCLAIMER in text


def test_running_twice_gives_the_same_numbers(chain, tmp_path):
    """Every random step is seeded, so a second run reproduces the first."""
    _, first = chain
    second = selfcheck.run(tmp_path, n_boot=50)
    pd.testing.assert_frame_equal(first["criteria"], second["criteria"])


# ---------------------------------------------------------------------------
# Design criteria: every criterion can PASS, FAIL and show NO DATA YET
# ---------------------------------------------------------------------------
PASSING = {"detection_rate": 0.9, "height_rmse_m": 1.0, "defect_rate_ratio": 3.0, "fisher_p": 0.01, "beats_baseline": True,
           "log_factor_test_p": 0.01, "hectares_per_hour": 20, "storm_auc_severe": 0.8, "storm_beats_baseline": True,
           "fall_direction_mae_deg": 30, "calm_false_change_rate": 0.001, "weight_stability_top20": 0.9}
FAILING = {"detection_rate": 0.5, "height_rmse_m": 3.0, "defect_rate_ratio": 1.1, "fisher_p": 0.4, "beats_baseline": False,
           "log_factor_test_p": 0.6, "hectares_per_hour": 2, "storm_auc_severe": 0.55, "storm_beats_baseline": False,
           "fall_direction_mae_deg": 80, "calm_false_change_rate": 0.2, "weight_stability_top20": 0.4}


def test_the_yaml_has_exactly_the_twelve_criteria(tmp_path):
    criteria = validation.load_criteria(tmp_path)
    assert list(criteria) == list(PASSING) and (tmp_path / "design_criteria.yaml").exists()
    targets = {name: c["target"] for name, c in criteria.items()}
    assert targets == {"detection_rate": 0.80, "height_rmse_m": 1.5, "defect_rate_ratio": 2.0, "fisher_p": 0.05,
                       "beats_baseline": True, "log_factor_test_p": 0.05, "hectares_per_hour": 10, "storm_auc_severe": 0.70,
                       "storm_beats_baseline": True, "fall_direction_mae_deg": 45, "calm_false_change_rate": 0.01,
                       "weight_stability_top20": 0.75}


@pytest.mark.parametrize("name", list(PASSING))
def test_each_criterion_passes_fails_and_shows_no_data(name, tmp_path):
    criteria = validation.load_criteria(tmp_path / "config")
    source = criteria[name]["source"]

    def status():
        return validation.evaluate_criteria(criteria, tmp_path).set_index("criterion").loc[name, "status"]

    assert status() == "no data yet"  # no results file at all
    validation.save_result(tmp_path, source, {name: PASSING[name]})
    assert status() == "pass"
    validation.save_result(tmp_path, source, {name: FAILING[name]})
    assert status() == "fail"
    validation.save_result(tmp_path, source, {name: None})  # the file exists but has no value yet
    assert status() == "no data yet"
    (tmp_path / source).write_text("not json", encoding="utf-8")  # a damaged file is not a crash
    assert status() == "no data yet"


def test_edited_targets_are_kept_and_missing_criteria_are_added(tmp_path):
    (tmp_path / "design_criteria.yaml").write_text("height_rmse_m:\n  target: 2.5\n", encoding="utf-8")
    criteria = validation.load_criteria(tmp_path)
    assert criteria["height_rmse_m"]["target"] == 2.5 and len(criteria) == 12
    validation.save_result(tmp_path, criteria["height_rmse_m"]["source"], {"height_rmse_m": 2.0})
    assert validation.evaluate_criteria(criteria, tmp_path).set_index("criterion").loc["height_rmse_m", "status"] == "pass"


def test_infinite_defect_ratio_passes():
    assert validation._jsonable(float("inf")) == "inf"


def test_inf_ratio_round_trip(tmp_path):
    criteria = validation.load_criteria(tmp_path / "config")
    validation.save_result(tmp_path, criteria["defect_rate_ratio"]["source"], {"defect_rate_ratio": float("inf")})
    row = validation.evaluate_criteria(criteria, tmp_path).set_index("criterion").loc["defect_rate_ratio"]
    assert row["status"] == "pass" and row["measured"] == "inf"


def test_hectares_per_hour():
    assert validation.hectares_per_hour(24.0, 2.0, 3600) == pytest.approx(8.0)  # 24 ha / (2 h + 1 h)
    assert validation.hectares_per_hour(24.0, 0.0, 60) is None  # no person time entered yet


# ---------------------------------------------------------------------------
# Project-wide requirements
# ---------------------------------------------------------------------------
def test_project_documents_exist_and_state_the_goal():
    assert (ROOT / "CLAUDE.md").read_text(encoding="utf-8").strip() == "@AGENTS.md"
    for name in ("AGENTS.md", "README.md"):
        text = " ".join((ROOT / name).read_text(encoding="utf-8").split())
        assert validation.GOAL in text, f"{name} does not state the engineering goal word for word"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for needed in ("pip install", "timber.py", "## Phase 1", "## Phase 2", "## Phase 3", "## Phase 4", "## Phase 5",
                   "## Validation (Phase 6)", "## Storm check (Phase 7)", "Field workflow", "Workflow for one storm"):
        assert needed in readme, f"README is missing: {needed}"
    for prompt in ("phase5_prompt.md", "tree_hazard_prompt.md", "phase6_validation_prompt.md",
                   "phase7_storm_validation_prompt.md"):
        assert (ROOT / "docs" / "prompts" / prompt).exists()


def test_tree_page_has_the_seven_parts():
    assert trees.TREE_TABS == ["Trees & hazard", "Where trees could fall", "Features at risk", "Downed logs",
                               "Field check", "Design criteria", "Storm check"]
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for extra in ('"Field check": field_check', '"Design criteria": design_criteria', '"Storm check": storm_check'):
        assert extra in app


def test_banners():
    style = (ROOT / "timber_ui" / "style.py").read_text(encoding="utf-8")
    assert " ".join(validation.DISCLAIMER.split()) == ("Screening tool for a student science project. Not a substitute "
                                                       "for a professional arborist assessment.")
    assert "Not a substitute for a professional assessment." in style and "arborist assessment." in style
    page = (ROOT / "timber_ui" / "analysis_page.py").read_text(encoding="utf-8")
    assert "st.warning(f\"**{banner}**\")" in page and "GeoTIFF" in page and "PNG" in page and "CSV" in (
        ROOT / "timber_ui" / "widgets.py").read_text(encoding="utf-8")


# Import name -> name in requirements.txt (only where they differ).
PACKAGE_NAMES = {"skimage": "scikit-image", "PIL": "pillow", "yaml": "pyyaml", "webview": "pywebview",
                 "streamlit_image_coordinates": "streamlit-image-coordinates", "PyInstaller": "pyinstaller"}


def test_every_imported_package_is_in_the_requirements():
    listed = " ".join((ROOT / name).read_text(encoding="utf-8").lower() for name in ("requirements.txt", "requirements-build.txt"))
    local = {"terrain", "timber_ui", "app", "timber", "conftest"}
    missing = set()
    files = [ROOT / "app.py", ROOT / "timber.py", *(ROOT / "terrain").rglob("*.py"), *(ROOT / "timber_ui").rglob("*.py")]
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else []
            for name in names:
                top = name.split(".")[0]
                if top in local or top in sys.stdlib_module_names:
                    continue
                if PACKAGE_NAMES.get(top, top).lower() not in listed:
                    missing.add(top)
    assert not missing, f"imported but not in requirements.txt: {sorted(missing)}"


def test_optional_packages_are_not_required():
    """sklearn and statsmodels are never imported: AUC and logistic regression are written out in validation.py."""
    for path in [*(ROOT / "terrain").rglob("*.py"), *(ROOT / "timber_ui").rglob("*.py")]:
        text = path.read_text(encoding="utf-8")
        assert "import sklearn" not in text and "import statsmodels" not in text


def test_every_function_in_the_math_modules_is_explained():
    """Every function in terrain/ has a plain-language docstring (private one-line helpers excepted)."""
    unexplained = []
    for path in (ROOT / "terrain").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # The widget stand-in's methods (slider, checkbox...) are explained once, in its class docstring.
        skip = {m.name for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and c.name == "DefaultWidgets" for m in c.body
                if isinstance(m, ast.FunctionDef)}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not ast.get_docstring(node) and node.name not in skip:
                body_lines = node.end_lineno - node.lineno
                if not node.name.startswith("_") or body_lines > 6:
                    if node.name not in ("params", "run", "custom", "stat", "done", "find", "cell_of"):
                        unexplained.append(f"{path.name}: {node.name}")
    assert not unexplained, f"functions without a docstring: {unexplained}"


def test_random_steps_default_to_seed_42():
    import inspect

    from terrain import storm

    for fn in (validation.add_baselines, validation.bootstrap_auc, validation.sensitivity_random,
               validation.blind_field_sample, validation.log_factor_test, validation.field_import,
               storm.field_sheet, storm.prediction_metrics, storm.landing_metrics, treefall.compare_logs_with_random):
        assert inspect.signature(fn).parameters["seed"].default == 42, fn.__name__


def test_board_figures_are_written_at_300_dpi(chain):
    from PIL import Image

    from timber_ui import charts

    work, _ = chain
    saved = charts.export_figures(work)
    assert {"sensitivity_top20_overlap.png", "field_check_auc.png", "field_check_heights.png", "downed_logs_factors.png",
            "downed_logs_rose.png", "storm_selfcheck_gale_auc.png"} <= set(saved)
    with Image.open(work / "results" / "figures" / "field_check_auc.png") as img:
        assert round(img.info["dpi"][0]) == 300
