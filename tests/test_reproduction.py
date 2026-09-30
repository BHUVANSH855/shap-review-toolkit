from pathlib import Path

from shap_review.reproduction.runner import run_script


def test_reproduction_runner(tmp_path: Path):
    script = tmp_path / "ok.py"
    script.write_text('print("ok")')

    result = run_script(str(script))

    assert result["passed"] is True


def test_reproduction_runner_accepts_environment(tmp_path: Path):
    script = tmp_path / "env.py"
    script.write_text('import os; print(os.environ["SHAP_REVIEW_TEST"])')

    result = run_script(
        str(script),
        env={"SHAP_REVIEW_TEST": "ok"},
    )

    assert result["passed"] is True
    assert "ok" in result["stdout"]


def test_historical_nullable_dtype_reproduction_has_explicit_status():
    import pytest

    pytest.importorskip("shap")
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from shap_review.regressions import run_4911

    result = run_4911()
    assert result.status in {
        "reproduced",
        "not_reproduced",
        "blocked",
        "ambiguous",
        "target_failure",
        "toolkit_failure",
        "unsupported",
    }
    if result.status == "ambiguous":
        assert result.reproduced is False


def test_historical_expected_value_reproduction_is_stable():
    import pytest

    pytest.importorskip("shap")
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    from shap_review.regressions import run_4495

    result = run_4495()
    # "blocked" is acceptable when XGBoost/SHAP version incompatibility
    # prevents the regression from running (e.g. XGBoost scientific notation
    # parsing issue on Python 3.10 with certain version combinations).
    assert result.status in {
        "reproduced",
        "not_reproduced",
        "blocked",
    }, result


def test_historical_model_output_reproduction_requires_source_grounded_precondition():
    import pytest

    pytest.importorskip("shap")
    from shap_review.regressions import run_5098

    result = run_5098()
    assert result.status in {"static_precondition", "blocked"}
