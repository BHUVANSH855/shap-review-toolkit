from pathlib import Path

from shap_review.reproduction.runner import run_script


def test_reproduction_runner(tmp_path: Path):
    script = tmp_path / "ok.py"
    script.write_text('print("ok")')

    result = run_script(str(script))

    assert result["passed"] is True


def test_reproduction_runner_accepts_environment(tmp_path: Path):
    script = tmp_path / "env.py"
    script.write_text(
        'import os; print(os.environ["SHAP_REVIEW_TEST"])'
    )

    result = run_script(
        str(script),
        env={"SHAP_REVIEW_TEST": "ok"},
    )

    assert result["passed"] is True
    assert "ok" in result["stdout"]