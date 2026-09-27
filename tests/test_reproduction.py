from pathlib import Path

from shap_review.reproduction.runner import run_script


def test_reproduction_runner(tmp_path: Path):
    p = tmp_path / "ok.py"
    p.write_text('print("ok")')
    r = run_script(str(p))
    assert r["passed"]


def test_reproduction_runner_accepts_environment(tmp_path: Path):
    p = tmp_path / "env.py"
    p.write_text('import os; print(os.environ["SHAP_REVIEW_TEST"])')
    r = run_script(str(p), env={"SHAP_REVIEW_TEST": "ok"})
    assert r["passed"] and "ok" in r["stdout"]
