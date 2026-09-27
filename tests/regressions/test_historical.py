import pytest

pytest.importorskip("shap")
from shap_review.regressions import run_4495, run_4911, run_5098


def test_4911_nullable_dtype():
    pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    r = run_4911()
    assert r.status in {"reproduced", "not_reproduced", "blocked"}
    assert r.status == "reproduced", r


def test_4495_expected_value_stability():
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    r = run_4495()
    assert r.status == "reproduced", r


def test_5098_source_grounded_precondition():
    r = run_5098()
    assert r.status in {"static_precondition", "blocked"}
