import pytest

pytest.importorskip("shap")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")
from shap_review.fuzzing.harnesses.treeexplainer import run_case
from shap_review.fuzzing.oracles.tree import evaluate_execution


def test_real_treeexplainer_execution():
    case = {
        "n_features": 3,
        "depth": 3,
        "n_trees": 2,
        "n_samples": 5,
        "dtype": "float64",
        "nan": False,
        "classification": False,
        "model_output": "raw",
        "seed": 7,
    }
    result = run_case(case)
    assert result["executed"] is True
    assert result.get("failed") is not True
    assert evaluate_execution(result)["valid"] is True


def test_real_classifier_treeexplainer_execution():
    case = {
        "n_features": 3,
        "depth": 3,
        "n_trees": 2,
        "n_samples": 5,
        "dtype": "float64",
        "nan": False,
        "classification": True,
        "model_output": "probability",
        "seed": 9,
    }
    result = run_case(case)
    assert result["executed"] is True
    assert result.get("failed") is not True
