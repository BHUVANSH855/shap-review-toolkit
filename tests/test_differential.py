from pathlib import Path

from shap_review.differential import compare, differential_scripts


def test_recursive_numeric_comparison():
    r = compare({"values": [1.0, 2.0]}, {"values": [1.0, 2.0000001]})
    assert r["equal"]


def test_differential_scripts(tmp_path: Path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text('import json; print(json.dumps({"value": [1.0, 2.0]}))')
    b.write_text('import json; print(json.dumps({"value": [1.0, 2.00000001]}))')
    r = differential_scripts(a, b)
    assert (
        r["status"] == "MATCH"
        and r["differential_agreement"] is True
        and r["reference_correctness"] == "UNKNOWN"
        and r["semantic_status"] == "NOT_EVALUATED"
    )


class FakeExplanation:
    def __init__(self):
        self.values = [1.0, 2.0]
        self.base_values = [0.5]
        self.data = [3, 4]
        self.feature_names = ["a", "b"]


def test_shap_semantic_normalization():
    from shap_review.differential import normalize_shap_result

    result = normalize_shap_result(FakeExplanation())
    assert result["values"] == [1.0, 2.0]
    assert result["feature_names"] == ["a", "b"]


def test_differential_rejects_feature_axis_broadcast():
    from shap_review.differential.semantic import compare_shap_contract

    left = {"values": [[1.0, 2.0], [3.0, 4.0]]}
    right = {"values": [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]}
    r = compare_shap_contract(left, right)
    assert r["semantic_equal"] is False
    assert "cannot be implicitly broadcast" in r["fields"]["values"]["reason"]


def test_differential_scalar_alignment_is_explicit():
    from shap_review.differential.semantic import evaluate_additivity

    r = evaluate_additivity([[1.0, 2.0]], 0.0, 3.0)
    assert r["applicable"] is True and r["passed"] is True
    assert r["broadcast"]["semantic_axis"] == "scalar"
