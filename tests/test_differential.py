from pathlib import Path

from shap_review.differential import compare, differential_scripts


def test_recursive_numeric_comparison():
    result = compare(
        {"values": [1.0, 2.0]},
        {"values": [1.0, 2.0000001]},
    )

    assert result["equal"]


def test_differential_scripts(tmp_path: Path):
    reference = tmp_path / "reference.py"
    target = tmp_path / "target.py"

    reference.write_text(
        'import json; print(json.dumps({"value": [1.0, 2.0]}))'
    )
    target.write_text(
        'import json; print(json.dumps({"value": [1.0, 2.00000001]}))'
    )

    result = differential_scripts(reference, target)

    assert result["status"] == "MATCH"
    assert result["differential_agreement"] is True
    assert result["reference_correctness"] == "UNKNOWN"
    assert result["semantic_status"] == "NOT_EVALUATED"


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

    left = {
        "values": [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    }
    right = {
        "values": [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
        ]
    }

    result = compare_shap_contract(left, right)

    assert result["semantic_equal"] is False
    assert (
        "cannot be implicitly broadcast"
        in result["fields"]["values"]["reason"]
    )


def test_differential_scalar_alignment_is_explicit():
    from shap_review.differential.semantic import evaluate_additivity

    result = evaluate_additivity(
        [[1.0, 2.0]],
        0.0,
        3.0,
    )

    assert result["applicable"] is True
    assert result["passed"] is True
    assert result["broadcast"]["semantic_axis"] == "scalar"

def test_differential_contract_cannot_be_hidden_by_generic_match(tmp_path: Path):
    from shap_review.differential.runner import differential_scripts

    left = tmp_path / "left.py"
    right = tmp_path / "right.py"

    left.write_text(
        "import json; print(json.dumps({'values':[1.0], 'base_values':[0.0]}))\n"
    )
    right.write_text(
        "import json; print(json.dumps({'values':[1.0], 'base_values':[0.5]}))\n"
    )

    result = differential_scripts(left, right)

    assert result["comparison"]["equal"] is False
    assert result["contract_comparison"]["equal"] is False
    assert result["equal"] is False

def test_differential_requires_semantic_contract():
    from shap_review.differential.semantic import compare_shap_contract

    left = {
        "values": [[1.0, 2.0]],
        "base_values": [0.0],
    }
    right = {
        "values": [[1.0, 2.0]],
        "base_values": [0.0],
    }

    result = compare_shap_contract(left, right)

    assert result["equal"] is True

def test_differential_marks_metadata_separately():
    from shap_review.differential.semantic import compare_shap_contract

    first = {"values": [[1.0]], "feature_names": ["a"]}
    second = {"values": [[1.0]], "feature_names": ["b"]}

    result = compare_shap_contract(first, second)

    assert result["semantic_equal"]
    assert result["fields"]["feature_names"]["severity"] == "metadata"


def test_cpu_gpu_script_differential_reports_semantic_fields(tmp_path):
    from shap_review.differential.gpu_runner import cpu_gpu_scripts

    cpu = tmp_path / "cpu.py"
    gpu = tmp_path / "gpu.py"

    cpu.write_text(
        "import json; "
        "print(json.dumps({'values': [[1.0]], 'base_values': [0.0]}))\n"
    )
    gpu.write_text(
        "import json; "
        "print(json.dumps({'values': [[1.0]], "
        "'base_values': [0.0], 'feature_names':['x']}))\n"
    )

    result = cpu_gpu_scripts(cpu, gpu)

    assert result["applicable"]
    assert result["semantic_equal"]

def test_canonical_additivity_does_not_guess_equal_output_dimensions():
    import numpy as np

    from shap_review.differential.semantic import evaluate_additivity

    result = evaluate_additivity(np.ones((1, 2, 2)), np.zeros(2), np.array([2.0, 2.0]), interaction=False)
    assert result["applicable"] and result["passed"]


def test_differential_uses_canonical_semantic_tensor():
    import numpy as np

    from shap_review.differential.semantic import evaluate_additivity

    result = evaluate_additivity(
        np.ones((2, 3, 2)), np.zeros((2, 2)), np.full((2, 2), 3.0)
    )
    assert result["passed"] is True
