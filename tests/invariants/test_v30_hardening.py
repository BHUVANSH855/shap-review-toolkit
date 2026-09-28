from pathlib import Path


def test_public_import_paths_exist():
    import shap_review
    import shap_review.fuzzing

    assert callable(shap_review.run_script)
    assert callable(shap_review.fuzzing.evaluate_execution)


def test_gpu_differential_uses_canonical_semantic_comparator():
    from shap_review.fuzzing.gpu_differential import CPUGPUDifferential

    assert CPUGPUDifferential()._compare({"values": [1.0]}, {"values": [1.0]})["semantic_equal"]


def test_declared_oracle_is_not_proven():
    from shap_review.contracts.oracles import classify_target_provenance

    result = classify_target_provenance("independent_function", True)
    assert result["oracle_independence"] == "DECLARED"
    assert result["independence_basis"] == "caller_declared"


def test_evidence_ambiguity_is_not_scoring_eligible():
    from types import SimpleNamespace
    from shap_review.evidence.validation import validate_evidence_item

    item = SimpleNamespace(kind="dynamic", origin="derived", execution_id=None, fixture_id=None, input_fingerprint=None, derived_from=())
    result = validate_evidence_item(item)
    assert result.valid is True
    assert result.status == "AMBIGUOUS"
    assert result.provenance_valid is False
    assert result.scoring_eligible is False


def test_release_validator_source_rules_present():
    root = Path(__file__).resolve().parents[2]
    text = (root / "MANIFEST.in").read_text()
    assert "global-exclude **/.out/*" in text


def test_contribution_alignment_does_not_broadcast_output_axis():
    import numpy as np
    from shap_review.contracts.tensor import SHAPAxisSpec, semantic_align

    axes = SHAPAxisSpec(sample_axis=0, feature_axis=1, output_axis=2)
    try:
        semantic_align(
            np.ones((2, 3, 1)),
            np.ones((2, 3, 2)),
            axes=axes,
            role="values",
        )
    except ValueError as exc:
        assert "does not authorize singleton broadcast" in str(exc)
    else:
        raise AssertionError("contribution output-axis broadcasting must be rejected")
