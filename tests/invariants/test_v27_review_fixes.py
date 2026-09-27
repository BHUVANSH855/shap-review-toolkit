from pathlib import Path

import numpy as np


def test_tensor_rejects_semantically_invalid_baseline_rank():
    from shap_review.contracts.tensor import SHAPSemanticTensor

    t = SHAPSemanticTensor.from_values(np.ones((2, 3)), base_values=np.ones((2, 3, 1)))
    try:
        t.reconstruction()
    except ValueError:
        return
    raise AssertionError("invalid baseline rank was accepted")


def test_differential_match_is_not_correctness():
    import tempfile

    from shap_review.differential.runner import differential_scripts

    with tempfile.TemporaryDirectory() as d:
        a = Path(d) / "a.py"
        b = Path(d) / "b.py"
        a.write_text('import json; print(json.dumps({"values":[[1.0]]}))')
        b.write_text('import json; print(json.dumps({"values":[[1.0]]}))')
        r = differential_scripts(a, b)
        assert r["comparison_status"] == "MATCH" and r["differential_agreement"] is True
        assert (
            r["reference_correctness"] == "UNKNOWN"
            and r["semantic_status"] == "NOT_EVALUATED"
        )


def test_sanitizer_clean_is_not_memory_safety_confirmation(tmp_path):
    from shap_review.reproduction.sanitizer import run_sanitized

    s = tmp_path / "ok.py"
    s.write_text('print("clean")')
    r = run_sanitized(s, "asan")
    assert r["verdict"] in {"SANITIZER_CLEAN", "SANITIZER_EXECUTION_FAILED"}
    assert r["memory_safety_confirmation"] is False


def test_evidence_same_transformation_is_correlated():
    from shap_review.evidence.graph import EvidenceGraph, EvidenceNode

    g = EvidenceGraph()
    g.add(EvidenceNode("a", "dynamic", "x", "p", "a", True, transformation="norm"))
    g.add(EvidenceNode("b", "dynamic", "x", "p", "b", True, transformation="norm"))
    assert (
        not g.is_independent("a", "b")
        and g.independence_reason("a", "b") == "shared-transformation"
    )


def test_backend_adapter_reports_actual_failure_stage():
    from shap_review.backends.adapter import MatrixBackendAdapter
    from shap_review.fuzzing.backend_matrix import BackendSpec

    class M:
        pass

    a = MatrixBackendAdapter(BackendSpec("x", "x", (), True, "1"), M())
    a.prepare = lambda **kw: {}

    def bad(ctx):
        raise RuntimeError("fit broke")

    a.fit = bad
    r = a.execute_case()
    assert r["stage"] == "fit" and r["execution_reason"] == "BACKEND_ERROR"


def test_semantic_align_rejects_feature_axis_broadcast():
    import numpy as np

    from shap_review.contracts.tensor import SHAPAxisSpec, semantic_align

    try:
        semantic_align(
            np.ones((2, 1, 3)), np.ones((2, 4, 3)), axes=SHAPAxisSpec(output_axis=2)
        )
    except ValueError:
        return
    raise AssertionError("feature-axis broadcasting was accepted")


def test_input_mutation_reports_structural_dimensions():
    import numpy as np

    from shap_review.contracts.oracles import InputMutationOracle

    before = np.array([[1, 2]], dtype=np.int64)
    after = np.array([[1, 2]], dtype=np.float32)
    r = InputMutationOracle().check(before, after)
    assert r.passed is False
    assert r.details["dtype_changed"] is True
    assert "buffer_identity_changed" in r.details


def test_backend_rejects_invalid_regression_output_contract():
    from shap_review.backends.adapter import MatrixBackendAdapter
    from shap_review.fuzzing.backend_matrix import BackendSpec

    class M:
        pass

    a = MatrixBackendAdapter(BackendSpec("x", "x", (), True, "1"), M())
    ok, reason = a.supports_case(
        classification=False, model_output="log_loss", interaction=False
    )
    assert ok is False and "regression" in reason


def test_backend_failure_stage_is_preserved():
    from shap_review.backends.adapter import MatrixBackendAdapter
    from shap_review.fuzzing.backend_matrix import BackendSpec

    class M:
        pass

    a = MatrixBackendAdapter(BackendSpec("x", "x", (), True, "1"), M())
    a.prepare = lambda **kw: {}
    a.fit = lambda ctx: (_ for _ in ()).throw(RuntimeError("fit broke"))
    r = a.execute_case()
    assert (
        r["stage"] == "fit"
        and r["execution_reason"] == "BACKEND_ERROR"
        and r["status"] == "BACKEND_ERROR"
    )


def test_sanitizer_finding_is_not_memory_safety_proof(tmp_path):
    from shap_review.reproduction.sanitizer import run_sanitized

    s = tmp_path / "bad.py"
    s.write_text('print("AddressSanitizer: heap-use-after-free")')
    r = run_sanitized(s, "asan")
    assert r["finding"] is True
    assert r["memory_safety_confirmation"] is False


def test_oracle_status_does_not_claim_certification():
    import numpy as np

    from shap_review.fuzzing.backend_matrix import _semantic_additivity

    _, _, oracle = _semantic_additivity(np.ones((1, 2)), np.array(0.0), np.array([2.0]))
    assert oracle["status"] != "CERTIFIED"
