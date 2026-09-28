from shap_review.invariants.evaluator import numeric_additivity, shape_equal


def test_additivity_passes():
    r = numeric_additivity([[1.0, 2.0], [3.0, 4.0]], [0.0, 1.0], [3.0, 8.0])
    assert r.passed


def test_additivity_fails():
    r = numeric_additivity([[1.0, 2.0]], [0.0], [10.0])
    assert not r.passed


def test_shape():
    assert shape_equal((2, 3), (2, 3)).passed
    assert not shape_equal((2, 3), (3, 2)).passed


def test_v28_shared_semantic_alignment_rejects_feature_broadcast():
    import numpy as np

    from shap_review.contracts.tensor import SHAPAxisSpec, semantic_align

    try:
        semantic_align(
            np.ones((2, 1)), np.ones((2, 3)), axes=SHAPAxisSpec(), role="values"
        )
    except ValueError as exc:
        assert "feature/interaction axis" in str(exc)
    else:
        raise AssertionError("feature-axis broadcasting must be rejected")


def test_tensor_rejects_semantically_invalid_baseline_rank():
    import numpy as np

    from shap_review.contracts.tensor import SHAPSemanticTensor

    tensor = SHAPSemanticTensor.from_values(
        np.ones((2, 3)),
        base_values=np.ones((2, 3, 1)),
    )

    try:
        tensor.reconstruction()
    except ValueError:
        return

    raise AssertionError("invalid baseline rank was accepted")


def test_evidence_same_transformation_is_correlated():
    from shap_review.evidence.graph import EvidenceGraph, EvidenceNode

    graph = EvidenceGraph()

    graph.add(
        EvidenceNode(
            "a",
            "dynamic",
            "x",
            "p",
            "a",
            True,
            transformation="norm",
        )
    )
    graph.add(
        EvidenceNode(
            "b",
            "dynamic",
            "x",
            "p",
            "b",
            True,
            transformation="norm",
        )
    )

    assert not graph.is_independent("a", "b")
    assert graph.independence_reason("a", "b") == "shared-transformation"


def test_semantic_align_rejects_feature_axis_broadcast():
    import numpy as np

    from shap_review.contracts.tensor import SHAPAxisSpec, semantic_align

    try:
        semantic_align(
            np.ones((2, 1, 3)),
            np.ones((2, 4, 3)),
            axes=SHAPAxisSpec(output_axis=2),
        )
    except ValueError:
        return

    raise AssertionError("feature-axis broadcasting was accepted")


def test_input_mutation_reports_structural_dimensions():
    import numpy as np

    from shap_review.contracts.oracles import InputMutationOracle

    before = np.array([[1, 2]], dtype=np.int64)
    after = np.array([[1, 2]], dtype=np.float32)

    result = InputMutationOracle().check(before, after)

    assert result.passed is False
    assert result.details["dtype_changed"] is True
    assert "buffer_identity_changed" in result.details


def test_oracle_status_does_not_claim_certification():
    import numpy as np

    from shap_review.fuzzing.backend_matrix import _semantic_additivity

    _, _, oracle = _semantic_additivity(
        np.ones((1, 2)),
        np.array(0.0),
        np.array([2.0]),
    )

    assert oracle["status"] != "CERTIFIED"