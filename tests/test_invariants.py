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
