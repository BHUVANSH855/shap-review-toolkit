import pytest

pytest.importorskip("shap")
import numpy as np


def test_semantic_additivity_failed_reconstruction_is_not_certified():
    from shap_review.fuzzing.backend_matrix import _semantic_additivity

    values = np.array([[1.0, 2.0], [1.0, 2.0]])
    base = np.array([0.0, 0.0])
    target = np.array([99.0, 99.0])
    passed, _, oracle = _semantic_additivity(values, base, target, backend="test")
    assert passed is False
    assert oracle["status"] == "SEMANTIC_MISMATCH"
    assert oracle["passed"] is False


def test_protocol_event_canonical_source_is_nearest_shap_frame():
    import random

    from shap_review.fuzzing.generators.protocol import ProtocolObject

    obj = ProtocolObject(random.Random(0), "shape")
    obj.begin_shap_call("call-1", "TreeExplainer.shap_values", "wrapper-source")
    obj._runtime_provenance = lambda: {
        "stack": ["shap.foo"],
        "shap_frames": ["shap.foo"],
        "shap_frame_observed": True,
        "nearest_shap_frame": "shap.foo",
    }
    obj._touch("shape")
    ev = obj._protocol_event_records[-1]
    assert ev["canonical_source"] == "shap.foo"
    assert ev["shap_call_id"] == "call-1"
    assert ev["callback_event_id"] == ev["event_id"]


def test_backend_summary_exposes_not_executed():
    from shap_review.fuzzing.backend_matrix import execute_installed_backend_matrix

    r = execute_installed_backend_matrix()
    assert "not_executed" in r["summary"]
    assert "execution_reasons" in r["summary"]


def test_catboost_reproducer_schema():
    from shap_review.regressions.suite import catboost_candidate_reproducer

    r = catboost_candidate_reproducer()
    assert r["id"] == "SHAP-CATBOOST-INTERVENTIONAL-RECON"
    assert "shap_version" in r and "catboost_version" in r
    assert r["confirmation_status"] in {"candidate_only", "confirmed", "unknown"}


def test_catboost_cross_version_requires_two_explicit_interpreters():
    from shap_review.regressions.suite import cross_version_catboost_validation

    r = cross_version_catboost_validation()
    assert r["status"] == "NOT_RUN"
    assert r["confirmation_status"] == "candidate_only"
