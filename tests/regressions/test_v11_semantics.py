import numpy as np

from shap_review.contracts import SHAPContract, validate_contract
from shap_review.contracts.oracles import (
    AdditivityOracle,
    ExpectedValueOracle,
    OutputSpaceOracle,
)
from shap_review.evidence.chain import build_chain


def test_additivity_multiclass_preserves_output_axis():
    c = SHAPContract(
        "TreeExplainer",
        "tree",
        model_output="raw",
        additivity_required=True,
        tolerance=1e-8,
    )
    vals = np.zeros((2, 3, 2))
    vals[:, :, 0] = 1
    vals[:, :, 1] = 2
    target = np.array([[3.0, 6.0], [3.0, 6.0]])
    r = AdditivityOracle().check(
        values=vals, base_values=np.zeros(2), model_output=target, contract=c
    )
    assert r.passed
    assert r.details["contribution_shape"] == [2, 2]


def test_additivity_interaction_reduces_both_feature_axes():
    c = SHAPContract("TreeExplainer", "tree", interaction=True, tolerance=1e-8)
    vals = np.ones((2, 3, 3))
    target = np.full(2, 9.0)
    r = AdditivityOracle().check(
        values=vals, base_values=np.zeros(2), model_output=target, contract=c
    )
    assert r.passed


def test_expected_value_supports_per_sample_base_values():
    r = ExpectedValueOracle().check(
        base_values=np.array([1.0, 1.0, 1.0]), expected_value=np.array(1.0)
    )
    assert r.passed


def test_output_space_detects_wrong_reconstruction():
    c = SHAPContract("TreeExplainer", "tree", model_output="raw", tolerance=1e-8)
    r = OutputSpaceOracle().check(
        contract=c,
        shap_values=np.array([[1.0, 1.0]]),
        base_values=np.array([0.0]),
        model_output=np.array([9.0]),
    )
    assert r.passed is False


def test_required_unavailable_semantics_are_inconclusive():
    c = SHAPContract("TreeExplainer", "tree", expected_value_required=True)
    result = validate_contract(
        c, np.ones((1, 2)), np.zeros(1), np.full(1, 2.0), expected_value=None
    )
    assert result["semantic_oracle"]["status"] == "INCONCLUSIVE"
    assert not result["valid"]


def test_evidence_same_execution_is_not_independent():
    chain = build_chain(
        entries=[
            {
                "kind": "dynamic",
                "source": "a",
                "claim": "a",
                "passed": True,
                "evidence_id": "a",
                "details": {"execution_id": "run1"},
            },
            {
                "kind": "differential",
                "source": "b",
                "claim": "b",
                "passed": True,
                "evidence_id": "b",
                "details": {"execution_id": "run1"},
            },
        ]
    )
    assert not chain.graph.is_independent("a", "b")
    assert chain.graph.independence_reason("a", "b") == "shared-execution-lineage"


def test_protocol_campaign_marks_harness_vs_real_target():
    from shap_review.fuzzing.protocol_campaign import ProtocolCampaign

    r = ProtocolCampaign(4).run(iterations=10)
    assert r["mode"] == "harness-self-test"
    assert "seed" in r


def test_interaction_multiclass_uses_feature_pair_axes_not_output_axis():
    from shap_review.contracts.oracles import InteractionOracle

    c = SHAPContract("TreeExplainer", "tree", interaction=True)
    vals = np.zeros((1, 2, 2, 3))
    vals[:, 0, 1, :] = 1
    vals[:, 1, 0, :] = 1
    r = InteractionOracle().check(
        values=vals, interaction_values=vals, axes=c.axis_spec
    )
    assert r.passed is True
    assert r.details["interaction_feature_axes"] == [1, 2]


def test_canonical_additivity_does_not_guess_equal_output_dimension():
    from shap_review.differential.semantic import evaluate_additivity

    vals = np.ones((1, 2, 2))
    r = evaluate_additivity(vals, np.zeros(2), np.array([2.0, 2.0]), interaction=False)
    assert r["applicable"] and r["passed"]


def test_protocol_campaign_reports_requested_vs_observed_adversarial_modes():
    from shap_review.fuzzing.protocol_campaign import ProtocolCampaign

    r = ProtocolCampaign(1).run(iterations=15)
    assert r["mutation_requested"] > 0
    assert r["reentry_requested"] > 0
    assert r["mutation_observed"] > 0
    assert r["reentry_observed"] > 0
