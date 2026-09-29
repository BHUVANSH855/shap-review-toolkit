from shap_review.invariants.evaluator import numeric_additivity, shape_equal


def test_additivity_passes():
    result = numeric_additivity(
        [[1.0, 2.0], [3.0, 4.0]],
        [0.0, 1.0],
        [3.0, 8.0],
    )
    assert result.passed


def test_additivity_fails():
    result = numeric_additivity(
        [[1.0, 2.0]],
        [0.0],
        [10.0],
    )
    assert not result.passed


def test_shape():
    assert shape_equal((2, 3), (2, 3)).passed
    assert not shape_equal((2, 3), (3, 2)).passed


def test_semantic_alignment_rejects_feature_broadcast():
    import numpy as np

    from shap_review.contracts.tensor import SHAPAxisSpec, semantic_align

    try:
        semantic_align(
            np.ones((2, 1)),
            np.ones((2, 3)),
            axes=SHAPAxisSpec(),
            role="values",
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


def test_public_package_version_matches_development_metadata():
    import shap_review
    from shap_review.version import VERSION

    assert shap_review.__version__ == VERSION


def test_canonical_capability_contract():
    from shap_review.version import CAPABILITIES, SCHEMA_VERSION, VERSION

    assert VERSION
    assert SCHEMA_VERSION
    assert len(CAPABILITIES) == 24
    assert "evidence" in CAPABILITIES
    assert list(CAPABILITIES) == list(dict.fromkeys(CAPABILITIES))


def test_shap_contract_validation():
    from shap_review.contracts import (
        SHAPContract,
        compare_contracts,
        validate_contract,
    )

    first = SHAPContract(
        "TreeExplainer",
        "tree",
        values_shape=(2, 3),
    )
    second = SHAPContract(
        "TreeExplainer",
        "tree",
        values_shape=(2, 3),
    )

    assert compare_contracts(first, second)["equal"]

    result = validate_contract(
        first,
        [[1, 2, 3], [4, 5, 6]],
    )

    assert result["values_shape_match"]

def test_evidence_graph_rejects_shared_ancestry_as_independent():
    from shap_review.evidence.chain import build_chain

    chain = build_chain(
        entries=[
            {
                "kind": "historical",
                "source": "issue",
                "claim": "known",
                "passed": True,
                "evidence_id": "root",
            },
            {
                "kind": "static",
                "source": "rule-a",
                "claim": "hit",
                "passed": True,
                "evidence_id": "a",
                "derived_from": ["root"],
            },
            {
                "kind": "dynamic",
                "source": "rule-b",
                "claim": "hit",
                "passed": True,
                "evidence_id": "b",
                "derived_from": ["root"],
            },
        ]
    )

    assert not chain.graph.is_independent("a", "b")


def test_semantic_contract_executes_additivity_oracle():
    import numpy as np

    from shap_review.contracts import SHAPContract, validate_contract

    contract = SHAPContract(
        "TreeExplainer",
        "tree",
        values_shape=(2, 3),
        base_values_shape=(2,),
        target_shape=(2,),
        tolerance=1e-6,
    )

    result = validate_contract(
        contract,
        np.ones((2, 3)),
        np.zeros(2),
        np.full(2, 3.0),
    )

    assert result["valid"]
    assert result["semantic_oracle"]["passed"]

def test_axis_roles_reject_conflicting_assignments():
    import pytest

    from shap_review.contracts.tensor import SHAPAxisSpec

    for spec in [
        SHAPAxisSpec(sample_axis=0, feature_axis=0),
        SHAPAxisSpec(sample_axis=0, feature_axis=1, output_axis=0),
        SHAPAxisSpec(sample_axis=0, feature_axis=1, interaction_feature_axes=(0, 2)),
        SHAPAxisSpec(sample_axis=0, feature_axis=1, output_axis=2, interaction_feature_axes=(1, 2)),
    ]:
        with pytest.raises(ValueError):
            spec.normalize(4)


def test_semantic_tensor_reconstructs_multiclass_outputs():
    import numpy as np

    from shap_review.contracts.tensor import SHAPSemanticTensor

    values = np.ones((2, 3, 2))
    base = np.zeros((2, 2))
    tensor = SHAPSemanticTensor.from_values(values, base_values=base)
    assert tensor.reconstruction().shape == (2, 2)
    assert np.allclose(tensor.reconstruction(), 3)


def test_semantic_tensor_reduces_interaction_feature_axes():
    import numpy as np

    from shap_review.contracts.tensor import SHAPSemanticTensor

    tensor = SHAPSemanticTensor.from_values(
        np.ones((2, 3, 3, 2)),
        base_values=np.zeros((2, 2)),
        interaction=True,
    )
    assert tensor.axis_spec.interaction_feature_axes == (1, 2)
    assert tensor.reduce_contributions().shape == (2, 2)


def test_interaction_axis_validation_rejects_unequal_feature_pair_sizes():
    import numpy as np

    from shap_review.contracts.oracles import InteractionOracle

    values = np.zeros((2, 3, 4, 5))
    result = InteractionOracle().check(values=values, interaction_values=values)
    assert result.passed is False
    assert "different sizes" in result.reason


def test_contract_wildcard_dimension_matches_any_size():
    import numpy as np

    from shap_review.contracts.shap_contract import SHAPContract, validate_contract

    contract = SHAPContract(
        explainer="TreeExplainer", model_family="tree", values_shape=(-1, 3),
        additivity_required=False, output_space_required=False,
    )
    result = validate_contract(contract, np.zeros((7, 3)))
    assert result["values_shape_match"]


def test_multiclass_additivity_preserves_output_axis():
    import numpy as np

    from shap_review.contracts import SHAPContract
    from shap_review.contracts.oracles import AdditivityOracle

    contract = SHAPContract("TreeExplainer", "tree", model_output="raw", additivity_required=True, tolerance=1e-8)
    values = np.zeros((2, 3, 2))
    values[:, :, 0] = 1
    values[:, :, 1] = 2
    result = AdditivityOracle().check(
        values=values, base_values=np.zeros(2), model_output=np.array([[3.0, 6.0], [3.0, 6.0]]), contract=contract
    )
    assert result.passed
    assert result.details["contribution_shape"] == [2, 2]


def test_interaction_additivity_reduces_feature_axes():
    import numpy as np

    from shap_review.contracts import SHAPContract
    from shap_review.contracts.oracles import AdditivityOracle

    contract = SHAPContract("TreeExplainer", "tree", interaction=True, tolerance=1e-8)
    result = AdditivityOracle().check(
        values=np.ones((2, 3, 3)), base_values=np.zeros(2),
        model_output=np.full(2, 9.0), contract=contract
    )
    assert result.passed


def test_expected_value_accepts_per_sample_base_values():
    import numpy as np

    from shap_review.contracts.oracles import ExpectedValueOracle

    result = ExpectedValueOracle().check(
        base_values=np.array([1.0, 1.0, 1.0]), expected_value=np.array(1.0)
    )
    assert result.passed


def test_required_semantics_without_available_value_are_inconclusive():
    import numpy as np

    from shap_review.contracts import SHAPContract, validate_contract

    contract = SHAPContract("TreeExplainer", "tree", expected_value_required=True)
    result = validate_contract(
        contract, np.ones((1, 2)), np.zeros(1), np.full(1, 2.0), expected_value=None
    )
    assert result["semantic_oracle"]["status"] == "INCONCLUSIVE"
    assert not result["valid"]


def test_wrong_output_reconstruction_is_not_certified():
    import numpy as np

    from shap_review.contracts import SHAPContract
    from shap_review.contracts.oracles import OutputSpaceOracle

    contract = SHAPContract("TreeExplainer", "tree", model_output="raw", tolerance=1e-8)
    result = OutputSpaceOracle().check(
        contract=contract, shap_values=np.array([[1.0, 1.0]]),
        base_values=np.array([0.0]), model_output=np.array([9.0])
    )
    assert result.passed is False


def test_failed_additivity_reconstruction_is_not_certified():
    import numpy as np

    from shap_review.fuzzing.backend_matrix import _semantic_additivity

    passed, _, oracle = _semantic_additivity(
        np.array([[1.0, 2.0], [1.0, 2.0]]), np.array([0.0, 0.0]),
        np.array([99.0, 99.0]), backend="test"
    )
    assert passed is False
    assert oracle["status"] == "SEMANTIC_MISMATCH"
    assert oracle["passed"] is False


def test_policy_required_output_space_remains_required():
    import numpy as np

    from shap_review.contracts.oracles import OraclePolicy, SHAPSemanticOracle
    from shap_review.contracts.shap_contract import SHAPContract

    contract = SHAPContract("TreeExplainer", "tree", additivity_required=True, output_space_required=False)
    result = SHAPSemanticOracle().evaluate(
        contract=contract, values=np.ones((2, 2)), base_values=np.zeros(2), model_output=None,
        policy=OraclePolicy(require_shape=False, require_additivity=False, require_output_space=True),
    )
    assert result["status"] == "INCONCLUSIVE"
    assert "OutputSpaceOracle" in result["required_checks"]


def test_required_additivity_without_base_values_is_inconclusive():
    import numpy as np

    from shap_review.contracts.oracles import OraclePolicy, SHAPSemanticOracle
    from shap_review.contracts.shap_contract import SHAPContract

    contract = SHAPContract("TreeExplainer", "tree", additivity_required=False)
    result = SHAPSemanticOracle().evaluate(
        contract=contract, values=np.ones((2, 2)), base_values=None, model_output=np.zeros(2),
        policy=OraclePolicy(require_shape=False, require_additivity=True, require_output_space=False),
    )
    assert result["status"] == "INCONCLUSIVE"
    additivity = next(x for x in result["results"] if x["name"] == "AdditivityOracle")
    assert additivity["passed"] is None and additivity["applicable"] is False


def test_expected_value_rejects_wrong_per_sample_baseline():
    import numpy as np

    from shap_review.contracts.oracles import ExpectedValueOracle

    result = ExpectedValueOracle().check(
        base_values=np.array([[1.0, 2.0], [1.0, 3.0]]),
        expected_value=np.array([1.0, 2.0]), semantics="per-sample",
    )
    assert result.passed is False


def test_required_oracles_without_runtime_inputs_are_inconclusive():
    import numpy as np

    from shap_review.contracts.oracles import OraclePolicy, SHAPSemanticOracle
    from shap_review.contracts.shap_contract import SHAPContract

    contract = SHAPContract("TreeExplainer", "tree", additivity_required=True, output_space_required=True)
    result = SHAPSemanticOracle().evaluate(
        contract=contract, values=np.ones((2, 3)), base_values=None, model_output=None,
        policy=OraclePolicy(require_shape=False, require_additivity=True, require_output_space=True),
    )
    assert result["status"] == "INCONCLUSIVE"
































def test_legacy_list_outputs_are_canonicalized():
    import numpy as np

    from shap_review.contracts.tensor import SHAPSemanticTensor

    values = [np.ones((2, 3)), np.full((2, 3), 2.0)]
    tensor = SHAPSemanticTensor.from_values(values, base_values=np.zeros((2, 2)))
    assert tensor.values.shape == (2, 3, 2)


def test_interaction_reduction_preserves_output_axis():
    import numpy as np

    from shap_review.contracts.tensor import SHAPAxisSpec, SHAPSemanticTensor

    values = np.ones((2, 3, 3, 2))
    base = np.zeros((2, 2))
    tensor = SHAPSemanticTensor.from_values(
        values, base_values=base, interaction=True,
        axis_spec=SHAPAxisSpec(output_axis=3, interaction_feature_axes=(1, 2)),
    )
    assert tensor.reduce_contributions().shape == (2, 2)
