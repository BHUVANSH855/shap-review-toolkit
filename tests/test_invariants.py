import pytest

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
        SHAPAxisSpec(
            sample_axis=0,
            feature_axis=1,
            output_axis=2,
            interaction_feature_axes=(1, 2),
        ),
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
        explainer="TreeExplainer",
        model_family="tree",
        values_shape=(-1, 3),
        additivity_required=False,
        output_space_required=False,
    )
    result = validate_contract(contract, np.zeros((7, 3)))
    assert result["values_shape_match"]


def test_multiclass_additivity_preserves_output_axis():
    import numpy as np

    from shap_review.contracts import SHAPContract
    from shap_review.contracts.oracles import AdditivityOracle

    contract = SHAPContract(
        "TreeExplainer",
        "tree",
        model_output="raw",
        additivity_required=True,
        tolerance=1e-8,
    )
    values = np.zeros((2, 3, 2))
    values[:, :, 0] = 1
    values[:, :, 1] = 2
    result = AdditivityOracle().check(
        values=values,
        base_values=np.zeros(2),
        model_output=np.array([[3.0, 6.0], [3.0, 6.0]]),
        contract=contract,
    )
    assert result.passed
    assert result.details["contribution_shape"] == [2, 2]


def test_interaction_additivity_reduces_feature_axes():
    import numpy as np

    from shap_review.contracts import SHAPContract
    from shap_review.contracts.oracles import AdditivityOracle

    contract = SHAPContract("TreeExplainer", "tree", interaction=True, tolerance=1e-8)
    result = AdditivityOracle().check(
        values=np.ones((2, 3, 3)),
        base_values=np.zeros(2),
        model_output=np.full(2, 9.0),
        contract=contract,
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
        contract=contract,
        shap_values=np.array([[1.0, 1.0]]),
        base_values=np.array([0.0]),
        model_output=np.array([9.0]),
    )
    assert result.passed is False


def test_failed_additivity_reconstruction_is_not_certified():
    import numpy as np

    from shap_review.fuzzing.backend_matrix import _semantic_additivity

    passed, _, oracle = _semantic_additivity(
        np.array([[1.0, 2.0], [1.0, 2.0]]),
        np.array([0.0, 0.0]),
        np.array([99.0, 99.0]),
        backend="test",
    )
    assert passed is False
    assert oracle["status"] == "SEMANTIC_MISMATCH"
    assert oracle["passed"] is False


def test_policy_required_output_space_remains_required():
    import numpy as np

    from shap_review.contracts.oracles import OraclePolicy, SHAPSemanticOracle
    from shap_review.contracts.shap_contract import SHAPContract

    contract = SHAPContract(
        "TreeExplainer", "tree", additivity_required=True, output_space_required=False
    )
    result = SHAPSemanticOracle().evaluate(
        contract=contract,
        values=np.ones((2, 2)),
        base_values=np.zeros(2),
        model_output=None,
        policy=OraclePolicy(
            require_shape=False, require_additivity=False, require_output_space=True
        ),
    )
    assert result["status"] == "INCONCLUSIVE"
    assert "OutputSpaceOracle" in result["required_checks"]


def test_required_additivity_without_base_values_is_inconclusive():
    import numpy as np

    from shap_review.contracts.oracles import OraclePolicy, SHAPSemanticOracle
    from shap_review.contracts.shap_contract import SHAPContract

    contract = SHAPContract("TreeExplainer", "tree", additivity_required=False)
    result = SHAPSemanticOracle().evaluate(
        contract=contract,
        values=np.ones((2, 2)),
        base_values=None,
        model_output=np.zeros(2),
        policy=OraclePolicy(
            require_shape=False, require_additivity=True, require_output_space=False
        ),
    )
    assert result["status"] == "INCONCLUSIVE"
    additivity = next(x for x in result["results"] if x["name"] == "AdditivityOracle")
    assert additivity["passed"] is None and additivity["applicable"] is False


def test_expected_value_rejects_wrong_per_sample_baseline():
    import numpy as np

    from shap_review.contracts.oracles import ExpectedValueOracle

    result = ExpectedValueOracle().check(
        base_values=np.array([[1.0, 2.0], [1.0, 3.0]]),
        expected_value=np.array([1.0, 2.0]),
        semantics="per-sample",
    )
    assert result.passed is False


def test_required_oracles_without_runtime_inputs_are_inconclusive():
    import numpy as np

    from shap_review.contracts.oracles import OraclePolicy, SHAPSemanticOracle
    from shap_review.contracts.shap_contract import SHAPContract

    contract = SHAPContract(
        "TreeExplainer", "tree", additivity_required=True, output_space_required=True
    )
    result = SHAPSemanticOracle().evaluate(
        contract=contract,
        values=np.ones((2, 3)),
        base_values=None,
        model_output=None,
        policy=OraclePolicy(
            require_shape=False, require_additivity=True, require_output_space=True
        ),
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
        values,
        base_values=base,
        interaction=True,
        axis_spec=SHAPAxisSpec(output_axis=3, interaction_feature_axes=(1, 2)),
    )
    assert tensor.reduce_contributions().shape == (2, 2)


# ============================================================================
# New tests — gaps identified in review
# ============================================================================

# ---------------------------------------------------------------------------
# C-1: dtype-adaptive tolerance in validate_contract / SHAPContract
# ---------------------------------------------------------------------------


def test_validate_contract_float32_uses_relaxed_tolerance():
    """float32 SHAP values must NOT fail validate_contract with default tolerance.

    float32 arithmetic can produce additivity errors up to ~5e-4.  Using the
    old hard-coded 1e-6 for both rtol and atol caused correct float32
    computations to fail as FAIL (false positive).  validate_contract() must
    now infer dtype and apply 5e-4 for float32.
    """
    import numpy as np

    from shap_review.contracts.shap_contract import SHAPContract, validate_contract

    np.random.seed(42)
    n, f = 4, 3
    values = np.random.rand(n, f).astype(np.float32)
    base = np.zeros(n, dtype=np.float32)
    target = values.sum(axis=1).astype(np.float32) + base

    # Inject a small float32-level error that is correct within 5e-4.
    target += np.float32(1e-4)

    contract = SHAPContract(
        "TreeExplainer",
        "tree",
        additivity_required=True,
        values_shape=(n, f),
    )
    result = validate_contract(contract, values, base, target)

    assert result["dtype"] == "float32"
    assert result["tolerance_source"] == "dtype_adaptive"
    assert result["effective_tolerance"] == 5e-4, (
        f"Expected 5e-4 for float32, got {result['effective_tolerance']}"
    )
    # Must NOT produce a false FAIL for a float32-correct computation.
    oracle_status = result["semantic_oracle"]["status"]
    assert oracle_status in {"PASS", "INCONCLUSIVE"}, (
        f"float32 correct computation must not FAIL, got {oracle_status}"
    )


def test_validate_contract_float64_uses_strict_tolerance():
    """float64 SHAP values use strict 1e-6 tolerance by default."""
    import numpy as np

    from shap_review.contracts.shap_contract import SHAPContract, validate_contract

    n, f = 4, 3
    values = np.ones((n, f), dtype=np.float64)
    base = np.zeros(n, dtype=np.float64)
    target = values.sum(axis=1) + base

    contract = SHAPContract("TreeExplainer", "tree", additivity_required=True)
    result = validate_contract(contract, values, base, target)

    assert result["dtype"] == "float64"
    assert result["effective_tolerance"] == 1e-6
    assert result["tolerance_source"] == "dtype_adaptive"


def test_validate_contract_explicit_tolerance_overrides_dtype_adaptive():
    """An explicit contract.tolerance always wins over dtype-adaptive selection."""
    import numpy as np

    from shap_review.contracts.shap_contract import SHAPContract, validate_contract

    values = np.ones((2, 3), dtype=np.float32)
    base = np.zeros(2, dtype=np.float32)
    target = values.sum(axis=1) + base

    # Caller explicitly requests strict tolerance even for float32.
    contract = SHAPContract("TreeExplainer", "tree", tolerance=1e-9)
    result = validate_contract(contract, values, base, target)

    assert result["tolerance_source"] == "caller_supplied"
    assert result["effective_tolerance"] == 1e-9


def test_dtype_tolerance_function_values():
    """dtype_tolerance() must return correct values for each dtype."""
    import numpy as np

    from shap_review.contracts.shap_contract import dtype_tolerance

    assert dtype_tolerance(np.dtype("float16")) == 1e-2
    assert dtype_tolerance(np.dtype("float32")) == 5e-4
    assert dtype_tolerance(np.dtype("float64")) == 1e-6
    assert dtype_tolerance(None) == 1e-6
    assert dtype_tolerance("float32") == 5e-4


def test_float32_additivity_error_within_tolerance_does_not_fail():
    """Additivity error of 3e-4 for float32 must not produce FAIL."""
    import numpy as np

    from shap_review.contracts.shap_contract import SHAPContract, validate_contract

    n, f = 3, 4
    values = np.random.RandomState(0).rand(n, f).astype(np.float32)
    base = np.zeros(n, dtype=np.float32)
    target = values.sum(axis=1).astype(np.float32) + base
    # Inject 3e-4 error — within float32 tolerance (5e-4) but outside old 1e-6.
    target = target + np.float32(3e-4)

    contract = SHAPContract("TreeExplainer", "tree", additivity_required=True)
    result = validate_contract(contract, values, base, target)

    results_by_name = {r["name"]: r for r in result["semantic_oracle"]["results"]}
    additivity = results_by_name.get("AdditivityOracle")
    if additivity and additivity.get("applicable"):
        assert additivity.get("passed") is not False, (
            "3e-4 additivity error for float32 must not FAIL "
            f"(got passed={additivity.get('passed')})"
        )


def test_float32_additivity_error_exceeding_tolerance_fails():
    """Additivity error of 1e-2 for float32 must FAIL even with relaxed tolerance."""
    import numpy as np

    from shap_review.contracts.shap_contract import SHAPContract, validate_contract

    n, f = 3, 4
    values = np.ones((n, f), dtype=np.float32)
    base = np.zeros(n, dtype=np.float32)
    # Large error well beyond 5e-4.
    target = values.sum(axis=1) + base + np.float32(1e-2)

    contract = SHAPContract("TreeExplainer", "tree", additivity_required=True)
    result = validate_contract(contract, values, base, target)

    results_by_name = {r["name"]: r for r in result["semantic_oracle"]["results"]}
    additivity = results_by_name.get("AdditivityOracle")
    if additivity and additivity.get("applicable"):
        assert additivity.get("passed") is False, (
            "1e-2 additivity error must FAIL even with float32 relaxed tolerance"
        )


# ---------------------------------------------------------------------------
# H-4: log-loss OutputSpaceOracle must be NOT_APPLICABLE, not INCONCLUSIVE
# ---------------------------------------------------------------------------


def test_log_loss_oracle_without_true_labels_is_not_applicable():
    """OutputSpaceOracle for log_loss without true_labels must return NOT_APPLICABLE.

    Previously the oracle computed labels then immediately returned INCONCLUSIVE,
    silently discarding the labels.  The corrected behaviour is to return
    NOT_APPLICABLE when true_labels are not supplied — explicitly signalling
    that the check was not attempted, not that it was inconclusive.
    """
    import numpy as np

    from shap_review.contracts.oracles import OutputSpaceOracle
    from shap_review.contracts.shap_contract import SHAPContract

    contract = SHAPContract(
        "TreeExplainer",
        "tree",
        model_output="log_loss",
        output_space_required=True,
    )
    oracle = OutputSpaceOracle()
    result = oracle.check(
        contract=contract,
        shap_values=np.ones((4, 3)),
        base_values=np.zeros(4),
        # No true_labels supplied.
    )
    # applicable=False → NOT_APPLICABLE, not INCONCLUSIVE.
    assert result.applicable is False
    assert result.passed is None
    assert "true_labels" in result.reason.lower() or "NOT_APPLICABLE" in (
        result.details or {}
    ).get("resolution", "")


def test_log_loss_oracle_status_is_not_applicable_not_inconclusive():
    """OracleStatus for log_loss without labels must be NOT_APPLICABLE."""
    import numpy as np

    from shap_review.contracts.oracles import OracleStatus, OutputSpaceOracle
    from shap_review.contracts.shap_contract import SHAPContract

    contract = SHAPContract("TreeExplainer", "tree", model_output="log_loss")
    result = OutputSpaceOracle().check(
        contract=contract,
        shap_values=np.ones((2, 3)),
        base_values=np.zeros(2),
    )
    assert result.status == OracleStatus.NOT_APPLICABLE, (
        f"Expected NOT_APPLICABLE, got {result.status}"
    )


# ---------------------------------------------------------------------------
# OracleStatus enum — must be used in oracle returns
# ---------------------------------------------------------------------------


def test_oracle_status_from_result_pass():
    from shap_review.contracts.oracles import OracleResult, OracleStatus

    r = OracleResult("test", True, True, "ok")
    assert r.status == OracleStatus.PASS
    assert r.to_dict()["status"] == "PASS"


def test_oracle_status_from_result_fail():
    from shap_review.contracts.oracles import OracleResult, OracleStatus

    r = OracleResult("test", True, False, "fail")
    assert r.status == OracleStatus.FAIL
    assert r.to_dict()["status"] == "FAIL"


def test_oracle_status_from_result_inconclusive():
    from shap_review.contracts.oracles import OracleResult, OracleStatus

    r = OracleResult("test", True, None, "unknown")
    assert r.status == OracleStatus.INCONCLUSIVE


def test_oracle_status_from_result_not_applicable():
    from shap_review.contracts.oracles import OracleResult, OracleStatus

    r = OracleResult("test", False, None, "n/a")
    assert r.status == OracleStatus.NOT_APPLICABLE


# ---------------------------------------------------------------------------
# ORACLE_REGISTRY thread safety
# ---------------------------------------------------------------------------


def test_oracle_registry_thread_safe_concurrent_read():
    """Concurrent oracle evaluation must not produce partial-registry results."""
    import threading

    import numpy as np

    from shap_review.contracts.oracles import SHAPSemanticOracle
    from shap_review.contracts.shap_contract import SHAPContract

    errors = []
    contract = SHAPContract("TreeExplainer", "tree")

    def evaluate():
        try:
            oracle = SHAPSemanticOracle()
            oracle.evaluate(
                contract=contract,
                values=np.ones((2, 3)),
                base_values=np.zeros(2),
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=evaluate) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent oracle evaluation raised: {errors}"


def test_oracle_registry_snapshot_isolates_from_later_mutations():
    """SHAPSemanticOracle snapshots registry at construction, not at evaluate() time."""

    from shap_review.contracts.oracles import (
        OracleResult,
        SHAPSemanticOracle,
        register_oracle,
    )

    class StubOracle:
        def check(self, **kwargs):
            return OracleResult("StubOracle", True, True, "stub")

    oracle = SHAPSemanticOracle()

    # Mutate the global registry AFTER construction.
    register_oracle("__test_stub__", StubOracle())

    # The existing oracle instance must not see the late registration.
    assert "__test_stub__" not in oracle.registry, (
        "Registry mutation after construction must not affect existing instances"
    )

    # Clean up.
    from shap_review.contracts.oracles import _REGISTRY_LOCK, ORACLE_REGISTRY

    with _REGISTRY_LOCK:
        ORACLE_REGISTRY.pop("__test_stub__", None)


# ---------------------------------------------------------------------------
# InteractionOracle — same-ndim INCONCLUSIVE (not false-positive PASS)
# ---------------------------------------------------------------------------


def test_interaction_oracle_same_ndim_is_inconclusive_not_pass():
    """Passing the interaction tensor as both values and interaction_values is INCONCLUSIVE.

    A symmetric-but-wrong tensor would pass symmetry, so reconstruction must
    be checked.  Without a separate reference tensor it cannot be, and the
    result must be INCONCLUSIVE (passed=None), not PASS (passed=True).
    """
    import numpy as np

    from shap_review.contracts.oracles import InteractionOracle, OracleStatus

    iv = np.array([[[1.0, 2.0], [2.0, 3.0]]])  # shape (1, 2, 2) — symmetric
    result = InteractionOracle().check(values=iv, interaction_values=iv)

    assert result.applicable is True
    assert result.passed is None, "same-ndim must be INCONCLUSIVE, not PASS"
    assert result.status == OracleStatus.INCONCLUSIVE
    assert result.details["reconstruction_checked"] is False


def test_interaction_oracle_asymmetric_same_ndim_is_also_inconclusive():
    """Asymmetric tensor with same ndim is INCONCLUSIVE (cannot check reconstruction)."""
    import numpy as np

    from shap_review.contracts.oracles import InteractionOracle, OracleStatus

    iv = np.array([[[1.0, 9.0], [2.0, 3.0]]])  # asymmetric
    result = InteractionOracle().check(values=iv, interaction_values=iv)

    # INCONCLUSIVE regardless of symmetry when ndims match.
    assert result.passed is None
    assert result.status == OracleStatus.INCONCLUSIVE


def test_axis_spec_rejects_class_axis_overlapping_interaction_axis():
    from shap_review.contracts.tensor import SHAPAxisSpec

    spec = SHAPAxisSpec(
        output_axis=3,
        class_axis=1,
        interaction_feature_axes=(1, 2),
    )

    with pytest.raises(ValueError, match="class_axis cannot overlap"):
        spec.normalize(4)


def test_axis_spec_rejects_class_axis_overlapping_sample_axis():
    from shap_review.contracts.tensor import SHAPAxisSpec

    spec = SHAPAxisSpec(
        class_axis=0,
    )

    with pytest.raises(
        ValueError, match="class_axis must be distinct from sample_axis"
    ):
        spec.normalize(3)


def test_infer_axis_spec_rejects_rank_two_interaction_tensor():
    import numpy as np

    from shap_review.contracts.tensor import infer_axis_spec

    with pytest.raises(ValueError, match="interaction tensors must have rank 3 or 4"):
        infer_axis_spec(np.zeros((2, 3)), interaction=True)
