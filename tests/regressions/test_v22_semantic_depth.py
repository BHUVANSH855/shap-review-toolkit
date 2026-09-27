import numpy as np


def test_multiclass_additivity_preserves_output_axis():
    from shap_review.contracts.tensor import SHAPAxisSpec, SHAPSemanticTensor

    values = np.ones((2, 3, 2), dtype=float)
    base = np.zeros((2, 2), dtype=float)
    tensor = SHAPSemanticTensor.from_values(
        values, base_values=base, axis_spec=SHAPAxisSpec(output_axis=2)
    )
    assert tensor.reduce_contributions().shape == (2, 2)


def test_interaction_reduction_preserves_output_axis():
    from shap_review.contracts.tensor import SHAPAxisSpec, SHAPSemanticTensor

    values = np.ones((2, 3, 3, 2), dtype=float)
    base = np.zeros((2, 2), dtype=float)
    tensor = SHAPSemanticTensor.from_values(
        values,
        base_values=base,
        interaction=True,
        axis_spec=SHAPAxisSpec(output_axis=3, interaction_feature_axes=(1, 2)),
    )
    assert tensor.reduce_contributions().shape == (2, 2)


def test_legacy_list_outputs_are_canonicalized():
    from shap_review.contracts.tensor import SHAPSemanticTensor

    values = [np.ones((2, 3)), np.full((2, 3), 2.0)]
    tensor = SHAPSemanticTensor.from_values(values, base_values=np.zeros((2, 2)))
    assert tensor.values.shape == (2, 3, 2)


def test_expected_value_semantics_reject_wrong_per_sample_baseline():
    from shap_review.contracts.oracles import ExpectedValueOracle

    r = ExpectedValueOracle().check(
        base_values=np.array([[1.0, 2.0], [1.0, 3.0]]),
        expected_value=np.array([1.0, 2.0]),
        semantics="per-sample",
    )
    assert r.passed is False


def test_required_unavailable_oracle_is_inconclusive():
    from shap_review.contracts.oracles import OraclePolicy, SHAPSemanticOracle
    from shap_review.contracts.shap_contract import SHAPContract

    c = SHAPContract(
        explainer="TreeExplainer",
        model_family="tree",
        additivity_required=True,
        output_space_required=True,
    )
    r = SHAPSemanticOracle().evaluate(
        contract=c,
        values=np.ones((2, 3)),
        base_values=None,
        model_output=None,
        policy=OraclePolicy(
            require_shape=False, require_additivity=True, require_output_space=True
        ),
    )
    assert r["status"] == "INCONCLUSIVE"


def test_output_space_intentional_wrong_target_fails():
    from shap_review.contracts.oracles import OutputSpaceOracle
    from shap_review.contracts.shap_contract import SHAPContract

    c = SHAPContract(
        explainer="TreeExplainer",
        model_family="tree",
        model_output="raw",
        tolerance=1e-8,
    )
    r = OutputSpaceOracle().check(
        contract=c,
        shap_values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        base_values=np.zeros(2),
        model_output=np.array([99.0, 99.0]),
    )
    assert r.passed is False


def test_api_era_detects_alias_treeexplainer_call():
    from shap_review.semantic.api_era import scan_api_era

    findings = scan_api_era(
        "from shap import TreeExplainer\ne=TreeExplainer(model)\ne(X)\n"
    )
    assert any(f["api"] == "Explainer.__call__" for f in findings)
