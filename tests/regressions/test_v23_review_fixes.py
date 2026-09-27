import numpy as np


def _contract(**kwargs):
    from shap_review.contracts.shap_contract import SHAPContract

    return SHAPContract(explainer="TreeExplainer", model_family="tree", **kwargs)


def test_policy_required_output_space_cannot_disappear_when_contract_disables_it():
    from shap_review.contracts.oracles import OraclePolicy, SHAPSemanticOracle

    c = _contract(additivity_required=True, output_space_required=False)
    r = SHAPSemanticOracle().evaluate(
        contract=c,
        values=np.ones((2, 2)),
        base_values=np.zeros(2),
        model_output=None,
        policy=OraclePolicy(
            require_shape=False, require_additivity=False, require_output_space=True
        ),
    )
    assert r["status"] == "INCONCLUSIVE"
    assert "OutputSpaceOracle" in r["required_checks"]


def test_required_additivity_without_base_values_is_inconclusive():
    from shap_review.contracts.oracles import OraclePolicy, SHAPSemanticOracle

    c = _contract(additivity_required=False)
    r = SHAPSemanticOracle().evaluate(
        contract=c,
        values=np.ones((2, 2)),
        base_values=None,
        model_output=np.zeros(2),
        policy=OraclePolicy(
            require_shape=False, require_additivity=True, require_output_space=False
        ),
    )
    assert r["status"] == "INCONCLUSIVE"
    add = next(x for x in r["results"] if x["name"] == "AdditivityOracle")
    assert add["passed"] is None and add["applicable"] is False


def test_api_era_does_not_classify_arbitrary_nested_calls_as_shap():
    from shap_review.semantic.api_era import scan_api_era

    findings = scan_api_era("foo = make_factory(); y = foo(X)\n")
    assert not any(f["api"] == "Explainer.__call__" for f in findings)


def test_api_era_detects_direct_shap_treeexplainer_callable():
    from shap_review.semantic.api_era import scan_api_era

    findings = scan_api_era("import shap\ny = shap.TreeExplainer(model)(X)\n")
    assert any(f["api"] == "Explainer.__call__" for f in findings)


def test_subprocess_environment_fingerprint_is_from_child_process(tmp_path):
    from shap_review.differential.runner import run_json_script

    script = tmp_path / "emit.py"
    script.write_text("import json; print(json.dumps({'value': 1}))")
    result = run_json_script(script, capture_environment=True)
    assert result["ok"] is True
    assert result["subprocess_environment"]["executable"]
    assert result["subprocess_environment"]["python"]
