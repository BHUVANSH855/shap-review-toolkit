from pathlib import Path

import numpy as np


def contract(**kwargs):
    from shap_review.contracts.shap_contract import SHAPContract

    return SHAPContract(explainer="TreeExplainer", model_family="tree", **kwargs)


def test_oracle_result_invariants_and_requirement_source():
    from shap_review.contracts.oracles import OraclePolicy, SHAPSemanticOracle

    c = contract(values_shape=(2, 2), output_space_required=False)
    r = SHAPSemanticOracle().evaluate(
        contract=c,
        values=np.ones((2, 2)),
        base_values=np.zeros(2),
        model_output=None,
        policy=OraclePolicy(
            require_shape=True, require_additivity=False, require_output_space=False
        ),
    )
    x = next(x for x in r["results"] if x["name"] == "ShapeOracle")
    assert x["applicable"] is True and x["passed"] is True
    assert x["requirement_source"] in {"contract", "policy", "policy+contract"}


def test_missing_registered_oracle_is_inconclusive():
    from shap_review.contracts.oracles import OraclePolicy, SHAPSemanticOracle

    c = contract(additivity_required=False)
    r = SHAPSemanticOracle(registry={}).evaluate(
        contract=c,
        values=np.ones((2, 2)),
        base_values=np.zeros(2),
        model_output=np.zeros(2),
        policy=OraclePolicy(
            require_shape=False, require_additivity=True, require_output_space=False
        ),
    )
    assert r["status"] == "INCONCLUSIVE"
    assert r["results"][0]["reason"] == "oracle not registered"


def test_differential_agreement_is_not_correctness():
    import tempfile

    from shap_review.differential import differential_scripts

    tmp_dir = Path(tempfile.mkdtemp(prefix="shap-v24-"))
    a = tmp_dir / "shap-v24-a.py"
    b = tmp_dir / "shap-v24-b.py"
    a.write_text('import json; print(json.dumps({"values":[1.0]}))')
    b.write_text('import json; print(json.dumps({"values":[1.0]}))')
    r = differential_scripts(a, b)
    assert r["status"] == "MATCH" and r["reference_correctness"] == "UNKNOWN"


def test_api_alias_reassignment_invalidates():
    from shap_review.semantic.api_era import scan_api_era

    src = "from shap import TreeExplainer\nTreeExplainer = unrelated_factory\ne = TreeExplainer(model)\ne(X)\n"
    assert not any(x["api"] == "Explainer.__call__" for x in scan_api_era(src))


def test_api_scope_isolated():
    from shap_review.semantic.api_era import scan_api_era

    src = "import shap\ndef foo():\n e=shap.TreeExplainer(model)\n return e\ndef bar():\n e=other_factory()\n e(X)\n"
    calls = [x for x in scan_api_era(src) if x["api"] == "Explainer.__call__"]
    assert len(calls) == 0 or all(x["scope"] == "foo" for x in calls)


def test_conditional_api_has_medium_confidence():
    from shap_review.semantic.api_era import scan_api_era

    src = "import shap\nif flag:\n e=shap.TreeExplainer(model)\nelse:\n e=other_factory()\ne(X)\n"
    fs = scan_api_era(src)
    assert any(
        x["api"] == "Explainer.__call__" and x["confidence"] == "medium" for x in fs
    )


def test_target_process_environment_capture(tmp_path):
    from shap_review.differential.runner import run_json_script

    s = tmp_path / "env.py"
    s.write_text(
        'import os,json; os.environ["CUDA_VISIBLE_DEVICES"]="target-mutated"; print(json.dumps({"value":1}))'
    )
    r = run_json_script(s, capture_environment=True)
    assert r["subprocess_environment"]["cuda_visible_devices"] == "target-mutated"


def test_backend_matrix_routes_through_adapter():
    from shap_review.backends.adapter import MatrixBackendAdapter

    assert hasattr(MatrixBackendAdapter, "execute_case")
