from pathlib import Path


def test_semantic_mapper(tmp_path: Path):
    from shap_review.semantic import SHAPSemanticMapper

    source = tmp_path / "_tree.py"
    source.write_text("class TreeExplainer:\n    pass\n")

    graph = SHAPSemanticMapper().map(tmp_path)

    assert any(node.name == "TreeExplainer" for node in graph.nodes)


def test_api_scope_shadowing_and_invalidation():
    from shap_review.semantic.api_era import scan_api_era

    source = """
import shap
e = shap.TreeExplainer(model)
e(X)
def f(shap):
    e = shap.TreeExplainer(model)
    e(X)
def g():
    import shap as s
    e = s.TreeExplainer(model)
    e = other
    e(X)
"""

    findings = scan_api_era(source)
    symbols = [finding["symbol"] for finding in findings]

    assert "e" in symbols
    assert all(
        finding["scope"] != "f" or finding["provenance"] != "proven"
        for finding in findings
        if finding["scope"] == "f"
    )
    assert not any(
        finding["scope"] == "g"
        and finding["symbol"] == "e"
        and finding["message"].startswith("callable")
        for finding in findings
    )


def test_target_provenance_does_not_overclaim_model_api():
    from shap_review.contracts.oracles import classify_target_provenance

    provenance = classify_target_provenance("model.predict", None)

    assert provenance["target_source"] == "model_api"
    assert provenance["oracle_independence"] == "UNKNOWN"

    provenance = classify_target_provenance(
        "independent_probability_fn",
        True,
    )

    assert provenance["oracle_independence"] == "DECLARED"
    assert provenance["independence_basis"] == "caller_declared"


def test_interaction_oracle_requires_symmetry_and_reconstruction():
    import numpy as np

    from shap_review.contracts.oracles import InteractionOracle

    values = np.array([[3.0, 5.0]])
    interaction_values = np.array(
        [[[1.0, 2.0], [2.0, 3.0]]]
    )

    result = InteractionOracle().check(
        values=values,
        interaction_values=interaction_values,
    )

    assert result.passed is True

    invalid_interaction_values = interaction_values.copy()
    invalid_interaction_values[0, 0, 1] = 99.0

    result = InteractionOracle().check(
        values=values,
        interaction_values=invalid_interaction_values,
    )

    assert result.passed is False


def test_oracle_result_invariants_and_requirement_source():
    import numpy as np

    from shap_review.contracts.oracles import OraclePolicy, SHAPSemanticOracle
    from shap_review.contracts.shap_contract import SHAPContract

    contract = SHAPContract(
        explainer="TreeExplainer",
        model_family="tree",
        values_shape=(2, 2),
        output_space_required=False,
    )

    result = SHAPSemanticOracle().evaluate(
        contract=contract,
        values=np.ones((2, 2)),
        base_values=np.zeros(2),
        model_output=None,
        policy=OraclePolicy(
            require_shape=True,
            require_additivity=False,
            require_output_space=False,
        ),
    )

    shape_oracle = next(
        item for item in result["results"] if item["name"] == "ShapeOracle"
    )

    assert shape_oracle["applicable"] is True
    assert shape_oracle["passed"] is True
    assert shape_oracle["requirement_source"] in {
        "contract",
        "policy",
        "policy+contract",
    }


def test_missing_registered_oracle_is_inconclusive():
    import numpy as np

    from shap_review.contracts.oracles import OraclePolicy, SHAPSemanticOracle
    from shap_review.contracts.shap_contract import SHAPContract

    contract = SHAPContract(
        explainer="TreeExplainer",
        model_family="tree",
        additivity_required=False,
    )

    result = SHAPSemanticOracle(registry={}).evaluate(
        contract=contract,
        values=np.ones((2, 2)),
        base_values=np.zeros(2),
        model_output=np.zeros(2),
        policy=OraclePolicy(
            require_shape=False,
            require_additivity=True,
            require_output_space=False,
        ),
    )

    assert result["status"] == "INCONCLUSIVE"
    assert result["results"][0]["reason"] == "oracle not registered"


def test_differential_agreement_is_not_correctness(tmp_path: Path):
    from shap_review.differential import differential_scripts

    reference = tmp_path / "reference.py"
    target = tmp_path / "target.py"

    reference.write_text(
        'import json; print(json.dumps({"values":[1.0]}))'
    )
    target.write_text(
        'import json; print(json.dumps({"values":[1.0]}))'
    )

    result = differential_scripts(reference, target)

    assert result["status"] == "MATCH"
    assert result["reference_correctness"] == "UNKNOWN"


def test_api_alias_reassignment_invalidates():
    from shap_review.semantic.api_era import scan_api_era

    source = (
        "from shap import TreeExplainer\n"
        "TreeExplainer = unrelated_factory\n"
        "e = TreeExplainer(model)\n"
        "e(X)\n"
    )

    assert not any(
        finding["api"] == "Explainer.__call__"
        for finding in scan_api_era(source)
    )


def test_api_scope_isolated():
    from shap_review.semantic.api_era import scan_api_era

    source = (
        "import shap\n"
        "def foo():\n"
        " e=shap.TreeExplainer(model)\n"
        " return e\n"
        "def bar():\n"
        " e=other_factory()\n"
        " e(X)\n"
    )

    calls = [
        finding
        for finding in scan_api_era(source)
        if finding["api"] == "Explainer.__call__"
    ]

    assert len(calls) == 0 or all(
        finding["scope"] == "foo"
        for finding in calls
    )


def test_conditional_api_has_medium_confidence():
    from shap_review.semantic.api_era import scan_api_era

    source = (
        "import shap\n"
        "if flag:\n"
        " e=shap.TreeExplainer(model)\n"
        "else:\n"
        " e=other_factory()\n"
        "e(X)\n"
    )

    findings = scan_api_era(source)

    assert any(
        finding["api"] == "Explainer.__call__"
        and finding["confidence"] == "medium"
        for finding in findings
    )


def test_api_nested_scope_is_preserved():
    from shap_review.semantic.api_era import scan_api_era

    source = """import shap
def foo():
    e=shap.TreeExplainer(model)
    def inner():
        e(X)
    inner()
"""

    findings = scan_api_era(source)

    calls = [
        finding
        for finding in findings
        if finding["api"] == "Explainer.__call__"
    ]

    assert calls
    assert calls[0]["scope"] == "inner"


def test_api_shadowing_does_not_promote_non_shap():
    from shap_review.semantic.api_era import scan_api_era

    source = """import shap
e=shap.TreeExplainer(model)
e=other_factory()
e(X)
"""

    findings = scan_api_era(source)

    assert not [
        finding
        for finding in findings
        if finding["api"] == "Explainer.__call__"
    ]


def test_semantic_broadcast_records_authorization():
    import numpy as np

    from shap_review.contracts.oracles import _semantic_align

    _, _, metadata = _semantic_align(
        np.array([[1.0, 2.0]]),
        np.array([[1.0, 2.0], [1.0, 2.0]]),
        role="target",
    )

    assert metadata["broadcast_applied"] is True
    assert metadata["semantic_axis"] == "singleton_dimension"


def test_backend_matrix_routes_through_adapter():
    from shap_review.backends.adapter import MatrixBackendAdapter

    assert hasattr(MatrixBackendAdapter, "execute_case")


def test_input_mutation_oracle_is_registered():
    from shap_review.contracts.oracles import ORACLE_REGISTRY

    assert "InputMutationOracle" in ORACLE_REGISTRY

def test_native_flow_detects_lifetime_after_boundary(tmp_path: Path):
    from shap_review.semantic.native_flow import correlate_boundary

    path = tmp_path / "x.cpp"
    path.write_text(
        """
void f(PyObject* x) {
    auto p = PyArray_DATA(x);
    if (!p) { return; }
    Py_DECREF(x);
    free(p);
}
"""
    )

    result = correlate_boundary(path, 3)

    assert result["lifetime_after_boundary"] is True

def test_native_flow_associates_symbol(tmp_path: Path):
    from shap_review.semantic.native_flow import correlate_boundary

    path = tmp_path / "x.cpp"
    path.write_text(
        """
void f(PyObject* x) {
    auto data = PyArray_DATA(x);
    if (!data) { return; }
    Py_DECREF(x);
    free(data);
}
"""
    )

    result = correlate_boundary(path, 3)

    assert result["same_function"]
    assert result["symbol"] == "data"
    assert result["lifetime_after_boundary"]


def test_native_flow_marks_heuristic_results_as_not_proven(tmp_path: Path):
    from shap_review.semantic.native_flow import correlate_boundary

    source = """void f(PyObject* x) {
    auto data = PyArray_DATA(x);
    if (!data) { return; }
    Py_DECREF(data);
}
"""

    path = tmp_path / "native.cpp"
    path.write_text(source, encoding="utf-8")

    result = correlate_boundary(path, boundary_line=2)

    assert result["analysis_mode"] == "triage"
    assert result["proof_status"] == "NOT_PROVEN"
    assert result["requires_runtime_or_control_flow_validation"] is True
    assert result["lifetime_after_boundary"] is True

def test_api_era_scanner():
    from shap_review.semantic.api_era import scan_api_era

    findings = scan_api_era(
        "explainer.shap_values(X)\n"
        "shap.Explainer(model)(X)"
    )

    assert any(finding["era"] == "legacy" for finding in findings)
    assert any(
        finding["api"] == "Explainer.__call__"
        for finding in findings
    )


def test_native_map_has_expected_layers(tmp_path: Path):
    from shap_review.native import map_shap_native

    (tmp_path / "shap/cutils").mkdir(parents=True)
    (tmp_path / "shap/cext").mkdir(parents=True)

    result = map_shap_native(str(tmp_path))

    names = {
        component["name"]
        for component in result["components"]
        if component["present"]
    }

    assert {"cutils", "cext"} <= names