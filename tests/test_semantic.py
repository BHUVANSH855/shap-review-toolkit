from pathlib import Path

from shap_review.semantic import SHAPSemanticMapper


def test_semantic_mapper(tmp_path: Path):
    p = tmp_path / "_tree.py"
    p.write_text("class TreeExplainer:\n    pass\n")
    g = SHAPSemanticMapper().map(tmp_path)
    assert any(n.name == "TreeExplainer" for n in g.nodes)


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
    symbols = [f["symbol"] for f in findings]
    assert "e" in symbols
    assert all(
        f["scope"] != "f" or f["provenance"] != "proven"
        for f in findings
        if f["scope"] == "f"
    )
    assert not any(
        f["scope"] == "g" and f["symbol"] == "e" and f["message"].startswith("callable")
        for f in findings
    )


def test_target_provenance_does_not_overclaim_model_api():
    from shap_review.contracts.oracles import classify_target_provenance

    p = classify_target_provenance("model.predict", None)
    assert p["target_source"] == "model_api"
    assert p["oracle_independence"] == "UNKNOWN"
    p = classify_target_provenance("independent_probability_fn", True)
    assert p["oracle_independence"] == "DECLARED"
    assert p["independence_basis"] == "caller_declared"


def test_interaction_oracle_requires_symmetry_and_reconstruction():
    import numpy as np

    from shap_review.contracts.oracles import InteractionOracle

    values = np.array([[3.0, 5.0]])
    iv = np.array([[[1.0, 2.0], [2.0, 3.0]]])
    r = InteractionOracle().check(values=values, interaction_values=iv)
    assert r.passed is True
    bad = iv.copy()
    bad[0, 0, 1] = 99.0
    r = InteractionOracle().check(values=values, interaction_values=bad)
    assert r.passed is False
