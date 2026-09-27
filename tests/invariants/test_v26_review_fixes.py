from pathlib import Path


def test_input_mutation_oracle_is_registered():
    from shap_review.contracts.oracles import ORACLE_REGISTRY

    assert "InputMutationOracle" in ORACLE_REGISTRY


def test_differential_match_is_not_semantic_pass(tmp_path: Path):
    from shap_review.differential import differential_scripts

    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text('import json; print(json.dumps({"values":[1,2]}))')
    b.write_text('import json; print(json.dumps({"values":[1,2]}))')
    r = differential_scripts(a, b)
    assert r["comparison_status"] == "MATCH"
    assert r["differential_agreement"] is True
    assert r["semantic_status"] == "NOT_EVALUATED"
    assert r["reference_correctness"] == "UNKNOWN"


def test_api_nested_scope_is_preserved():
    from shap_review.semantic.api_era import scan_api_era

    src = """import shap\ndef foo():\n    e=shap.TreeExplainer(model)\n    def inner():\n        e(X)\n    inner()\n"""
    findings = scan_api_era(src)
    calls = [x for x in findings if x["api"] == "Explainer.__call__"]
    assert calls and calls[0]["scope"] == "inner"


def test_api_shadowing_does_not_promote_non_shap():
    from shap_review.semantic.api_era import scan_api_era

    src = """import shap\ne=shap.TreeExplainer(model)\ne=other_factory()\ne(X)\n"""
    findings = scan_api_era(src)
    assert not [x for x in findings if x["api"] == "Explainer.__call__"]


def test_semantic_broadcast_records_authorization():
    import numpy as np

    from shap_review.contracts.oracles import _semantic_align

    _, _, meta = _semantic_align(
        np.array([[1.0, 2.0]]), np.array([[1.0, 2.0], [1.0, 2.0]]), role="target"
    )
    assert meta["broadcast_applied"] is True
    assert meta["semantic_axis"] == "singleton_dimension"


def test_evidence_graph_detects_shared_input_fingerprint():
    from shap_review.evidence.graph import EvidenceGraph, EvidenceNode

    g = EvidenceGraph()
    g.add(
        EvidenceNode(
            "a",
            "dynamic",
            "runtime",
            "p",
            "a",
            True,
            execution_id="r1",
            input_fingerprint="same",
        )
    )
    g.add(
        EvidenceNode(
            "b",
            "dynamic",
            "runtime",
            "p",
            "b",
            True,
            execution_id="r2",
            input_fingerprint="same",
        )
    )
    assert not g.is_independent("a", "b")
    assert g.independence_reason("a", "b") == "shared-input-lineage"


def test_sanitizer_clean_never_confirms_memory_safety():
    from shap_review.evidence.model import EvidenceChain, EvidenceItem, EvidenceKind

    item = EvidenceItem(
        EvidenceKind.SANITIZER,
        "asan",
        "no finding",
        True,
        evidence_id="s1",
        execution_id="run1",
        fixture_id="fx1",
        details={"finding": False},
    )
    chain = EvidenceChain()
    chain.add(item)
    assert chain.verdict() == "SANITIZER_CLEAN"


def test_backend_matrix_routes_through_adapter(monkeypatch):
    import pytest

    pytest.importorskip("shap")
    import shap_review.fuzzing.backend_matrix as bm
    from shap_review.backends.adapter import MatrixBackendAdapter

    called = {"n": 0}
    original = MatrixBackendAdapter.execute_case

    def wrapped(self, **kwargs):
        called["n"] += 1
        return {
            "backend": self.name,
            "status": "PASS_WITH_FINDINGS",
            "execution_status": "EXECUTED",
            "semantic_status": "INCONCLUSIVE",
            "execution_reason": "COMPLETED",
        }

    monkeypatch.setattr(MatrixBackendAdapter, "execute_case", wrapped)
    monkeypatch.setattr(
        bm, "discover_backends", lambda: [bm.BackendSpec("fake", "json", (), True, "1")]
    )
    monkeypatch.setattr(bm.importlib, "import_module", lambda name: __import__("json"))
    out = bm.execute_installed_backend_matrix(exhaustive=False)
    assert called["n"] == 9
    assert out["summary"]["executed"] == 9
    monkeypatch.setattr(MatrixBackendAdapter, "execute_case", original)


def test_archive_validator_rejects_pytest_cache(tmp_path: Path):
    import subprocess
    import sys
    import zipfile

    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("pkg/.pytest_cache/x", "x")
    p = subprocess.run(
        [sys.executable, str(root / "scripts/validate_distribution.py"), str(archive)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode != 0
