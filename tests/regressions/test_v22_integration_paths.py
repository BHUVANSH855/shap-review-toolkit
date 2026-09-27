from shap_review.cli import dispatch


def test_cli_capability_path():
    r = dispatch("capabilities")
    assert "fuzz-protocol" in r["commands"]
    assert "semantic-oracle" in r["commands"]


def test_cli_backend_discovery_path():
    r = dispatch("fuzz-backends", arguments={"execute": False})
    assert "backends" in r and "matrix" in r


def test_evidence_same_revision_and_environment_is_correlated():
    from shap_review.evidence.graph import EvidenceGraph, EvidenceNode

    g = EvidenceGraph()
    env = {"python": "3.13", "shap": "0.50.0"}
    g.add(
        EvidenceNode(
            "a",
            "dynamic",
            "a",
            "p1",
            "a",
            True,
            execution_id="e1",
            repository_revision="r1",
            environment=env,
        )
    )
    g.add(
        EvidenceNode(
            "b",
            "differential",
            "b",
            "p2",
            "b",
            True,
            execution_id="e2",
            repository_revision="r1",
            environment=env,
        )
    )
    assert not g.is_independent("a", "b")
    assert g.independence_reason("a", "b") == "shared-revision-environment"


def test_native_map_can_report_source_symbols(tmp_path):
    from shap_review.native.shap_map import map_shap_native

    p = tmp_path / "shap" / "cext"
    p.mkdir(parents=True)
    (p / "_cext.cpp").write_text("int tree_shap_value(double x) { return 0; }\n")
    r = map_shap_native(str(tmp_path))
    assert any(s["symbol"] == "tree_shap_value" for s in r["symbols"])


def test_backend_adapter_protocol_is_exported():
    from shap_review.backends import BackendAdapter, BackendExecution

    assert BackendAdapter is not None and "status" in BackendExecution.__annotations__
