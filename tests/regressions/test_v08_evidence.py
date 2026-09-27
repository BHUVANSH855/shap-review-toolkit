from pathlib import Path


def test_evidence_verdict_progression():
    from shap_review.evidence.model import EvidenceChain, EvidenceItem, EvidenceKind

    c = EvidenceChain()
    c.add(EvidenceItem(EvidenceKind.STATIC, "x", "candidate", True))
    assert c.verdict() == "STATIC_CANDIDATE"

    c.add(EvidenceItem(EvidenceKind.DYNAMIC, "runtime", "executed", True))
    assert (
        c.verdict() == "STATIC_CANDIDATE"
    )  # v0.25: ambiguous runtime evidence is not scored independently

    c.add(EvidenceItem(EvidenceKind.REPRODUCTION, "repro", "reproduced", True))
    assert (
        c.verdict() == "STATIC_CANDIDATE"
    )  # v0.25: reproduction without execution/fixture provenance is ambiguous


def test_differential_requires_semantic_contract():
    from shap_review.differential.semantic import compare_shap_contract

    a = {"values": [[1.0, 2.0]], "base_values": [0.0]}
    b = {"values": [[1.0, 2.0]], "base_values": [0.0]}
    r = compare_shap_contract(a, b)
    assert r["equal"] is True


def test_native_flow_detects_lifetime_after_boundary(tmp_path: Path):
    from shap_review.semantic.native_flow import correlate_boundary

    p = tmp_path / "x.cpp"
    p.write_text("""
    void f(PyObject* x) {
        auto p = PyArray_DATA(x);
        if (!p) { return; }
        Py_DECREF(x);
        free(p);
    }
    """)
    r = correlate_boundary(p, 3)
    assert r["lifetime_after_boundary"] is True


def test_protocol_campaign_executes_protocols():
    from shap_review.fuzzing.protocol_campaign import ProtocolCampaign

    def target(obj, case):
        _ = obj.shape
        if case["protocol"] == "call":
            obj()

    r = ProtocolCampaign(1).run(target, 20)
    assert r["executed"] == 20
    assert r["protocols_observed"]
    assert r["mutation_observed"] >= 0


def test_provider_versions():
    from adapters.claude.adapter import ClaudeAdapter
    from adapters.gemini.adapter import GeminiAdapter
    from adapters.openai.adapter import OpenAIAdapter

    for adapter in (ClaudeAdapter(), OpenAIAdapter(), GeminiAdapter()):
        assert (
            adapter.invoke("version", {})["version"]
            == __import__("shap_review.version", fromlist=["VERSION"]).VERSION
        )
        assert adapter.invoke("evidence", {})["supported"] is True
