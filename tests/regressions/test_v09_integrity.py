import json
from pathlib import Path


def test_canonical_release_contract():
    from shap_review.version import CAPABILITIES, SCHEMA_VERSION, VERSION

    assert VERSION == __import__("shap_review.version", fromlist=["VERSION"]).VERSION
    assert (
        SCHEMA_VERSION
        == __import__("shap_review.version", fromlist=["SCHEMA_VERSION"]).SCHEMA_VERSION
    )
    assert len(CAPABILITIES) == 24
    assert "evidence" in CAPABILITIES


def test_evidence_dependency_is_not_independent():
    from shap_review.evidence import (
        EvidenceChain,
        EvidenceItem,
        EvidenceKind,
        EvidenceOrigin,
    )

    c = EvidenceChain()
    c.add(
        EvidenceItem(
            EvidenceKind.STATIC,
            "historical-rule",
            "same issue",
            True,
            origin=EvidenceOrigin.DERIVED,
            derived_from=("SHAP-EVID-1",),
        )
    )
    assert c.independent_items() == []
    assert c.verdict() == "UNVALIDATED"


def test_finding_classification():
    from shap_review.evidence.provenance import classify_finding

    assert (
        classify_finding(
            historical_issue="4911",
            reproduced=True,
            discovered_by_current_analysis=False,
        ).value
        == "historical_reproduction"
    )
    assert classify_finding(reproduced=True).value == "novel_reproduction"


def test_evidence_gate_rejects_static_confirmation():
    from shap_review.evidence import EvidenceChain, EvidenceItem, EvidenceKind
    from shap_review.findings.lifecycle import evidence_transition
    from shap_review.types import FindingStatus

    c = EvidenceChain([EvidenceItem(EvidenceKind.STATIC, "x", "candidate", True)])
    try:
        evidence_transition(FindingStatus.REPRODUCED, FindingStatus.EVIDENCE_VALID, c)
    except ValueError:
        pass
    else:
        raise AssertionError("static evidence must not pass confirmation gate")


def test_shap_contract():
    from shap_review.contracts import SHAPContract, compare_contracts, validate_contract

    a = SHAPContract("TreeExplainer", "tree", values_shape=(2, 3))
    b = SHAPContract("TreeExplainer", "tree", values_shape=(2, 3))
    assert compare_contracts(a, b)["equal"]
    r = validate_contract(a, [[1, 2, 3], [4, 5, 6]])
    assert r["values_shape_match"]


def test_provider_manifests_match_canonical():
    from shap_review.version import CAPABILITIES, SCHEMA_VERSION, VERSION

    for provider in ("claude", "openai", "gemini"):
        for name in (
            "tool-schema.json",
            "tool-definition.json",
            "function-declaration.json",
        ):
            p = Path("adapters") / provider / name
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            assert d["version"] == VERSION
            assert d["schema_version"] == SCHEMA_VERSION
            assert d["commands"] == list(CAPABILITIES)


def test_protocol_campaign_records_real_events():
    from shap_review.fuzzing.protocol_campaign import ProtocolCampaign

    def target(obj, case):
        _ = obj.shape
        if case["protocol"] == "call":
            obj()

    r = ProtocolCampaign(1).run(target, 10)
    assert r["executed"] == 10
    assert "shape" in r["protocols_observed"]
    assert all("protocol_events" in item for item in r["results"])


def test_native_flow_associates_symbol(tmp_path):
    from shap_review.semantic.native_flow import correlate_boundary

    p = tmp_path / "x.cpp"
    p.write_text("""
    void f(PyObject* x) {
        auto data = PyArray_DATA(x);
        if (!data) { return; }
        Py_DECREF(x);
        free(data);
    }
    """)
    r = correlate_boundary(p, 3)
    assert r["same_function"]
    assert r["symbol"] == "data"
    assert r["lifetime_after_boundary"]
