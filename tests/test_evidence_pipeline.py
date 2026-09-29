from pathlib import Path

from shap_review.engine import ReviewEngine
from shap_review.runtime_bridge import (
    _candidate_fingerprint,
    attach_dynamic_evidence,
)

FIX = Path(__file__).parent / "fixtures"


def run(path):
    return ReviewEngine().analyze(path, path / ".out")


def test_issue_4911_signal_is_evidence_linked():
    candidates = run(FIX / "shap_4911_nullable_dtype")

    assert any(
        candidate.bug_class == "SHAP-05"
        and any(
            evidence.source == "SHAP-EVID-4911"
            for evidence in candidate.evidence
        )
        for candidate in candidates
    )


def test_issue_5098_signal_is_evidence_linked():
    candidates = run(FIX / "shap_5098_model_output")

    assert any(
        candidate.bug_class == "SHAP-02"
        and any(
            evidence.source == "SHAP-EVID-5098"
            for evidence in candidate.evidence
        )
        for candidate in candidates
    )


def test_issue_4869_is_test_gap_not_defect():
    candidates = run(FIX / "shap_4869_interaction_gap")

    assert any(
        candidate.bug_class == "SHAP-01"
        and not candidate.validation_required
        for candidate in candidates
    )


def test_evidence_corpus_has_canonical_primary_records():
    from shap_review.evidence import EvidenceCorpus

    corpus = EvidenceCorpus.from_directory(
        Path(__file__).parents[1] / "data/evidence/issues"
    )

    assert len(corpus.records) >= 7
    assert not corpus.validate()


def test_dynamic_evidence_without_provenance_is_ambiguous_not_independent():
    from shap_review.evidence.model import (
        EvidenceChain,
        EvidenceItem,
        EvidenceKind,
    )

    item = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "runtime",
        "claim",
        True,
        evidence_id="e1",
    )

    chain = EvidenceChain()
    chain.add(item)

    assert item.independent is False


def test_derived_evidence_without_ancestry_is_not_independent():
    from shap_review.evidence.model import (
        EvidenceChain,
        EvidenceItem,
        EvidenceKind,
        EvidenceOrigin,
    )

    item = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "runtime",
        "claim",
        True,
        origin=EvidenceOrigin.DERIVED,
        evidence_id="e1",
        execution_id="run1",
    )

    chain = EvidenceChain()
    chain.add(item)

    assert item.independent is False


def test_same_producer_can_be_independent_across_runs():
    from shap_review.evidence.model import (
        EvidenceChain,
        EvidenceItem,
        EvidenceKind,
    )

    first = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "runtime",
        "a",
        True,
        evidence_id="a",
        execution_id="run-a",
        producer="oracle",
        input_fingerprint="input-a",
    )

    second = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "runtime",
        "b",
        True,
        evidence_id="b",
        execution_id="run-b",
        producer="oracle",
        input_fingerprint="input-b",
    )

    chain = EvidenceChain()
    chain.add(first)
    chain.add(second)

    assert len(chain.independent_items()) == 2


def test_evidence_graph_detects_shared_input_fingerprint():
    from shap_review.evidence.graph import EvidenceGraph, EvidenceNode

    graph = EvidenceGraph()

    graph.add(
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
    graph.add(
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

    assert not graph.is_independent("a", "b")
    assert graph.independence_reason("a", "b") == "shared-input-lineage"


def test_sanitizer_clean_never_confirms_memory_safety():
    from shap_review.evidence.model import (
        EvidenceChain,
        EvidenceItem,
        EvidenceKind,
    )

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


def test_runtime_anomaly_with_matching_bug_class_is_not_attached_without_correlation():
    candidates = run(FIX / "shap_4911_nullable_dtype")

    target = next(
        candidate
        for candidate in candidates
        if candidate.bug_class == "SHAP-05"
    )

    original_chain_size = len(target.evidence_chain.items)
    original_validation_required = target.validation_required

    bridge_result = {
        "available": True,
        "anomalies": [
            {
                "case": {
                    "n_features": 4,
                    "classification": False,
                    "model_output": "raw",
                },
                "execution": {
                    "executed": True,
                    "input_unchanged": False,
                },
                "classification": {
                    "kind": "INPUT_MUTATION",
                    "bug_class": "SHAP-05",
                    "confidence": 0.7,
                },
                "provenance": {
                    "execution_id": "unrelated-run",
                    "input_fingerprint": "unrelated-input",
                    "environment_fingerprint": "env-a",
                    "producer": "runtime-bridge",
                },
            }
        ],
        "classifications": [
            {
                "kind": "INPUT_MUTATION",
                "bug_class": "SHAP-05",
                "confidence": 0.7,
            }
        ],
    }

    updated = attach_dynamic_evidence([target], bridge_result)
    result = updated[0]

    assert len(result.evidence_chain.items) == original_chain_size
    assert result.validation_required == original_validation_required
    assert "dynamic-evidence" not in result.tags
    assert "runtime-bridge" not in result.tags


def test_correlated_runtime_anomaly_can_attach_to_matching_candidate():
    candidates = run(FIX / "shap_4911_nullable_dtype")

    target = next(
        candidate
        for candidate in candidates
        if candidate.bug_class == "SHAP-05"
    )

    candidate_fingerprint = _candidate_fingerprint(target)

    bridge_result = {
        "available": True,
        "anomalies": [
            {
                "case": {
                    "n_features": 4,
                    "classification": False,
                    "model_output": "raw",
                },
                "execution": {
                    "executed": True,
                    "input_unchanged": False,
                },
                "classification": {
                    "kind": "INPUT_MUTATION",
                    "bug_class": "SHAP-05",
                    "confidence": 0.7,
                },
                "provenance": {
                    "execution_id": "correlated-run",
                    "input_fingerprint": "matching-input",
                    "candidate_fingerprint": candidate_fingerprint,
                    "environment_fingerprint": "env-a",
                    "producer": "runtime-bridge",
                    "shap_version": "test",
                },
            }
        ],
        "classifications": [
            {
                "kind": "INPUT_MUTATION",
                "bug_class": "SHAP-05",
                "confidence": 0.7,
            }
        ],
    }

    updated = attach_dynamic_evidence([target], bridge_result)
    result = updated[0]

    dynamic = [
        item
        for item in result.evidence_chain.items
        if item.source == "runtime-bridge"
    ]

    assert len(dynamic) == 1
    assert dynamic[0].origin.value == "execution"
    assert dynamic[0].details["execution_id"] == "correlated-run"
    assert dynamic[0].details["candidate_fingerprint"] == candidate_fingerprint
    assert "dynamic-evidence" in result.tags
    assert "runtime-bridge" in result.tags
    assert result.validation_required == target.validation_required


def test_runtime_bridge_preserves_environment_provenance():
    from shap_review.runtime_bridge import run_bridge

    result = run_bridge(iterations=0, seed=42)

    assert "provenance" in result

    provenance = result["provenance"]
    assert "environment_fingerprint" in provenance
    assert "python_executable" in provenance
    assert "python_version" in provenance
    assert "shap_version" in provenance
    assert "shap_source_path" in provenance

    if result["available"]:
        assert provenance["producer"] == "runtime-bridge"
        assert provenance["execution_id"]
        assert provenance["seed"] == 42
        assert provenance["iterations"] == 0


def test_evidence_graph_preserves_item_provenance():
    from shap_review.evidence.graph import EvidenceGraph
    from shap_review.evidence.model import (
        EvidenceItem,
        EvidenceKind,
        EvidenceOrigin,
    )

    item = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "runtime",
        "claim",
        False,
        origin=EvidenceOrigin.EXECUTION,
        evidence_id="runtime-1",
        execution_id="exec-1",
        producer="runtime-bridge",
        fixture_id="fixture-1",
        input_fingerprint="input-1",
        repository_revision="revision-1",
        environment_fingerprint="environment-1",
        transformation="normalization",
    )

    graph = EvidenceGraph()
    node = graph.add_item(item)

    assert node.execution_id == "exec-1"
    assert node.producer == "runtime-bridge"
    assert node.fixture_id == "fixture-1"
    assert node.input_fingerprint == "input-1"
    assert node.repository_revision == "revision-1"
    assert node.transformation == "normalization"
    assert node.environment == {}


def test_evidence_graph_detects_shared_ancestry():
    from shap_review.evidence.graph import EvidenceGraph, EvidenceNode

    graph = EvidenceGraph()

    graph.add(
        EvidenceNode(
            "root",
            "static",
            "source",
            "scanner",
            "root",
            True,
        )
    )
    graph.add(
        EvidenceNode(
            "left",
            "dynamic",
            "runtime",
            "runner-a",
            "left",
            True,
            parent_ids=("root",),
        )
    )
    graph.add(
        EvidenceNode(
            "right",
            "dynamic",
            "runtime",
            "runner-b",
            "right",
            True,
            parent_ids=("root",),
        )
    )

    assert not graph.is_independent("left", "right")
    assert graph.independence_reason("left", "right") == "shared-ancestry"


def test_evidence_verdict_stays_static_when_runtime_provenance_is_ambiguous():
    from shap_review.evidence.model import EvidenceChain, EvidenceItem, EvidenceKind

    chain = EvidenceChain()

    chain.add(
        EvidenceItem(
            EvidenceKind.STATIC,
            "x",
            "candidate",
            True,
            evidence_id="static-1",
        )
    )

    chain.add(
        EvidenceItem(
            EvidenceKind.DYNAMIC,
            "runtime",
            "executed",
            True,
            evidence_id="dynamic-1",
        )
    )

    assert chain.verdict() == "STATIC_CANDIDATE"

    chain.add(
        EvidenceItem(
            EvidenceKind.REPRODUCTION,
            "repro",
            "reproduced",
            True,
            evidence_id="reproduction-1",
        )
    )

    assert chain.verdict() == "STATIC_CANDIDATE"


def test_evidence_dependency_is_not_independent():
    from shap_review.evidence import (
        EvidenceChain,
        EvidenceItem,
        EvidenceKind,
        EvidenceOrigin,
    )

    chain = EvidenceChain()
    chain.add(
        EvidenceItem(
            EvidenceKind.STATIC,
            "historical-rule",
            "same issue",
            True,
            origin=EvidenceOrigin.DERIVED,
            derived_from=("SHAP-EVID-1",),
        )
    )

    assert chain.independent_items() == []
    assert chain.verdict() == "UNVALIDATED"


def test_evidence_gate_rejects_static_confirmation():
    from shap_review.evidence import EvidenceChain, EvidenceItem, EvidenceKind
    from shap_review.findings.lifecycle import evidence_transition
    from shap_review.types import FindingStatus

    chain = EvidenceChain(
        [EvidenceItem(EvidenceKind.STATIC, "x", "candidate", True)]
    )

    try:
        evidence_transition(
            FindingStatus.REPRODUCED,
            FindingStatus.EVIDENCE_VALID,
            chain,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("static evidence must not pass confirmation gate")

def test_promotion_requires_runtime_evidence():
    import pytest

    from shap_review.evidence.chain import build_chain
    from shap_review.findings.lifecycle import evidence_transition
    from shap_review.types import FindingStatus

    chain = build_chain(
        entries=[
            {
                "kind": "static",
                "source": "s",
                "claim": "candidate",
                "passed": True,
                "evidence_id": "s",
            }
        ]
    )

    with pytest.raises(ValueError):
        evidence_transition(
            FindingStatus.REPRO_PENDING,
            FindingStatus.REPRODUCED,
            chain,
        )