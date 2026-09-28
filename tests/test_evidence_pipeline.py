from pathlib import Path

from shap_review.engine import ReviewEngine


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
    )

    second = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "runtime",
        "b",
        True,
        evidence_id="b",
        execution_id="run-b",
        producer="oracle",
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
