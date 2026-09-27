from shap_review.evidence.model import (
    EvidenceChain,
    EvidenceItem,
    EvidenceKind,
    EvidenceOrigin,
)


def test_dynamic_evidence_without_provenance_is_ambiguous_not_independent():
    item = EvidenceItem(
        EvidenceKind.DYNAMIC, "runtime", "claim", True, evidence_id="e1"
    )
    chain = EvidenceChain()
    chain.add(item)
    assert item.independent is False


def test_derived_evidence_without_ancestry_is_not_independent():
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
    a = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "runtime",
        "a",
        True,
        evidence_id="a",
        execution_id="run-a",
        producer="oracle",
    )
    b = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "runtime",
        "b",
        True,
        evidence_id="b",
        execution_id="run-b",
        producer="oracle",
    )
    chain = EvidenceChain()
    chain.add(a)
    chain.add(b)
    assert len(chain.independent_items()) == 2
