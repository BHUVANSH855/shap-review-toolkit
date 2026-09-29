import pytest

from shap_review.findings.lifecycle import transition
from shap_review.types import FindingStatus


def test_valid_transition():
    assert (
        transition(
            FindingStatus.CANDIDATE,
            FindingStatus.INVESTIGATING,
        )
        == FindingStatus.INVESTIGATING
    )


def test_invalid_transition():
    with pytest.raises(ValueError):
        transition(
            FindingStatus.CANDIDATE,
            FindingStatus.CONFIRMED,
        )

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

    chain = EvidenceChain(
        [EvidenceItem(EvidenceKind.STATIC, "x", "candidate", True)]
    )

    with pytest.raises(ValueError):
        evidence_transition(
            FindingStatus.REPRODUCED,
            FindingStatus.EVIDENCE_VALID,
            chain,
        )

def test_historical_and_novel_reproduction_are_classified_by_provenance():
    from shap_review.evidence.provenance import classify_finding

    assert classify_finding(
        historical_issue="4911", reproduced=True, discovered_by_current_analysis=False
    ).value == "historical_reproduction"
    assert classify_finding(reproduced=True).value == "novel_reproduction"


