from __future__ import annotations

from shap_review.evidence.chain import build_chain
from shap_review.evidence.provenance import classify_finding
from shap_review.types import Candidate, Finding, FindingStatus

from .fingerprint import candidate_fingerprint


def promote_reproduced(
    candidate: Candidate,
    summary: str,
    evidence: list,
    reproducer: str | None = None,
    *,
    historical_issue: str | None = None,
    discovered_by_current_analysis: bool = True,
) -> Finding:
    entries = []
    for e in evidence:
        if hasattr(e, "kind"):
            entries.append(
                {
                    "kind": e.kind,
                    "source": e.source,
                    "claim": e.note,
                    "passed": True,
                    "confidence": min(
                        1.0, max(0.1, e.strength / 10 if e.strength else 0.1)
                    ),
                }
            )
        elif isinstance(e, dict):
            entries.append(e)
    chain = build_chain(entries=entries)
    classification = classify_finding(
        historical_issue=historical_issue,
        reproduced=True,
        discovered_by_current_analysis=discovered_by_current_analysis,
    )
    return Finding(
        f"SHAPR-{candidate.candidate_id[-12:]}",
        candidate.candidate_id,
        FindingStatus.REPRODUCED,
        candidate.bug_class,
        candidate.invariant,
        summary,
        evidence,
        reproducer,
        fingerprint=candidate_fingerprint(candidate),
        evidence_chain=chain,
        classification=classification,
    )
