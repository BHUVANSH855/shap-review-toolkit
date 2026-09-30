from __future__ import annotations

from collections import defaultdict

from shap_review.evidence.chain import build_chain
from shap_review.types import Candidate, FindingClassification

# Evidence tiers are deliberately explicit. Static observations are weaker than
# historical evidence, runtime reproduction, differential evidence, or sanitizer
# evidence. These values are used for aggregation only; they are not a proof
# score and must never independently certify a candidate.
EVIDENCE_TIER = {
    "source": 1,
    "documentation": 2,
    "test-gap": 2,
    "issue": 4,
    "pull_request": 5,
    "dynamic": 7,
    "differential": 8,
    "sanitizer": 10,
}

# Classification ordering represents the strongest claim already established by
# an input candidate. Aggregation may preserve an existing classification, but
# must never manufacture a stronger classification from score or evidence count.
_CLASSIFICATION_RANK = {
    FindingClassification.NOVEL_CANDIDATE: 10,
    FindingClassification.HISTORICAL_CANDIDATE: 20,
    FindingClassification.REGRESSION_CANDIDATE: 30,
    FindingClassification.NOVEL_REPRODUCTION: 40,
    FindingClassification.HISTORICAL_REPRODUCTION: 50,
    FindingClassification.KNOWN_ISSUE_REPRODUCED: 60,
}

# Evidence kinds which represent actual execution or stronger independent
# confirmation. Their presence matters more than the number of static analyzers
# reporting the same location.
_RUNTIME_EVIDENCE = frozenset(
    {
        "dynamic",
        "differential",
        "sanitizer",
    }
)

_HISTORICAL_EVIDENCE = frozenset(
    {
        "issue",
        "pull_request",
    }
)


def _strongest_classification(
    candidates: list[Candidate],
) -> FindingClassification:
    """Return the strongest classification explicitly present in a group.

    The aggregator preserves classifications supplied by upstream analyzers.
    Evidence accumulation alone cannot promote a candidate to a reproduction or
    known issue.
    """
    return max(
        (candidate.classification for candidate in candidates),
        key=lambda classification: _CLASSIFICATION_RANK.get(classification, 0),
    )


def _evidence_origin(kind: str) -> str:
    """Return the provenance origin for an evidence kind."""
    return "external" if kind in _HISTORICAL_EVIDENCE else "source"


def _evidence_confidence(strength: int) -> float:
    """Normalize evidence strength into the evidence-chain confidence range."""
    if strength:
        return min(1.0, max(0.1, strength / 10))
    return 1.0


def _evidence_key(evidence: object) -> tuple[str, str, str]:
    """Return the stable deduplication key for an evidence record."""
    return (
        getattr(evidence, "kind", ""),
        getattr(evidence, "source", ""),
        getattr(evidence, "note", ""),
    )


def _confidence_from_evidence(
    evidence: list[object],
    classification: FindingClassification,
) -> str:
    """Derive conservative display confidence from evidence quality.

    Runtime or sanitizer evidence is required before an unvalidated candidate
    can reach high confidence. Historical evidence alone can strengthen a
    candidate but cannot turn it into a runtime-confirmed finding.
    """
    kinds = {getattr(item, "kind", "") for item in evidence}
    score = sum(
        EVIDENCE_TIER.get(
            kind,
            max(1, int(getattr(item, "strength", 1))),
        )
        for item in evidence
        for kind in [getattr(item, "kind", "")]
    )

    independent_kinds = len(kinds)
    runtime_kinds = kinds & _RUNTIME_EVIDENCE
    historical_kinds = kinds & _HISTORICAL_EVIDENCE

    # Diversity is useful, but only as a small bonus. Multiple static findings
    # are not independent confirmation of the underlying behavior.
    score += max(0, independent_kinds - 1) * 2

    if (
        classification
        in {
            FindingClassification.NOVEL_REPRODUCTION,
            FindingClassification.HISTORICAL_REPRODUCTION,
            FindingClassification.KNOWN_ISSUE_REPRODUCED,
        }
        and runtime_kinds
        and score >= 14
    ):
        return "high"

    if runtime_kinds and score >= 10:
        return "high"

    if historical_kinds and runtime_kinds and score >= 12:
        return "high"

    if score >= 7:
        return "medium"

    return "low"


class CandidateAggregator:
    """Merge duplicate candidates while preserving provenance and claims.

    Aggregation is deliberately non-certifying: merging candidates can combine
    evidence and preserve the strongest classification already supplied by an
    analyzer, but it cannot invent a reproduction, regression, or known issue.
    """

    def merge(self, candidates: list[Candidate]) -> list[Candidate]:
        groups: dict[tuple[object, ...], list[Candidate]] = defaultdict(list)

        for candidate in candidates:
            key = (
                candidate.bug_class,
                candidate.file,
                candidate.line,
                candidate.invariant,
            )
            groups[key].append(candidate)

        out: list[Candidate] = []

        for group in groups.values():
            base = group[0]

            evidence = []
            tags: list[str] = []
            messages: list[str] = []
            seen_evidence: set[tuple[str, str, str]] = set()
            seen_messages: set[str] = set()

            for candidate in group:
                if candidate.message and candidate.message not in seen_messages:
                    seen_messages.add(candidate.message)
                    messages.append(candidate.message)

                tags.extend(candidate.tags)

                for evidence_ref in candidate.evidence:
                    key = _evidence_key(evidence_ref)
                    if key in seen_evidence:
                        continue

                    seen_evidence.add(key)
                    evidence.append(evidence_ref)

            classification = _strongest_classification(group)
            confidence = _confidence_from_evidence(evidence, classification)

            chain_entries = [
                {
                    "kind": evidence_ref.kind,
                    "source": evidence_ref.source,
                    "claim": evidence_ref.note,
                    "passed": True,
                    "confidence": _evidence_confidence(evidence_ref.strength),
                    "origin": _evidence_origin(evidence_ref.kind),
                }
                for evidence_ref in evidence
            ]

            chain = build_chain(entries=chain_entries)

            out.append(
                Candidate(
                    candidate_id=base.candidate_id,
                    bug_class=base.bug_class,
                    invariant=base.invariant,
                    file=base.file,
                    line=base.line,
                    symbol=base.symbol,
                    message=" ".join(messages),
                    evidence=evidence,
                    validation_required=any(
                        candidate.validation_required for candidate in group
                    ),
                    confidence=confidence,
                    tags=sorted(set(tags)),
                    evidence_chain=chain,
                    classification=classification,
                )
            )

        return sorted(
            out,
            key=lambda candidate: (
                candidate.file,
                candidate.line or 0,
                candidate.bug_class,
            ),
        )
