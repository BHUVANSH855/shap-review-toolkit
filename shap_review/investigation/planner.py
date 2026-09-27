from __future__ import annotations

from dataclasses import dataclass

from shap_review.types import Candidate


@dataclass
class InvestigationPlan:
    candidate_id: str
    steps: list[str]


def plan(candidate: Candidate) -> InvestigationPlan:
    return InvestigationPlan(
        candidate.candidate_id,
        [
            "inspect source context",
            "trace callers/data flow",
            "compare invariant preconditions",
            "inspect relevant tests",
            "attempt minimal reproducer",
            "run sanitizer if native",
            "record evidence or false positive",
        ],
    )
