from __future__ import annotations

from shap_review.investigation.planner import plan
from shap_review.types import Candidate, EvidenceRef


def candidate_plan(candidate_dict: dict) -> dict:
    c = Candidate(
        candidate_dict.get("candidate_id", "unknown"),
        candidate_dict.get("bug_class", "unknown"),
        candidate_dict.get("invariant", "unknown"),
        candidate_dict.get("file", ""),
        candidate_dict.get("line"),
        candidate_dict.get("symbol"),
        candidate_dict.get("message", ""),
        [EvidenceRef(**e) for e in candidate_dict.get("evidence", [])],
        candidate_dict.get("validation_required", True),
        candidate_dict.get("confidence", "medium"),
        candidate_dict.get("tags", []),
    )
    return {"candidate": c.__dict__, "plan": plan(c).__dict__}
