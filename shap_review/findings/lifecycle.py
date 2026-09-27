from __future__ import annotations

from shap_review.types import FindingStatus

from .promotion import PromotionPolicy

ALLOWED = {
    FindingStatus.DISCOVERED: {FindingStatus.CANDIDATE},
    FindingStatus.CANDIDATE: {
        FindingStatus.INVESTIGATING,
        FindingStatus.FALSE_POSITIVE,
    },
    FindingStatus.INVESTIGATING: {
        FindingStatus.REPRO_PENDING,
        FindingStatus.FALSE_POSITIVE,
    },
    FindingStatus.REPRO_PENDING: {
        FindingStatus.REPRODUCED,
        FindingStatus.FALSE_POSITIVE,
    },
    FindingStatus.REPRODUCED: {
        FindingStatus.EVIDENCE_VALID,
        FindingStatus.FALSE_POSITIVE,
    },
    FindingStatus.EVIDENCE_VALID: {FindingStatus.CONFIRMED},
    FindingStatus.CONFIRMED: {
        FindingStatus.REPORTED,
        FindingStatus.FIXED,
        FindingStatus.TRACKED,
    },
    FindingStatus.REPORTED: {FindingStatus.FIXED, FindingStatus.TRACKED},
    FindingStatus.TRACKED: {FindingStatus.FIXED},
    FindingStatus.FALSE_POSITIVE: set(),
    FindingStatus.FIXED: set(),
}


def transition(current: FindingStatus, next_status: FindingStatus) -> FindingStatus:
    if next_status not in ALLOWED.get(current, set()):
        raise ValueError(f"invalid finding transition {current} -> {next_status}")
    return next_status


def evidence_transition(
    current, next_status, evidence_chain=None, reproducer=None, policy=None
):
    PromotionPolicy() if policy is None else policy
    policy = policy or PromotionPolicy()
    policy.validate(current, next_status, evidence_chain, reproducer)
    return next_status
