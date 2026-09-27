from shap_review.utils import fingerprint


def candidate_fingerprint(candidate) -> str:
    return fingerprint(
        candidate.bug_class,
        candidate.invariant,
        candidate.file,
        str(candidate.line),
        candidate.symbol or "",
    )
