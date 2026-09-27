from shap_review.utils import fingerprint


def failure_fingerprint(kind: str, message: str, stack: str = "") -> str:
    return fingerprint(kind, message, stack[:2000])
