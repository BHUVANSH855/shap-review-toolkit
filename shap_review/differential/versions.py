from __future__ import annotations

from pathlib import Path

from .comparator import compare
from .runner import run_json_script
from .semantic import compare_shap_contract


def differential_versions(
    reference_script: str | Path,
    candidate_script: str | Path,
    python_reference: str,
    python_candidate: str,
    timeout: int = 120,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict:
    """Run the same logical regression under two explicitly selected interpreters."""
    left = run_json_script(reference_script, timeout=timeout, python=python_reference)
    right = run_json_script(candidate_script, timeout=timeout, python=python_candidate)
    out = {
        "reference_python": python_reference,
        "candidate_python": python_candidate,
        "reference": left,
        "candidate": right,
        "isolated_interpreters": python_reference != python_candidate,
    }
    if not left.get("ok") or not right.get("ok"):
        out.update({"status": "execution_failed", "equal": False})
        return out
    out["comparison"] = compare(left["value"], right["value"], rtol=rtol, atol=atol)
    out["contract_comparison"] = compare_shap_contract(
        left["value"], right["value"], rtol=rtol, atol=atol
    )
    out["equal"] = bool(
        out["comparison"]["equal"] and out["contract_comparison"].get("equal", True)
    )
    out["status"] = "match" if out["equal"] else "mismatch"
    return out
