from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .comparator import compare
from .semantic import compare_shap_contract, normalize_shap_result

# Capture wrapper script — extracted from inline python -c string so it
# works correctly on Windows paths containing backslashes.
_CAPTURE_WRAPPER = (
    Path(__file__).resolve().parents[1] / "resources" / "capture_wrapper.py"
)


def run_json_script(
    script: str | Path,
    timeout: int = 60,
    python: str | None = None,
    env: dict | None = None,
    capture_environment: bool = False,
    environment_depth: str = "basic",
) -> dict:
    executable = python or sys.executable
    proc_env = os.environ.copy()
    proc_env.update(env or {})

    try:
        if capture_environment:
            p = subprocess.run(
                [
                    executable,
                    str(_CAPTURE_WRAPPER),
                    str(script),
                    environment_depth,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=proc_env,
                check=False,
            )
            payload = json.loads(p.stdout) if p.stdout.strip() else {}
            return {
                "ok": (
                    p.returncode == 0
                    and payload.get("returncode", 1) == 0
                    and payload.get("parse_error") is None
                ),
                "timeout": False,
                "returncode": payload.get("returncode", p.returncode),
                "stdout": payload.get("stdout", "")[-20000:],
                "stderr": p.stderr[-20000:],
                "value": payload.get("value"),
                "subprocess_environment": payload.get("environment"),
                "parse_error": payload.get("parse_error"),
                "execution_error": payload.get("error"),
            }

        p = subprocess.run(
            [executable, str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=proc_env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "timeout": True,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "execution_status": "NOT_EXECUTED",
            "execution_reason": "TIMEOUT",
            "semantic_status": "NOT_EVALUATED",
        }

    r = {
        "ok": p.returncode == 0,
        "timeout": False,
        "returncode": p.returncode,
        "stdout": p.stdout[-20000:],
        "stderr": p.stderr[-20000:],
    }
    if p.returncode == 0:
        try:
            r["value"] = json.loads(p.stdout)
        except json.JSONDecodeError as exc:
            r.update({"ok": False, "parse_error": str(exc)})
    return r


def differential_scripts(
    reference: str | Path,
    candidate: str | Path,
    timeout: int = 60,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    semantic: bool = True,
) -> dict:
    left = run_json_script(reference, timeout, capture_environment=True)
    right = run_json_script(candidate, timeout, capture_environment=True)
    out = {"reference": left, "candidate": right, "equal": None}

    if left.get("timeout") or right.get("timeout"):
        out.update(
            {
                "equal": False,
                "status": "EXECUTION_FAILED",
                "comparison_status": "EXECUTION_FAILED",
                "comparison_reason": "TIMEOUT",
                "semantic_status": "NOT_EVALUATED",
                "differential_agreement": None,
                "reference_correctness": "UNKNOWN",
            }
        )
        return out

    if not left.get("ok") or not right.get("ok"):
        out.update(
            {
                "equal": False,
                "status": "EXECUTION_FAILED",
                "comparison_status": "EXECUTION_FAILED",
                "comparison_reason": "EXECUTION_FAILED",
                "semantic_status": "NOT_EVALUATED",
                "differential_agreement": None,
                "reference_correctness": "UNKNOWN",
            }
        )
        return out

    lv = normalize_shap_result(left["value"]) if semantic else left["value"]
    rv = normalize_shap_result(right["value"]) if semantic else right["value"]

    out["comparison"] = compare(lv, rv, rtol=rtol, atol=atol)
    out["contract_comparison"] = compare_shap_contract(
        left["value"], right["value"], rtol=rtol, atol=atol
    )

    agree = bool(out["comparison"]["equal"] and out["contract_comparison"]["equal"])

    out.update(
        {
            "equal": agree,
            "differential_agreement": agree,
            "reference_correctness": "UNKNOWN",
            "comparison_status": "MATCH" if agree else "MISMATCH",
            "status": "MATCH" if agree else "MISMATCH",
            "semantic_status": "NOT_EVALUATED" if agree else "INCONCLUSIVE",
            "semantic_disagreement": (
                out["comparison"]["equal"] != out["contract_comparison"]["equal"]
            ),
            "comparison_reason": (
                "differential agreement does not establish correctness"
            ),
            "semantic_note": (
                "MATCH means reference and candidate agree; "
                "it is not a correctness proof"
            ),
        }
    )
    return out