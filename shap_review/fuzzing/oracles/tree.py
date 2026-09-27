from __future__ import annotations


def evaluate_execution(result: dict) -> dict:
    if not result.get("executed"):
        return {"valid": False, "reason": "not-executed"}
    if result.get("failed"):
        return {
            "valid": False,
            "reason": "runtime-exception",
            "exception": result.get("exception"),
            "message": result.get("message"),
        }
    checks = {
        "finite": result.get("finite", False),
        "input_unchanged": result.get("input_unchanged", False),
        "additivity_checked": result.get("additivity_checked", False),
        "additivity_passed": result.get("additivity_passed", False),
        "additivity_within_tolerance": result.get("max_additivity_error", float("inf"))
        <= result.get("additivity_tolerance", 0.0),
        "shape_consistent": bool(result.get("shape_compatible", False))
        and tuple(result.get("reconstruction_shape", ()))
        == tuple(result.get("target_shape", ())),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "max_additivity_error": result.get("max_additivity_error"),
    }
