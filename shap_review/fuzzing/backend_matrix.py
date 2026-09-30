from __future__ import annotations

from shap_review.backends.registry import (
    DEFAULT_BACKENDS,
    BackendSpec,
    discover_backends,
)

__all__ = [
    "DEFAULT_BACKENDS",
    "BackendSpec",
    "_semantic_additivity",
    "discover_backends",
    "execute_installed_backend_matrix",
    "matrix_dimensions",
]

EXECUTION_STATUSES = ("EXECUTED", "NOT_EXECUTED")

EXECUTION_REASONS = (
    "COMPLETED",
    "SHAP_ERROR",
    "BACKEND_ERROR",
    "ADAPTER_ERROR",
    "TOOLKIT_ERROR",
    "UNSUPPORTED",
    "SKIPPED",
    "TIMEOUT",
    "ERROR",
)

SEMANTIC_STATUSES = ("PASS", "FAIL", "INCONCLUSIVE", "NOT_EVALUATED")

STATUSES = (
    "EXECUTED",
    "SEMANTIC_PASS",
    "SEMANTIC_FAIL",
    "PASS_WITH_FINDINGS",
    "SKIPPED",
    "UNSUPPORTED",
    "ADAPTER_ERROR",
    "BACKEND_ERROR",
    "SHAP_ERROR",
    "TOOLKIT_ERROR",
    "ERROR",
)


def _semantic_additivity(
    values,
    base,
    target,
    *,
    interaction=False,
    output_space="raw",
    backend=None,
):
    """Compatibility wrapper for the canonical semantic additivity oracle."""
    from shap_review.backends.adapter import MatrixBackendAdapter

    return MatrixBackendAdapter._semantic_additivity(
        values,
        base,
        target,
        interaction=interaction,
        output_space=output_space,
    )


def matrix_dimensions():
    return {
        "backend": [s.name for s in DEFAULT_BACKENDS],
        "model_output": ["raw", "probability", "log_loss"],
        "input_representation": ["ndarray", "dataframe", "sparse"],
        "missing_values": [False, True],
        "classification": [False, True],
        "interaction": [False, True],
    }


def execute_installed_backend_matrix(*, exhaustive: bool = False):
    """Execute scheduled backend cases through concrete adapters."""
    try:
        import shap  # noqa: F401
    except ImportError as exc:
        return {
            "status": "SKIPPED",
            "reason": f"SHAP unavailable: {exc}",
            "results": [],
            "summary": {
                "scheduled": 0,
                "executed": 0,
                "not_executed": 0,
                "findings": 0,
                "errors": 0,
                "execution_reasons": {"SKIPPED": 0},
            },
        }

    from shap_review.backends.adapter import MatrixBackendAdapter

    results = []
    discovered = discover_backends()

    for spec in discovered:
        if not spec.available:
            results.append(
                {
                    "backend": spec.name,
                    "status": "SKIPPED",
                    "execution_status": "NOT_EXECUTED",
                    "execution_reason": "SKIPPED",
                    "semantic_status": "NOT_EVALUATED",
                    "reason": "dependency unavailable",
                }
            )
            continue

        import importlib

        mod = importlib.import_module(spec.import_name)
        adapter = MatrixBackendAdapter(spec, mod)

        if exhaustive:
            cases = [
                (classification, rep, missing, interaction, output)
                for classification in (False, True)
                for rep in ("ndarray", "dataframe", "sparse")
                for missing in (False, True)
                for interaction in (False, True)
                for output in (
                    ("raw", "probability", "log_loss") if classification else ("raw",)
                )
            ]
        else:
            cases = [
                (False, "ndarray", False, False, "raw"),
                (False, "dataframe", False, False, "raw"),
                (False, "sparse", False, False, "raw"),
                (False, "ndarray", False, True, "raw"),
                (True, "ndarray", False, False, "probability"),
                (True, "dataframe", False, False, "probability"),
                (True, "ndarray", True, False, "probability"),
                (True, "ndarray", False, False, "log_loss"),
                (True, "dataframe", False, False, "log_loss"),
            ]

        for classification, rep, missing, interaction, output in cases:
            case = {
                "classification": classification,
                "representation": rep,
                "missing_values": missing,
                "interaction": interaction,
                "model_output": output,
            }

            supported, support_reason = adapter.supports_case(**case)

            if not supported:
                result = {
                    "backend": spec.name,
                    "status": "UNSUPPORTED",
                    "execution_status": "NOT_EXECUTED",
                    "execution_reason": "UNSUPPORTED",
                    "semantic_status": "NOT_EVALUATED",
                    "stage": "capability",
                    "reason": support_reason,
                    "classification": classification,
                    "input_representation": rep,
                    "missing_values": missing,
                    "interaction": interaction,
                    "model_output": output,
                }
            else:
                result = adapter.execute_case(**case)

            status = result.get("status")

            if status in {"SEMANTIC_PASS", "SEMANTIC_FAIL"}:
                result.update(
                    execution_status="EXECUTED",
                    execution_reason="COMPLETED",
                    semantic_status=("PASS" if status == "SEMANTIC_PASS" else "FAIL"),
                )
            elif status == "PASS_WITH_FINDINGS":
                result.update(
                    execution_status="EXECUTED",
                    execution_reason="COMPLETED",
                    semantic_status="INCONCLUSIVE",
                )
            elif status in {
                "SKIPPED",
                "UNSUPPORTED",
                "ADAPTER_ERROR",
                "BACKEND_ERROR",
                "SHAP_ERROR",
                "TOOLKIT_ERROR",
                "ERROR",
            }:
                result.update(
                    execution_status="NOT_EXECUTED",
                    execution_reason=status,
                    semantic_status="NOT_EVALUATED",
                )

            results.append(result)

    findings = [
        result
        for result in results
        if result.get("status") in {"SEMANTIC_FAIL", "PASS_WITH_FINDINGS"}
    ]

    errors = [
        result
        for result in results
        if result.get("execution_status") == "NOT_EXECUTED"
        and result.get("execution_reason") not in {"SKIPPED", "UNSUPPORTED"}
    ]

    cases_per_backend = 48 if exhaustive else 9
    scheduled = len(discovered) * cases_per_backend
    executed = sum(result.get("execution_status") == "EXECUTED" for result in results)
    not_executed = len(results) - executed

    reasons = {
        reason: sum(1 for result in results if result.get("execution_reason") == reason)
        for reason in sorted(
            {result.get("execution_reason") for result in results} - {None}
        )
    }

    summary = {
        "scheduled": scheduled,
        "executed": executed,
        "not_executed": not_executed,
        "execution_status": {
            "EXECUTED": executed,
            "NOT_EXECUTED": not_executed,
        },
        "execution_reasons": reasons,
        "semantic_status": {
            "PASS": sum(result.get("semantic_status") == "PASS" for result in results),
            "FAIL": sum(result.get("semantic_status") == "FAIL" for result in results),
            "INCONCLUSIVE": sum(
                result.get("semantic_status") == "INCONCLUSIVE" for result in results
            ),
            "NOT_EVALUATED": sum(
                result.get("semantic_status") == "NOT_EVALUATED" for result in results
            ),
        },
        "semantic_pass": sum(
            result.get("status") == "SEMANTIC_PASS" for result in results
        ),
        "semantic_fail": sum(
            result.get("status") == "SEMANTIC_FAIL" for result in results
        ),
        "pass_with_findings": sum(
            result.get("status") == "PASS_WITH_FINDINGS" for result in results
        ),
        "errors": len(errors),
        "skipped": sum(result.get("status") == "SKIPPED" for result in results),
        "unsupported": sum(result.get("status") == "UNSUPPORTED" for result in results),
    }

    return {
        "status": "PASS_WITH_FINDINGS" if findings or errors else "PASS",
        "results": results,
        "dimensions": matrix_dimensions(),
        "summary": summary,
        "matrix_note": (
            "Each case is routed through a concrete BackendAdapter and "
            "records execution status/reason separately from semantic "
            "status. Differential agreement is not correctness proof."
        ),
    }
