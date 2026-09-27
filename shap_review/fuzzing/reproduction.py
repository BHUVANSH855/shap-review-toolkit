from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from shap_review.reproduction.environment import snapshot


@dataclass(frozen=True)
class ReproductionResult:
    case_id: str
    runs: int
    stable: bool
    outcomes: list[str]
    current_version: str | None
    notes: str
    environment: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


def reproduce_callable(
    case_id: str,
    runner: Callable[[], Any],
    runs: int = 2,
    metadata: dict[str, Any] | None = None,
) -> ReproductionResult:
    """Repeat a deterministic finding runner and record whether the outcome is stable."""
    outcomes = []
    count = max(2, runs)
    for _ in range(count):
        try:
            outcomes.append(json.dumps(runner(), sort_keys=True, default=str))
        except Exception as exc:
            outcomes.append(f"EXCEPTION:{type(exc).__name__}:{exc}")
    try:
        import shap

        version = getattr(shap, "__version__", None)
    except Exception:
        version = None
    meta = dict(metadata or {})
    meta.setdefault("argv", __import__("sys").argv[:])
    meta.setdefault("cwd", __import__("os").getcwd())
    meta.setdefault("seed", meta.get("seed"))
    meta.setdefault("model_id", meta.get("model_id"))
    meta.setdefault("input_hash", meta.get("input_hash"))
    meta.setdefault("exact_command", " ".join(__import__("sys").argv))
    return ReproductionResult(
        case_id,
        count,
        len(set(outcomes)) == 1,
        outcomes,
        version,
        "Repeated in the current environment.",
        snapshot(),
        meta,
    )


def write_reproduction_bundle(result: ReproductionResult, path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(result), fh, indent=2, sort_keys=True)
    return path
