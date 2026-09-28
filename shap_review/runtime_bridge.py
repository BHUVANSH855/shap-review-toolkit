"""Runtime validation bridge for the static SHAP review pipeline.

The bridge executes a bounded TreeExplainer campaign and exposes runtime
observations separately from static candidates. Runtime observations are only
attached to candidates when their provenance explicitly correlates with the
candidate. Uncorrelated runtime anomalies remain campaign evidence and are not
used to weaken candidate validation requirements.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shap_review.types import Candidate

log = logging.getLogger(__name__)

_BRIDGE_SEED = 42
_BRIDGE_ITERATIONS = 8


def _runtime_available() -> bool:
    """Return whether the runtime dependencies required by the bridge exist."""
    for module_name in ("shap", "sklearn", "pandas", "numpy"):
        try:
            import_module(module_name)
        except ImportError:
            return False
    return True


def _environment_fingerprint() -> str:
    """Return a stable fingerprint for the current runtime environment."""
    versions: dict[str, str | None] = {}

    for module_name in ("shap", "numpy", "pandas", "sklearn"):
        try:
            module = import_module(module_name)
            versions[module_name] = getattr(module, "__version__", None)
        except ImportError:
            versions[module_name] = None

    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": versions,
    }

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _case_fingerprint(case: dict[str, Any]) -> str:
    """Return a stable fingerprint for one generated fuzz case."""
    encoded = json.dumps(case, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _candidate_fingerprint(candidate: Candidate) -> str:
    """Return a stable identity for the static candidate being reviewed."""
    payload = {
        "candidate_id": candidate.candidate_id,
        "bug_class": candidate.bug_class,
        "invariant": candidate.invariant,
        "file": candidate.file,
        "line": candidate.line,
        "symbol": candidate.symbol,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _shap_provenance() -> dict[str, Any]:
    """Collect the installed SHAP runtime provenance."""
    try:
        shap = import_module("shap")
    except ImportError:
        return {
            "shap_version": None,
            "shap_source_path": None,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "environment_fingerprint": _environment_fingerprint(),
        }

    return {
        "shap_version": getattr(shap, "__version__", None),
        "shap_source_path": getattr(shap, "__file__", None),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "environment_fingerprint": _environment_fingerprint(),
    }


def run_bridge(
    iterations: int = _BRIDGE_ITERATIONS,
    seed: int = _BRIDGE_SEED,
) -> dict[str, Any]:
    """Run a bounded runtime campaign and return provenance-rich observations.

    Runtime observations are intentionally independent of static candidate
    matching. The caller must establish candidate-specific correlation before
    attaching an observation to a candidate.
    """
    provenance = _shap_provenance()

    if not _runtime_available():
        return {
            "available": False,
            "results": [],
            "anomalies": [],
            "classifications": [],
            "provenance": provenance,
        }

    try:
        from shap_review.fuzzing.engine import TreeExplainerFuzzer
    except ImportError as exc:
        log.debug("Runtime bridge fuzzer import failed: %s", exc)
        return {
            "available": True,
            "results": [],
            "anomalies": [],
            "classifications": [],
            "provenance": provenance,
            "error": str(exc),
        }

    try:
        campaign = TreeExplainerFuzzer(seed=seed).run(iterations=iterations)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        log.debug("Runtime bridge campaign failed: %s", exc)
        return {
            "available": True,
            "results": [],
            "anomalies": [],
            "classifications": [],
            "provenance": provenance,
            "error": str(exc),
        }

    execution_id = hashlib.sha256(
        f"runtime-bridge:{seed}:{iterations}:{provenance['environment_fingerprint']}"
        .encode()
    ).hexdigest()

    results: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []

    for item in campaign["results"]:
        case = item.get("case", {})
        exec_result = item.get("execution", {})
        oracle = item.get("oracle", {})

        record = {
            **item,
            "provenance": {
                **provenance,
                "execution_id": execution_id,
                "input_fingerprint": _case_fingerprint(case),
                "producer": "runtime-bridge",
                "seed": seed,
                "iterations": iterations,
            },
        }
        results.append(record)

        if not exec_result.get("executed"):
            continue

        if oracle.get("valid", True):
            continue

        classification = _classify_anomaly(exec_result, oracle)
        anomaly = {
            **record,
            "classification": classification,
        }
        anomalies.append(anomaly)
        classifications.append(classification)

    return {
        "available": True,
        "results": results,
        "anomalies": anomalies,
        "classifications": classifications,
        "provenance": {
            **provenance,
            "execution_id": execution_id,
            "producer": "runtime-bridge",
            "seed": seed,
            "iterations": iterations,
        },
        "campaign_summary": {
            "iterations": campaign["iterations"],
            "executed": campaign["executed_cases"],
            "validated": campaign["validated_cases"],
            "failures": campaign["failures"],
        },
    }


def _classify_anomaly(
    exec_result: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    """Classify an oracle failure without overstating its meaning."""
    if exec_result.get("failed"):
        exc_type = exec_result.get("exception", "")
        message = str(exec_result.get("message", "")).lower()

        if (
            exc_type == "NotImplementedError"
            or "not supported" in message
            or "unsupported" in message
        ):
            return {
                "kind": "UNSUPPORTED_CONFIG",
                "bug_class": None,
                "confidence": 0.0,
            }

        if exc_type in {"MemoryError", "RecursionError"}:
            return {
                "kind": "RESOURCE_LIMIT",
                "bug_class": None,
                "confidence": 0.0,
            }

        if "shap" in message or "tree" in message or "explainer" in message:
            return {
                "kind": "SHAP_EXCEPTION",
                "bug_class": "SHAP-01",
                "confidence": 0.5,
            }

        return {
            "kind": "UNKNOWN_EXCEPTION",
            "bug_class": None,
            "confidence": 0.0,
        }

    if not oracle.get("additivity_passed", True):
        error = exec_result.get("max_additivity_error", float("inf"))
        tolerance = exec_result.get("additivity_tolerance", 1e-5)

        if error > tolerance * 100:
            return {
                "kind": "ADDITIVITY_FAIL",
                "bug_class": "SHAP-01",
                "confidence": 0.65,
            }

        return {
            "kind": "ADDITIVITY_BORDERLINE",
            "bug_class": "SHAP-01",
            "confidence": 0.3,
        }

    if not exec_result.get("input_unchanged", True):
        return {
            "kind": "INPUT_MUTATION",
            "bug_class": "SHAP-05",
            "confidence": 0.7,
        }

    if not exec_result.get("finite", True):
        return {
            "kind": "NON_FINITE_OUTPUT",
            "bug_class": "SHAP-01",
            "confidence": 0.5,
        }

    return {
        "kind": "ORACLE_FAIL_UNKNOWN",
        "bug_class": None,
        "confidence": 0.0,
    }


def _correlates_with_candidate(
    candidate: Candidate,
    anomaly: dict[str, Any],
) -> bool:
    """Return whether runtime evidence is explicitly correlated to a candidate.

    The bridge does not infer correlation merely from bug class. A future
    candidate-specific runtime campaign may provide a matching fingerprint.
    """
    provenance = anomaly.get("provenance", {})
    expected_fingerprint = provenance.get("candidate_fingerprint")

    if not expected_fingerprint:
        return False

    return expected_fingerprint == _candidate_fingerprint(candidate)


def attach_dynamic_evidence(
    candidates: list[Candidate],
    bridge_result: dict[str, Any],
) -> list[Candidate]:
    """Attach only explicitly correlated runtime evidence to candidates.

    Uncorrelated runtime anomalies remain available through ``bridge_result`` but
    are never attached to unrelated candidates, never promoted by bug-class
    coincidence, and never used to clear ``validation_required``.
    """
    if not bridge_result.get("available") or not bridge_result.get("anomalies"):
        return candidates

    from shap_review.evidence.chain import build_chain
    from shap_review.evidence.model import EvidenceItem, EvidenceKind, EvidenceOrigin
    from shap_review.types import Candidate as CandidateType

    updated: list[Candidate] = []

    for candidate in candidates:
        correlated = [
            anomaly
            for anomaly in bridge_result["anomalies"]
            if _correlates_with_candidate(candidate, anomaly)
            and anomaly.get("classification", {}).get("bug_class")
            == candidate.bug_class
            and anomaly.get("classification", {}).get("confidence", 0) > 0
        ]

        if not correlated:
            updated.append(candidate)
            continue

        best = max(
            correlated,
            key=lambda item: item["classification"]["confidence"],
        )

        classification = best["classification"]
        execution = best.get("execution", {})
        case = best.get("case", {})
        provenance = best.get("provenance", {})

        dynamic_item = EvidenceItem(
            kind=EvidenceKind.DYNAMIC,
            source="runtime-bridge",
            claim=(
                f"Runtime campaign detected {classification['kind']} on the "
                f"correlated case: n_features={case.get('n_features')}, "
                f"classification={case.get('classification')}, "
                f"model_output={case.get('model_output')}"
            ),
            passed=False,
            confidence=classification["confidence"],
            origin=EvidenceOrigin.EXECUTION,
            evidence_id=(
                f"bridge-{candidate.candidate_id}-"
                f"{provenance.get('execution_id', 'unknown')}"
            ),
            details={
                "kind": classification["kind"],
                "max_additivity_error": execution.get("max_additivity_error"),
                "input_unchanged": execution.get("input_unchanged"),
                "finite": execution.get("finite"),
                "case": case,
                "execution_id": provenance.get("execution_id"),
                "fixture_id": provenance.get("fixture_id"),
                "input_fingerprint": provenance.get("input_fingerprint"),
                "candidate_fingerprint": provenance.get("candidate_fingerprint"),
                "repository_revision": provenance.get("repository_revision"),
                "environment_fingerprint": provenance.get(
                    "environment_fingerprint"
                ),
                "shap_source_path": provenance.get("shap_source_path"),
                "shap_version": provenance.get("shap_version"),
                "python_executable": provenance.get("python_executable"),
                "python_version": provenance.get("python_version"),
                "producer": provenance.get("producer"),
                "bridge_seed": provenance.get("seed"),
                "bridge_iterations": provenance.get("iterations"),
            },
        )

        existing_entries = [
            {
                "kind": (
                    item.kind.value
                    if hasattr(item.kind, "value")
                    else item.kind
                ),
                "source": item.source,
                "claim": item.claim,
                "passed": item.passed,
                "confidence": item.confidence,
                "origin": (
                    item.origin.value
                    if hasattr(item.origin, "value")
                    else item.origin
                ),
                "evidence_id": item.evidence_id,
            }
            for item in candidate.evidence_chain.items
        ]

        existing_entries.append(
            {
                "kind": "dynamic",
                "source": dynamic_item.source,
                "claim": dynamic_item.claim,
                "passed": False,
                "confidence": dynamic_item.confidence,
                "origin": "execution",
                "evidence_id": dynamic_item.evidence_id,
                "details": dynamic_item.details,
            }
        )

        new_chain = build_chain(entries=existing_entries)

        score = new_chain.evidence_strength_score()
        new_confidence = (
            "high"
            if score >= 1.5
            else "medium"
            if score >= 0.75
            else candidate.confidence
        )

        updated.append(
            CandidateType(
                candidate_id=candidate.candidate_id,
                bug_class=candidate.bug_class,
                invariant=candidate.invariant,
                file=candidate.file,
                line=candidate.line,
                symbol=candidate.symbol,
                message=candidate.message,
                evidence=candidate.evidence,
                validation_required=candidate.validation_required,
                confidence=new_confidence,
                tags=sorted(
                    set(candidate.tags)
                    | {"dynamic-evidence", "runtime-bridge"}
                ),
                evidence_chain=new_chain,
            )
        )

    return updated