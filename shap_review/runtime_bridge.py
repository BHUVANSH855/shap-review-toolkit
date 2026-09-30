"""Runtime validation bridge for the static SHAP review pipeline.

The bridge executes a bounded TreeExplainer campaign against the installed
SHAP runtime and exposes runtime observations separately from static
candidates.

Runtime evidence is intentionally conservative:

* runtime anomalies are not static findings by themselves;
* bug-class similarity is not sufficient for candidate correlation;
* candidate-specific fingerprints are required before dynamic evidence is
  attached to a candidate;
* runtime evidence never clears ``validation_required``;
* campaign provenance is retained so maintainers can reproduce the observation;
* unsupported configurations and resource limits are not classified as bugs.

The bridge therefore acts as an evidence producer, not as an automatic
finding confirmer.
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

from shap_review.evidence.chain import build_chain
from shap_review.evidence.model import EvidenceItem, EvidenceKind, EvidenceOrigin

log = logging.getLogger(__name__)

_BRIDGE_SEED = 42
_BRIDGE_ITERATIONS = 8
_BRIDGE_PRODUCER = "runtime-bridge"


def _runtime_available() -> bool:
    """Return whether all runtime dependencies required by the bridge exist."""
    for module_name in ("shap", "sklearn", "pandas", "numpy"):
        try:
            import_module(module_name)
        except ImportError:
            return False
    return True


def _package_versions() -> dict[str, str | None]:
    """Return versions of packages participating in the runtime campaign."""
    versions: dict[str, str | None] = {}

    for module_name in ("shap", "numpy", "pandas", "sklearn"):
        try:
            module = import_module(module_name)
        except ImportError:
            versions[module_name] = None
        else:
            versions[module_name] = getattr(module, "__version__", None)

    return versions


def _environment_fingerprint() -> str:
    """Return a stable fingerprint for the runtime environment."""
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": _package_versions(),
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _case_fingerprint(case: dict[str, Any]) -> str:
    """Return a stable fingerprint for one generated fuzz case."""
    encoded = json.dumps(
        case,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _candidate_fingerprint(candidate: Candidate) -> str:
    """Return a stable identity for a static candidate."""
    payload = {
        "candidate_id": candidate.candidate_id,
        "bug_class": candidate.bug_class,
        "invariant": candidate.invariant,
        "file": candidate.file,
        "line": candidate.line,
        "symbol": candidate.symbol,
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _shap_provenance() -> dict[str, Any]:
    """Collect provenance for the installed SHAP runtime."""
    try:
        shap = import_module("shap")
    except ImportError:
        return {
            "shap_version": None,
            "shap_source_path": None,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "environment_fingerprint": _environment_fingerprint(),
        }

    source_path = getattr(shap, "__file__", None)

    return {
        "shap_version": getattr(shap, "__version__", None),
        "shap_source_path": source_path,
        "shap_source_root": (
            str(source_path.rsplit("/", 1)[0])
            if source_path and "/" in str(source_path)
            else None
        ),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "environment_fingerprint": _environment_fingerprint(),
    }


def _empty_bridge_result(
    provenance: dict[str, Any],
    *,
    available: bool,
    error: str | None = None,
) -> dict[str, Any]:
    """Construct a consistent empty bridge result."""
    result: dict[str, Any] = {
        "available": available,
        "results": [],
        "anomalies": [],
        "classifications": [],
        "provenance": provenance,
        "campaign_summary": {
            "iterations": 0,
            "executed": 0,
            "validated": 0,
            "failures": 0,
            "timeouts": 0,
        },
    }

    if error:
        result["error"] = error

    return result


def run_bridge(
    iterations: int = _BRIDGE_ITERATIONS,
    seed: int = _BRIDGE_SEED,
) -> dict[str, Any]:
    """Run a bounded runtime campaign and return provenance-rich observations.

    The campaign operates against the installed SHAP runtime. It does not
    inspect or execute the repository under static review.

    Runtime anomalies are deliberately kept separate from static candidates.
    Candidate correlation must be established explicitly before an anomaly
    can become candidate evidence.
    """
    iterations = max(0, int(iterations))
    seed = int(seed)

    provenance = _shap_provenance()

    if not _runtime_available():
        return _empty_bridge_result(
            provenance,
            available=False,
            error="required runtime dependencies are unavailable",
        )

    try:
        from shap_review.fuzzing.engine import TreeExplainerFuzzer
    except ImportError as exc:
        log.debug("Runtime bridge fuzzer import failed: %s", exc)
        return _empty_bridge_result(
            provenance,
            available=True,
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        campaign = TreeExplainerFuzzer(seed=seed).run(iterations=iterations)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        log.debug("Runtime bridge campaign failed: %s", exc)
        return _empty_bridge_result(
            provenance,
            available=True,
            error=f"{type(exc).__name__}: {exc}",
        )

    execution_id = hashlib.sha256(
        (
            f"{_BRIDGE_PRODUCER}:{seed}:{iterations}:"
            f"{provenance['environment_fingerprint']}"
        ).encode()
    ).hexdigest()

    campaign_provenance = {
        **provenance,
        "execution_id": execution_id,
        "producer": _BRIDGE_PRODUCER,
        "seed": seed,
        "iterations": iterations,
    }

    results: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []
    classifications: list[dict[str, Any]] = []

    for item in campaign.get("results", []):
        case = item.get("case", {})
        execution = item.get("execution", {})
        oracle = item.get("oracle", {})

        input_fingerprint = _case_fingerprint(case)

        record = {
            **item,
            "provenance": {
                **campaign_provenance,
                "input_fingerprint": input_fingerprint,
                # A bridge campaign is not candidate-specific unless an
                # explicit candidate fingerprint is supplied by the caller.
                "candidate_fingerprint": None,
            },
        }

        results.append(record)

        if not execution.get("executed"):
            continue

        if oracle.get("valid", True):
            continue

        classification = _classify_anomaly(execution, oracle)

        anomaly = {
            **record,
            "classification": classification,
        }

        anomalies.append(anomaly)
        classifications.append(classification)

    campaign_summary = campaign.get("provenance", {})

    return {
        "available": True,
        "results": results,
        "anomalies": anomalies,
        "classifications": classifications,
        "provenance": campaign_provenance,
        "campaign_summary": {
            "iterations": campaign.get("iterations", iterations),
            "executed": campaign.get("executed_cases", 0),
            "validated": campaign.get("validated_cases", 0),
            "failures": campaign.get("failures", 0),
            "timeouts": campaign.get("timeout_cases", 0),
            "unique_failures": campaign.get("unique_failures", 0),
            "duplicate_failures": campaign.get("duplicate_failures", 0),
            "campaign_execution_id": campaign_provenance.get("execution_id"),
            "campaign_target_fingerprint": campaign_summary.get("target_fingerprint"),
        },
    }


def _traceback_contains_shap_frame(traceback: str) -> bool:
    """Return True only when a SHAP module frame appears in the traceback.

    Keyword matching on exception messages is too broad — sklearn, numpy and
    pandas raise messages containing "tree", "explainer" and "shap" for
    reasons unrelated to SHAP bugs.  Checking for a SHAP file path in the
    traceback frame lines is a much stronger signal: it means execution was
    actually inside SHAP code when the exception occurred.

    Frame lines in Python tracebacks look like:
      File "/path/to/shap/_explanation.py", line 42, in shap_values
    We match on the file path segment only, not on the function name or
    message, to avoid false positives from error messages quoting SHAP APIs.
    """
    if not traceback:
        return False
    for line in traceback.splitlines():
        # Match only "File ..." lines — not exception message lines.
        stripped = line.strip()
        if not stripped.startswith("File "):
            continue
        # The path segment is between the first pair of quotes.
        if (
            '"/shap/' in stripped
            or "\\shap\\" in stripped
            or "/site-packages/shap/" in stripped
        ):
            return True
        # Editable installs: path contains "shap" as a directory component.
        # Be conservative: require it to be a directory boundary, not a
        # substring of another package name (e.g. "reshap", "shapper").
        import re as _re

        if _re.search(r"[/\\]shap[/\\]", stripped):
            return True
    return False


def _classify_anomaly(
    exec_result: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    """Classify a runtime anomaly without overstating its meaning.

    Classification is deliberately conservative. A runtime exception does
    not automatically imply a specific SHAP bug class unless the evidence
    strongly identifies the failure mode.  Exception message keyword matching
    is intentionally avoided — use traceback frame inspection instead so that
    sklearn/numpy/pandas errors whose messages happen to contain "shap",
    "tree", or "explainer" are not misclassified as SHAP bugs.
    """
    if exec_result.get("timeout"):
        return {
            "kind": "TIMEOUT",
            "bug_class": None,
            "confidence": 0.0,
            "requires_followup": True,
        }

    if exec_result.get("failed"):
        exc_type = str(exec_result.get("exception", ""))
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
                "requires_followup": False,
            }

        if exc_type in {"MemoryError", "RecursionError"}:
            return {
                "kind": "RESOURCE_LIMIT",
                "bug_class": None,
                "confidence": 0.0,
                "requires_followup": True,
            }

        if _traceback_contains_shap_frame(exec_result.get("traceback", "")):
            return {
                "kind": "SHAP_EXCEPTION",
                "bug_class": None,
                "confidence": 0.25,
                "requires_followup": True,
            }

        return {
            "kind": "UNKNOWN_EXCEPTION",
            "bug_class": None,
            "confidence": 0.0,
            "requires_followup": True,
        }

    if not oracle.get("additivity_passed", True):
        error = float(exec_result.get("max_additivity_error", float("inf")))
        tolerance = float(exec_result.get("additivity_tolerance", 1e-5))

        if error > tolerance * 100:
            return {
                "kind": "ADDITIVITY_FAIL",
                "bug_class": "SHAP-01",
                "confidence": 0.65,
                "requires_followup": True,
            }

        return {
            "kind": "ADDITIVITY_BORDERLINE",
            "bug_class": "SHAP-01",
            "confidence": 0.3,
            "requires_followup": True,
        }

    if not exec_result.get("input_unchanged", True):
        return {
            "kind": "INPUT_MUTATION",
            "bug_class": "SHAP-05",
            "confidence": 0.7,
            "requires_followup": True,
        }

    if not exec_result.get("finite", True):
        return {
            "kind": "NON_FINITE_OUTPUT",
            "bug_class": None,
            "confidence": 0.35,
            "requires_followup": True,
        }

    return {
        "kind": "ORACLE_FAIL_UNKNOWN",
        "bug_class": None,
        "confidence": 0.0,
        "requires_followup": True,
    }


def _correlates_with_candidate(
    candidate: Candidate,
    anomaly: dict[str, Any],
) -> bool:
    """Return whether runtime evidence is explicitly correlated.

    Correlation requires an exact candidate fingerprint. Bug-class equality,
    file equality, message similarity, or execution similarity alone is not
    sufficient.
    """
    provenance = anomaly.get("provenance", {})
    expected_fingerprint = provenance.get("candidate_fingerprint")

    if not expected_fingerprint:
        return False

    return expected_fingerprint == _candidate_fingerprint(candidate)


def _candidate_dynamic_evidence(
    candidate: Candidate,
    anomaly: dict[str, Any],
) -> EvidenceItem:
    """Convert one correlated runtime anomaly into an evidence item."""
    classification = anomaly.get("classification", {})
    execution = anomaly.get("execution", {})
    case = anomaly.get("case", {})
    provenance = anomaly.get("provenance", {})

    confidence = float(classification.get("confidence", 0.0))

    return EvidenceItem(
        kind=EvidenceKind.DYNAMIC,
        source=_BRIDGE_PRODUCER,
        claim=(
            f"Runtime campaign detected {classification.get('kind', 'UNKNOWN')} "
            f"on a correlated case: "
            f"n_features={case.get('n_features')}, "
            f"classification={case.get('classification')}, "
            f"model_output={case.get('model_output')}"
        ),
        passed=False,
        confidence=confidence,
        origin=EvidenceOrigin.EXECUTION,
        evidence_id=(
            f"bridge-{candidate.candidate_id}-"
            f"{provenance.get('execution_id', 'unknown')}-"
            f"{provenance.get('input_fingerprint', 'unknown')[:16]}"
        ),
        details={
            "kind": classification.get("kind"),
            "bug_class": classification.get("bug_class"),
            "requires_followup": classification.get("requires_followup", True),
            "max_additivity_error": execution.get("max_additivity_error"),
            "additivity_tolerance": execution.get("additivity_tolerance"),
            "input_unchanged": execution.get("input_unchanged"),
            "finite": execution.get("finite"),
            "timeout": execution.get("timeout", False),
            "case": case,
            "execution_id": provenance.get("execution_id"),
            "fixture_id": provenance.get("fixture_id"),
            "input_fingerprint": provenance.get("input_fingerprint"),
            "candidate_fingerprint": provenance.get("candidate_fingerprint"),
            "repository_revision": provenance.get("repository_revision"),
            "environment_fingerprint": provenance.get("environment_fingerprint"),
            "shap_source_path": provenance.get("shap_source_path"),
            "shap_version": provenance.get("shap_version"),
            "python_executable": provenance.get("python_executable"),
            "python_version": provenance.get("python_version"),
            "platform": provenance.get("platform"),
            "packages": provenance.get("packages"),
            "producer": provenance.get("producer"),
            "bridge_seed": provenance.get("seed"),
            "bridge_iterations": provenance.get("iterations"),
        },
    )


def _chain_entries(candidate: Candidate) -> list[dict[str, Any]]:
    """Serialize an existing candidate evidence chain."""
    return [
        {
            "kind": (item.kind.value if hasattr(item.kind, "value") else item.kind),
            "source": item.source,
            "claim": item.claim,
            "passed": item.passed,
            "confidence": item.confidence,
            "origin": (
                item.origin.value if hasattr(item.origin, "value") else item.origin
            ),
            "evidence_id": item.evidence_id,
            "details": getattr(item, "details", {}),
        }
        for item in candidate.evidence_chain.items
    ]


def attach_dynamic_evidence(
    candidates: list[Candidate],
    bridge_result: dict[str, Any],
) -> list[Candidate]:
    """Attach only explicitly correlated runtime evidence.

    Uncorrelated anomalies remain available in ``bridge_result`` but are never
    attached to unrelated candidates.

    Runtime evidence also never changes ``validation_required``. A dynamic
    observation strengthens the evidence record; it does not replace the
    maintainer's need for candidate-specific validation.
    """
    if not bridge_result.get("available"):
        return candidates

    anomalies = bridge_result.get("anomalies", [])
    if not anomalies:
        return candidates

    from shap_review.types import Candidate as CandidateType

    updated: list[Candidate] = []

    for candidate in candidates:
        correlated = [
            anomaly
            for anomaly in anomalies
            if _correlates_with_candidate(candidate, anomaly)
            and anomaly.get("classification", {}).get("bug_class")
            == candidate.bug_class
            and float(anomaly.get("classification", {}).get("confidence", 0.0)) > 0
        ]

        if not correlated:
            updated.append(candidate)
            continue

        best = max(
            correlated,
            key=lambda item: float(
                item.get("classification", {}).get("confidence", 0.0)
            ),
        )

        dynamic_item = _candidate_dynamic_evidence(candidate, best)

        entries = _chain_entries(candidate)

        # Do not duplicate the same runtime evidence if attach_dynamic_evidence
        # is called more than once with the same bridge result.
        existing_ids = {
            entry.get("evidence_id") for entry in entries if entry.get("evidence_id")
        }

        if dynamic_item.evidence_id not in existing_ids:
            entries.append(
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

        new_chain = build_chain(entries=entries)
        score = new_chain.evidence_strength_score()

        if score >= 1.5:
            new_confidence = "high"
        elif score >= 0.75:
            new_confidence = "medium"
        else:
            new_confidence = candidate.confidence

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
                    set(candidate.tags) | {"dynamic-evidence", "runtime-bridge"}
                ),
                evidence_chain=new_chain,
            )
        )

    return updated
