"""Runtime oracle bridge — connects the fuzzer/harness to the static analysis pipeline.

When SHAP and sklearn are installed, runs a small fixed-seed fuzzing campaign
and converts any oracle failures into DYNAMIC EvidenceItems that are attached
to matching static candidates by bug class.

This bridges the previously disconnected static analysis path and the runtime
oracle path, without redesigning either system.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shap_review.types import Candidate

log = logging.getLogger(__name__)

# Bug classes that runtime additivity failures map to.
_ADDITIVITY_BUG_CLASSES = {"SHAP-01", "SHAP-02"}
_INPUT_BUG_CLASSES = {"SHAP-05"}

_BRIDGE_SEED = 42
_BRIDGE_ITERATIONS = 8  # small enough to be fast; large enough to cover key paths


def _runtime_available() -> bool:
    for mod in ("shap", "sklearn", "pandas", "numpy"):
        try:
            __import__(mod)
        except ImportError:
            return False
    return True


def run_bridge(iterations: int = _BRIDGE_ITERATIONS, seed: int = _BRIDGE_SEED) -> dict:
    """Run a small deterministic fuzzing campaign and return classified results.

    Returns a dict with:
        - 'available': whether SHAP was installed
        - 'results': list of oracle result dicts (only executed cases)
        - 'anomalies': list of cases where oracle failed (genuine signal)
        - 'classifications': list of anomaly classifications
    """
    if not _runtime_available():
        return {
            "available": False,
            "results": [],
            "anomalies": [],
            "classifications": [],
        }

    try:
        from shap_review.fuzzing.engine import TreeExplainerFuzzer

        campaign = TreeExplainerFuzzer(seed=seed).run(iterations=iterations)
    except Exception as exc:
        log.debug("Runtime bridge campaign failed: %s", exc)
        return {
            "available": True,
            "results": [],
            "anomalies": [],
            "classifications": [],
            "error": str(exc),
        }

    anomalies = []
    classifications = []
    for item in campaign["results"]:
        exec_result = item.get("execution", {})
        oracle = item.get("oracle", {})
        if not exec_result.get("executed"):
            continue
        if oracle.get("valid", True):
            continue
        # Classify the anomaly so it isn't confused with toolkit or backend errors
        classification = _classify_anomaly(exec_result, oracle)
        anomalies.append(item)
        classifications.append(classification)

    return {
        "available": True,
        "results": campaign["results"],
        "anomalies": anomalies,
        "classifications": classifications,
        "campaign_summary": {
            "iterations": campaign["iterations"],
            "executed": campaign["executed_cases"],
            "validated": campaign["validated_cases"],
            "failures": campaign["failures"],
        },
    }


def _classify_anomaly(exec_result: dict, oracle: dict) -> dict:
    """Classify why an oracle failed — critical for preventing false positives."""
    if exec_result.get("failed"):
        exc_type = exec_result.get("exception", "")
        msg = exec_result.get("message", "").lower()
        if (
            exc_type in ("NotImplementedError",)
            or "not supported" in msg
            or "unsupported" in msg
        ):
            return {"kind": "UNSUPPORTED_CONFIG", "bug_class": None, "confidence": 0.0}
        if exc_type in ("MemoryError", "RecursionError"):
            return {"kind": "RESOURCE_LIMIT", "bug_class": None, "confidence": 0.0}
        if "shap" in msg or "tree" in msg or "explainer" in msg:
            return {"kind": "SHAP_EXCEPTION", "bug_class": "SHAP-01", "confidence": 0.5}
        return {"kind": "UNKNOWN_EXCEPTION", "bug_class": None, "confidence": 0.0}

    if not oracle.get("additivity_passed", True):
        err = exec_result.get("max_additivity_error", float("inf"))
        tol = exec_result.get("additivity_tolerance", 1e-5)
        if err > tol * 100:
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
        return {"kind": "INPUT_MUTATION", "bug_class": "SHAP-05", "confidence": 0.7}

    if not exec_result.get("finite", True):
        return {"kind": "NON_FINITE_OUTPUT", "bug_class": "SHAP-01", "confidence": 0.5}

    return {"kind": "ORACLE_FAIL_UNKNOWN", "bug_class": None, "confidence": 0.0}


def attach_dynamic_evidence(
    candidates: list[Candidate], bridge_result: dict
) -> list[Candidate]:
    """Attach DYNAMIC EvidenceItems from bridge anomalies to matching static candidates.

    Only attaches when the classification has a non-None bug_class and confidence > 0.
    Evidence is marked EXTERNAL (not derived) because it comes from a separate execution.
    """
    if not bridge_result.get("available") or not bridge_result.get("anomalies"):
        return candidates

    from shap_review.evidence.model import EvidenceItem, EvidenceKind, EvidenceOrigin

    anomalies_by_class: dict[str, list[dict]] = {}
    for anomaly, classification in zip(
        bridge_result["anomalies"], bridge_result["classifications"]
    ):
        bc = classification.get("bug_class")
        if bc and classification.get("confidence", 0) > 0:
            anomalies_by_class.setdefault(bc, []).append(
                {"anomaly": anomaly, "classification": classification}
            )

    if not anomalies_by_class:
        return candidates

    updated = []
    for candidate in candidates:
        matching = anomalies_by_class.get(candidate.bug_class, [])
        if not matching:
            updated.append(candidate)
            continue

        # Build a DYNAMIC evidence item for the first matching anomaly
        best = max(matching, key=lambda x: x["classification"]["confidence"])
        cls = best["classification"]
        exec_r = best["anomaly"].get("execution", {})
        case = best["anomaly"].get("case", {})

        dynamic_item = EvidenceItem(
            kind=EvidenceKind.DYNAMIC,
            source="runtime-bridge",
            claim=(
                f"Fixed-seed fuzzer (seed={_BRIDGE_SEED}) detected "
                f"{cls['kind']} on case: n_features={case.get('n_features')}, "
                f"classification={case.get('classification')}, "
                f"model_output={case.get('model_output')}"
            ),
            passed=False,
            confidence=cls["confidence"],
            origin=EvidenceOrigin.EXTERNAL,  # separate execution, not derived
            evidence_id=f"bridge-{candidate.bug_class}-{_BRIDGE_SEED}",
            details={
                "kind": cls["kind"],
                "max_additivity_error": exec_r.get("max_additivity_error"),
                "input_unchanged": exec_r.get("input_unchanged"),
                "finite": exec_r.get("finite"),
                "case": case,
                "bridge_seed": _BRIDGE_SEED,
                "bridge_iterations": len(bridge_result["results"]),
            },
        )

        # Rebuild the evidence chain to include the dynamic item
        from shap_review.evidence.chain import build_chain

        existing_entries = [
            {
                "kind": item.kind.value if hasattr(item.kind, "value") else item.kind,
                "source": item.source,
                "claim": item.claim,
                "passed": item.passed,
                "confidence": item.confidence,
                "origin": item.origin.value
                if hasattr(item.origin, "value")
                else item.origin,
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
                "origin": "external",
                "evidence_id": dynamic_item.evidence_id,
                "details": dynamic_item.details,
            }
        )
        new_chain = build_chain(entries=existing_entries)

        # Create updated candidate with new chain and elevated confidence
        from shap_review.types import Candidate as CandidateType

        new_conf = (
            "high"
            if new_chain.evidence_strength_score() >= 1.5
            else "medium"
            if new_chain.evidence_strength_score() >= 0.75
            else candidate.confidence
        )
        updated_candidate = CandidateType(
            candidate_id=candidate.candidate_id,
            bug_class=candidate.bug_class,
            invariant=candidate.invariant,
            file=candidate.file,
            line=candidate.line,
            symbol=candidate.symbol,
            message=candidate.message,
            evidence=candidate.evidence,
            validation_required=False,  # dynamic evidence reduces requirement
            confidence=new_conf,
            tags=sorted(set(candidate.tags) | {"dynamic-evidence", "runtime-bridge"}),
            evidence_chain=new_chain,
        )
        updated.append(updated_candidate)

    return updated
