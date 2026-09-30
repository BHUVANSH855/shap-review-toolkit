from __future__ import annotations

from dataclasses import dataclass

from shap_review.types import Candidate


@dataclass
class InvestigationPlan:
    candidate_id: str
    steps: list[str]


_NATIVE_BUG_CLASSES = {
    "SHAP-06",
    "SHAP-10",
}

_NATIVE_TAGS = {
    "native",
    "borrowed-reference",
    "buffer-validation",
    "exception-propagation",
    "reentry",
    "cuda",
    "nanobind",
}

_DIFFERENTIAL_TAGS = {
    "differential",
    "cpu-gpu",
    "cuda",
}

_TEST_GAP_TAGS = {
    "test-gap",
    "test_gap",
    "missing-test",
    "missing_test",
}


def _contains_any(values: set[str], candidates: set[str]) -> bool:
    return bool(values & candidates)


def plan(candidate: Candidate) -> InvestigationPlan:
    """Build a candidate-specific investigation workflow.

    The planner deliberately remains prescriptive rather than executing any
    investigation itself.  Execution belongs to the corresponding discovery,
    oracle, differential, fuzzing, reproduction, and sanitizer subsystems.

    The returned steps are ordered from cheapest/static validation toward
    stronger runtime evidence so that investigators do not jump directly to
    expensive confirmation workflows.
    """
    tags = {str(tag).lower() for tag in candidate.tags}
    bug_class = str(candidate.bug_class).upper()
    invariant = str(candidate.invariant).upper()

    steps = [
        "inspect source context around the candidate location",
        "trace callers and relevant data/control flow",
    ]

    if bug_class == "SHAP-01" or "ATTR" in invariant or "additivity" in tags:
        steps.extend(
            [
                "identify the model output space and expected-value semantics",
                "reconstruct the model output from SHAP values plus the baseline",
                "validate the reconstruction with the semantic additivity oracle",
            ]
        )

    elif bug_class == "SHAP-02" or "OUT" in invariant or "output-space" in tags:
        steps.extend(
            [
                "identify the requested model-output space",
                "trace model-output, baseline, and SHAP-value transformations",
                "validate output-space semantics with the contract oracle",
                "compare reconstruction against an independent model output",
            ]
        )

    elif bug_class == "SHAP-03" or "SHAPE" in invariant or "shape" in tags:
        steps.extend(
            [
                "identify the expected tensor axes and output dimensionality",
                "trace shape transformations across the explanation path",
                "validate the result with the semantic tensor contract",
                "exercise single-output and multi-output cases where applicable",
            ]
        )

    elif bug_class == "SHAP-04" or "dispatch" in tags:
        steps.extend(
            [
                "identify the dispatch predicate and supported model/input combination",
                "trace the selected implementation and fallback path",
                "test supported and unsupported combinations explicitly",
                "verify that unsupported combinations are rejected rather than misclassified",
            ]
        )

    elif bug_class == "SHAP-05" or "INPUT" in invariant or "dtype" in tags:
        steps.extend(
            [
                "identify the input representation and dtype entering SHAP",
                "trace conversions between Python, NumPy, pandas, and native representations",
                "test nullable, non-contiguous, and unusual dtype representations where applicable",
                "validate input immutability and semantic preservation",
            ]
        )

    elif bug_class in _NATIVE_BUG_CLASSES or _contains_any(tags, _NATIVE_TAGS):
        steps.extend(
            [
                "trace the Python-to-native boundary and ownership assumptions",
                "inspect exception paths and early returns",
                "inspect borrowed-reference, allocation, and lifetime transitions",
                "validate the suspicious path with the native-flow analysis",
                "attempt a minimal runtime reproducer",
                "run ASan/UBSan when the failure reaches native code",
            ]
        )

    elif bug_class == "SHAP-07" or "STATE" in invariant or "state" in tags:
        steps.extend(
            [
                "identify mutable state created during explainer construction",
                "compare state before and after repeated explanation calls",
                "test repeated, reordered, and independent invocations",
                "validate state invariants against an independent prediction",
            ]
        )

    elif bug_class == "SHAP-08" or "numerical" in tags:
        steps.extend(
            [
                "identify the numerical quantity and its expected tolerance",
                "test float32 and float64 representations separately",
                "check finite values, cancellation, overflow, and underflow behavior",
                "validate the result using an independent numerical reconstruction",
            ]
        )

    elif bug_class == "SHAP-09" or "masker" in tags:
        steps.extend(
            [
                "identify the masker contract and expected input representation",
                "trace masker transformations into the explainer",
                "exercise representative and edge-case masker inputs",
                "validate output shape, dtype, and semantic preservation",
            ]
        )

    elif bug_class == "SHAP-10" or "resource" in tags:
        steps.extend(
            [
                "identify the resource being allocated or released",
                "trace success, failure, and early-return cleanup paths",
                "check repeated execution for leaks or stale state",
                "attempt a minimal runtime reproducer",
                "run sanitizer validation when native resources are involved",
            ]
        )

    else:
        steps.extend(
            [
                "compare the observed behavior with the stated invariant",
                "identify the smallest input that can exercise the candidate",
                "attempt a minimal reproducer",
            ]
        )

    if _contains_any(tags, _DIFFERENTIAL_TAGS):
        steps.extend(
            [
                "compare the candidate path against the independent reference implementation",
                "verify that differential agreement is not being mistaken for correctness",
            ]
        )

    if _contains_any(tags, _TEST_GAP_TAGS):
        steps.extend(
            [
                "identify the missing regression assertion",
                "add a focused regression test covering the candidate invariant",
            ]
        )

    steps.extend(
        [
            "record runtime, differential, reproduction, or sanitizer evidence",
            "classify the candidate as confirmed, inconclusive, or false positive",
            "preserve the evidence and provenance needed for maintainer review",
        ]
    )

    return InvestigationPlan(candidate.candidate_id, steps)
