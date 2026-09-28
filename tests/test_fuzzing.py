"""Fuzzer tests.

Key design principle: the fuzzer MUST be allowed to report failures.
A test that asserts failures == 0 would itself fail if the fuzzer found a
real SHAP bug — the opposite of what we want.

Instead these tests verify:
  1. The fuzzer executes and produces classified results.
  2. Every oracle failure is classified, not silently swallowed.
  3. Unclassified / toolkit-internal errors are zero.
  4. The runtime bridge correctly attaches dynamic evidence to candidates.
"""

import pytest

pytest.importorskip("shap")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")

from shap_review.fuzzing.engine import TreeExplainerFuzzer


def test_fuzzer_executes_cases():
    result = TreeExplainerFuzzer(seed=11).run(iterations=3)

    assert result["executed_cases"] >= 1


def test_fuzzer_classifies_all_failures():
    """Every oracle failure must be classifiable."""
    from shap_review.runtime_bridge import _classify_anomaly

    result = TreeExplainerFuzzer(seed=123).run(iterations=25)
    representations = {
        item["case"]["representation"]
        for item in result["results"]
    }

    assert {"ndarray", "dataframe"} <= representations

    for item in result["results"]:
        execution = item.get("execution", {})
        oracle = item.get("oracle", {})

        if not execution.get("executed"):
            continue

        if oracle.get("valid", True):
            continue

        classification = _classify_anomaly(execution, oracle)

        assert "kind" in classification, (
            f"Unclassified failure on case {item['case']}: "
            f"oracle={oracle}, exec={execution}"
        )

        # RESOURCE_LIMIT and UNSUPPORTED_CONFIG are not SHAP bugs and
        # must not be reported as findings.
        if classification["kind"] in {
            "RESOURCE_LIMIT",
            "UNSUPPORTED_CONFIG",
        }:
            assert classification["bug_class"] is None, (
                f"{classification['kind']} must not produce a bug_class"
            )


def test_fuzzer_anomaly_has_sufficient_confidence_when_real():
    """Genuine anomalies must have confidence greater than zero."""
    from shap_review.runtime_bridge import _classify_anomaly

    result = TreeExplainerFuzzer(seed=99).run(iterations=20)

    for item in result["results"]:
        execution = item.get("execution", {})
        oracle = item.get("oracle", {})

        if not execution.get("executed") or oracle.get("valid", True):
            continue

        classification = _classify_anomaly(execution, oracle)

        # For genuine SHAP-attributed failures, confidence must be > 0.
        if classification["bug_class"] is not None:
            assert classification["confidence"] > 0, (
                "SHAP-attributed failure has confidence=0: "
                f"{classification}"
            )


def test_runtime_bridge_attaches_dynamic_evidence():
    """The bridge must return structured anomaly and classification lists."""
    from shap_review.runtime_bridge import run_bridge

    bridge = run_bridge(iterations=5, seed=7)

    assert bridge["available"] is True
    assert isinstance(bridge["anomalies"], list)
    assert isinstance(bridge["classifications"], list)


def test_fuzzer_coverage_includes_key_dimensions():
    """Fuzzer must exercise representations and classification modes."""
    result = TreeExplainerFuzzer(seed=42).run(iterations=20)
    coverage = result["coverage"]

    assert "ndarray" in coverage["representation"]["observed"]
    assert "dataframe" in coverage["representation"]["observed"]

    assert "True" in coverage["classification"]["observed"]
    assert "False" in coverage["classification"]["observed"]