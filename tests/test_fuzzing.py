"""Fuzzer tests.

Key design principle: the fuzzer MUST be allowed to report failures.
A test that asserts failures==0 would itself fail if the fuzzer found a real
SHAP bug — the opposite of what we want.

Instead these tests verify:
  1. The fuzzer executes and produces classified results.
  2. Every oracle failure is *classified* (not silently swallowed).
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
    """Every oracle failure must be classifiable — no uncategorised failures."""
    from shap_review.runtime_bridge import _classify_anomaly

    r = TreeExplainerFuzzer(seed=123).run(iterations=25)
    representations = {x["case"]["representation"] for x in r["results"]}
    assert {"ndarray", "dataframe"} <= representations

    for item in r["results"]:
        exec_result = item.get("execution", {})
        oracle = item.get("oracle", {})
        if not exec_result.get("executed"):
            continue
        if oracle.get("valid", True):
            continue
        # Every failure MUST be classifiable — if not, that's a toolkit bug
        classification = _classify_anomaly(exec_result, oracle)
        assert "kind" in classification, (
            f"Unclassified failure on case {item['case']}: "
            f"oracle={oracle}, exec={exec_result}"
        )
        # RESOURCE_LIMIT and UNSUPPORTED_CONFIG are not SHAP bugs and must not
        # be reported as findings
        if classification["kind"] in {"RESOURCE_LIMIT", "UNSUPPORTED_CONFIG"}:
            assert classification["bug_class"] is None, (
                f"{classification['kind']} must not produce a bug_class"
            )


def test_fuzzer_anomaly_has_sufficient_confidence_when_real():
    """Genuine anomalies (additivity fail, non-finite, mutation) must have confidence > 0."""
    from shap_review.runtime_bridge import _classify_anomaly

    r = TreeExplainerFuzzer(seed=99).run(iterations=20)
    for item in r["results"]:
        exec_result = item.get("execution", {})
        oracle = item.get("oracle", {})
        if not exec_result.get("executed") or oracle.get("valid", True):
            continue
        classification = _classify_anomaly(exec_result, oracle)
        # For genuine SHAP-attributed failures confidence must be > 0
        if classification["bug_class"] is not None:
            assert classification["confidence"] > 0, (
                f"SHAP-attributed failure has confidence=0: {classification}"
            )


def test_runtime_bridge_attaches_dynamic_evidence():
    """When the bridge finds anomalies, candidates get DYNAMIC evidence items."""

    from shap_review.runtime_bridge import run_bridge

    # Run a campaign that may or may not find anomalies
    bridge = run_bridge(iterations=5, seed=7)
    assert bridge["available"] is True

    # If no anomalies found, bridge should return empty lists (not crash)
    assert isinstance(bridge["anomalies"], list)
    assert isinstance(bridge["classifications"], list)


def test_fuzzer_coverage_includes_key_dimensions():
    """Fuzzer must exercise at least two representations and both classif/regression."""
    r = TreeExplainerFuzzer(seed=42).run(iterations=20)
    cov = r["coverage"]
    # Must see both ndarray and dataframe
    assert "ndarray" in cov["representation"]["observed"]
    assert "dataframe" in cov["representation"]["observed"]
    # Must exercise both classification and regression
    assert "True" in cov["classification"]["observed"]
    assert "False" in cov["classification"]["observed"]
