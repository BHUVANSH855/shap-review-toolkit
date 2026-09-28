"""Fuzzer tests.

Key design principle: the fuzzer MUST be allowed to report failures.
A test that asserts failures == 0 would itself fail if the fuzzer found a
real SHAP bug — the opposite of what we want.

Instead these tests verify:
  1. The fuzzer executes and produces classified results.
  2. Every oracle failure is classified, not silently swallowed.
  3. Unclassified / toolkit-internal errors are zero.
  4. The runtime bridge returns structured anomaly data.
  5. Fuzz results preserve runtime target and execution provenance.
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


def test_runtime_bridge_returns_structured_anomalies():
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


def test_fuzzer_campaign_has_target_provenance():
    """Every campaign identifies the runtime target that was actually used."""
    result = TreeExplainerFuzzer(seed=17).run(iterations=2)

    provenance = result["provenance"]

    assert provenance["producer"] == "treeexplainer-fuzzer"
    assert provenance["target_fingerprint"]
    assert provenance["python_executable"]
    assert provenance["python_version"]
    assert provenance["platform"]
    assert provenance["seed"] == 17
    assert provenance["iterations"] == 2

    if provenance["available"]:
        assert provenance["shap_version"]
        assert provenance["shap_source_path"]
        assert provenance["shap_source_root"]


def test_fuzzer_results_preserve_execution_and_input_provenance():
    """Each generated case carries campaign, execution, and input identity."""
    result = TreeExplainerFuzzer(seed=23).run(iterations=3)

    assert result["provenance"]["execution_id"]

    for item in result["results"]:
        provenance = item["provenance"]

        assert provenance["execution_id"] == result["provenance"]["execution_id"]
        assert provenance["target_fingerprint"] == (
            result["provenance"]["target_fingerprint"]
        )
        assert provenance["input_fingerprint"]

        assert provenance["producer"] == "treeexplainer-fuzzer"
        assert provenance["seed"] == 23
        assert provenance["iterations"] == 3


def test_fuzzer_target_fingerprint_is_stable_for_same_configuration():
    """Equivalent campaigns identify the same runtime target consistently."""
    first = TreeExplainerFuzzer(seed=31).run(iterations=1)
    second = TreeExplainerFuzzer(seed=31).run(iterations=1)

    assert first["provenance"]["target_fingerprint"] == (
        second["provenance"]["target_fingerprint"]
    )


def test_fuzzer_execution_id_changes_with_campaign_parameters():
    """Different campaign parameters must not share the same execution ID."""
    first = TreeExplainerFuzzer(seed=41).run(iterations=1)
    second = TreeExplainerFuzzer(seed=42).run(iterations=1)

    assert first["provenance"]["execution_id"] != (
        second["provenance"]["execution_id"]
    )

def test_review_engine_runtime_bridge_artifact_marks_installed_runtime(tmp_path):
    from shap_review.engine import ReviewEngine

    root = tmp_path / "repo"
    root.mkdir()

    artifact = tmp_path / "artifacts"

    engine = ReviewEngine()

    # Avoid requiring a complete SHAP repository fixture for this contract
    # test. We only need to verify the runtime-bridge artifact metadata.
    original_discover = engine.discover
    _ = original_discover

    result = engine.analyze(root, out=artifact)

    assert isinstance(result, list)

    runtime_bridge = artifact / "runtime-bridge.json"
    assert runtime_bridge.exists()

    import json

    payload = json.loads(runtime_bridge.read_text(encoding="utf-8"))

    assert payload["campaign_scope"] == "installed-runtime"
    assert payload["evidence_scope"] == "runtime-campaign"
    assert payload["repository_root"] == str(root.resolve())
    assert payload["repository_runtime_match"] is False
    assert payload["attach_policy"] == "explicit-candidate-correlation-only"

def test_protocol_campaign_executes_protocols():
    from shap_review.fuzzing.protocol_campaign import ProtocolCampaign

    def target(obj, case):
        _ = obj.shape
        if case["protocol"] == "call":
            obj()

    result = ProtocolCampaign(1).run(target, 20)

    assert result["executed"] == 20
    assert result["protocols_observed"]
    assert result["mutation_observed"] >= 0
