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
    representations = {item["case"]["representation"] for item in result["results"]}

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
                f"SHAP-attributed failure has confidence=0: {classification}"
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
        assert (
            provenance["target_fingerprint"]
            == (result["provenance"]["target_fingerprint"])
        )
        assert provenance["input_fingerprint"]

        assert provenance["producer"] == "treeexplainer-fuzzer"
        assert provenance["seed"] == 23
        assert provenance["iterations"] == 3


def test_fuzzer_target_fingerprint_is_stable_for_same_configuration():
    """Equivalent campaigns identify the same runtime target consistently."""
    first = TreeExplainerFuzzer(seed=31).run(iterations=1)
    second = TreeExplainerFuzzer(seed=31).run(iterations=1)

    assert (
        first["provenance"]["target_fingerprint"]
        == (second["provenance"]["target_fingerprint"])
    )


def test_fuzzer_execution_id_changes_with_campaign_parameters():
    """Different campaign parameters must not share the same execution ID."""
    first = TreeExplainerFuzzer(seed=41).run(iterations=1)
    second = TreeExplainerFuzzer(seed=42).run(iterations=1)

    assert first["provenance"]["execution_id"] != (second["provenance"]["execution_id"])


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


def test_protocol_every_declared_protocol_is_triggerable():
    from shap_review.fuzzing.generators.protocol import PROTOCOLS
    from shap_review.fuzzing.protocol_campaign import ProtocolCampaign

    campaign = ProtocolCampaign(1)
    result = campaign.run(iterations=len(PROTOCOLS) * 3)

    assert result["coverage_percent"] == 100.0
    assert all(result["protocol_coverage"].values())


def test_backend_matrix_has_primary_tree_backends():
    from shap_review.fuzzing.backend_matrix import (
        DEFAULT_BACKENDS,
        matrix_dimensions,
    )

    names = {backend.name for backend in DEFAULT_BACKENDS}

    assert {"sklearn", "xgboost", "lightgbm", "catboost"} <= names
    assert "backend" in matrix_dimensions()


def test_protocol_campaign_records_real_protocol_events():
    from shap_review.fuzzing.protocol_campaign import ProtocolCampaign

    def target(obj, case):
        _ = obj.shape
        if case["protocol"] == "call":
            obj()

    result = ProtocolCampaign(1).run(target, 10)
    assert result["executed"] == 10
    assert "shape" in result["protocols_observed"]
    assert all("protocol_events" in item for item in result["results"])


def test_protocol_campaign_self_test_marks_harness_execution():
    from shap_review.fuzzing.protocol_campaign import ProtocolCampaign

    result = ProtocolCampaign(4).run(iterations=10)
    assert result["mode"] == "harness-self-test"
    assert "seed" in result


def test_protocol_campaign_reports_requested_and_observed_adversarial_modes():
    from shap_review.fuzzing.protocol_campaign import ProtocolCampaign

    result = ProtocolCampaign(1).run(iterations=15)
    assert result["mutation_requested"] > 0
    assert result["reentry_requested"] > 0
    assert result["mutation_observed"] > 0
    assert result["reentry_observed"] > 0


def test_protocol_campaign_scenarios_report_runtime_boundaries():
    target = __import__(
        "shap_review.fuzzing.protocol_campaign",
        fromlist=["make_default_treeexplainer_protocol_target"],
    ).make_default_treeexplainer_protocol_target()
    from shap_review.fuzzing.protocol_campaign import ProtocolCampaign

    result = ProtocolCampaign(seed=3).run(
        target_callable=target,
        iterations=6,
        scenarios=("shape-dtype", "array-coercion", "indexing"),
    )
    boundaries = {
        item["case"]["scenario"]: item["scenario_boundary"]
        for item in result["results"]
        if item["scenario_boundary"]
    }
    assert "np.asarray" in boundaries["array-coercion"]
    assert "obj[0]" in boundaries["indexing"]
    assert "shape/dtype" in boundaries["shape-dtype"]


def test_protocol_reentry_records_nested_execution_without_harness_callback():
    import random

    from shap_review.fuzzing.generators.protocol import ProtocolObject

    parent = ProtocolObject(
        random.Random(0),
        "shape",
        reentry=True,
        reentry_target=lambda obj: obj.events.append("nested-execution"),
    )
    parent.state["outer_execution_id"] = "outer-1"
    parent._touch("shape")
    assert parent.reentered
    assert parent.reentry_depth == 0
    assert not any(
        e.endswith(":shap") for e in parent.events if e.startswith("reentry:")
    )
    assert any(e.startswith("reentry:shape") for e in parent.events)


def test_protocol_mutation_events_are_typed():
    import random

    from shap_review.fuzzing.generators.protocol import ProtocolObject

    obj = ProtocolObject(random.Random(0), "shape", mutation="on-access")
    _ = obj.shape
    mutated = obj.mutation_events
    assert mutated
    assert {e["type"] for e in mutated} <= {
        "value",
        "protocol_metadata",
        "strides",
        "buffer_replacement",
        "payload_identity",
        "writeability",
    }
    assert all(e["changed"] for e in mutated)
    assert all(e["value_before"] != e["value_after"] for e in mutated)
    assert all(e["object_identity_changed"] is False for e in mutated)


def test_backend_matrix_dimensions_are_explicit():
    from shap_review.fuzzing.backend_matrix import matrix_dimensions

    dimensions = matrix_dimensions()
    assert "sparse" in dimensions["input_representation"]
    assert "log_loss" in dimensions["model_output"]
    assert True in dimensions["interaction"]


def test_protocol_coverage_separates_harness_and_target():
    from shap_review.fuzzing.protocol_campaign import (
        ProtocolCampaign,
        make_default_treeexplainer_protocol_target,
    )

    target = make_default_treeexplainer_protocol_target()
    result = ProtocolCampaign(seed=1).run(target_callable=target, iterations=10)
    assert "harness_triggered_protocols" in result
    assert "target_observed_protocols" in result
    assert "shap_observed_protocols" in result
    assert (
        result["target_observed_coverage_percent"]
        >= result["shap_observed_coverage_percent"]
    )


def test_backend_status_taxonomy_includes_failure_stages():
    from shap_review.fuzzing.backend_matrix import STATUSES, matrix_dimensions

    assert {"SHAP_ERROR", "BACKEND_ERROR", "ADAPTER_ERROR"}.issubset(STATUSES)
    assert "log_loss" in matrix_dimensions()["model_output"]


def test_protocol_event_preserves_exact_shap_call_provenance():
    from shap_review.fuzzing.protocol_campaign import (
        ProtocolCampaign,
        make_default_treeexplainer_protocol_target,
    )

    result = ProtocolCampaign(seed=4).run(
        target_callable=make_default_treeexplainer_protocol_target(), iterations=10
    )
    events = [
        event
        for row in result["results"]
        for event in row.get("protocol_events_detail", [])
    ]
    shap_events = [event for event in events if event.get("causal_to_active_shap_call")]
    assert all(
        event.get("shap_call_id")
        and event.get("callback_event_id")
        and event.get("shap_function")
        for event in shap_events
    )
    assert all(event.get("causal_confidence") == "high" for event in shap_events)


def test_natural_reentry_requires_an_exact_shap_callback():
    import random

    from shap_review.fuzzing.generators.protocol import ProtocolObject

    obj = ProtocolObject(
        random.Random(0),
        "shape",
        reentry=True,
        reentry_target=lambda o: o.events.append("nested"),
        allow_harness_reentry=False,
    )
    obj._touch("shape")
    assert not obj.reentered


def test_fuzz_treeexplainer_cli_command_is_available():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "shap_review.cli",
            "fuzz-treeexplainer",
            "--iterations",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_catboost_candidate_regression_is_explicit():
    from shap_review.regressions.suite import run_catboost_interventional

    result = run_catboost_interventional()
    assert result.id == "SHAP-CATBOOST-INTERVENTIONAL-RECON"
    if result.reproduced:
        assert result.details["confirmation_status"] == "candidate_only"


def test_protocol_mutations_distinguish_metadata_from_storage():
    import random

    from shap_review.fuzzing.generators.protocol import ProtocolObject

    shape_obj = ProtocolObject(random.Random(1), "shape", mutation="shape")
    _ = shape_obj.shape
    shape_event = shape_obj.mutation_events[-1]
    assert shape_event["protocol_shape_changed"] is True
    assert shape_event["storage_shape_changed"] is False

    dtype_obj = ProtocolObject(random.Random(1), "dtype", mutation="dtype")
    _ = dtype_obj.dtype
    dtype_event = dtype_obj.mutation_events[-1]
    assert dtype_event["protocol_dtype_changed"] is True
    assert dtype_event["storage_dtype_changed"] is False

    write_obj = ProtocolObject(random.Random(1), "shape", mutation="writeability")
    _ = write_obj.shape
    write_event = write_obj.mutation_events[-1]
    assert (
        write_event["writeable_before"] is True
        and write_event["writeable_after"] is False
    )


def test_backend_execution_summary_exposes_not_executed_reasons():
    from shap_review.fuzzing.backend_matrix import execute_installed_backend_matrix

    result = execute_installed_backend_matrix()
    assert "not_executed" in result["summary"]
    assert "execution_reasons" in result["summary"]


def test_catboost_reproducer_schema_is_explicit():
    from shap_review.regressions.suite import catboost_candidate_reproducer

    result = catboost_candidate_reproducer()
    assert result["id"] == "SHAP-CATBOOST-INTERVENTIONAL-RECON"
    assert "shap_version" in result and "catboost_version" in result
    assert result["confirmation_status"] in {"candidate_only", "confirmed", "unknown"}


def test_cross_version_catboost_validation_is_not_run_without_explicit_interpreters():
    from shap_review.regressions.suite import cross_version_catboost_validation

    result = cross_version_catboost_validation()
    assert result["status"] == "NOT_RUN"
    assert result["confirmation_status"] == "candidate_only"


def test_real_treeexplainer_protocol_target_has_boundary_metadata():
    from shap_review.fuzzing.protocol_campaign import (
        make_default_treeexplainer_protocol_target,
    )

    target = make_default_treeexplainer_protocol_target()
    assert getattr(target, "__shap_boundary__", None) == "TreeExplainer.shap_values"


def test_treeexplainer_harness_uses_canonical_reducer():
    from pathlib import Path

    text = (
        Path(__file__).parents[1] / "shap_review/fuzzing/harnesses/treeexplainer.py"
    ).read_text()
    assert "def _sum_shap" not in text
    assert "SHAPSemanticTensor" in text


def test_backend_discovery_cli_reports_matrix():
    from shap_review.cli import dispatch

    result = dispatch("fuzz-backends", arguments={"execute": False})
    assert "backends" in result and "matrix" in result


def test_semantic_oracle_cli_dispatches_a_valid_contract():
    from shap_review.cli import dispatch

    payload = {
        "contract": {
            "explainer": "TreeExplainer",
            "model_family": "tree",
            "model_output": "raw",
            "values_shape": [-1, 2],
            "target_shape": [-1],
            "additivity_required": True,
            "output_space_required": True,
            "tolerance": 1e-6,
        },
        "values": [[1, 2], [3, 4]],
        "base_values": [0, 0],
        "target": [3, 7],
    }
    result = dispatch("semantic-oracle", arguments=payload)
    assert result["valid"] is True


def test_protocol_event_uses_nearest_shap_frame_as_canonical_source():
    import random

    from shap_review.fuzzing.generators.protocol import ProtocolObject

    obj = ProtocolObject(random.Random(0), "shape")
    obj.begin_shap_call("call-1", "TreeExplainer.shap_values", "wrapper-source")
    obj._runtime_provenance = lambda: {
        "stack": ["shap.foo"],
        "shap_frames": ["shap.foo"],
        "shap_frame_observed": True,
        "nearest_shap_frame": "shap.foo",
    }
    obj._touch("shape")
    event = obj._protocol_event_records[-1]
    assert event["canonical_source"] == "shap.foo"
    assert event["shap_call_id"] == "call-1"
    assert event["callback_event_id"] == event["event_id"]
