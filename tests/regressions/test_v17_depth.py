import pytest

pytest.importorskip("shap")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")
from shap_review.fuzzing.backend_matrix import matrix_dimensions
from shap_review.fuzzing.protocol_campaign import (
    ProtocolCampaign,
    make_default_treeexplainer_protocol_target,
)
from shap_review.version import SCHEMA_VERSION, VERSION


def test_v17_metadata():
    assert VERSION
    assert SCHEMA_VERSION


def test_scenarios_have_distinct_runtime_boundaries():
    target = make_default_treeexplainer_protocol_target()
    result = ProtocolCampaign(seed=3).run(
        target_callable=target,
        iterations=6,
        scenarios=("shape-dtype", "array-coercion", "indexing"),
    )
    boundaries = {
        r["case"]["scenario"]: r["scenario_boundary"]
        for r in result["results"]
        if r["scenario_boundary"]
    }
    assert "np.asarray" in boundaries["array-coercion"]
    assert "obj[0]" in boundaries["indexing"]
    assert "shape/dtype" in boundaries["shape-dtype"]


def test_nested_reentry_has_parent_and_child_ids():
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


def test_mutation_events_are_typed():
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


def test_matrix_dimensions_are_explicit():
    dims = matrix_dimensions()
    assert "sparse" in dims["input_representation"]
    assert "log_loss" in dims["model_output"]
    assert True in dims["interaction"]


def test_protocol_coverage_separates_harness_and_target():
    target = make_default_treeexplainer_protocol_target()
    result = ProtocolCampaign(seed=1).run(target_callable=target, iterations=10)
    assert "harness_triggered_protocols" in result
    assert "target_observed_protocols" in result
    assert "shap_observed_protocols" in result
    assert (
        result["target_observed_coverage_percent"]
        >= result["shap_observed_coverage_percent"]
    )


def test_backend_status_taxonomy_and_log_loss_dimension():
    from shap_review.fuzzing.backend_matrix import STATUSES

    assert (
        "SHAP_ERROR" in STATUSES
        and "BACKEND_ERROR" in STATUSES
        and "ADAPTER_ERROR" in STATUSES
    )
    assert "log_loss" in matrix_dimensions()["model_output"]


def test_reproduction_runner_is_stable():
    from shap_review.fuzzing.reproduction import reproduce_callable

    result = reproduce_callable("demo", lambda: {"ok": True}, runs=2)
    assert result.stable and result.runs == 2


def test_protocol_event_has_exact_shap_call_provenance():
    target = make_default_treeexplainer_protocol_target()
    result = ProtocolCampaign(seed=4).run(target_callable=target, iterations=10)
    detailed = [
        e for r in result["results"] for e in r.get("protocol_events_detail", [])
    ]
    shap_events = [e for e in detailed if e.get("causal_to_active_shap_call")]
    assert all(
        e.get("shap_call_id") and e.get("callback_event_id") and e.get("shap_function")
        for e in shap_events
    )
    assert all(e.get("causal_confidence") == "high" for e in shap_events)


def test_natural_reentry_requires_exact_shap_callback():
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


def test_cli_parser_exposes_fuzz_treeexplainer():
    # parser-level availability is exercised by the subprocess-style dispatch path in the CLI
    import subprocess
    import sys

    p = subprocess.run(
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
    assert p.returncode == 0, p.stderr


def test_catboost_candidate_regression_is_explicit():
    from shap_review.regressions.suite import run_catboost_interventional

    r = run_catboost_interventional()
    assert r.id == "SHAP-CATBOOST-INTERVENTIONAL-RECON"
    if r.reproduced:
        assert r.details["confirmation_status"] == "candidate_only"


def test_mutation_semantics_distinguish_protocol_metadata_from_storage():
    import random

    from shap_review.fuzzing.generators.protocol import ProtocolObject

    shape_obj = ProtocolObject(random.Random(1), "shape", mutation="shape")
    _ = shape_obj.shape
    m = shape_obj.mutation_events[-1]
    assert m["protocol_shape_changed"] is True
    assert m["storage_shape_changed"] is False
    dtype_obj = ProtocolObject(random.Random(1), "dtype", mutation="dtype")
    _ = dtype_obj.dtype
    d = dtype_obj.mutation_events[-1]
    assert d["protocol_dtype_changed"] is True
    assert d["storage_dtype_changed"] is False
    write_obj = ProtocolObject(random.Random(1), "shape", mutation="writeability")
    _ = write_obj.shape
    w = write_obj.mutation_events[-1]
    assert w["writeable_before"] is True and w["writeable_after"] is False
