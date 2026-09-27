import json
from pathlib import Path

import numpy as np

from shap_review.contracts.tensor import SHAPSemanticTensor
from shap_review.differential.semantic import evaluate_additivity
from shap_review.fuzzing.backend_matrix import execute_installed_backend_matrix
from shap_review.fuzzing.protocol_campaign import ProtocolCampaign
from shap_review.version import SCHEMA_VERSION, VERSION


def test_v14_version():
    assert VERSION
    assert SCHEMA_VERSION


def test_canonical_semantic_tensor_multiclass():
    values = np.ones((3, 4, 2))
    base = np.zeros((3, 2))
    t = SHAPSemanticTensor.from_values(values, base_values=base)
    assert t.axis_spec.feature_axis == 1
    assert t.axis_spec.output_axis == 2
    assert t.reduce_contributions().shape == (3, 2)


def test_canonical_semantic_tensor_interaction():
    values = np.ones((2, 3, 3, 2))
    base = np.zeros((2, 2))
    t = SHAPSemanticTensor.from_values(values, base_values=base, interaction=True)
    assert t.axis_spec.interaction_feature_axes == (1, 2)
    assert t.reduce_contributions().shape == (2, 2)


def test_differential_uses_canonical_tensor():
    values = np.ones((2, 3, 2))
    base = np.zeros((2, 2))
    target = np.full((2, 2), 3.0)
    result = evaluate_additivity(values, base, target)
    assert result["passed"] is True


def test_backend_statuses_are_semantic():
    result = execute_installed_backend_matrix()
    assert result["status"] in {"PASS", "PASS_WITH_FINDINGS", "SKIPPED"}
    for row in result["results"]:
        assert row["status"] in {
            "SEMANTIC_PASS",
            "SEMANTIC_FAIL",
            "PASS_WITH_FINDINGS",
            "SKIPPED",
            "ERROR",
            "UNSUPPORTED",
            "ADAPTER_ERROR",
            "BACKEND_ERROR",
            "SHAP_ERROR",
            "TOOLKIT_ERROR",
        }


def test_protocol_scenarios_are_reported():
    out = ProtocolCampaign(seed=0).run(iterations=6)
    assert out["scenarios"]
    assert set(out["scenario_stats"]) == set(out["scenarios"])


def test_release_metadata_consistency():
    root = Path(__file__).parents[2]
    for rel in [
        "plugins/shap-review-toolkit/.claude-plugin/plugin.json",
        "adapters/claude/tool-schema.json",
        "adapters/openai/tool-schema.json",
        "adapters/openai/tool-definition.json",
        "adapters/gemini/tool-schema.json",
        "adapters/gemini/function-declaration.json",
    ]:
        data = json.loads((root / rel).read_text())
        assert data.get("version") == VERSION
        assert data.get("schema_version", SCHEMA_VERSION) == SCHEMA_VERSION
