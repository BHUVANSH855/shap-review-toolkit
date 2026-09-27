import json
from pathlib import Path

import numpy as np

from shap_review.cli import dispatch
from shap_review.contracts.tensor import SHAPAxisSpec, SHAPSemanticTensor
from shap_review.version import CAPABILITIES, SCHEMA_VERSION, VERSION


def test_v15_metadata():
    assert VERSION
    assert SCHEMA_VERSION
    assert "semantic-oracle" in CAPABILITIES


def test_axis_roles_reject_conflicts():
    for spec in [
        SHAPAxisSpec(sample_axis=0, feature_axis=0),
        SHAPAxisSpec(sample_axis=0, feature_axis=1, output_axis=0),
        SHAPAxisSpec(sample_axis=0, feature_axis=1, interaction_feature_axes=(0, 2)),
        SHAPAxisSpec(
            sample_axis=0,
            feature_axis=1,
            output_axis=2,
            interaction_feature_axes=(1, 2),
        ),
    ]:
        try:
            spec.normalize(4)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid semantic axis roles were accepted")


def test_reconstruction_uses_canonical_tensor():
    values = np.ones((2, 3, 2))
    base = np.zeros((2, 2))
    tensor = SHAPSemanticTensor.from_values(values, base_values=base)
    assert tensor.reconstruction().shape == (2, 2)
    assert np.allclose(tensor.reconstruction(), 3)


def test_provider_command_sets_match_capabilities():
    root = Path(__file__).parents[2]
    for rel in [
        "plugins/shap-review-toolkit/.claude-plugin/plugin.json",
        "adapters/claude/tool-schema.json",
        "adapters/openai/tool-definition.json",
        "adapters/openai/tool-schema.json",
        "adapters/gemini/function-declaration.json",
        "adapters/gemini/tool-schema.json",
    ]:
        data = json.loads((root / rel).read_text())
        assert data.get("commands") == list(CAPABILITIES)
        try:
            assert data["parameters"]["properties"]["command"]["enum"] == list(
                CAPABILITIES
            )
        except KeyError:
            pass


def test_dispatch_meta_commands():
    assert dispatch("version")["version"] == VERSION
    assert dispatch("capabilities")["schema_version"] == SCHEMA_VERSION
    assert dispatch("evidence")["supported"] is True


def test_semantic_oracle_dispatch():
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
    out = dispatch("semantic-oracle", arguments=payload)
    assert out["valid"] is True


def test_treeexplainer_harness_uses_canonical_reducer():
    text = (
        Path(__file__).parents[2] / "shap_review/fuzzing/harnesses/treeexplainer.py"
    ).read_text()
    assert "def _sum_shap" not in text
    assert "SHAPSemanticTensor" in text
