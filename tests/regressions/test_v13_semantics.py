import json
from pathlib import Path

import numpy as np

from shap_review.contracts.oracles import InteractionOracle
from shap_review.contracts.shap_contract import SHAPContract, validate_contract
from shap_review.version import SCHEMA_VERSION, VERSION


def test_v13_version():
    assert VERSION
    assert SCHEMA_VERSION


def test_interaction_axes_reject_unequal_feature_pair():
    values = np.zeros((2, 3, 4, 5))
    result = InteractionOracle().check(values=values, interaction_values=values)
    assert result.passed is False
    assert "different sizes" in result.reason


def test_contract_shape_wildcard():
    c = SHAPContract(
        explainer="TreeExplainer",
        model_family="tree",
        values_shape=(-1, 3),
        additivity_required=False,
        output_space_required=False,
    )
    out = validate_contract(c, np.zeros((7, 3)))
    assert out["values_shape_match"]


def test_provider_versions_match_release():
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
