import json
from pathlib import Path

from shap_review.version import SCHEMA_VERSION, VERSION


def test_release_metadata_is_synchronized():
    root = Path(__file__).parents[2]
    assert VERSION
    assert SCHEMA_VERSION
    assert f'version = "{VERSION}"' in (root / "pyproject.toml").read_text()
    plugin = json.loads(
        (root / "plugins/shap-review-toolkit/.claude-plugin/plugin.json").read_text()
    )
    assert plugin["version"] == VERSION
    assert "fuzz-protocol" in plugin["commands"]


def test_real_treeexplainer_target_is_constructible_when_dependencies_exist():
    try:
        from shap_review.fuzzing.protocol_campaign import (
            make_default_treeexplainer_protocol_target,
        )

        target = make_default_treeexplainer_protocol_target()
        assert getattr(target, "__shap_boundary__", None) == "TreeExplainer.shap_values"
    except ImportError:
        pass
