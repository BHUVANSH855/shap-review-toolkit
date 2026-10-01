import subprocess
import sys
from pathlib import Path


def test_distribution_hygiene_script():
    root = Path(__file__).resolve().parents[1]
    validator = root / "scripts" / "validate_distribution.py"

    result = subprocess.run(
        [sys.executable, str(validator)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_canonical_distribution_contract():
    from shap_review.version import CAPABILITIES, SCHEMA_VERSION, VERSION

    assert VERSION
    assert SCHEMA_VERSION
    assert len(CAPABILITIES) == 24
    assert "evidence" in CAPABILITIES


def test_release_metadata_is_synchronized():
    import json
    from pathlib import Path

    from shap_review.version import SCHEMA_VERSION, VERSION

    root = Path(__file__).parents[1]
    assert VERSION and SCHEMA_VERSION
    assert f'version = "{VERSION}"' in (root / "pyproject.toml").read_text()
    plugin = json.loads(
        (root / "plugins/shap-review-toolkit/.claude-plugin/plugin.json").read_text()
    )
    assert plugin["version"] == VERSION
    # Commands live in the commands/ directory, not in plugin.json
    # (Claude Code schema does not allow commands array in plugin.json)
    commands_dir = root / "plugins/shap-review-toolkit/commands"
    assert (commands_dir / "fuzz-protocol.md").exists(), "fuzz-protocol command missing"


def test_cli_reports_canonical_metadata():
    from shap_review.cli import dispatch
    from shap_review.version import SCHEMA_VERSION, VERSION

    assert dispatch("version")["version"] == VERSION
    assert dispatch("capabilities")["schema_version"] == SCHEMA_VERSION
    assert dispatch("evidence")["supported"] is True


def test_cli_capability_contract_exposes_current_commands():
    from shap_review.cli import dispatch

    result = dispatch("capabilities")
    assert "fuzz-protocol" in result["commands"]
    assert "semantic-oracle" in result["commands"]
