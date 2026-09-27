import pytest

pytest.importorskip("shap")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")
import subprocess
import sys
from pathlib import Path


def test_fuzzer_dimensions_are_execution_backed():
    from shap_review.fuzzing.engine import TreeExplainerFuzzer

    r = TreeExplainerFuzzer(seed=123).run(25)
    assert r["executed_cases"] == 25
    assert r["validated_cases"] == 25
    assert r["coverage"]["interaction_executed"] >= 1


def test_history_reports_changed_symbols(tmp_path: Path):
    from shap_review.discovery.history import HistoryScanner

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "x.py").write_text("def old_name():\n    return 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=T",
            "-c",
            "user.email=t@e",
            "commit",
            "-m",
            "initial",
        ],
        check=True,
        capture_output=True,
    )
    (tmp_path / "x.py").write_text("def new_name():\n    return 2\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=T",
            "-c",
            "user.email=t@e",
            "commit",
            "-m",
            "fix #4911",
        ],
        check=True,
        capture_output=True,
    )
    r = HistoryScanner().analyze_issue(tmp_path, 4911)
    assert "new_name" in r["semantic_summary"]["symbols_changed"]
    assert "old_name" in r["semantic_summary"]["symbols_changed"]


def test_adapter_contract_is_common():
    from adapters.claude.adapter import ClaudeAdapter
    from adapters.gemini.adapter import GeminiAdapter
    from adapters.openai.adapter import OpenAIAdapter

    adapters = [ClaudeAdapter(), OpenAIAdapter(), GeminiAdapter()]
    assert len({tuple(a.capabilities.commands) for a in adapters}) == 1
    for a in adapters:
        r = a.invoke("capabilities", {})
        assert r["provider"] == a.provider
        assert (
            r["schema_version"]
            == __import__(
                "shap_review.version", fromlist=["SCHEMA_VERSION"]
            ).SCHEMA_VERSION
        )


def test_distribution_hygiene_script():
    p = Path(__file__).parents[2] / "scripts" / "validate_distribution.py"
    r = subprocess.run(
        [sys.executable, str(p)], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_adapter_version_capability_is_common():
    from adapters.claude.adapter import ClaudeAdapter
    from adapters.gemini.adapter import GeminiAdapter
    from adapters.openai.adapter import OpenAIAdapter

    for adapter in (ClaudeAdapter(), OpenAIAdapter(), GeminiAdapter()):
        result = adapter.invoke("version", {})
        assert (
            result["toolkit_version"]
            == __import__("shap_review.version", fromlist=["VERSION"]).VERSION
        )
        assert (
            result["schema_version"]
            == __import__(
                "shap_review.version", fromlist=["SCHEMA_VERSION"]
            ).SCHEMA_VERSION
        )


def test_differential_contract_cannot_be_hidden_by_generic_match(tmp_path: Path):
    from shap_review.differential.runner import differential_scripts

    left = tmp_path / "left.py"
    right = tmp_path / "right.py"
    left.write_text(
        "import json; print(json.dumps({'values':[1.0], 'base_values':[0.0]}))\n"
    )
    right.write_text(
        "import json; print(json.dumps({'values':[1.0], 'base_values':[0.5]}))\n"
    )
    result = differential_scripts(left, right)
    assert result["comparison"]["equal"] is False
    assert result["contract_comparison"]["equal"] is False
    assert result["equal"] is False
