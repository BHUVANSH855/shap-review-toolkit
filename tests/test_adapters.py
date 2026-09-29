import json
from pathlib import Path

import pytest

from adapters import ClaudeAdapter, GeminiAdapter, OpenAIAdapter


def test_all_frontends_share_capability_contract():
    adapters = [
        ClaudeAdapter(),
        OpenAIAdapter(),
        GeminiAdapter(),
    ]

    capabilities = [
        adapter.invoke("capabilities", {})
        for adapter in adapters
    ]

    assert {capability["provider"] for capability in capabilities} == {
        "claude",
        "openai",
        "gemini",
    }

    assert all(
        "differential" in capability["commands"]
        for capability in capabilities
    )
    assert all(
        "sanitizer" in capability["commands"]
        for capability in capabilities
    )

    assert (
        capabilities[0]["commands"]
        == capabilities[1]["commands"]
        == capabilities[2]["commands"]
    )


def test_backend_adapter_reports_actual_failure_stage():
    from shap_review.backends.adapter import MatrixBackendAdapter
    from shap_review.fuzzing.backend_matrix import BackendSpec

    class Model:
        pass

    adapter = MatrixBackendAdapter(
        BackendSpec("x", "x", (), True, "1"),
        Model(),
    )
    adapter.prepare = lambda **kwargs: {}

    def failing_fit(context):
        raise RuntimeError("fit broke")

    adapter.fit = failing_fit

    result = adapter.execute_case()

    assert result["stage"] == "fit"
    assert result["execution_reason"] == "BACKEND_ERROR"


def test_backend_rejects_invalid_regression_output_contract():
    from shap_review.backends.adapter import MatrixBackendAdapter
    from shap_review.fuzzing.backend_matrix import BackendSpec

    class Model:
        pass

    adapter = MatrixBackendAdapter(
        BackendSpec("x", "x", (), True, "1"),
        Model(),
    )

    supported, reason = adapter.supports_case(
        classification=False,
        model_output="log_loss",
        interaction=False,
    )

    assert supported is False
    assert "regression" in reason


def test_backend_failure_stage_is_preserved():
    from shap_review.backends.adapter import MatrixBackendAdapter
    from shap_review.fuzzing.backend_matrix import BackendSpec

    class Model:
        pass

    adapter = MatrixBackendAdapter(
        BackendSpec("x", "x", (), True, "1"),
        Model(),
    )
    adapter.prepare = lambda **kwargs: {}

    def failing_fit(context):
        raise RuntimeError("fit broke")

    adapter.fit = failing_fit

    result = adapter.execute_case()

    assert result["stage"] == "fit"
    assert result["execution_reason"] == "BACKEND_ERROR"
    assert result["status"] == "BACKEND_ERROR"

def test_adapter_contract_is_common():
    from adapters.claude.adapter import ClaudeAdapter
    from adapters.gemini.adapter import GeminiAdapter
    from adapters.openai.adapter import OpenAIAdapter

    adapters = [ClaudeAdapter(), OpenAIAdapter(), GeminiAdapter()]

    assert len({tuple(adapter.capabilities.commands) for adapter in adapters}) == 1

    for adapter in adapters:
        result = adapter.invoke("capabilities", {})

        assert result["provider"] == adapter.provider
        assert (
            result["schema_version"]
            == __import__(
                "shap_review.version",
                fromlist=["SCHEMA_VERSION"],
            ).SCHEMA_VERSION
        )


def test_adapter_version_capability_is_common():
    from adapters.claude.adapter import ClaudeAdapter
    from adapters.gemini.adapter import GeminiAdapter
    from adapters.openai.adapter import OpenAIAdapter
    from shap_review.version import SCHEMA_VERSION, VERSION

    for adapter in (ClaudeAdapter(), GeminiAdapter(), OpenAIAdapter()):
        result = adapter.invoke("version", {})

        assert result["toolkit_version"] == VERSION
        assert result["schema_version"] == SCHEMA_VERSION

def test_provider_versions_expose_canonical_metadata():
    from adapters.claude.adapter import ClaudeAdapter
    from adapters.gemini.adapter import GeminiAdapter
    from adapters.openai.adapter import OpenAIAdapter
    from shap_review.version import VERSION

    for adapter in (ClaudeAdapter(), GeminiAdapter(), OpenAIAdapter()):
        result = adapter.invoke("version", {})

        assert result["version"] == VERSION
        assert result.get("supported", True)
        assert adapter.invoke("evidence", {})["supported"] is True

def _declared_command_enum(manifest: dict) -> list[str] | None:
    """Return a nested command enum when the manifest exposes one."""
    if manifest.get("type") == "function":
        function = manifest["function"]
        parameters = function["parameters"]
        properties = parameters["properties"]
        command = properties["command"]
        return command["enum"]

    if "parameters" in manifest:
        properties = manifest["parameters"]["properties"]
        command = properties["command"]
        return command["enum"]

    if "inputSchema" in manifest:
        properties = manifest["inputSchema"]["properties"]
        command = properties["command"]
        return command["enum"]

    function = manifest.get("function", {})
    if "parameters" in function:
        properties = function["parameters"]["properties"]
        command = properties["command"]
        return command["enum"]

    return None


def _assert_provider_manifest_contract(
    provider: str,
    name: str,
    manifest: dict,
    capabilities: list[str],
    schema_version: str,
    version: str,
) -> None:
    """Validate common metadata and provider command declarations."""
    assert manifest["version"] == version, (
        f"{provider}/{name} has stale version metadata"
    )
    assert manifest["schema_version"] == schema_version, (
        f"{provider}/{name} has stale schema metadata"
    )
    assert manifest["commands"] == capabilities, (
        f"{provider}/{name} top-level command list drifted"
    )

    declared = _declared_command_enum(manifest)

    if declared is not None:
        assert declared == capabilities, (
            f"{provider}/{name} declared command enum drifted"
        )


def test_provider_manifests_match_canonical():
    from shap_review.version import CAPABILITIES, SCHEMA_VERSION, VERSION

    capabilities = list(CAPABILITIES)
    checked = 0

    for provider in ("claude", "openai", "gemini"):
        for name in (
            "tool-schema.json",
            "tool-definition.json",
            "function-declaration.json",
        ):
            path = Path("adapters") / provider / name

            if not path.exists():
                continue

            checked += 1
            manifest = json.loads(path.read_text(encoding="utf-8"))

            _assert_provider_manifest_contract(
                provider,
                name,
                manifest,
                capabilities,
                SCHEMA_VERSION,
                VERSION,
            )

    assert checked > 0, "no provider manifests were discovered"


def test_openai_nested_command_enum_matches_capabilities():
    from shap_review.version import CAPABILITIES

    path = Path("adapters/openai/tool-definition.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))

    declared = (
        manifest["function"]
        ["parameters"]
        ["properties"]
        ["command"]
        ["enum"]
    )

    assert declared == list(CAPABILITIES)


def test_provider_manifest_missing_top_level_commands_fails():
    from shap_review.version import CAPABILITIES, SCHEMA_VERSION, VERSION

    broken = {
        "version": VERSION,
        "schema_version": SCHEMA_VERSION,
    }

    with pytest.raises(KeyError):
        _assert_provider_manifest_contract(
            "broken",
            "tool-schema.json",
            broken,
            list(CAPABILITIES),
            SCHEMA_VERSION,
            VERSION,
        )

def test_backend_adapter_protocol_is_exported():
    from shap_review.backends import BackendAdapter, BackendExecution

    assert BackendAdapter is not None
    assert "status" in BackendExecution.__annotations__
