"""Adapter layer tests.

Covers:
- All three provider adapters (Claude, OpenAI, Gemini) share the same
  command contract and return correct envelope structure.
- ``BaseReviewAdapter`` correctly routes built-in commands.
- ``BaseReviewAdapter`` rejects unknown commands with ``AdapterError``.
- ``BaseReviewAdapter`` encodes ``TypeError`` / ``FileNotFoundError`` from
  the engine as structured error envelopes rather than propagating exceptions.
- ``AdapterCapabilities.to_dict()`` surfaces the security note and the
  dynamic-evidence note.
- ``AdapterRequest.validate()`` raises ``AdapterError`` for unknown commands.
- Provider identity is correctly set per-instance, not inherited from the
  base class.
- JSON manifests (tool-schema.json, tool-definition.json,
  function-declaration.json) match the canonical capabilities list from
  ``shap_review.version``.
- ``CONFIRMED_MEMORY_SAFETY`` verdict is absent from all manifests (it never
  existed in the evidence model).
- Backend adapter stage attribution and UNSUPPORTED classification.
- Evidence command surfaces the independence note and dynamic-evidence note.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from adapters import ClaudeAdapter, GeminiAdapter, OpenAIAdapter
from adapters.base import _SCRIPT_EXECUTION_COMMANDS, BaseReviewAdapter
from adapters.interface import AdapterCapabilities, AdapterError, AdapterRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_adapters():
    return [ClaudeAdapter(), OpenAIAdapter(), GeminiAdapter()]


def _declared_command_enum(manifest: dict) -> list[str] | None:
    """Return the nested command enum from a provider manifest, if present."""
    if manifest.get("type") == "function":
        return manifest["function"]["parameters"]["properties"]["command"]["enum"]
    if "parameters" in manifest:
        props = manifest["parameters"]["properties"]
        if "command" in props and "enum" in props["command"]:
            return props["command"]["enum"]
    return None


def _assert_provider_manifest_contract(
    provider: str,
    name: str,
    manifest: dict,
    capabilities: list[str],
    schema_version: str,
    version: str,
) -> None:
    # version and schema_version are only checked when present at top level.
    # function-declaration.json (bare Gemini function) intentionally omits them.
    if "version" in manifest:
        assert manifest["version"] == version, (
            f"{provider}/{name} has stale version metadata"
        )
    if "schema_version" in manifest:
        assert manifest["schema_version"] == schema_version, (
            f"{provider}/{name} has stale schema metadata"
        )
    # top-level "commands" is optional; bare function declarations omit it.
    if "commands" in manifest:
        assert manifest["commands"] == capabilities, (
            f"{provider}/{name} top-level command list drifted"
        )
    declared = _declared_command_enum(manifest)
    if declared is not None:
        assert declared == capabilities, (
            f"{provider}/{name} declared command enum drifted"
        )


# ---------------------------------------------------------------------------
# Provider identity
# ---------------------------------------------------------------------------


def test_each_provider_has_distinct_identity():
    """Provider string must be unique per adapter class, not inherited from base."""
    adapters = _all_adapters()
    providers = [a.provider for a in adapters]
    assert sorted(providers) == ["claude", "gemini", "openai"]


def test_capabilities_provider_matches_adapter_provider():
    """capabilities.provider must reflect the concrete adapter, not 'generic'."""
    for adapter in _all_adapters():
        assert adapter.capabilities.provider == adapter.provider, (
            f"{type(adapter).__name__}.capabilities.provider is wrong"
        )


def test_base_adapter_provider_is_generic():
    adapter = BaseReviewAdapter()
    assert adapter.provider == "generic"
    assert adapter.capabilities.provider == "generic"


# ---------------------------------------------------------------------------
# Built-in commands
# ---------------------------------------------------------------------------


def test_all_adapters_share_same_command_list():
    command_sets = [tuple(a.capabilities.commands) for a in _all_adapters()]
    assert len(set(command_sets)) == 1, "command lists diverged across providers"


def test_capabilities_command_returns_correct_provider():
    for adapter in _all_adapters():
        result = adapter.invoke("capabilities")
        assert result["provider"] == adapter.provider
        assert result["command"] == "capabilities"


def test_capabilities_includes_all_core_commands():
    from shap_review.version import CAPABILITIES

    for adapter in _all_adapters():
        result = adapter.invoke("capabilities")
        assert result["result"]["commands"] == list(CAPABILITIES)


def test_capabilities_surfaces_security_note():
    """The security note about script execution must appear in capabilities."""
    result = ClaudeAdapter().invoke("capabilities")
    caps = result["result"]
    note = caps.get("security_note", "")
    assert "subprocess" in note.lower() or "script" in note.lower(), (
        "capabilities must warn about subprocess script execution"
    )


def test_capabilities_surfaces_dynamic_evidence_note():
    """The dynamic evidence limitation must appear in capabilities."""
    result = ClaudeAdapter().invoke("capabilities")
    caps = result["result"]
    note = caps.get("dynamic_evidence_note", "")
    assert "candidate" in note.lower() or "bridge" in note.lower(), (
        "capabilities must document the dynamic evidence bridge limitation"
    )


def test_version_command_returns_canonical_metadata():
    from shap_review.version import SCHEMA_VERSION, VERSION

    for adapter in _all_adapters():
        result = adapter.invoke("version")
        assert result["result"]["version"] == VERSION
        assert result["result"]["toolkit_version"] == VERSION
        assert result["result"]["schema_version"] == SCHEMA_VERSION
        assert result["result"]["provider"] == adapter.provider


def test_evidence_command_returns_structured_metadata():
    result = ClaudeAdapter().invoke("evidence")
    ev = result["result"]
    assert ev["supported"] is True
    assert ev["independence_tracking"] is True
    assert ev["provenance_tracking"] is True


def test_evidence_command_surfaces_independence_note():
    """Evidence command must warn about graph=None bypass."""
    result = ClaudeAdapter().invoke("evidence")
    note = result["result"].get("independence_note", "")
    assert "build_chain" in note or "graph" in note.lower(), (
        "evidence command must document the EvidenceChain graph bypass risk"
    )


def test_evidence_command_surfaces_dynamic_evidence_note():
    """Evidence command must warn that bridge does not enrich candidates."""
    result = ClaudeAdapter().invoke("evidence")
    note = result["result"].get("dynamic_evidence_note", "")
    assert "fingerprint" in note.lower() or "candidate" in note.lower(), (
        "evidence command must document the dynamic evidence non-enrichment"
    )


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


def test_response_envelope_always_contains_metadata():
    """Every invoke() response must contain provider, version, schema_version, command."""
    from shap_review.version import SCHEMA_VERSION, VERSION

    for adapter in _all_adapters():
        result = adapter.invoke("version")
        assert result["provider"] == adapter.provider
        assert result["version"] == VERSION
        assert result["schema_version"] == SCHEMA_VERSION
        assert result["command"] == "version"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unknown_command_raises_adapter_error():
    """Unknown commands must raise AdapterError, not propagate to the caller."""
    for adapter in _all_adapters():
        with pytest.raises(AdapterError) as exc_info:
            adapter.invoke("nonexistent-command-xyz")
        assert exc_info.value.command == "nonexistent-command-xyz"
        assert exc_info.value.provider == adapter.provider


def test_argument_error_is_encoded_not_raised():
    """TypeError from the engine must be returned as a structured error envelope."""
    adapter = ClaudeAdapter()
    # Patch dispatch to raise TypeError simulating bad arguments.
    with patch("adapters.base.dispatch", side_effect=TypeError("bad args")):
        result = adapter.invoke("analyze", {"root": "."})
    assert "error" in result
    assert result["error"]["kind"] == "ARGUMENT_ERROR"
    assert result["error"]["layer"] == "adapter"
    assert "result" not in result


def test_file_not_found_is_encoded_not_raised():
    """FileNotFoundError from the engine must be returned as a structured error."""
    adapter = OpenAIAdapter()
    with patch(
        "adapters.base.dispatch", side_effect=FileNotFoundError("/no/such/path")
    ):
        result = adapter.invoke("analyze", {"root": "/no/such/path"})
    assert "error" in result
    assert result["error"]["kind"] == "PATH_NOT_FOUND"
    assert "result" not in result


def test_unexpected_engine_error_is_encoded_not_raised():
    """Unexpected engine failures must be encoded, not propagated."""
    adapter = GeminiAdapter()
    with patch("adapters.base.dispatch", side_effect=RuntimeError("boom")):
        result = adapter.invoke("analyze", {"root": "."})
    assert "error" in result
    assert result["error"]["kind"] == "ENGINE_ERROR"
    assert "RuntimeError" in result["error"]["message"]


# ---------------------------------------------------------------------------
# Security — script execution commands
# ---------------------------------------------------------------------------


def test_script_execution_commands_are_documented():
    """The set of script-execution commands must be non-empty and stable."""
    assert "differential" in _SCRIPT_EXECUTION_COMMANDS
    assert "reproduce" in _SCRIPT_EXECUTION_COMMANDS
    assert "sanitizer" in _SCRIPT_EXECUTION_COMMANDS
    assert "differential-versions" in _SCRIPT_EXECUTION_COMMANDS


def test_script_execution_commands_log_warning(caplog):
    """Invoking a script-execution command must emit a security warning."""
    import logging

    adapter = ClaudeAdapter()
    # Patch dispatch so we don't need a real script file.
    with patch("adapters.base.dispatch", return_value={"status": "ok"}):
        with caplog.at_level(logging.WARNING, logger="adapters.base"):
            adapter.invoke("reproduce", {"root": ".", "script": "x.py"})
    assert any(
        "subprocess" in r.message.lower() or "script" in r.message.lower()
        for r in caplog.records
    ), "script-execution command must emit a security warning log"


# ---------------------------------------------------------------------------
# AdapterRequest validation
# ---------------------------------------------------------------------------


def test_adapter_request_validate_rejects_unknown_command():
    req = AdapterRequest(command="bad-command", provider="test")
    with pytest.raises(AdapterError) as exc_info:
        req.validate()
    assert exc_info.value.command == "bad-command"


def test_adapter_request_validate_accepts_known_command():
    req = AdapterRequest(command="version", provider="test")
    req.validate()  # must not raise


# ---------------------------------------------------------------------------
# AdapterCapabilities
# ---------------------------------------------------------------------------


def test_adapter_capabilities_to_dict_is_complete():
    from shap_review.version import CAPABILITIES, SCHEMA_VERSION, VERSION

    caps = AdapterCapabilities(provider="test")
    d = caps.to_dict()
    assert d["provider"] == "test"
    assert d["version"] == VERSION
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["commands"] == list(CAPABILITIES)
    assert "security_note" in d
    assert "dynamic_evidence_note" in d


def test_adapter_capabilities_version_defaults_to_current():
    from shap_review.version import VERSION

    caps = AdapterCapabilities(provider="x")
    assert caps.version == VERSION


# ---------------------------------------------------------------------------
# JSON manifests
# ---------------------------------------------------------------------------


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
                provider, name, manifest, capabilities, SCHEMA_VERSION, VERSION
            )

    assert checked > 0, "no provider manifests were discovered"


def test_manifests_do_not_contain_nonexistent_verdict():
    """CONFIRMED_MEMORY_SAFETY never existed in the evidence model."""
    for provider in ("claude", "openai", "gemini"):
        for name in (
            "tool-schema.json",
            "tool-definition.json",
            "function-declaration.json",
        ):
            path = Path("adapters") / provider / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            assert "CONFIRMED_MEMORY_SAFETY" not in text, (
                f"{provider}/{name} references nonexistent verdict "
                "CONFIRMED_MEMORY_SAFETY"
            )


def test_manifests_contain_security_note():
    """All manifests with a top-level security_note field must document subprocess."""
    for provider in ("claude", "openai", "gemini"):
        for name in (
            "tool-schema.json",
            "tool-definition.json",
            "function-declaration.json",
        ):
            path = Path("adapters") / provider / name
            if not path.exists():
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            # function-declaration.json (Gemini) intentionally omits security_note
            # at top level because it is a bare function declaration without metadata.
            if "security_note" in manifest:
                note = manifest["security_note"]
                assert "subprocess" in note.lower() or "script" in note.lower(), (
                    f"{provider}/{name} security_note does not mention subprocess"
                )


def test_openai_tool_definition_nested_command_enum_matches_capabilities():
    from shap_review.version import CAPABILITIES

    path = Path("adapters/openai/tool-definition.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    declared = manifest["function"]["parameters"]["properties"]["command"]["enum"]
    assert declared == list(CAPABILITIES)


def test_manifests_contain_dynamic_evidence_note_in_evidence_model():
    """Evidence model block in manifests must document the bridge limitation."""
    for provider in ("claude", "openai", "gemini"):
        for name in ("tool-schema.json", "tool-definition.json"):
            path = Path("adapters") / provider / name
            if not path.exists():
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            ev = manifest.get("evidence_model", {})
            note = ev.get("dynamic_evidence_note", "")
            assert note, (
                f"{provider}/{name} evidence_model is missing dynamic_evidence_note"
            )


# ---------------------------------------------------------------------------
# Backend adapter (backend matrix — these live in shap_review.backends)
# ---------------------------------------------------------------------------


def test_backend_adapter_reports_actual_failure_stage():
    from shap_review.backends.adapter import MatrixBackendAdapter
    from shap_review.fuzzing.backend_matrix import BackendSpec

    adapter = MatrixBackendAdapter(
        BackendSpec("x", "x", (), True, "1"),
        MagicMock(),
    )
    adapter.prepare = lambda **kwargs: {}

    def failing_fit(context):
        raise RuntimeError("fit broke")

    adapter.fit = failing_fit
    result = adapter.execute_case()
    assert result["stage"] == "fit"
    assert result["execution_reason"] == "BACKEND_ERROR"


def test_backend_rejects_unsupported_regression_output():
    from shap_review.backends.adapter import MatrixBackendAdapter
    from shap_review.fuzzing.backend_matrix import BackendSpec

    adapter = MatrixBackendAdapter(
        BackendSpec("x", "x", (), True, "1"),
        MagicMock(),
    )
    supported, reason = adapter.supports_case(
        classification=False,
        model_output="log_loss",
        interaction=False,
    )
    assert supported is False
    assert "regression" in reason


def test_backend_unsupported_status_is_not_error():
    """UNSUPPORTED must be distinguishable from SHAP_ERROR / BACKEND_ERROR."""
    from shap_review.backends.adapter import MatrixBackendAdapter
    from shap_review.fuzzing.backend_matrix import BackendSpec

    adapter = MatrixBackendAdapter(
        BackendSpec("x", "x", (), True, "1"),
        MagicMock(),
    )
    # Trigger UNSUPPORTED via supports_case returning False.
    adapter.prepare = lambda **kwargs: (_ for _ in ()).throw(
        NotImplementedError("unsupported combination")
    )
    result = adapter.execute_case()
    assert result["status"] == "UNSUPPORTED"
    assert result["execution_reason"] == "UNSUPPORTED"
    # UNSUPPORTED must not be conflated with a genuine SHAP failure.
    assert result["status"] not in {"SHAP_ERROR", "BACKEND_ERROR"}


def test_backend_adapter_protocol_is_exported():
    from shap_review.backends import BackendAdapter, BackendExecution

    assert BackendAdapter is not None
    assert "status" in BackendExecution.__annotations__


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_adapter_repr_contains_provider_and_version():
    from shap_review.version import VERSION

    for adapter in _all_adapters():
        r = repr(adapter)
        assert adapter.provider in r
        assert VERSION in r
