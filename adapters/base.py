"""Base adapter implementation shared by all provider adapters.

``BaseReviewAdapter`` handles command routing, input validation, error
classification, and response envelope construction.  Provider-specific
adapters (Claude, OpenAI, Gemini) subclass this and override ``provider``
and ``capabilities`` only.

Security
--------
The commands listed in ``_SCRIPT_EXECUTION_COMMANDS`` invoke arbitrary Python
scripts via subprocess.  ``BaseReviewAdapter`` does not sandbox or validate
the script paths — that responsibility lies with the caller.  Subclasses
intended for untrusted environments should override ``invoke`` to restrict or
reject these commands before delegating to ``super().invoke()``.
"""

from __future__ import annotations

import logging
from typing import Any

from adapters.interface import CORE_COMMANDS, AdapterCapabilities, AdapterError
from shap_review.cli import dispatch
from shap_review.engine import ReviewEngine
from shap_review.version import SCHEMA_VERSION, VERSION

log = logging.getLogger(__name__)

# Commands that execute arbitrary Python scripts via subprocess.
# Callers must ensure these are only triggered with trusted, controlled paths.
_SCRIPT_EXECUTION_COMMANDS: frozenset[str] = frozenset(
    {
        "differential",
        "cpu-gpu-differential",
        "differential-versions",
        "reproduce",
        "sanitizer",
    }
)


class BaseReviewAdapter:
    """Routing base for all SHAP Review Toolkit provider adapters.

    Subclasses must override ``provider`` and reconstruct ``capabilities``
    with the correct ``provider`` argument.  They must **not** override
    analysis logic — all analysis is owned by ``shap_review.engine``.

    Parameters
    ----------
    engine:
        Optional pre-constructed ``ReviewEngine``.  When ``None`` a default
        engine is created.  Injecting a shared engine allows tests and
        integration callers to control the engine lifecycle.
    """

    provider: str = "generic"

    def __init__(self, engine: ReviewEngine | None = None) -> None:
        self.engine: ReviewEngine = engine or ReviewEngine()
        # Capabilities are constructed per-instance so that the correct
        # provider string is always reflected even when ``provider`` is
        # overridden on a subclass without also overriding ``capabilities``.
        self.capabilities: AdapterCapabilities = AdapterCapabilities(
            provider=self.provider,
            version=VERSION,
            commands=CORE_COMMANDS,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke(
        self, command: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Route ``command`` to the core engine and return a structured result.

        The response envelope always contains ``provider``, ``version``,
        ``schema_version``, and ``command``.  On success it also contains
        ``result``.  On error it contains ``error`` with structured
        classification.

        Parameters
        ----------
        command:
            Must be one of the strings in ``self.capabilities.commands``.
        arguments:
            Command arguments.  ``None`` is treated as an empty dict.

        Returns
        -------
        dict
            Always JSON-serialisable.  Never raises for expected failures.

        Raises
        ------
        AdapterError
            When ``command`` is not in ``self.capabilities.commands``.
            This is the only exception that propagates; all other failures
            are captured and encoded in the response envelope.
        """
        arguments = arguments or {}

        if command not in self.capabilities.commands:
            raise AdapterError(
                command,
                self.provider,
                f"unknown command; valid commands are: {sorted(self.capabilities.commands)}",
            )

        if command in _SCRIPT_EXECUTION_COMMANDS:
            log.warning(
                "[%s] command=%r executes arbitrary Python scripts via subprocess; "
                "ensure all path arguments are trusted and controlled.",
                self.provider,
                command,
            )

        envelope: dict[str, Any] = {
            "provider": self.provider,
            "version": VERSION,
            "schema_version": SCHEMA_VERSION,
            "command": command,
        }

        # Built-in commands handled directly without touching the engine.
        builtin = self._builtin(command)
        if builtin is not None:
            envelope["result"] = builtin
            return envelope

        # All other commands are routed to the core CLI dispatcher.
        try:
            result = dispatch(
                command,
                root=arguments.get("root", "."),
                arguments=arguments,
                engine=self.engine,
            )
        except TypeError as exc:
            # Missing or wrong-typed arguments from the provider payload.
            log.debug("[%s] command=%r argument error: %s", self.provider, command, exc)
            envelope["error"] = self._error_payload(
                "ARGUMENT_ERROR",
                f"Invalid arguments for command {command!r}: {exc}",
                command,
            )
            return envelope
        except FileNotFoundError as exc:
            # Root path or script path does not exist.
            log.debug("[%s] command=%r path not found: %s", self.provider, command, exc)
            envelope["error"] = self._error_payload(
                "PATH_NOT_FOUND",
                str(exc),
                command,
            )
            return envelope
        except Exception as exc:
            # Unexpected core engine failure. Encode rather than propagate so
            # the provider layer always receives a structured response.
            log.exception(
                "[%s] command=%r unexpected engine failure",
                self.provider,
                command,
                exc_info=exc,
            )
            envelope["error"] = self._error_payload(
                "ENGINE_ERROR",
                f"{type(exc).__name__}: {exc}",
                command,
            )
            return envelope

        envelope["result"] = result
        return envelope

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"provider={self.provider!r}, "
            f"version={VERSION!r}, "
            f"engine={self.engine!r}"
            f")"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _builtin(self, command: str) -> dict[str, Any] | None:
        """Return a pre-computed result for built-in commands, or ``None``."""
        if command == "capabilities":
            return self.capabilities.to_dict()
        if command == "version":
            return {
                "version": VERSION,
                "toolkit_version": VERSION,
                "schema_version": SCHEMA_VERSION,
                "provider": self.provider,
            }
        if command == "evidence":
            return {
                "model": "EvidenceChain",
                "supported": True,
                "schema_version": SCHEMA_VERSION,
                "independence_tracking": True,
                "provenance_tracking": True,
                # Explicit note: the graph must be present for pairwise
                # independence detection.  Direct EvidenceChain() construction
                # without build_chain() bypasses correlation detection.
                "independence_note": (
                    "Pairwise independence is graph-derived via EvidenceGraph. "
                    "Always use build_chain() to construct EvidenceChain instances; "
                    "direct construction bypasses correlation detection."
                ),
                # Explicit note about dynamic evidence bridge status.
                "dynamic_evidence_note": (
                    "attach_dynamic_evidence() correlates only when anomaly "
                    "provenance contains an explicit candidate_fingerprint. "
                    "The generic bridge campaign does not set this field; "
                    "candidates are therefore not enriched with dynamic "
                    "evidence in the current base implementation."
                ),
            }
        return None

    @staticmethod
    def _error_payload(kind: str, message: str, command: str) -> dict[str, Any]:
        """Return a structured error dict suitable for the response envelope."""
        return {
            "kind": kind,
            "message": message,
            "command": command,
            # Distinguish adapter-layer errors from SHAP/engine failures.
            "layer": "adapter",
        }
