"""Shared interface definitions for the SHAP Review Toolkit adapter layer.

All provider adapters (Claude, OpenAI, Gemini) implement ``ReviewAdapter`` and
extend ``BaseReviewAdapter``.  The interface is intentionally narrow: adapters
route commands to the core engine and return structured results.  They never
own analysis logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from shap_review.version import CAPABILITIES, SCHEMA_VERSION, VERSION

# The canonical command set is owned by shap_review.version.CAPABILITIES so
# that the adapter layer and the core CLI always agree without duplication.
CORE_COMMANDS: tuple[str, ...] = CAPABILITIES


class AdapterError(Exception):
    """Raised when an adapter cannot fulfil a request.

    Attributes
    ----------
    command:
        The command that failed.
    provider:
        The adapter provider that raised this error.
    reason:
        Human-readable explanation.
    """

    def __init__(self, command: str, provider: str, reason: str) -> None:
        super().__init__(f"[{provider}] command={command!r}: {reason}")
        self.command = command
        self.provider = provider
        self.reason = reason


@dataclass(frozen=True)
class AdapterRequest:
    """A validated, immutable request routed to the core engine.

    Parameters
    ----------
    command:
        One of the strings in ``CORE_COMMANDS``.
    arguments:
        Provider-decoded keyword arguments forwarded to ``dispatch()``.
    provider:
        Originating adapter identity (for logging and error attribution).
    """

    command: str
    arguments: dict[str, Any] = field(default_factory=dict)
    provider: str = "unknown"

    def validate(self) -> None:
        """Raise ``AdapterError`` if the command is not in ``CORE_COMMANDS``."""
        if self.command not in CORE_COMMANDS:
            raise AdapterError(
                self.command,
                self.provider,
                f"unknown command; valid commands are: {sorted(CORE_COMMANDS)}",
            )


@dataclass(frozen=True)
class AdapterCapabilities:
    """Immutable capabilities declaration returned by the ``capabilities`` command.

    Parameters
    ----------
    provider:
        Provider identity string (e.g. ``"claude"``, ``"openai"``, ``"gemini"``).
    version:
        Toolkit version string.  Defaults to the canonical ``VERSION`` constant
        so that all adapters automatically reflect the installed version.
    commands:
        Tuple of supported command names.  Defaults to ``CORE_COMMANDS``.
    structured_output:
        Whether the adapter guarantees JSON-serialisable structured output.
    schema_version:
        Evidence schema version string.

    Notes
    -----
    In the original implementation ``provider`` was a class-level string and
    ``capabilities`` was also a class-level ``AdapterCapabilities`` instance
    constructed at class definition time.  Because both referred to
    ``VERSION`` at definition time, any subclass that redeclared ``provider``
    without also redeclaring ``capabilities`` would inherit the parent's
    ``AdapterCapabilities(provider="generic", ...)``.  This class now takes
    ``provider`` as a constructor argument so each subclass supplies it
    explicitly, making the coupling visible and correct.
    """

    provider: str
    version: str = VERSION
    commands: tuple[str, ...] = CORE_COMMANDS
    structured_output: bool = True
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable capabilities dict."""
        return {
            "provider": self.provider,
            "version": self.version,
            "toolkit_version": self.version,
            "schema_version": self.schema_version,
            "commands": list(self.commands),
            "structured_output": self.structured_output,
            # Explicit note that dynamic evidence enrichment requires
            # candidate-specific campaigns that the generic bridge does not
            # currently provide.  Consumers must not assume that
            # ``runtime-bridge.json`` anomalies enriched candidate evidence.
            "dynamic_evidence_note": (
                "runtime-bridge campaign runs against the installed SHAP, not "
                "the repository under review; candidate enrichment requires "
                "explicit candidate-fingerprint correlation which the generic "
                "bridge does not set."
            ),
            # Explicit security note surfaced to any consumer that reads
            # capabilities before issuing commands.
            "security_note": (
                "Commands 'differential', 'reproduce', 'sanitizer', and "
                "'differential-versions' execute arbitrary Python scripts via "
                "subprocess.  Only invoke these commands with trusted, "
                "controlled input paths."
            ),
        }


class ReviewAdapter(Protocol):
    """Structural interface that all concrete adapters must satisfy.

    Conforming classes must expose a ``capabilities`` attribute and an
    ``invoke`` method with the documented signature.  The ``ReviewAdapter``
    protocol is used for static type checking; runtime isinstance checks are
    not required.
    """

    capabilities: AdapterCapabilities

    def invoke(self, command: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute ``command`` with ``arguments`` and return a structured result.

        Parameters
        ----------
        command:
            One of the strings in ``CORE_COMMANDS``.
        arguments:
            Keyword arguments specific to the command.

        Returns
        -------
        dict
            Always a JSON-serialisable dict.  Never raises for expected
            runtime or model failures; those are encoded in the result.

        Raises
        ------
        AdapterError
            When ``command`` is not in ``self.capabilities.commands``.
        """
        ...
