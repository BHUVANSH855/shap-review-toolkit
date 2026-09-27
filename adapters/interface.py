from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from shap_review.version import CAPABILITIES, SCHEMA_VERSION, VERSION

CORE_COMMANDS = CAPABILITIES


@dataclass(frozen=True)
class AdapterRequest:
    command: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AdapterCapabilities:
    provider: str
    version: str = VERSION
    commands: tuple[str, ...] = CORE_COMMANDS
    structured_output: bool = True
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "version": self.version,
            "toolkit_version": self.version,
            "schema_version": self.schema_version,
            "commands": list(self.commands),
            "structured_output": self.structured_output,
        }


class ReviewAdapter(Protocol):
    capabilities: AdapterCapabilities

    def invoke(self, command: str, arguments: dict[str, Any]) -> dict[str, Any]: ...
