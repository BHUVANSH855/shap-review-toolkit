from __future__ import annotations

from adapters.interface import CORE_COMMANDS, AdapterCapabilities
from shap_review.cli import dispatch
from shap_review.engine import ReviewEngine
from shap_review.version import SCHEMA_VERSION, VERSION


class BaseReviewAdapter:
    provider = "generic"
    version = VERSION
    capabilities = AdapterCapabilities(
        provider=provider, version=VERSION, commands=CORE_COMMANDS
    )

    def __init__(self, engine: ReviewEngine | None = None):
        self.engine = engine or ReviewEngine()

    def invoke(self, command: str, arguments: dict) -> dict:
        if command not in self.capabilities.commands:
            raise ValueError(f"unsupported command for {self.provider}: {command}")
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
            }
        result = dispatch(
            command,
            root=arguments.get("root", "."),
            arguments=arguments,
            engine=self.engine,
        )
        return {
            "provider": self.provider,
            "version": VERSION,
            "schema_version": self.capabilities.schema_version,
            "command": command,
            "result": result,
        }
