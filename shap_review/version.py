from __future__ import annotations

VERSION = "0.0.0.dev0"
SCHEMA_VERSION = "2"

# Command order must match the adapter JSON manifests exactly.
CAPABILITIES: tuple[str, ...] = (
    "map",
    "health",
    "analyze",
    "report",
    "hotspots",
    "explore",
    "investigate",
    "reproduce",
    "fuzz",
    "fuzz-treeexplainer",
    "fuzz-protocol",
    "fuzz-backends",
    "differential",
    "cpu-gpu-differential",
    "regressions",
    "history",
    "sanitizer",
    "differential-versions",
    "evidence",
    "semantic-oracle",
    "native-map",
    "api-era",
    "capabilities",
    "version",
)


def release_metadata(provider: str = "generic") -> dict:
    """Return structured development metadata for the capabilities command."""
    return {
        "version": VERSION,
        "toolkit_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "provider": provider,
        "capabilities": list(CAPABILITIES),
        "commands": list(CAPABILITIES),
    }