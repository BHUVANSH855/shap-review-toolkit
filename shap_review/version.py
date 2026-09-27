"""Canonical release metadata for SHAP Review Toolkit."""

VERSION = "0.28.0"
SCHEMA_VERSION = "2.9"
CAPABILITIES = (
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
    return {
        "provider": provider,
        "version": VERSION,
        "toolkit_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "commands": list(CAPABILITIES),
    }
