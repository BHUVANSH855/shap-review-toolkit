from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendSpec:
    """Description of a supported SHAP model backend."""

    name: str
    import_name: str
    families: tuple[str, ...]
    available: bool = False
    version: str | None = None


DEFAULT_BACKENDS = (
    BackendSpec("sklearn", "sklearn", ("tree", "linear", "ensemble")),
    BackendSpec("xgboost", "xgboost", ("tree", "gbtree")),
    BackendSpec("lightgbm", "lightgbm", ("tree", "gbdt")),
    BackendSpec("catboost", "catboost", ("tree", "categorical")),
)


def discover_backends(specs=DEFAULT_BACKENDS):
    """Return backend specifications with availability and version metadata.

    Only import-related failures are treated as unavailable optional
    dependencies. Unexpected import failures are allowed to propagate so
    broken installations are not silently classified as unavailable.
    """
    discovered = []

    for spec in specs:
        try:
            module = importlib.import_module(spec.import_name)
        except (ImportError, OSError):
            discovered.append(spec)
            continue

        discovered.append(
            BackendSpec(
                name=spec.name,
                import_name=spec.import_name,
                families=spec.families,
                available=True,
                version=getattr(module, "__version__", None),
            )
        )

    return discovered
