from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceValidation:
    valid: bool
    status: str
    reason: str
    independent: bool | None = None
    missing: tuple[str, ...] = ()
    schema_valid: bool = True
    provenance_valid: bool = True
    scoring_eligible: bool = True


_RUNTIME_KINDS = {
    "dynamic",
    "reproduction",
    "differential",
    "sanitizer",
}


def _kind_value(item: Any) -> str | None:
    kind = getattr(item, "kind", None)
    return getattr(kind, "value", kind)


def _origin_value(item: Any) -> str | None:
    origin = getattr(item, "origin", None)
    return getattr(origin, "value", origin)


def _has_runtime_provenance(item: Any) -> bool:
    """Return whether runtime-style evidence has a concrete provenance anchor."""
    execution_id = getattr(item, "execution_id", None)
    fixture_id = getattr(item, "fixture_id", None)
    input_fingerprint = getattr(item, "input_fingerprint", None)
    repository_revision = getattr(item, "repository_revision", None)
    environment_fingerprint = getattr(item, "environment_fingerprint", None)

    return bool(
        execution_id
        and (
            fixture_id
            or input_fingerprint
            or repository_revision
            or environment_fingerprint
        )
    )


def validate_evidence_item(item: Any) -> EvidenceValidation:
    """Validate evidence structure and provenance without conflating them.

    ``valid`` means that the evidence item is structurally acceptable to enter
    an evidence chain. Runtime-style evidence can therefore be structurally
    valid while remaining ``AMBIGUOUS`` and ineligible for scoring when the
    required provenance is incomplete.

    Static and historical evidence may remain lightweight because they describe
    source material rather than a runtime observation.
    """
    kind = _kind_value(item)
    origin = _origin_value(item)
    missing: list[str] = []

    if kind in _RUNTIME_KINDS:
        if not getattr(item, "execution_id", None):
            missing.append("execution_id")

        if not (
            getattr(item, "fixture_id", None)
            or getattr(item, "input_fingerprint", None)
            or getattr(item, "repository_revision", None)
            or getattr(item, "environment_fingerprint", None)
        ):
            missing.append(
                "fixture_id_or_input_fingerprint_or_repository_revision_or_environment_fingerprint"
            )

    if origin == "derived" and not getattr(item, "derived_from", ()):
        missing.append("derived_from")

    if missing:
        return EvidenceValidation(
            valid=True,
            status="AMBIGUOUS",
            reason=(
                "evidence is structurally valid but lacks sufficient provenance "
                "for independent scoring"
            ),
            independent=False,
            missing=tuple(missing),
            schema_valid=True,
            provenance_valid=False,
            scoring_eligible=False,
        )

    derived = bool(getattr(item, "derived_from", ())) or origin == "derived"

    return EvidenceValidation(
        valid=True,
        status="VALID",
        reason="required structural and provenance fields are present",
        independent=not derived,
        missing=(),
        schema_valid=True,
        provenance_valid=True,
        scoring_eligible=not derived,
    )
