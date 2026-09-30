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


_KNOWN_KINDS = {
    "static",
    "dynamic",
    "reproduction",
    "differential",
    "sanitizer",
    "historical",
}

_RUNTIME_KINDS = {
    "dynamic",
    "reproduction",
    "differential",
    "sanitizer",
}

_KNOWN_ORIGINS = {
    "source",
    "execution",
    "external",
    "derived",
}


def _kind_value(item: Any) -> str | None:
    kind = getattr(item, "kind", None)
    value = getattr(kind, "value", kind)

    if value is None:
        return None

    return str(value)


def _origin_value(item: Any) -> str | None:
    origin = getattr(item, "origin", None)
    value = getattr(origin, "value", origin)

    if value is None:
        return None

    return str(value)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _has_runtime_provenance(item: Any) -> bool:
    """Return whether runtime evidence has a concrete provenance anchor."""
    execution_id = getattr(item, "execution_id", None)
    fixture_id = getattr(item, "fixture_id", None)
    input_fingerprint = getattr(item, "input_fingerprint", None)
    repository_revision = getattr(item, "repository_revision", None)
    environment_fingerprint = getattr(item, "environment_fingerprint", None)

    return bool(
        _non_empty_string(execution_id)
        and (
            _non_empty_string(fixture_id)
            or _non_empty_string(input_fingerprint)
            or _non_empty_string(repository_revision)
            or _non_empty_string(environment_fingerprint)
        )
    )


def _missing_runtime_provenance(item: Any) -> list[str]:
    """Return missing runtime provenance fields."""
    missing: list[str] = []

    if not _non_empty_string(getattr(item, "execution_id", None)):
        missing.append("execution_id")

    if not any(
        _non_empty_string(getattr(item, field, None))
        for field in (
            "fixture_id",
            "input_fingerprint",
            "repository_revision",
            "environment_fingerprint",
        )
    ):
        missing.append(
            "fixture_id_or_input_fingerprint_or_repository_revision_or_environment_fingerprint"
        )

    return missing


def _is_derived(item: Any, origin: str | None) -> bool:
    return bool(
        getattr(item, "derived_from", ())
        or getattr(item, "parent_evidence_ids", ())
        or origin == "derived"
    )


def _has_full_schema(item: Any) -> bool:
    """Return whether the object exposes the canonical evidence payload."""
    return all(
        hasattr(item, field)
        for field in (
            "source",
            "claim",
            "confidence",
            "passed",
        )
    )


def validate_evidence_item(item: Any) -> EvidenceValidation:
    """Validate evidence structure, provenance, and scoring eligibility.

    Validation deliberately distinguishes three states:

    * ``INVALID`` — the evidence violates the expected schema.
    * ``AMBIGUOUS`` — the structure is usable, but provenance is insufficient
      for independent scoring.
    * ``VALID`` — the evidence is structurally valid and sufficiently
      attributable for its declared role.

    Runtime evidence must have an execution identifier plus at least one
    concrete execution anchor. Derived evidence is never independently
    scoreable because its claim depends on other evidence.

    Evidence IDs are intentionally optional at validation time. The chain
    builder may assign stable IDs when an item does not provide one.
    """
    missing: list[str] = []

    kind = _kind_value(item)
    origin = _origin_value(item)

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    if kind is None:
        missing.append("kind")
    elif kind not in _KNOWN_KINDS:
        return EvidenceValidation(
            valid=False,
            status="INVALID",
            reason=f"unknown evidence kind: {kind!r}",
            independent=False,
            missing=("kind",),
            schema_valid=False,
            provenance_valid=False,
            scoring_eligible=False,
        )

    if origin is None:
        missing.append("origin")
    elif origin not in _KNOWN_ORIGINS:
        return EvidenceValidation(
            valid=False,
            status="INVALID",
            reason=f"unknown evidence origin: {origin!r}",
            independent=False,
            missing=("origin",),
            schema_valid=False,
            provenance_valid=False,
            scoring_eligible=False,
        )

    # Lightweight callers may provide only provenance fields. Preserve that
    # compatibility rather than treating omitted payload fields as malformed.
    if _has_full_schema(item):
        if not _non_empty_string(getattr(item, "source", None)):
            missing.append("source")

        if not _non_empty_string(getattr(item, "claim", None)):
            missing.append("claim")

        confidence = getattr(item, "confidence", None)

        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            missing.append("confidence")
        elif not 0 <= float(confidence) <= 1:
            missing.append("confidence_range")

        passed = getattr(item, "passed", None)

        if passed not in {True, False, None}:
            missing.append("passed")

    if missing:
        return EvidenceValidation(
            valid=False,
            status="INVALID",
            reason="required evidence schema fields are missing or malformed",
            independent=False,
            missing=tuple(dict.fromkeys(missing)),
            schema_valid=False,
            provenance_valid=False,
            scoring_eligible=False,
        )

    # ------------------------------------------------------------------
    # Derived evidence
    # ------------------------------------------------------------------

    derived = _is_derived(item, origin)

    if origin == "derived" and not (
        getattr(item, "derived_from", ()) or getattr(item, "parent_evidence_ids", ())
    ):
        return EvidenceValidation(
            valid=True,
            status="AMBIGUOUS",
            reason=(
                "evidence is structurally valid but is marked as derived "
                "without identifying its parent evidence"
            ),
            independent=False,
            missing=("derived_from_or_parent_evidence_ids",),
            schema_valid=True,
            provenance_valid=False,
            scoring_eligible=False,
        )

    if derived:
        return EvidenceValidation(
            valid=True,
            status="DERIVED",
            reason=(
                "evidence is structurally valid but depends on other evidence "
                "and therefore cannot participate as independent evidence"
            ),
            independent=False,
            missing=(),
            schema_valid=True,
            provenance_valid=True,
            scoring_eligible=False,
        )

    # ------------------------------------------------------------------
    # Runtime provenance
    # ------------------------------------------------------------------

    if kind in _RUNTIME_KINDS:
        runtime_missing = _missing_runtime_provenance(item)

        if runtime_missing or not _has_runtime_provenance(item):
            return EvidenceValidation(
                valid=True,
                status="AMBIGUOUS",
                reason=(
                    "runtime evidence is structurally valid but lacks sufficient "
                    "execution provenance for independent scoring"
                ),
                independent=False,
                missing=tuple(runtime_missing),
                schema_valid=True,
                provenance_valid=False,
                scoring_eligible=False,
            )

        if origin not in {"execution", "source"}:
            return EvidenceValidation(
                valid=True,
                status="AMBIGUOUS",
                reason=(
                    "runtime evidence has execution provenance but an origin "
                    "that does not establish direct execution lineage"
                ),
                independent=False,
                missing=("execution_origin",),
                schema_valid=True,
                provenance_valid=False,
                scoring_eligible=False,
            )

    # ------------------------------------------------------------------
    # External evidence
    # ------------------------------------------------------------------

    if kind == "historical" and origin != "external":
        return EvidenceValidation(
            valid=True,
            status="AMBIGUOUS",
            reason=(
                "historical evidence should identify an external provenance "
                "source before participating in independent scoring"
            ),
            independent=False,
            missing=("external_origin",),
            schema_valid=True,
            provenance_valid=False,
            scoring_eligible=False,
        )

    # ------------------------------------------------------------------
    # Valid independent evidence
    # ------------------------------------------------------------------

    return EvidenceValidation(
        valid=True,
        status="VALID",
        reason="required structural and provenance fields are present",
        independent=True,
        missing=(),
        schema_valid=True,
        provenance_valid=True,
        scoring_eligible=True,
    )
