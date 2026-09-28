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


def validate_evidence_item(item: Any) -> EvidenceValidation:
    """Validate minimum provenance while preserving legacy static evidence.

    IDs may be assigned by ``EvidenceChain`` when omitted. Runtime evidence is
    fail-closed unless it carries an execution, fixture, or explicit ancestry.
    Historical/static references may remain lightweight because they describe
    source material rather than a runtime observation.
    """
    kind = getattr(getattr(item, "kind", None), "value", getattr(item, "kind", None))
    origin = getattr(
        getattr(item, "origin", None), "value", getattr(item, "origin", None)
    )
    missing: list[str] = []
    if kind in {"dynamic", "reproduction", "differential", "sanitizer"}:
        if not getattr(item, "execution_id", None) or (
            not getattr(item, "fixture_id", None)
            and not getattr(item, "input_fingerprint", None)
        ):
            missing.append("execution_id_and_fixture_or_input_fingerprint")
    if (
        origin == "derived"
        and kind not in {"historical", "static"}
        and not getattr(item, "derived_from", ())
    ):
        missing.append("derived_from")
    if missing:
        return EvidenceValidation(
            True,
            "AMBIGUOUS",
            "runtime provenance is missing; evidence is not eligible for independent scoring",
            independent=False,
            missing=tuple(missing),
            schema_valid=True,
            provenance_valid=False,
            scoring_eligible=False,
        )
    return EvidenceValidation(
        True,
        "VALID",
        "minimum provenance present",
        independent=not bool(getattr(item, "derived_from", ())),
        schema_valid=True,
        provenance_valid=True,
        scoring_eligible=not bool(getattr(item, "derived_from", ())),
    )
