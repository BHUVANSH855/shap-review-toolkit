from __future__ import annotations

from typing import Any

from .graph import EvidenceGraph, EvidenceNode
from .model import EvidenceChain, EvidenceItem, EvidenceKind, EvidenceOrigin

_LEGACY_KIND_MAP = {
    "issue": "historical",
    "pull_request": "historical",
    "test-gap": "static",
    "source": "static",
    "documentation": "static",
    "paper": "historical",
    "dynamic": "dynamic",
    "differential": "differential",
    "sanitizer": "sanitizer",
    "reproduction": "reproduction",
}

_EXTERNAL_KINDS = {"issue", "pull_request"}
_DERIVED_KINDS = {"test-gap"}


def _default_origin(raw_kind: str) -> str:
    """Return the safest provenance origin for a raw evidence kind."""
    if raw_kind in _EXTERNAL_KINDS:
        return "external"
    if raw_kind in _DERIVED_KINDS:
        return "derived"
    return "source"


def _normalise_kind(raw_kind: Any) -> EvidenceKind:
    """Convert legacy/current evidence kinds into the canonical enum."""
    kind = _LEGACY_KIND_MAP.get(str(raw_kind), str(raw_kind))
    return EvidenceKind(kind)


def _metadata(entry: dict[str, Any], key: str) -> Any:
    """Read provenance metadata from the entry or its details mapping."""
    details = entry.get("details") or {}
    return entry.get(key, details.get(key))


def _normalise_parent_ids(entry: dict[str, Any]) -> tuple[str, ...]:
    """Return stable, duplicate-free parent evidence IDs."""
    raw = _metadata(entry, "parent_evidence_ids") or ()

    if isinstance(raw, str):
        raw = (raw,)

    result: list[str] = []
    seen: set[str] = set()

    for parent in raw:
        parent_id = str(parent).strip()
        if parent_id and parent_id not in seen:
            seen.add(parent_id)
            result.append(parent_id)

    return tuple(result)


def _build_entries(
    entries: list[dict[str, Any]] | None,
    groups: dict[str, Any],
) -> list[dict[str, Any]]:
    """Normalise explicit entries and legacy grouped evidence."""
    if entries is not None:
        return list(entries)

    result: list[dict[str, Any]] = []

    for name, values in groups.items():
        for evidence in values or []:
            result.append({**evidence, "kind": name})

    return result


def _make_evidence_item(
    entry: dict[str, Any],
    index: int,
    used_ids: set[str],
) -> EvidenceItem:
    """Construct one canonical evidence item with validated provenance."""
    if "kind" not in entry:
        raise ValueError(f"Evidence entry {index} is missing required field 'kind'")

    raw_kind = str(entry["kind"])
    kind = _normalise_kind(raw_kind)

    origin = str(entry.get("origin", _default_origin(raw_kind)))

    try:
        evidence_origin = EvidenceOrigin(origin)
    except ValueError as exc:
        raise ValueError(
            f"Evidence entry {index} has invalid origin {origin!r}"
        ) from exc

    evidence_id = str(entry.get("evidence_id", f"e{index}")).strip()

    if not evidence_id:
        raise ValueError(f"Evidence entry {index} has an empty evidence_id")

    if evidence_id in used_ids:
        raise ValueError(f"Duplicate evidence_id {evidence_id!r} in evidence chain")

    used_ids.add(evidence_id)

    details = dict(entry.get("details") or {})

    execution_id = _metadata(entry, "execution_id")
    producer = _metadata(entry, "producer")
    fixture_id = _metadata(entry, "fixture_id")
    input_fingerprint = _metadata(entry, "input_fingerprint")
    repository_revision = _metadata(entry, "repository_revision")
    environment_fingerprint = _metadata(entry, "environment_fingerprint")
    transformation = _metadata(entry, "transformation")

    confidence = float(entry.get("confidence", 1.0))

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"Evidence entry {evidence_id!r} has invalid confidence "
            f"{confidence}; expected a value in [0, 1]"
        )

    derived_from = tuple(
        str(parent) for parent in entry.get("derived_from", ()) if str(parent)
    )

    return EvidenceItem(
        kind=kind,
        source=str(entry.get("source", "unknown")),
        claim=str(entry.get("claim", "")),
        passed=entry.get("passed"),
        confidence=confidence,
        origin=evidence_origin,
        derived_from=derived_from,
        evidence_id=evidence_id,
        details=details,
        execution_id=execution_id,
        producer=producer,
        fixture_id=fixture_id,
        input_fingerprint=input_fingerprint,
        repository_revision=repository_revision,
        environment_fingerprint=environment_fingerprint,
        parent_evidence_ids=_normalise_parent_ids(entry),
        transformation=transformation,
    )


def _add_graph_items(chain: EvidenceChain) -> EvidenceGraph:
    """Build an evidence graph while preserving unresolved provenance."""
    graph = EvidenceGraph()
    evidence_ids = {item.evidence_id for item in chain.items}

    pending = list(chain.items)

    while pending:
        progressed = False

        for item in pending[:]:
            parents = tuple(
                parent for parent in item.derived_from if parent in evidence_ids
            )

            unresolved = tuple(
                parent for parent in parents if parent not in graph.nodes
            )

            if unresolved:
                continue

            graph.add_item(
                item,
                producer=item.producer or "unknown",
                execution_id=item.execution_id,
                repository_revision=item.repository_revision,
                environment=(
                    {"fingerprint": item.environment_fingerprint}
                    if item.environment_fingerprint
                    else {}
                ),
                transformation=item.transformation,
                fixture_id=item.fixture_id,
                input_fingerprint=item.input_fingerprint,
            )

            pending.remove(item)
            progressed = True

        if progressed:
            continue

        # Some evidence may intentionally refer to provenance that is not
        # present in the current chain. Preserve the evidence as a terminal
        # node rather than silently dropping it.
        for item in pending:
            graph.nodes[item.evidence_id] = EvidenceNode(
                item.evidence_id,
                item.kind.value,
                item.source,
                item.producer or "builder",
                item.claim,
                item.passed,
                item.confidence,
                tuple(parent for parent in item.derived_from if parent in graph.nodes),
                item.execution_id,
                item.repository_revision,
                (
                    {"fingerprint": item.environment_fingerprint}
                    if item.environment_fingerprint
                    else {}
                ),
                item.transformation,
                item.fixture_id,
                item.input_fingerprint,
            )

        break

    return graph


def build_chain(
    *, entries: list[dict[str, Any]] | None = None, **groups: Any
) -> EvidenceChain:
    """Build an evidence chain with validated and preserved provenance.

    The builder accepts both the current explicit ``entries`` representation
    and the historical grouped representation used by older callers.

    Evidence IDs must be unique within a chain, confidence values must remain
    within ``[0, 1]``, and invalid provenance origins are rejected instead of
    being silently converted. Unresolved parent references are preserved as
    terminal graph nodes so evidence is never silently discarded.
    """
    chain = EvidenceChain()
    raw_entries = _build_entries(entries, groups)

    used_ids: set[str] = set()

    for index, entry in enumerate(raw_entries, 1):
        chain.add(
            _make_evidence_item(
                entry,
                index,
                used_ids,
            )
        )

    chain.graph = _add_graph_items(chain)
    return chain
