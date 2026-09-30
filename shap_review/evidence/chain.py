from __future__ import annotations

from typing import Any

from .graph import EvidenceGraph, EvidenceNode
from .model import EvidenceChain, EvidenceItem, EvidenceKind, EvidenceOrigin


def build_chain(
    *, entries: list[dict[str, Any]] | None = None, **groups
) -> EvidenceChain:
    """Build an evidence chain and preserve provenance consistently."""
    chain = EvidenceChain()

    if entries is None:
        entries = []
        for name, values in groups.items():
            for evidence in values or []:
                entries.append({**evidence, "kind": name})

    legacy = {
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

    for index, entry in enumerate(entries, 1):
        raw_kind = entry["kind"]
        kind = legacy.get(raw_kind, raw_kind)

        if raw_kind in {"issue", "pull_request"}:
            default_origin = "external"
        elif raw_kind == "test-gap":
            default_origin = "derived"
        else:
            default_origin = "source"

        origin = entry.get("origin", default_origin)
        details = dict(entry.get("details", {}))

        execution_id = entry.get("execution_id", details.get("execution_id"))
        producer = entry.get("producer", details.get("producer"))
        fixture_id = entry.get("fixture_id", details.get("fixture_id"))
        input_fingerprint = entry.get(
            "input_fingerprint",
            details.get("input_fingerprint"),
        )
        repository_revision = entry.get(
            "repository_revision",
            details.get("repository_revision"),
        )
        environment_fingerprint = entry.get(
            "environment_fingerprint",
            details.get("environment_fingerprint"),
        )
        parent_evidence_ids = tuple(
            entry.get(
                "parent_evidence_ids",
                details.get("parent_evidence_ids", ()),
            )
        )
        transformation = entry.get(
            "transformation",
            details.get("transformation"),
        )

        chain.add(
            EvidenceItem(
                kind=EvidenceKind(kind),
                source=str(entry.get("source", "unknown")),
                claim=str(entry.get("claim", "")),
                passed=entry.get("passed"),
                confidence=float(entry.get("confidence", 1.0)),
                origin=EvidenceOrigin(origin),
                derived_from=tuple(entry.get("derived_from", ())),
                evidence_id=str(entry.get("evidence_id", f"e{index}")),
                details=details,
                execution_id=execution_id,
                producer=producer,
                fixture_id=fixture_id,
                input_fingerprint=input_fingerprint,
                repository_revision=repository_revision,
                environment_fingerprint=environment_fingerprint,
                parent_evidence_ids=parent_evidence_ids,
                transformation=transformation,
            )
        )

    graph = EvidenceGraph()
    ids = {item.evidence_id for item in chain.items}
    pending = list(chain.items)

    while pending:
        progressed = False

        for item in pending[:]:
            parents = tuple(parent for parent in item.derived_from if parent in ids)

            if not all(parent in graph.nodes for parent in parents):
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

        # Preserve unresolved provenance as terminal nodes instead of silently
        # dropping evidence whose parent relationships cannot be resolved.
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

    chain.graph = graph
    return chain
