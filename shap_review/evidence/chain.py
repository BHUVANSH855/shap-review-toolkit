from __future__ import annotations

from typing import Any

from .graph import EvidenceGraph
from .model import EvidenceChain, EvidenceItem, EvidenceKind, EvidenceOrigin


def build_chain(
    *, entries: list[dict[str, Any]] | None = None, **groups
) -> EvidenceChain:
    chain = EvidenceChain()
    if entries is None:
        entries = []
        for name, values in groups.items():
            for e in values or []:
                entries.append({**e, "kind": name})
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
    for idx, e in enumerate(entries, 1):
        raw_kind = e["kind"]
        kind = legacy.get(raw_kind, raw_kind)
        # Historical evidence (issues, PRs) comes from an external corpus record —
        # it is an independently verifiable public artifact, NOT a derived observation.
        # Using "derived" caused every corpus-linked candidate to score 0.0.
        # "source" evidence (AST patterns) originates from the analysed source tree.
        # Only dynamic/reproduction/differential evidence computed inside a single
        # run should be marked "derived".
        if raw_kind in {"issue", "pull_request"}:
            default_origin = "external"
        elif raw_kind in {"test-gap"}:
            default_origin = "derived"
        else:
            default_origin = "source"
        origin = e.get("origin", default_origin)
        chain.add(
            EvidenceItem(
                kind=EvidenceKind(kind),
                source=str(e.get("source", "unknown")),
                claim=str(e.get("claim", "")),
                passed=e.get("passed"),
                confidence=float(e.get("confidence", 1.0)),
                origin=EvidenceOrigin(origin),
                derived_from=tuple(e.get("derived_from", ())),
                evidence_id=str(e.get("evidence_id", f"e{idx}")),
                details=dict(e.get("details", {})),
                execution_id=e.get(
                    "execution_id", e.get("details", {}).get("execution_id")
                ),
                producer=e.get("producer", e.get("details", {}).get("producer")),
                fixture_id=e.get("fixture_id", e.get("details", {}).get("fixture_id")),
            )
        )
    graph = EvidenceGraph()
    ids = {i.evidence_id for i in chain.items}
    pending = list(chain.items)
    while pending:
        progressed = False
        for item in pending[:]:
            parents = tuple(p for p in item.derived_from if p in ids)
            if all(p in graph.nodes for p in parents):
                graph.add_item(
                    item,
                    producer=str(
                        item.details.get("producer", item.producer or "unknown")
                    ),
                    execution_id=item.details.get("execution_id"),
                    repository_revision=item.details.get("repository_revision"),
                    environment=item.details.get("environment"),
                    transformation=item.details.get("transformation"),
                    fixture_id=item.fixture_id,
                )
                pending.remove(item)
                progressed = True
        if not progressed:
            # Preserve unresolved provenance as a terminal dependent node rather than silently dropping it.
            for item in pending:
                graph.nodes[item.evidence_id] = __import__(
                    "shap_review.evidence.graph", fromlist=["EvidenceNode"]
                ).EvidenceNode(
                    item.evidence_id,
                    item.kind.value,
                    item.source,
                    "builder",
                    item.claim,
                    item.passed,
                    item.confidence,
                    tuple(p for p in item.derived_from if p in graph.nodes),
                    item.details.get("execution_id"),
                    item.details.get("repository_revision"),
                    dict(item.details.get("environment", {})),
                    item.details.get("transformation"),
                    item.fixture_id,
                )
            break
    chain.graph = graph
    return chain
