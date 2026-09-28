from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .model import EvidenceItem


@dataclass(frozen=True)
class EvidenceNode:
    evidence_id: str
    kind: str
    source: str
    producer: str
    claim: str
    passed: bool | None
    confidence: float = 1.0
    parent_ids: tuple[str, ...] = ()
    execution_id: str | None = None
    repository_revision: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    transformation: str | None = None
    fixture_id: str | None = None
    input_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceGraph:
    """Provenance DAG used to prevent derived evidence from masquerading as independent."""

    def __init__(self) -> None:
        self.nodes: dict[str, EvidenceNode] = {}

    def add(self, node: EvidenceNode) -> None:
        if node.evidence_id in self.nodes:
            raise ValueError(f"duplicate evidence id: {node.evidence_id}")
        missing = [p for p in node.parent_ids if p not in self.nodes]
        if missing:
            raise ValueError(f"missing evidence parents: {missing}")
        self.nodes[node.evidence_id] = node

    def add_item(
        self,
        item: EvidenceItem,
        *,
        producer: str | None = None,
        execution_id: str | None = None,
        repository_revision: str | None = None,
        environment: dict[str, Any] | None = None,
        transformation: str | None = None,
        fixture_id: str | None = None,
        input_fingerprint: str | None = None,
    ) -> EvidenceNode:
        """Add an EvidenceItem while preserving its canonical provenance."""
        evidence_id = item.evidence_id or f"{item.kind.value}:{len(self.nodes) + 1}"

        node = EvidenceNode(
            evidence_id=evidence_id,
            kind=item.kind.value,
            source=item.source,
            producer=producer if producer is not None else (item.producer or "unknown"),
            claim=item.claim,
            passed=item.passed,
            confidence=item.confidence,
            parent_ids=tuple(item.derived_from),
            execution_id=(
                execution_id
                if execution_id is not None
                else item.execution_id
            ),
            repository_revision=(
                repository_revision
                if repository_revision is not None
                else item.repository_revision
            ),
            environment=dict(environment or {}),
            transformation=(
                transformation
                if transformation is not None
                else item.transformation
            ),
            fixture_id=(
                fixture_id if fixture_id is not None else item.fixture_id
            ),
            input_fingerprint=(
                input_fingerprint
                if input_fingerprint is not None
                else item.input_fingerprint
            ),
        )
        self.add(node)
        return node

    def ancestors(self, evidence_id: str) -> set[str]:
        seen: set[str] = set()
        stack = [evidence_id]
        while stack:
            current = stack.pop()
            if current in seen or current not in self.nodes:
                continue
            seen.add(current)
            stack.extend(self.nodes[current].parent_ids)
        seen.discard(evidence_id)
        return seen

    def is_independent(self, evidence_id: str, other_id: str) -> bool:
        if (
            evidence_id == other_id
            or evidence_id not in self.nodes
            or other_id not in self.nodes
        ):
            return False
        left, right = self.nodes[evidence_id], self.nodes[other_id]
        # Evidence produced by the same execution/fixture is correlated even if the
        # graph has no explicit parent edge. A missing execution id is not assumed
        # to be independent; it is simply unknown to this rule.
        if (
            left.execution_id
            and right.execution_id
            and left.execution_id == right.execution_id
        ):
            return False
        if left.fixture_id and right.fixture_id and left.fixture_id == right.fixture_id:
            return False
        if (
            left.input_fingerprint
            and right.input_fingerprint
            and left.input_fingerprint == right.input_fingerprint
        ):
            return False
        if (
            left.repository_revision
            and right.repository_revision
            and left.repository_revision == right.repository_revision
            and left.environment
            and right.environment
            and left.environment == right.environment
        ):
            # Same revision alone does not imply correlation, but matching
            # revision and environment are treated as correlated.
            return False
        if (
            left.transformation
            and right.transformation
            and left.transformation == right.transformation
        ):
            return False
        # Producer identity alone does not prove correlation. The same validator
        # may produce independent observations in separate executions.
        left_ancestry = self.ancestors(evidence_id)
        right_ancestry = self.ancestors(other_id)

        return not (
            evidence_id in right_ancestry
            or other_id in left_ancestry
            or left_ancestry.intersection(right_ancestry)
        )

    def independence_reason(self, evidence_id: str, other_id: str) -> str:
        if evidence_id not in self.nodes or other_id not in self.nodes:
            return "unknown-evidence"
        a, b = self.nodes[evidence_id], self.nodes[other_id]
        if a.execution_id and b.execution_id and a.execution_id == b.execution_id:
            return "shared-execution-lineage"
        if a.fixture_id and b.fixture_id and a.fixture_id == b.fixture_id:
            return "shared-fixture"
        if (
            a.input_fingerprint
            and b.input_fingerprint
            and a.input_fingerprint == b.input_fingerprint
        ):
            return "shared-input-lineage"
        if (
            a.repository_revision
            and b.repository_revision
            and a.repository_revision == b.repository_revision
            and a.environment
            and b.environment
            and a.environment == b.environment
        ):
            return "shared-revision-environment"
        if (
            a.transformation
            and b.transformation
            and a.transformation == b.transformation
        ):
            return "shared-transformation"
        left_ancestors = self.ancestors(evidence_id)
        right_ancestors = self.ancestors(other_id)

        if (
            evidence_id in right_ancestors
            or other_id in left_ancestors
            or left_ancestors.intersection(right_ancestors)
        ):
            return "shared-ancestry"

        # Producer identity alone is not sufficient to establish correlation.
        return "independent"

    def independent_pairs(self) -> list[tuple[str, str]]:
        ids = list(self.nodes)
        return [
            (a, b)
            for i, a in enumerate(ids)
            for b in ids[i + 1 :]
            if self.is_independent(a, b)
        ]

    def root_ids(self) -> list[str]:
        return sorted(eid for eid, node in self.nodes.items() if not node.parent_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "roots": self.root_ids(),
        }
