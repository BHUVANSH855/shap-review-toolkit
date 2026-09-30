from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .model import EvidenceItem


@dataclass(frozen=True)
class EvidenceNode:
    """A single provenance-aware node in the evidence graph."""

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
    """Provenance DAG used to prevent derived evidence from masquerading as independent.

    The graph treats explicit ancestry, execution lineage, fixture identity,
    input identity, and selected provenance metadata as correlation signals.
    Independence is therefore conservative: absence of a known correlation is
    not interpreted as proof of experimental independence.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, EvidenceNode] = {}

    # ------------------------------------------------------------------
    # Graph construction and validation
    # ------------------------------------------------------------------

    def add(self, node: EvidenceNode) -> None:
        """Add a provenance node while enforcing DAG invariants."""
        if not node.evidence_id:
            raise ValueError("evidence id must not be empty")

        if node.evidence_id in self.nodes:
            raise ValueError(f"duplicate evidence id: {node.evidence_id}")

        if node.evidence_id in node.parent_ids:
            raise ValueError(
                f"evidence node cannot reference itself as a parent: {node.evidence_id}"
            )

        missing = [parent for parent in node.parent_ids if parent not in self.nodes]
        if missing:
            raise ValueError(f"missing evidence parents: {missing}")

        if not 0.0 <= node.confidence <= 1.0:
            raise ValueError(
                f"evidence confidence must be between 0 and 1: {node.confidence!r}"
            )

        # Parents are required to exist before the child is inserted, so any
        # newly introduced edge can only point backward in the construction
        # order. This prevents cycles in normal graph construction.
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
            producer=(
                producer if producer is not None else (item.producer or "unknown")
            ),
            claim=item.claim,
            passed=item.passed,
            confidence=item.confidence,
            parent_ids=tuple(item.derived_from),
            execution_id=(
                execution_id if execution_id is not None else item.execution_id
            ),
            repository_revision=(
                repository_revision
                if repository_revision is not None
                else item.repository_revision
            ),
            environment=dict(environment or {}),
            transformation=(
                transformation if transformation is not None else item.transformation
            ),
            fixture_id=fixture_id if fixture_id is not None else item.fixture_id,
            input_fingerprint=(
                input_fingerprint
                if input_fingerprint is not None
                else item.input_fingerprint
            ),
        )

        self.add(node)
        return node

    def validate(self) -> None:
        """Validate graph structure and provenance invariants.

        Raises
        ------
        ValueError
            If the graph contains duplicate IDs, missing parents, cycles, or
            invalid confidence values.
        """
        for evidence_id, node in self.nodes.items():
            if evidence_id != node.evidence_id:
                raise ValueError(
                    f"graph key does not match evidence id: {evidence_id!r}"
                )

            if not 0.0 <= node.confidence <= 1.0:
                raise ValueError(
                    f"invalid confidence for {evidence_id}: {node.confidence!r}"
                )

            if evidence_id in node.parent_ids:
                raise ValueError(
                    f"evidence node cannot reference itself: {evidence_id}"
                )

            missing = [parent for parent in node.parent_ids if parent not in self.nodes]
            if missing:
                raise ValueError(
                    f"missing evidence parents for {evidence_id}: {missing}"
                )

        # Explicit DFS cycle detection makes validation robust even if the
        # graph was constructed or mutated outside add().
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(evidence_id: str) -> None:
            if evidence_id in visiting:
                raise ValueError(f"cycle detected in evidence graph at {evidence_id!r}")

            if evidence_id in visited:
                return

            visiting.add(evidence_id)

            for parent in self.nodes[evidence_id].parent_ids:
                visit(parent)

            visiting.remove(evidence_id)
            visited.add(evidence_id)

        for evidence_id in self.nodes:
            visit(evidence_id)

    # ------------------------------------------------------------------
    # Ancestry
    # ------------------------------------------------------------------

    def ancestors(self, evidence_id: str) -> set[str]:
        """Return all transitive ancestors of an evidence node."""
        if evidence_id not in self.nodes:
            return set()

        seen: set[str] = set()
        stack = list(self.nodes[evidence_id].parent_ids)

        while stack:
            current = stack.pop()

            if current in seen:
                continue

            node = self.nodes.get(current)
            if node is None:
                continue

            seen.add(current)
            stack.extend(node.parent_ids)

        return seen

    def root_ids(self) -> list[str]:
        """Return evidence nodes with no explicit parents."""
        return sorted(
            evidence_id
            for evidence_id, node in self.nodes.items()
            if not node.parent_ids
        )

    # ------------------------------------------------------------------
    # Correlation / independence
    # ------------------------------------------------------------------

    def _correlation_reason(
        self,
        evidence_id: str,
        other_id: str,
    ) -> str | None:
        """Return a concrete correlation reason, or ``None`` if unknown."""
        if evidence_id not in self.nodes or other_id not in self.nodes:
            return "unknown-evidence"

        if evidence_id == other_id:
            return "same-evidence"

        left = self.nodes[evidence_id]
        right = self.nodes[other_id]

        if (
            left.execution_id
            and right.execution_id
            and left.execution_id == right.execution_id
        ):
            return "shared-execution-lineage"

        if left.fixture_id and right.fixture_id and left.fixture_id == right.fixture_id:
            return "shared-fixture"

        if (
            left.input_fingerprint
            and right.input_fingerprint
            and left.input_fingerprint == right.input_fingerprint
        ):
            return "shared-input-lineage"

        if (
            left.repository_revision
            and right.repository_revision
            and left.repository_revision == right.repository_revision
            and left.environment
            and right.environment
            and left.environment == right.environment
        ):
            return "shared-revision-environment"

        if (
            left.transformation
            and right.transformation
            and left.transformation == right.transformation
        ):
            return "shared-transformation"

        left_ancestors = self.ancestors(evidence_id)
        right_ancestors = self.ancestors(other_id)

        if evidence_id in right_ancestors or other_id in left_ancestors:
            return "direct-ancestry"

        if left_ancestors.intersection(right_ancestors):
            return "shared-ancestry"

        return None

    def is_independent(self, evidence_id: str, other_id: str) -> bool:
        """Return whether no known correlation exists between two nodes.

        This is deliberately conservative.  ``True`` means the graph found no
        known shared lineage, execution, fixture, or input identity.  It does
        not prove that the observations are statistically or experimentally
        independent.
        """
        if evidence_id == other_id:
            return False

        if evidence_id not in self.nodes or other_id not in self.nodes:
            return False

        return self._correlation_reason(evidence_id, other_id) is None

    def independence_reason(self, evidence_id: str, other_id: str) -> str:
        """Explain why two evidence nodes are or are not considered independent."""
        reason = self._correlation_reason(evidence_id, other_id)

        if reason is not None:
            return reason

        return "no-known-correlation"

    def independent_pairs(self) -> list[tuple[str, str]]:
        """Return unordered pairs with no known provenance correlation."""
        ids = list(self.nodes)

        return [
            (left, right)
            for index, left in enumerate(ids)
            for right in ids[index + 1 :]
            if self.is_independent(left, right)
        ]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph without losing provenance metadata."""
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "roots": self.root_ids(),
        }
