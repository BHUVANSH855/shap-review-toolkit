from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .validation import validate_evidence_item


class EvidenceKind(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    DIFFERENTIAL = "differential"
    HISTORICAL = "historical"
    SANITIZER = "sanitizer"
    REPRODUCTION = "reproduction"


class EvidenceOrigin(str, Enum):
    SOURCE = "source"
    EXECUTION = "execution"
    EXTERNAL = "external"
    DERIVED = "derived"


KIND_WEIGHT = {
    EvidenceKind.STATIC: 1.0,
    EvidenceKind.HISTORICAL: 1.25,
    EvidenceKind.DYNAMIC: 1.5,
    EvidenceKind.DIFFERENTIAL: 1.75,
    EvidenceKind.REPRODUCTION: 2.0,
    EvidenceKind.SANITIZER: 2.5,
}


@dataclass(frozen=True)
class EvidenceItem:
    kind: EvidenceKind
    source: str
    claim: str
    passed: bool | None
    confidence: float = 1.0
    origin: EvidenceOrigin = EvidenceOrigin.SOURCE
    derived_from: tuple[str, ...] = ()
    evidence_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    execution_id: str | None = None
    producer: str | None = None
    fixture_id: str | None = None
    input_fingerprint: str | None = None
    repository_revision: str | None = None
    environment_fingerprint: str | None = None
    parent_evidence_ids: tuple[str, ...] = ()
    transformation: str | None = None

    @property
    def intrinsically_valid(self) -> bool:
        if (
            self.derived_from
            or self.parent_evidence_ids
            or self.origin == EvidenceOrigin.DERIVED
        ):
            return False

        if self.kind in {
            EvidenceKind.DYNAMIC,
            EvidenceKind.REPRODUCTION,
            EvidenceKind.DIFFERENTIAL,
            EvidenceKind.SANITIZER,
        }:
            return bool(
                self.execution_id
                and (
                    self.fixture_id
                    or self.input_fingerprint
                    or self.repository_revision
                    or self.environment_fingerprint
                )
            )

        return True

    @property
    def independent(self) -> bool:
        """Compatibility field; true only when the item is intrinsically valid.

        Pairwise independence is determined by ``EvidenceGraph`` and is not
        represented by this item-level property.
        """
        return self.intrinsically_valid

    def to_dict(self):
        d = asdict(self)
        d["kind"] = self.kind.value
        d["origin"] = self.origin.value
        d["intrinsically_valid"] = self.intrinsically_valid
        d["independence_scope"] = "graph-derived-pairwise"
        d["independent"] = self.independent
        validation = validate_evidence_item(self)
        d["validation_status"] = validation.status
        d["provenance_valid"] = validation.provenance_valid
        d["scoring_eligible"] = validation.scoring_eligible
        return d


@dataclass
class EvidenceChain:
    items: list[EvidenceItem] = field(default_factory=list)
    graph: Any = None

    def add(self, item: EvidenceItem) -> None:
        validation = validate_evidence_item(item)
        if not validation.valid:
            raise ValueError(
                f"invalid evidence {getattr(item, 'evidence_id', '<unknown>')}: "
                f"{validation.reason}; missing={validation.missing}"
            )
        self.items.append(item)

    def kinds(self):
        return {i.kind for i in self.items}

    @staticmethod
    def _scoring_eligible(item: EvidenceItem) -> bool:
        """Return whether an item is eligible to participate in scoring."""
        validation = validate_evidence_item(item)
        return bool(
            validation.scoring_eligible
            and item.intrinsically_valid
        )

    def independent_items(self):
        eligible = [
            item
            for item in self.items
            if self._scoring_eligible(item)
        ]

        if self.graph is None:
            return eligible

        selected = []
        for item in eligible:
            if item.evidence_id not in self.graph.nodes:
                continue

            if all(
                self.graph.is_independent(item.evidence_id, other.evidence_id)
                for other in selected
            ):
                selected.append(item)

        return selected

    def independent_kinds(self):
        return len({i.kind for i in self.independent_items()})

    def strongest_positive(self):
        return max(
            (
                KIND_WEIGHT[i.kind] * max(0, min(1, i.confidence))
                for i in self.independent_items()
                if i.passed is True
            ),
            default=0.0,
        )

    def evidence_strength_score(self):
        pos = sum(
            KIND_WEIGHT[i.kind] * max(0, min(1, i.confidence))
            for i in self.independent_items()
            if i.passed is True
        )
        neg = sum(
            KIND_WEIGHT[i.kind] * max(0, min(1, i.confidence))
            for i in self.independent_items()
            if i.passed is False
        )
        return max(0.0, pos - neg * 0.75)

    def score(self):
        """Backward-compatible alias for the heuristic evidence-strength score."""
        return self.evidence_strength_score()

    def verdict(self):
        items = self.independent_items()
        kinds = {i.kind for i in items}
        score = self.evidence_strength_score()
        sanitizer_findings = [
            i
            for i in items
            if i.kind == EvidenceKind.SANITIZER
            and i.passed is True
            and i.details.get("finding", True)
        ]
        sanitizer_clean = [
            i
            for i in items
            if i.kind == EvidenceKind.SANITIZER
            and i.passed is True
            and i.details.get("finding") is False
        ]
        if sanitizer_findings:
            return "SANITIZER_FINDING"
        if sanitizer_clean:
            return "SANITIZER_CLEAN"
        if (
            EvidenceKind.REPRODUCTION in kinds
            and EvidenceKind.DYNAMIC in kinds
            and score >= 3.0
        ):
            return "CONFIRMED"
        if EvidenceKind.DIFFERENTIAL in kinds and score >= 3.0:
            return "HIGH_CONFIDENCE"
        if EvidenceKind.DYNAMIC in kinds and score >= 2.0:
            return "DYNAMICALLY_VALIDATED"
        if EvidenceKind.HISTORICAL in kinds and EvidenceKind.STATIC in kinds:
            return "HISTORICALLY_CORRELATED"
        if EvidenceKind.STATIC in kinds:
            return "STATIC_CANDIDATE"
        return "UNVALIDATED"

    def to_dict(self):
        return {
            "verdict": self.verdict(),
            "evidence_strength_score": round(self.evidence_strength_score(), 3),
            "score": round(self.evidence_strength_score(), 3),
            "independent_evidence_kinds": self.independent_kinds(),
            "items": [i.to_dict() for i in self.items],
            "independence_policy": (
                "Pairwise independence is graph-derived; execution, "
                "fixture/input lineage, revision/environment, ancestry and "
                "transformations are considered."
            ),
            "score_semantics": (
                "Heuristic evidence-strength score for prioritization, "
                "not probability."
            ),
        }