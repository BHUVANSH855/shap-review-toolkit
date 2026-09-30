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

# Evidence kinds that require execution provenance before they can participate
# in scoring. This is intentionally stricter than simply checking ``passed``.
_EXECUTION_KINDS = frozenset(
    {
        EvidenceKind.DYNAMIC,
        EvidenceKind.DIFFERENTIAL,
        EvidenceKind.REPRODUCTION,
        EvidenceKind.SANITIZER,
    }
)


def _clamp_confidence(value: float) -> float:
    """Clamp confidence to the closed [0, 1] interval."""
    return max(0.0, min(1.0, float(value)))


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
    def normalized_confidence(self) -> float:
        """Return confidence constrained to the scoring range."""
        return _clamp_confidence(self.confidence)

    @property
    def has_execution_provenance(self) -> bool:
        """Return whether execution evidence has enough provenance to score."""
        return bool(
            self.execution_id
            and (
                self.fixture_id
                or self.input_fingerprint
                or self.repository_revision
                or self.environment_fingerprint
            )
        )

    @property
    def intrinsically_valid(self) -> bool:
        """Return whether the item is structurally valid on its own.

        Derived evidence is deliberately excluded from intrinsic independence.
        Pairwise independence remains the responsibility of ``EvidenceGraph``.
        """
        if (
            self.derived_from
            or self.parent_evidence_ids
            or self.origin == EvidenceOrigin.DERIVED
        ):
            return False

        if self.kind in _EXECUTION_KINDS:
            return self.has_execution_provenance

        return True

    @property
    def independent(self) -> bool:
        """Return intrinsic validity, not pairwise graph independence.

        Two intrinsically valid items may still be correlated. Pairwise
        independence is determined by ``EvidenceGraph``.
        """
        return self.intrinsically_valid

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["origin"] = self.origin.value
        data["confidence"] = self.normalized_confidence
        data["intrinsically_valid"] = self.intrinsically_valid
        data["independence_scope"] = "graph-derived-pairwise"
        data["independent"] = self.independent

        validation = validate_evidence_item(self)
        data["validation_status"] = validation.status
        data["provenance_valid"] = validation.provenance_valid
        data["scoring_eligible"] = validation.scoring_eligible

        return data


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

    def kinds(self) -> set[EvidenceKind]:
        return {item.kind for item in self.items}

    @staticmethod
    def _scoring_eligible(item: EvidenceItem) -> bool:
        """Return whether an evidence item may participate in scoring."""
        validation = validate_evidence_item(item)

        return bool(
            validation.scoring_eligible
            and item.intrinsically_valid
            and item.confidence > 0
        )

    def scoring_items(self) -> list[EvidenceItem]:
        """Return all structurally valid items eligible for scoring."""
        return [item for item in self.items if self._scoring_eligible(item)]

    def independent_items(self) -> list[EvidenceItem]:
        """Return a conservatively selected independent evidence set.

        With a provenance graph attached, pairwise independence is established
        by ``EvidenceGraph``. Without a graph, only one scoring-eligible item
        is admitted. This prevents callers that bypass ``build_chain()`` from
        accidentally converting correlated evidence into confidence.
        """
        eligible = self.scoring_items()

        if self.graph is None:
            return eligible[:1]

        selected: list[EvidenceItem] = []

        # Stable ordering makes scoring deterministic rather than dependent on
        # whichever producer happened to insert evidence first.
        ordered = sorted(
            eligible,
            key=lambda item: (
                -KIND_WEIGHT[item.kind],
                -item.normalized_confidence,
                item.evidence_id,
            ),
        )

        for item in ordered:
            if item.evidence_id not in self.graph.nodes:
                continue

            if all(
                self.graph.is_independent(
                    item.evidence_id,
                    other.evidence_id,
                )
                for other in selected
            ):
                selected.append(item)

        return selected

    def independent_kinds(self) -> int:
        return len({item.kind for item in self.independent_items()})

    def strongest_positive(self) -> float:
        return max(
            (
                KIND_WEIGHT[item.kind] * item.normalized_confidence
                for item in self.independent_items()
                if item.passed is True
            ),
            default=0.0,
        )

    def positive_score(self) -> float:
        """Return the weighted score contributed by positive evidence."""
        return sum(
            KIND_WEIGHT[item.kind] * item.normalized_confidence
            for item in self.independent_items()
            if item.passed is True
        )

    def negative_score(self) -> float:
        """Return the weighted score contributed by negative evidence."""
        return sum(
            KIND_WEIGHT[item.kind] * item.normalized_confidence
            for item in self.independent_items()
            if item.passed is False
        )

    def inconclusive_items(self) -> list[EvidenceItem]:
        """Return evidence that neither confirms nor refutes the claim."""
        return [item for item in self.independent_items() if item.passed is None]

    def evidence_strength_score(self) -> float:
        """Return a heuristic net evidence score.

        This score is intentionally not a probability. Negative evidence is
        discounted rather than treated as a symmetric inverse of positive
        evidence because a failed validation can have several explanations.
        """
        return max(
            0.0,
            self.positive_score() - self.negative_score() * 0.75,
        )

    def score(self) -> float:
        """Backward-compatible alias for ``evidence_strength_score``."""
        return self.evidence_strength_score()

    def _sanitizer_finding(self, items: list[EvidenceItem]) -> bool:
        return any(
            item.kind == EvidenceKind.SANITIZER
            and item.passed is True
            and item.details.get("finding", True) is True
            for item in items
        )

    def _sanitizer_clean(self, items: list[EvidenceItem]) -> bool:
        return any(
            item.kind == EvidenceKind.SANITIZER
            and item.passed is True
            and item.details.get("finding") is False
            for item in items
        )

    def _has_confirming_reproduction(self, items: list[EvidenceItem]) -> bool:
        """Return whether a reproduction provides explicit positive confirmation."""
        return any(
            item.kind == EvidenceKind.REPRODUCTION
            and item.passed is True
            and bool(
                item.details.get("reproduced")
                or item.details.get("confirmed")
                or item.details.get("failure_observed")
            )
            for item in items
        )

    def _has_validated_dynamic(self, items: list[EvidenceItem]) -> bool:
        return any(
            item.kind == EvidenceKind.DYNAMIC and item.passed is True for item in items
        )

    def _has_validated_differential(self, items: list[EvidenceItem]) -> bool:
        return any(
            item.kind == EvidenceKind.DIFFERENTIAL and item.passed is True
            for item in items
        )

    def verdict(self) -> str:
        """Classify the current evidence conservatively.

        Verdicts describe the strength of available evidence; they are not
        probabilities and do not establish that a bug exists without the
        underlying evidence being independently reviewed.
        """
        items = self.independent_items()
        kinds = {item.kind for item in items}
        score = self.evidence_strength_score()

        if not items:
            return "UNVALIDATED"

        if self._sanitizer_finding(items):
            return "SANITIZER_FINDING"

        # A clean sanitizer run is informative but must not erase a positive
        # reproduction or dynamic finding.
        if self._sanitizer_clean(items):
            if self._has_confirming_reproduction(items) or self._has_validated_dynamic(
                items
            ):
                return "CONTRADICTORY_EVIDENCE"
            return "SANITIZER_CLEAN"

        # Confirmation requires an explicitly successful reproduction and
        # independent dynamic evidence. Mere presence of both kinds is not
        # sufficient.
        if (
            self._has_confirming_reproduction(items)
            and self._has_validated_dynamic(items)
            and score >= 3.0
        ):
            return "CONFIRMED"

        # Differential evidence is strong corroboration, but it does not by
        # itself establish a bug. Require an actual positive differential
        # observation and enough net evidence.
        if (
            self._has_validated_differential(items)
            and score >= 3.0
            and (EvidenceKind.REPRODUCTION in kinds or EvidenceKind.DYNAMIC in kinds)
        ):
            return "HIGH_CONFIDENCE"

        if self._has_validated_dynamic(items) and score >= 2.0:
            return "DYNAMICALLY_VALIDATED"

        if (
            EvidenceKind.HISTORICAL in kinds
            and EvidenceKind.STATIC in kinds
            and score > 0
        ):
            return "HISTORICALLY_CORRELATED"

        if EvidenceKind.STATIC in kinds:
            return "STATIC_CANDIDATE"

        return "UNVALIDATED"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the chain with its effective scoring state."""
        score = self.evidence_strength_score()
        independent = self.independent_items()

        return {
            "verdict": self.verdict(),
            "evidence_strength_score": round(score, 3),
            "score": round(score, 3),
            "positive_score": round(self.positive_score(), 3),
            "negative_score": round(self.negative_score(), 3),
            "independent_evidence_kinds": len({item.kind for item in independent}),
            "independent_evidence_count": len(independent),
            "inconclusive_evidence_count": len(
                [item for item in independent if item.passed is None]
            ),
            "graph_attached": self.graph is not None,
            "items": [item.to_dict() for item in self.items],
            "independence_policy": (
                "Pairwise independence is graph-derived; execution, "
                "fixture/input lineage, revision/environment, ancestry and "
                "transformations are considered. When no graph is attached "
                "only a single item participates in scoring. Always use "
                "build_chain()."
            ),
            "score_semantics": (
                "Heuristic evidence-strength score for prioritization, not probability."
            ),
        }
