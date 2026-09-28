from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class FindingClassification(str, Enum):
    NOVEL_CANDIDATE = "novel_candidate"
    NOVEL_REPRODUCTION = "novel_reproduction"
    HISTORICAL_CANDIDATE = "historical_candidate"
    HISTORICAL_REPRODUCTION = "historical_reproduction"
    KNOWN_ISSUE_REPRODUCED = "known_issue_reproduced"
    REGRESSION_CANDIDATE = "regression_candidate"


class FindingStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    CANDIDATE = "CANDIDATE"
    INVESTIGATING = "INVESTIGATING"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    REPRO_PENDING = "REPRO_PENDING"
    REPRODUCED = "REPRODUCED"
    EVIDENCE_VALID = "EVIDENCE_VALID"
    CONFIRMED = "CONFIRMED"
    REPORTED = "REPORTED"
    FIXED = "FIXED"
    TRACKED = "TRACKED"


class ExecutionStatus(str, Enum):
    EXECUTED = "EXECUTED"
    NOT_EXECUTED = "NOT_EXECUTED"


class ExecutionStage(str, Enum):
    SETUP = "setup"
    REPRESENTATION = "representation"
    MODEL_BUILD = "model_build"
    MODEL_FIT = "model_fit"
    EXPLAINER_CREATE = "explainer_create"
    EXPLAINER_EXECUTE = "explainer_execute"
    OUTPUT_NORMALIZATION = "output_normalization"
    ORACLE = "oracle"


class ExecutionReason(str, Enum):
    COMPLETED = "COMPLETED"
    SHAP_ERROR = "SHAP_ERROR"
    BACKEND_ERROR = "BACKEND_ERROR"
    ADAPTER_ERROR = "ADAPTER_ERROR"
    TOOLKIT_ERROR = "TOOLKIT_ERROR"
    UNSUPPORTED = "UNSUPPORTED"
    SKIPPED = "SKIPPED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class SemanticStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_EVALUATED = "NOT_EVALUATED"


class DifferentialStatus(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    INCONCLUSIVE = "INCONCLUSIVE"
    EXECUTION_FAILED = "EXECUTION_FAILED"


class RegressionStatus(str, Enum):
    REPRODUCED = "reproduced"
    NOT_REPRODUCED = "not_reproduced"
    BLOCKED = "blocked"
    AMBIGUOUS = "ambiguous"
    TARGET_FAILURE = "target_failure"
    TOOLKIT_FAILURE = "toolkit_failure"
    UNSUPPORTED = "unsupported"
    STATIC_PRECONDITION = "static_precondition"
    CANDIDATE_REPRODUCED = "candidate_reproduced"


@dataclass
class EvidenceRef:
    kind: str
    source: str
    note: str = ""
    strength: int = 0


@dataclass
class Candidate:
    candidate_id: str
    bug_class: str
    invariant: str
    file: str
    line: int | None
    symbol: str | None
    message: str
    evidence: list[EvidenceRef] = field(default_factory=list)
    validation_required: bool = True
    confidence: str = "medium"
    tags: list[str] = field(default_factory=list)
    evidence_chain: Any = None
    classification: FindingClassification = FindingClassification.NOVEL_CANDIDATE

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["classification"] = self.classification.value
        if self.evidence_chain is not None:
            d["evidence_chain"] = self.evidence_chain.to_dict()
        return d


@dataclass
class Finding:
    finding_id: str
    candidate_id: str
    status: FindingStatus
    bug_class: str
    invariant: str
    summary: str
    evidence: list[EvidenceRef] = field(default_factory=list)
    reproducer: str | None = None
    sanitizer_evidence: list[str] = field(default_factory=list)
    regression_test: str | None = None
    fingerprint: str | None = None
    notes: list[str] = field(default_factory=list)
    evidence_chain: Any = None
    classification: FindingClassification = FindingClassification.NOVEL_CANDIDATE

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["classification"] = self.classification.value
        if self.evidence_chain is not None:
            d["evidence_chain"] = self.evidence_chain.to_dict()
        return d


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
