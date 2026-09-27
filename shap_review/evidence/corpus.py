from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class EvidenceRecord:
    id: str
    type: str
    title: str
    url: str
    bug_class: str
    invariant: str
    affected_components: tuple[str, ...]
    observed_behavior: str
    expected_behavior: str
    status: str = "documented"
    reported_version: str = ""
    fixed_version: str = ""
    source_date: str = ""
    reproduction: str = ""
    regression_test: str = ""
    fix: str = ""
    provenance: str = "primary"
    observation: str = ""
    supports: tuple[str, ...] = ()

    def to_dict(self):
        return asdict(self)

    def validate(self) -> list[str]:
        errors = []
        if not self.id or not self.id.startswith("SHAP-EVID-"):
            errors.append("id must start with SHAP-EVID-")
        if self.type not in {
            "issue",
            "documentation",
            "paper",
            "pull_request",
            "test",
            "architecture",
        }:
            errors.append("unsupported evidence type")
        if not self.title:
            errors.append("missing title")
        if urlparse(self.url).scheme not in {"http", "https"}:
            errors.append("invalid URL")
        if not self.bug_class.startswith("SHAP-"):
            errors.append("invalid bug class")
        if not self.invariant.startswith("INV-"):
            errors.append("invalid invariant")
        if not self.affected_components:
            errors.append("missing affected components")
        if not self.observed_behavior or not self.expected_behavior:
            errors.append("missing observed/expected behavior")
        if self.provenance not in {"primary", "secondary"}:
            errors.append("invalid provenance")
        return errors


DEFAULT_EVIDENCE = [
    EvidenceRecord(
        "SHAP-EVID-4911",
        "issue",
        "TreeExplainer fails with pandas nullable dtypes in background data",
        "https://github.com/shap/shap/issues/4911",
        "SHAP-05",
        "INV-INPUT-001",
        ("TreeExplainer", "Tabular", "_cext"),
        "A pandas nullable DataFrame can produce an object-dtype array that reaches numeric native processing and fails.",
        "Supported numeric input should be converted to a numeric representation or rejected with a clear contract error.",
        reported_version="0.51.0",
        source_date="2026-04-29",
        reproduction="data/regressions/SHAP-4911/reproduce.py",
        regression_test="tests/regressions/test_historical.py::test_4911_nullable_dtype",
        provenance="primary",
    ),
    EvidenceRecord(
        "SHAP-EVID-5098",
        "issue",
        "TreeExplainer ignores model_output when model is already a TreeEnsemble",
        "https://github.com/shap/shap/issues/5098",
        "SHAP-02",
        "INV-OUT-001",
        ("TreeExplainer", "TreeEnsemble"),
        "When a pre-built TreeEnsemble is passed, the requested model_output can diverge from the internal model_output used by downstream checks.",
        "Requested output space must be propagated consistently into the internal model representation and explanation path.",
        reported_version="main at 2edcb04b",
        source_date="2026-07-24",
        reproduction="data/regressions/SHAP-5098/reproduce.py",
        provenance="primary",
    ),
    EvidenceRecord(
        "SHAP-EVID-4495",
        "issue",
        "TreeExplainer.shap_values() mutates expected_value for XGBoost models",
        "https://github.com/shap/shap/issues/4495",
        "SHAP-07",
        "INV-STATE-001",
        ("TreeExplainer", "expected_value", "XGBoost"),
        "expected_value can change value and shape after shap_values() is called, even though the explainer object was already initialized.",
        "expected_value should be correct after initialization and remain stable across equivalent calls.",
        reported_version="0.51.0",
        source_date="2026-04-12",
        reproduction="data/regressions/SHAP-4495/reproduce.py",
        regression_test="tests/regressions/test_historical.py::test_4495_expected_value_stability",
        provenance="primary",
    ),
    EvidenceRecord(
        "SHAP-EVID-4869",
        "issue",
        "Missing multiclass/multioutput/additivity coverage",
        "https://github.com/shap/shap/issues/4869",
        "SHAP-01",
        "INV-ATTR-001",
        ("TreeExplainer", "tests"),
        "Important output and additivity combinations lacked direct regression coverage.",
        "Contract-sensitive output combinations should have explicit assertions for shape and local accuracy.",
        source_date="2026-04-20",
        provenance="primary",
    ),
    EvidenceRecord(
        "SHAP-EVID-1539",
        "issue",
        "Tree SHAP numerical/additivity edge cases",
        "https://github.com/shap/shap/issues/1539",
        "SHAP-08",
        "INV-ATTR-001",
        ("TreeExplainer",),
        "Numerical edge cases have historically exercised additivity behavior.",
        "Supported explanations should satisfy the local-accuracy contract within documented numerical tolerance.",
        source_date="2020-05-01",
        provenance="primary",
    ),
    EvidenceRecord(
        "SHAP-EVID-2778",
        "issue",
        "Sparse/additivity edge case history",
        "https://github.com/shap/shap/issues/2778",
        "SHAP-08",
        "INV-ATTR-001",
        ("TreeExplainer", "sparse"),
        "Sparse and unusual input representations have historically exercised additivity behavior.",
        "Supported representations should preserve the explanation contract.",
        source_date="2022-01-01",
        provenance="primary",
    ),
]


class EvidenceCorpus:
    def __init__(self, records=None):
        self.records = list(records or DEFAULT_EVIDENCE)

    def by_bug_class(self, bug_class):
        return [r for r in self.records if r.bug_class == bug_class]

    def get(self, evidence_id):
        return next((r for r in self.records if r.id == evidence_id), None)

    def validate(self):
        return {r.id: r.validate() for r in self.records if r.validate()}

    def as_dict(self):
        return {r.id: r.to_dict() for r in self.records}

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def from_directory(cls, directory: Path):
        records = []
        for p in sorted(directory.glob("*.json")):
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                # Only canonical evidence records participate in the corpus; legacy
                # issue-shape files remain useful to humans but must not duplicate
                # the source-of-truth records.
                if raw.get("id", "").startswith("SHAP-EVID-"):
                    records.append(EvidenceRecord(**raw))
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        return cls(records or DEFAULT_EVIDENCE)
