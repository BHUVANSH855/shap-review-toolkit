from __future__ import annotations

from pathlib import Path

from shap_review.analyzers import (
    CudaAnalyzer,
    NanobindAnalyzer,
    NativeBoundaryAnalyzer,
    PythonContractAnalyzer,
    SHAPSemanticAnalyzer,
)
from shap_review.candidates import CandidateAggregator
from shap_review.discovery import (
    BuildScanner,
    ComponentScanner,
    DependencyScanner,
    HistoryScanner,
    NativeScanner,
    RepositoryScanner,
    TestScanner,
)
from shap_review.evidence import EvidenceCorpus
from shap_review.regressions import run_all
from shap_review.semantic import SHAPSemanticMapper
from shap_review.semantic_ir import SemanticIRBuilder
from shap_review.types import Candidate, write_json
from shap_review.utils import fingerprint


class ReviewEngine:
    def _corpus(self, root):
        local = Path(root) / "data/evidence/issues"
        bundled = Path(__file__).resolve().parent / "resources/evidence/issues"
        return EvidenceCorpus.from_directory(local if local.exists() else bundled)

    def discover(self, root: str | Path, out: str | Path | None = None) -> dict:
        root = Path(root).resolve()
        artifact = Path(out) if out else root / ".shap-review" / "discovery"
        artifact.mkdir(parents=True, exist_ok=True)
        data = {
            "repository": RepositoryScanner().scan(root).__dict__,
            "build": BuildScanner().scan(root),
            "native": NativeScanner().scan(root),
            "components": ComponentScanner().scan(root),
            "dependencies": DependencyScanner().scan(root),
            "tests": TestScanner().scan(root),
            "history": HistoryScanner().scan(root),
        }
        write_json(artifact / "discovery.json", data)
        graph = SHAPSemanticMapper().map(root)
        graph.save(artifact / "semantic-graph.json")
        ir = SemanticIRBuilder().build(root)
        ir.save(artifact / "analysis-ir.json")
        corpus = self._corpus(root)
        corpus.save(artifact / "evidence-corpus.json")
        write_json(artifact / "evidence-validation.json", corpus.validate())
        issue_numbers = []
        for record in corpus.records:
            issue_numbers.extend(record.id.replace("SHAP-EVID-", "").split("-", 1)[:1])
        history = HistoryScanner().scan(root, issue_numbers=issue_numbers)
        write_json(artifact / "history.json", history)
        data["history"] = history
        return data | {
            "semantic_graph": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
            "analysis_ir": {
                "symbols": len(ir.symbols),
                "calls": len(ir.calls),
                "flows": len(ir.flows),
                "properties": len(ir.properties),
                "boundaries": len(ir.boundaries),
            },
            "evidence_records": len(corpus.records),
            "evidence_errors": len(corpus.validate()),
        }

    def analyze(
        self, root: str | Path, out: str | Path | None = None
    ) -> list[Candidate]:
        root = Path(root).resolve()
        artifact = Path(out) if out else root / ".shap-review" / "candidates"
        artifact.mkdir(parents=True, exist_ok=True)
        ir = SemanticIRBuilder().build(root)
        analyzers = [
            PythonContractAnalyzer(),
            SHAPSemanticAnalyzer(ir),
            NativeBoundaryAnalyzer(ir),
            NanobindAnalyzer(ir),
            CudaAnalyzer(ir),
        ]
        raw = []
        for analyzer in analyzers:
            raw.extend(analyzer.analyze(root))
        candidates = CandidateAggregator().merge(raw)
        payload = []
        for c in candidates:
            payload.append(c.to_dict())
        write_json(artifact / "signals.json", [{"analyzer": a.name} for a in analyzers])
        write_json(artifact / "candidates.json", payload)
        write_json(
            artifact / "summary.json",
            {
                "raw_signals": len(raw),
                "candidates": len(candidates),
                "confidence": {
                    "high": sum(c.confidence == "high" for c in candidates),
                    "medium": sum(c.confidence == "medium" for c in candidates),
                    "low": sum(c.confidence == "low" for c in candidates),
                },
            },
        )
        return candidates

    def regressions(self, out: str | Path | None = None) -> dict:
        results = [r.to_dict() for r in run_all()]
        payload = {
            "results": results,
            "reproduced": sum(r["reproduced"] for r in results),
            "blocked": sum(r["status"] == "blocked" for r in results),
        }
        if out:
            write_json(Path(out) / "historical-regressions.json", payload)
        return payload

    def full_scan(self, root: str | Path) -> dict:
        root = Path(root).resolve()
        self.discover(root)
        candidates = self.analyze(root)
        regressions = self.regressions(root / ".shap-review")
        return {
            "root": str(root),
            "candidates": len(candidates),
            "candidate_fingerprints": [
                fingerprint(c.bug_class, c.file, str(c.line), c.message)
                for c in candidates
            ],
            "historical_regressions": regressions,
        }
