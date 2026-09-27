from pathlib import Path

from shap_review.analyzers.common.base import Analyzer
from shap_review.semantic.native import analyze_native_file
from shap_review.types import Candidate, EvidenceRef


class NanobindAnalyzer(Analyzer):
    name = "nanobind"

    def __init__(self, ir=None):
        self.ir = ir

    def analyze(self, root):
        root = Path(root)
        out = []
        if not self.ir:
            from shap_review.semantic_ir import SemanticIRBuilder

            self.ir = SemanticIRBuilder().build(root)
        cache = {}
        for b in self.ir.boundaries:
            if b.technology != "nanobind":
                continue
            cache.setdefault(b.file, analyze_native_file(root, root / b.file))
            evidence = next(
                (
                    e
                    for e in cache[b.file]
                    if e.line == b.line and e.operation == b.operation
                ),
                None,
            )
            flow = list(evidence.data_flow) if evidence else []
            risks = (
                list(evidence.risk_flags)
                if evidence
                else ["native-context-unavailable"]
            )
            msg = f"nanobind `{b.operation}` is a Python/native contract boundary; validate dtype/shape/stride conversion, ownership, exceptions, GIL state, and device/lifetime semantics."
            if flow:
                msg += " Observed flow: " + ", ".join(flow) + "."
            if risks:
                msg += " Risks: " + ", ".join(risks) + "."
            out.append(
                Candidate(
                    f"SHAP-CAND-NB-{b.file}-{b.line}-{b.operation}",
                    "SHAP-05" if b.operation == "ndarray-conversion" else "SHAP-06",
                    "INV-INPUT-001"
                    if b.operation == "ndarray-conversion"
                    else "INV-NATIVE-001",
                    b.file,
                    b.line,
                    evidence.function if evidence else None,
                    msg,
                    [
                        EvidenceRef(
                            "architecture",
                            "SHAP nanobind native targets",
                            "_cutils uses nanobind",
                            4,
                        ),
                        EvidenceRef(
                            "source", "native data-flow context", "; ".join(flow), 2
                        ),
                    ],
                    tags=["nanobind", b.operation] + risks,
                )
            )
        return out
