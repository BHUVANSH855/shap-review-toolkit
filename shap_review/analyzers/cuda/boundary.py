from pathlib import Path

from shap_review.analyzers.common.base import Analyzer
from shap_review.semantic.native import analyze_native_file
from shap_review.types import Candidate, EvidenceRef


class CudaAnalyzer(Analyzer):
    name = "cuda"

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
            if b.technology != "cuda":
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
            msg = f"CUDA `{b.operation}` boundary requires explicit allocation/error/synchronization validation and a CPU differential oracle where semantics permit."
            if flow:
                msg += " Observed flow: " + ", ".join(flow) + "."
            if risks:
                msg += " Risks: " + ", ".join(risks) + "."
            out.append(
                Candidate(
                    f"SHAP-CAND-CUDA-{b.file}-{b.line}-{b.operation}",
                    "SHAP-10",
                    "INV-NATIVE-001",
                    b.file,
                    b.line,
                    evidence.function if evidence else None,
                    msg,
                    [
                        EvidenceRef(
                            "architecture",
                            "SHAP CUDA target",
                            "optional GPU Tree SHAP path",
                            3,
                        ),
                        EvidenceRef(
                            "source", "native data-flow context", "; ".join(flow), 2
                        ),
                    ],
                    tags=["cuda", b.operation] + risks,
                )
            )
        return out
