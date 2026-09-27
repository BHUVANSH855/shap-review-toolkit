from __future__ import annotations

import re
from pathlib import Path

from shap_review.analyzers.common.base import Analyzer
from shap_review.semantic.native import analyze_native_file
from shap_review.semantic_ir import AnalysisIR
from shap_review.types import Candidate, EvidenceRef

_VALIDATION_RE = re.compile(
    r"(?:check|validate|ensure|assert|shape|dtype|strides|contig|PyErr_Occurred|PyArray_Check|PyArray_NDIM|PyArray_ISCONTIGUOUS)"
)
_OWNERSHIP_RE = re.compile(
    r"(?:Py_INCREF|Py_DECREF|Py_XINCREF|Py_XDECREF|Py_NewRef|Py_XNewRef|release\(|borrow|owned|reference|refcount)"
)


class NativeBoundaryAnalyzer(Analyzer):
    name = "native-boundary"

    def __init__(self, ir: AnalysisIR | None = None):
        self.ir = ir

    def analyze(self, root):
        root = Path(root)
        out = []
        if not self.ir:
            from shap_review.semantic_ir import SemanticIRBuilder

            self.ir = SemanticIRBuilder().build(root)
        native_cache = {}
        for b in self.ir.boundaries:
            source = root / b.file
            if source.exists() and b.file not in native_cache:
                native_cache[b.file] = analyze_native_file(root, source)
            evidence = next(
                (
                    e
                    for e in native_cache.get(b.file, [])
                    if e.line == b.line and e.operation == b.operation
                ),
                None,
            )
            context = ""
            if source.exists():
                lines = source.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                context = "\n".join(lines[max(0, b.line - 4) : b.line + 2])
            if b.operation in {"raw-buffer", "element-access"}:
                msg = "Raw NumPy buffer access requires validated dtype, shape, stride, and lifetime assumptions."
                if not _VALIDATION_RE.search(context):
                    msg += " No nearby validation marker was detected; inspect the full control flow before treating this as a defect."
                tag = "buffer-validation"
            elif b.operation == "borrowed-item":
                msg = "Borrowed Python reference crosses a native boundary; verify lifetime across allocation, callbacks, and re-entry."
                if not _OWNERSHIP_RE.search(context):
                    msg += " No nearby ownership/lifetime marker was detected."
                tag = "borrowed-reference"
            elif b.operation == "exception":
                msg = "Python exception boundary: verify every failure return is paired with correct exception propagation."
                tag = "exception-propagation"
            elif b.operation == "python-callback":
                msg = "Native code invokes Python-controlled behavior; verify re-entry, exception, mutation, and lifetime handling."
                tag = "reentry"
            else:
                msg = f"Native boundary `{b.operation}` requires ownership, exception, and lifetime validation."
                tag = b.operation
            if evidence and evidence.data_flow:
                msg += " Data-flow context: " + ", ".join(evidence.data_flow) + "."
            if evidence and evidence.risk_flags:
                msg += " Risk flags: " + ", ".join(evidence.risk_flags) + "."
            out.append(
                Candidate(
                    f"SHAP-CAND-NATIVE-{b.file}-{b.line}-{b.operation}",
                    "SHAP-06",
                    "INV-NATIVE-001",
                    b.file,
                    b.line,
                    evidence.function if evidence else None,
                    msg,
                    [
                        EvidenceRef(
                            "source",
                            "semantic native-boundary detection",
                            "native boundary requires path-sensitive validation",
                            2,
                        )
                    ],
                    tags=["native", b.technology, tag],
                )
            )
        return out
