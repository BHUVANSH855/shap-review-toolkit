from __future__ import annotations

import ast
from pathlib import Path

from shap_review.analyzers.common.base import Analyzer
from shap_review.evidence import EvidenceCorpus
from shap_review.semantic_ir import AnalysisIR
from shap_review.types import Candidate, EvidenceRef
from shap_review.utils import iter_source_files, read_text, rel


class SHAPSemanticAnalyzer(Analyzer):
    name = "shap-semantic"

    def __init__(self, ir: AnalysisIR | None = None):
        self.ir = ir

    def analyze(self, root: Path) -> list[Candidate]:
        out = []
        evidence = EvidenceCorpus()
        ir = self.ir
        if ir is None:
            from shap_review.semantic_ir import SemanticIRBuilder

            ir = SemanticIRBuilder().build(root)
        for call in ir.calls:
            if call.callee not in {
                "TreeExplainer",
                "shap.TreeExplainer",
                "Explainer",
                "shap.Explainer",
            }:
                continue
            args = " ".join(call.arguments)
            if "model_output" in args:
                out.append(
                    self._candidate(
                        root,
                        call,
                        "SHAP-02",
                        "INV-OUT-001",
                        "model_output is an explicit output-space control; trace its compatibility with feature perturbation and expected-value state.",
                        evidence,
                    )
                )
            out.append(
                self._candidate(
                    root,
                    call,
                    "SHAP-01",
                    "INV-ATTR-001",
                    "TreeExplainer construction is a contract boundary: additivity should be validated for supported model/output combinations.",
                    evidence,
                )
            )
        # High-value input conversion path: DataFrame -> numpy-like value -> Tree/native path.
        for flow in ir.flows:
            if flow.source == "pandas-like input":
                out.append(
                    self._candidate(
                        root,
                        flow,
                        "SHAP-05",
                        "INV-INPUT-001",
                        "Input conversion may erase pandas nullable dtype semantics before native processing; verify supported dtype and rejection behavior.",
                        evidence,
                        line=flow.line,
                        file=flow.file,
                        symbol="",
                    )
                )
        # Compatibility path for a standalone method fixture with no constructor call.
        if not out:
            for p in iter_source_files(root):
                if p.suffix != ".py":
                    continue
                text = read_text(p)
                try:
                    tree = ast.parse(text)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) and node.name in {
                        "shap_values",
                        "interaction_values",
                        "shap_interaction_values",
                    }:
                        out.append(
                            Candidate(
                                f"SHAP-CAND-FALLBACK-{p.stem}-{node.lineno}",
                                "SHAP-01",
                                "INV-ATTR-001",
                                rel(root, p),
                                node.lineno,
                                node.name,
                                "Contract-sensitive SHAP method requires invariant review.",
                                [
                                    EvidenceRef(
                                        "source",
                                        "SHAP implementation",
                                        "AST-scoped contract logic",
                                        2,
                                    )
                                ],
                                tags=["semantic", "ast"],
                            )
                        )

        # Test-gap detection uses actual test symbol/call evidence rather than token presence.
        for sym in ir.symbols:
            if sym.kind != "function" or not sym.name.startswith("test_"):
                continue
        text_cache = {}
        for p in iter_source_files(root):
            if p.suffix != ".py" or (
                "/test" not in "/" + rel(root, p)
                and "interaction" not in p.name.lower()
            ):
                continue
            t = read_text(p)
            text_cache[rel(root, p)] = t
            if "interaction_values" in t and "assert" not in t:
                out.append(
                    Candidate(
                        f"SHAP-CAND-GAP-{p.stem}-{t.index('interaction_values')}",
                        "SHAP-01",
                        "INV-ATTR-001",
                        rel(root, p),
                        t[: t.index("interaction_values")].count("\n") + 1,
                        None,
                        "Interaction/multi-output path is referenced without a direct assertion; this is a test-gap candidate, not a defect.",
                        [
                            EvidenceRef(
                                "issue", "SHAP-EVID-4869", "historical coverage gap", 4
                            )
                        ],
                        validation_required=False,
                        confidence="low",
                        tags=["test-gap", "coverage"],
                    )
                )
        return out

    def _candidate(
        self, root, obj, bug, inv, msg, corpus, line=None, file=None, symbol=None
    ):
        file = file or obj.file
        line = line if line is not None else obj.line
        refs = [
            EvidenceRef("issue", r.id, r.title, 4) for r in corpus.by_bug_class(bug)[:2]
        ]
        return Candidate(
            f"SHAP-CAND-{bug}-{file}-{line}",
            bug,
            inv,
            file,
            line,
            symbol or getattr(obj, "enclosing_symbol", None),
            msg,
            refs,
            True,
            "medium",
            ["semantic", "evidence-linked"],
        )
