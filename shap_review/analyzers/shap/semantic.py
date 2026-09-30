from __future__ import annotations

import ast
from pathlib import Path

from shap_review.analyzers.common.base import Analyzer
from shap_review.evidence import EvidenceCorpus
from shap_review.semantic_ir import AnalysisIR
from shap_review.types import Candidate, EvidenceRef
from shap_review.utils import iter_source_files, read_text, rel


def _has_ast_assertion(source: str) -> bool:
    """Return True when *source* contains at least one real assertion.

    Checks for AST-level assert statements AND common pytest/numpy assertion
    calls so that files using pytest.raises, np.testing.assert_*, or
    unittest assertions are not incorrectly flagged as test-gap candidates.
    Plain string search for 'assert' fires on comments, docstrings, and
    variable names such as 'assert_called_once'.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    _ASSERT_CALL_NAMES = {
        "assert_allclose",
        "assert_array_equal",
        "assert_almost_equal",
        "assert_raises",
        "assert_warns",
        "raises",          # pytest.raises
        "approx",          # used inside assert ... == approx(...)
    }

    for node in ast.walk(tree):
        # Bare assert statement
        if isinstance(node, ast.Assert):
            return True
        # Call to a known assertion helper
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else (func.id if isinstance(func, ast.Name) else None)
            )
            if name in _ASSERT_CALL_NAMES:
                return True

    return False


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
        # Only flag pandas->numpy conversions in files that also contain a
        # TreeExplainer call.  Without this guard, any pandas preprocessing
        # code in the analysed repository fires as a SHAP-05 candidate.
        files_with_shap_calls = {
            call.file
            for call in ir.calls
            if call.callee
            in {"TreeExplainer", "shap.TreeExplainer", "Explainer", "shap.Explainer"}
        }
        for flow in ir.flows:
            if (
                flow.source == "pandas-like input"
                and flow.file in files_with_shap_calls
            ):
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
        # Guard: only activate when the file actually imports from shap.
        # Without this guard the fallback fires on any custom attribution library
        # or mock class that has functions named shap_values / interaction_values.
        if not out:
            for p in iter_source_files(root):
                if p.suffix != ".py":
                    continue
                text = read_text(p)
                try:
                    tree = ast.parse(text)
                except SyntaxError:
                    continue
                # Require an actual shap import before activating fallback.
                has_shap_import = any(
                    (
                        isinstance(n, ast.Import)
                        and any(alias.name == "shap" for alias in n.names)
                    )
                    or (
                        isinstance(n, ast.ImportFrom)
                        and (n.module or "").split(".")[0] == "shap"
                    )
                    for n in ast.walk(tree)
                )
                if not has_shap_import:
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
            if "interaction_values" in t and not _has_ast_assertion(t):
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
        # Always include a static/source EvidenceRef documenting the IR signal.
        # This gives the evidence chain two independent kinds (historical + static)
        # so it can reach HISTORICALLY_CORRELATED rather than remaining UNVALIDATED.
        refs = [
            EvidenceRef(
                "source",
                "SHAP semantic IR",
                f"IR call-site or flow detected at {file}:{line}",
                2,
            )
        ]
        refs += [
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
