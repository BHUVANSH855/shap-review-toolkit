from __future__ import annotations

import ast
import re
from pathlib import Path

from shap_review.utils import iter_source_files, read_text, rel

from .model import (
    AnalysisIR,
    CallSite,
    NativeBoundary,
    Symbol,
    TestCoverage,
    ValueFlow,
    ValueProperty,
)

PY_SUFFIXES = {".py", ".pyi", ".pyx", ".c", ".cc", ".cpp", ".h", ".hpp", ".cu"}


# Word-bounded regex patterns for native boundary detection.
# Plain substring matching fires on C comments/string literals; word-bounded
# regex prevents false NativeBoundary entries from commented-out code.
_NATIVE_PATTERNS: list[tuple] = [
    (re.compile(r"\bnanobind\b"), "nanobind", "binding", True),
    (re.compile(r"\bnb::ndarray\b"), "nanobind", "ndarray-conversion", True),
    (re.compile(r"\bnb::object\b"), "nanobind", "object-handle", True),
    (re.compile(r"\bPyArray_DATA\b"), "numpy-c-api", "raw-buffer", False),
    (re.compile(r"\bPyArray_GETPTR\w*\b"), "numpy-c-api", "element-access", False),
    (re.compile(r"\bPyErr_\w+\b"), "python-c-api", "exception", True),
    (re.compile(r"\bPyList_GET_ITEM\b"), "python-c-api", "borrowed-item", True),
    (re.compile(r"\bcudaMemcpy\w*\b"), "cuda", "device-copy", False),
    (re.compile(r"\bcudaMalloc\w*\b"), "cuda", "allocation", False),
    (re.compile(r"\bPyObject_Call\w*\b"), "python-c-api", "python-callback", True),
]


class SemanticIRBuilder:
    def build(self, root: str | Path) -> AnalysisIR:
        root = Path(root).resolve()
        ir = AnalysisIR(str(root))
        for path in iter_source_files(root):
            if path.suffix not in PY_SUFFIXES:
                continue
            rp = rel(root, path)
            text = read_text(path)
            if path.suffix == ".py":
                self._python_file(ir, path, rp, text)
            else:
                self._native_file(ir, rp, text)
        return ir

    def _python_file(self, ir, path, rp, text):
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            return
        parents = {id(n): p for p in ast.walk(tree) for n in ast.iter_child_nodes(p)}
        # Precompute function ranges once per file O(n) for O(1) enclosing lookup.
        ranges = self._function_ranges(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = self._qualified(node, parents)
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                ir.symbols.append(Symbol(node.name, kind, rp, node.lineno, q))
                if node.name.startswith("test_") or "tests" in rp.split("/"):
                    asserts = tuple(
                        sorted(
                            {
                                self._assert_kind(n)
                                for n in ast.walk(node)
                                if isinstance(n, (ast.Assert, ast.Call))
                                and self._is_assert(n)
                            }
                        )
                    )
                    ir.tests.append(TestCoverage(q, rp, node.lineno, asserts))
            if isinstance(node, ast.Call):
                callee = self._callee(node.func)
                enclosing = self._enclosing_from_ranges(ranges, node.lineno)
                args = tuple(
                    [self._expr(a) for a in node.args[:6]]
                    + [f"{kw.arg}={self._expr(kw.value)}" for kw in node.keywords[:8]]
                )
                ir.calls.append(CallSite(callee, rp, node.lineno, enclosing, args))
                if callee in {
                    "TreeExplainer",
                    "shap.TreeExplainer",
                    "Explainer",
                    "shap.Explainer",
                }:
                    ir.flows.append(
                        ValueFlow(
                            callee, "SHAP.explainer", rp, node.lineno, "constructor"
                        )
                    )
                if callee.endswith("to_numpy") or callee in {
                    "DataFrame.values",
                    "values",
                }:
                    ir.flows.append(
                        ValueFlow(
                            "pandas-like input",
                            "numpy-like input",
                            rp,
                            node.lineno,
                            callee,
                            self._conversion_properties(node),
                        )
                    )
                if callee in {"predict", "predict_proba", "get_booster.predict"}:
                    ir.flows.append(
                        ValueFlow("model", "prediction output", rp, node.lineno, callee)
                    )
            if isinstance(node, ast.Assign):
                self._assignment(ir, rp, node)
            if isinstance(node, ast.AnnAssign) and node.value:
                self._assignment(ir, rp, node)
            if isinstance(node, ast.Attribute) and node.attr in {
                "values",
                "base_values",
                "output_names",
                "feature_names",
                "expected_value",
                "model_output",
            }:
                ir.properties.append(
                    ValueProperty(
                        self._expr(node), node.attr, rp, node.lineno, "attribute-access"
                    )
                )

    @staticmethod
    def _function_ranges(tree):
        """Precompute sorted (start, end, name) tuples for fast enclosing lookup."""
        ranges = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                end = getattr(node, "end_lineno", node.lineno)
                ranges.append((node.lineno, end, node.name))
        ranges.sort(key=lambda t: t[0])
        return ranges

    @staticmethod
    def _enclosing_from_ranges(ranges, lineno):
        """Return the innermost enclosing scope name for lineno."""
        best = ""
        for start, end, name in ranges:
            if start > lineno:
                break
            if start <= lineno <= end:
                best = name
        return best

    def _assignment(self, ir, rp, node):
        value = getattr(node, "value", None)
        targets = getattr(node, "targets", None) or [getattr(node, "target", None)]
        if not value:
            return
        src = self._expr(value)
        transform = self._transform(value)
        for t in targets:
            dst = self._expr(t)
            ir.flows.append(ValueFlow(src, dst, rp, node.lineno, transform))
            for prop in self._properties(value):
                ir.properties.append(
                    ValueProperty(dst, prop, rp, node.lineno, transform)
                )

    @staticmethod
    def _transform(node):
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return SemanticIRBuilder._callee(node.func)
        if isinstance(node, ast.Subscript):
            return "subscript"
        return type(node).__name__

    @staticmethod
    def _properties(node):
        s = ast.unparse(node) if hasattr(ast, "unparse") else ""
        props = []
        if "to_numpy" in s:
            props.append("numpy-conversion")
        if ".values" in s:
            props.append("values-extraction")
        if "astype" in s:
            props.append("dtype-cast")
        if "reshape" in s:
            props.append("reshape")
        if "copy(" in s:
            props.append("copy")
        return props

    @staticmethod
    def _conversion_properties(node):
        return tuple(SemanticIRBuilder._properties(node))

    @staticmethod
    def _callee(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            b = SemanticIRBuilder._callee(node.value)
            return f"{b}.{node.attr}" if b else node.attr
        return type(node).__name__

    @staticmethod
    def _expr(node):
        try:
            return ast.unparse(node)
        except Exception:
            return type(node).__name__

    @staticmethod
    def _qualified(node, parents):
        parts = [node.name]
        cur = parents.get(id(node))
        while cur:
            if isinstance(cur, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                parts.append(cur.name)
            cur = parents.get(id(cur))
        return ".".join(reversed(parts))

    @staticmethod
    def _is_assert(n):
        if isinstance(n, ast.Assert):
            return True
        return (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id.startswith("assert")
        )

    @staticmethod
    def _assert_kind(n):
        return "assert" if isinstance(n, ast.Assert) else getattr(n.func, "id", "call")

    def _native_file(self, ir, rp, text):
        """Scan a native file using word-bounded regex to avoid false positives
        from C comments and string literals."""
        for i, line in enumerate(text.splitlines(), 1):
            for pattern, tech, op, controlled in _NATIVE_PATTERNS:
                if pattern.search(line):
                    ir.boundaries.append(NativeBoundary(tech, rp, i, op, controlled))
