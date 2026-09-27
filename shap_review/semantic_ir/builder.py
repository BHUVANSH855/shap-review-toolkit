from __future__ import annotations

import ast
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
                enclosing = self._enclosing(tree, node.lineno)
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
    def _enclosing(tree, lineno):
        best = ""
        for n in ast.walk(tree):
            if isinstance(
                n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) and n.lineno <= lineno <= getattr(n, "end_lineno", n.lineno):
                best = n.name
        return best

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
        patterns = [
            ("nanobind", "nanobind", "binding", True),
            ("nb::ndarray", "nanobind", "ndarray-conversion", True),
            ("nb::object", "nanobind", "object-handle", True),
            ("PyArray_DATA", "numpy-c-api", "raw-buffer", False),
            ("PyArray_GETPTR", "numpy-c-api", "element-access", False),
            ("PyErr_", "python-c-api", "exception", True),
            ("PyList_GET_ITEM", "python-c-api", "borrowed-item", True),
            ("cudaMemcpy", "cuda", "device-copy", False),
            ("cudaMalloc", "cuda", "allocation", False),
            ("PyObject_Call", "python-c-api", "python-callback", True),
        ]
        for i, line in enumerate(text.splitlines(), 1):
            for needle, tech, op, controlled in patterns:
                if needle in line:
                    ir.boundaries.append(NativeBoundary(tech, rp, i, op, controlled))
