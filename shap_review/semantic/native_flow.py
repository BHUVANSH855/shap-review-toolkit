from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FlowFact:
    file: str
    line: int
    function: str | None
    kind: str
    expression: str
    order: int
    branch_depth: int = 0
    symbol: str | None = None
    component: str | None = None

    def to_dict(self):
        return asdict(self)


_FUNC = re.compile(
    r"(?:static\s+)?(?:inline\s+)?[\w:<>~*&\s]+\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
)
_BOUNDARY = re.compile(
    r"\b(PyArray_DATA|PyArray_GETPTR\w*|PyObject_Call\w*|PyObject_GetAttr\w*|PyList_GET_ITEM|nb::ndarray|nb::object|cudaMemcpy\w*|cudaMalloc\w*)\b"
)
_FREE = re.compile(
    r"\b(free|delete|PyMem_Free|Py_DECREF|Py_XDECREF|release|cudaFree)\s*\(?\s*([A-Za-z_]\w*)?"
)
_ERROR = re.compile(r"\b(return\s+NULL|return\s+-1|PyErr_|throw)\b")
_VALIDATION = re.compile(r"\b(if|assert|PyArg_|PyErr_Set)\b")


def function_ranges(lines):
    out = []
    current = None
    start = None
    depth = 0
    for i, line in enumerate(lines, 1):
        m = _FUNC.search(line)
        if m and current is None:
            current, start, depth = m.group(1), i, line.count("{") - line.count("}")
            continue
        if current is not None:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                out.append((start, i, current))
                current = None
                start = None
    if current is not None:
        out.append((start, len(lines), current))
    return out


def function_for(ranges, line):
    for a, b, n in ranges:
        if a <= line <= b:
            return n
    return None


def _boundary_symbol(line):
    m = re.search(
        r"\b([A-Za-z_]\w*)\s*=\s*(?:[^;]*?)\b(?:PyArray_DATA|PyArray_GETPTR\w*)\s*\(",
        line,
    )
    return m.group(1) if m else None


def analyze_flow(path):
    p = Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    ranges = function_ranges(lines)
    facts = []
    order = 0
    depth = 0
    for i, line in enumerate(lines, 1):
        st = line.strip()
        depth = max(0, depth - st.count("}"))
        kind = None
        symbol = None
        if _BOUNDARY.search(st):
            kind = "native-boundary"
            symbol = _boundary_symbol(st)
        elif _FREE.search(st):
            kind = "lifetime-end"
            symbol = _FREE.search(st).group(2)
        elif _ERROR.search(st):
            kind = "error-path"
        elif _VALIDATION.search(st):
            kind = "validation"
        if kind:
            order += 1
            facts.append(
                FlowFact(
                    str(p),
                    i,
                    function_for(ranges, i),
                    kind,
                    st[:300],
                    order,
                    depth,
                    symbol,
                )
            )
        depth += st.count("{")
    return facts


def correlate_boundary(path, boundary_line, radius=80):
    facts = analyze_flow(path)
    boundary = next((f for f in facts if f.line == boundary_line), None)
    relevant = [f for f in facts if abs(f.line - boundary_line) <= radius]
    function = (
        boundary.function
        if boundary
        else next((f.function for f in relevant if f.function), None)
    )
    same = [f for f in relevant if f.function == function]
    symbol = boundary.symbol if boundary else None
    lifetime = [
        f
        for f in same
        if f.kind == "lifetime-end"
        and f.line > boundary_line
        and (symbol is None or f.symbol in (None, symbol))
    ]
    errors = [f for f in same if f.kind == "error-path" and f.line > boundary_line]
    return {
        "boundary_line": boundary_line,
        "function": function,
        "symbol": symbol,
        "before": [f.to_dict() for f in same if f.line < boundary_line],
        "after": [f.to_dict() for f in same if f.line >= boundary_line],
        "lifetime_after_boundary": bool(lifetime),
        "error_after_boundary": bool(errors),
        "validation_before_boundary": any(
            f.kind == "validation" and f.line < boundary_line for f in same
        ),
        "same_function": True if boundary else False,
        "analysis_mode": "triage",
        "component_hints": ["cutils", "cext", "cext-gpu"]
        if any(x in str(path) for x in ("shap/cutils", "shap/cext"))
        else [],
    }
