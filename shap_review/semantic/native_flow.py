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
    r"\b("
    r"PyArray_DATA|PyArray_GETPTR\w*|PyObject_Call\w*|"
    r"PyObject_GetAttr\w*|PyList_GET_ITEM|nb::ndarray|nb::object|"
    r"cudaMemcpy\w*|cudaMalloc\w*"
    r")\b"
)
_FREE = re.compile(
    r"\b("
    r"free|delete|PyMem_Free|Py_DECREF|Py_XDECREF|release|cudaFree"
    r")\s*\(?\s*([A-Za-z_]\w*)?"
)
_ERROR = re.compile(r"\b(return\s+NULL|return\s+-1|PyErr_|throw)\b")
_VALIDATION = re.compile(r"\b(if|assert|PyArg_|PyErr_Set)\b")


def function_ranges(lines):
    out = []
    current = None
    start = None
    depth = 0

    for i, line in enumerate(lines, 1):
        match = _FUNC.search(line)

        if match and current is None:
            current = match.group(1)
            start = i
            depth = line.count("{") - line.count("}")
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
    for start, end, name in ranges:
        if start <= line <= end:
            return name
    return None


def _boundary_symbol(line):
    match = re.search(
        r"\b([A-Za-z_]\w*)\s*=\s*(?:[^;]*?)\b"
        r"(?:PyArray_DATA|PyArray_GETPTR\w*)\s*\(",
        line,
    )
    return match.group(1) if match else None


def analyze_flow(path):
    p = Path(path)
    lines = p.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    ranges = function_ranges(lines)
    facts = []
    order = 0
    depth = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        depth = max(0, depth - stripped.count("}"))

        kind = None
        symbol = None

        if _BOUNDARY.search(stripped):
            kind = "native-boundary"
            symbol = _boundary_symbol(stripped)
        elif _FREE.search(stripped):
            kind = "lifetime-end"
            symbol = _FREE.search(stripped).group(2)
        elif _ERROR.search(stripped):
            kind = "error-path"
        elif _VALIDATION.search(stripped):
            kind = "validation"

        if kind:
            order += 1
            facts.append(
                FlowFact(
                    str(p),
                    i,
                    function_for(ranges, i),
                    kind,
                    stripped[:300],
                    order,
                    depth,
                    symbol,
                )
            )

        depth += stripped.count("{")

    return facts


def correlate_boundary(path, boundary_line, radius=80):
    """Correlate nearby native-flow facts without claiming proof.

    This analysis is intentionally heuristic. It uses textual boundary
    detection, approximate function ranges, source ordering, and nearby
    lifetime/error markers. A returned correlation is therefore a triage
    signal, not a confirmed ownership, lifetime, control-flow, or
    memory-safety finding.
    """
    facts = analyze_flow(path)
    boundary = next(
        (fact for fact in facts if fact.line == boundary_line),
        None,
    )

    relevant = [
        fact
        for fact in facts
        if abs(fact.line - boundary_line) <= radius
    ]

    function = (
        boundary.function
        if boundary
        else next(
            (fact.function for fact in relevant if fact.function),
            None,
        )
    )

    same = [
        fact
        for fact in relevant
        if fact.function == function
    ]

    symbol = boundary.symbol if boundary else None

    lifetime = [
        fact
        for fact in same
        if (
            fact.kind == "lifetime-end"
            and fact.line > boundary_line
            and (symbol is None or fact.symbol in (None, symbol))
        )
    ]

    errors = [
        fact
        for fact in same
        if fact.kind == "error-path" and fact.line > boundary_line
    ]

    return {
        "boundary_line": boundary_line,
        "function": function,
        "symbol": symbol,
        "before": [
            fact.to_dict()
            for fact in same
            if fact.line < boundary_line
        ],
        "after": [
            fact.to_dict()
            for fact in same
            if fact.line >= boundary_line
        ],
        "lifetime_after_boundary": bool(lifetime),
        "error_after_boundary": bool(errors),
        "validation_before_boundary": any(
            fact.kind == "validation"
            and fact.line < boundary_line
            for fact in same
        ),
        "same_function": boundary is not None,
        "analysis_mode": "triage",
        "analysis_basis": [
            "textual-boundary-matching",
            "approximate-function-range",
            "source-order-correlation",
            "nearby-lifetime-marker",
            "nearby-error-marker",
        ],
        "proof_status": "NOT_PROVEN",
        "requires_runtime_or_control_flow_validation": True,
        "component_hints": (
            ["cutils", "cext", "cext-gpu"]
            if any(
                component in str(path)
                for component in ("shap/cutils", "shap/cext")
            )
            else []
        ),
    }