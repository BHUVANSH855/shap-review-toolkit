from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from shap_review.utils import rel


@dataclass(frozen=True)
class NativeEvidence:
    file: str
    line: int
    technology: str
    operation: str
    function: str | None
    validation: tuple[str, ...]
    ownership: tuple[str, ...]
    exception: tuple[str, ...]
    reentry: tuple[str, ...]
    data_flow: tuple[str, ...]
    risk_flags: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


_BOUNDARY = {
    "PyArray_DATA": ("numpy-c-api", "raw-buffer"),
    "PyArray_GETPTR": ("numpy-c-api", "element-access"),
    "PyList_GET_ITEM": ("python-c-api", "borrowed-item"),
    "PyErr_": ("python-c-api", "exception"),
    "PyObject_Call": ("python-c-api", "python-callback"),
    "nb::ndarray": ("nanobind", "ndarray-conversion"),
    "nb::object": ("nanobind", "object-handle"),
    "cudaMemcpy": ("cuda", "device-copy"),
    "cudaMalloc": ("cuda", "allocation"),
}
_VALID = re.compile(
    r"dtype|shape|stride|contig|PyArray_Check|PyArray_NDIM|validate|check|assert|bounds|size",
    re.IGNORECASE,
)
_OWN = re.compile(
    r"Py_(?:INCREF|DECREF|XINCREF|XDECREF)|NewRef|release\(|borrow|owned|refcount|keep_alive|reference",
    re.IGNORECASE,
)
_EXC = re.compile(
    r"PyErr_|throw|catch|error|exception|return\s+NULL|return\s+-1", re.IGNORECASE
)
_REENTRY = re.compile(
    r"PyObject_Call|nb::object|nb::handle|callback|gil|GIL|py::", re.IGNORECASE
)
_FUNC = re.compile(
    r"(?:static\s+)?(?:inline\s+)?[\w:<>~*&\s]+\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
)


def _function_at(lines, line_no):
    current = None
    depth = 0
    for line in lines[:line_no]:
        m = _FUNC.search(line)
        if m:
            current = m.group(1)
            depth = line.count("{") - line.count("}")
        else:
            depth += line.count("{") - line.count("}")
        if depth <= 0:
            current = None
    return current


def analyze_native_file(root, path):
    root = Path(root)
    p = Path(path)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for i, line in enumerate(lines, 1):
        for needle, (tech, op) in _BOUNDARY.items():
            if needle not in line:
                continue
            lo = max(0, i - 20)
            hi = min(len(lines), i + 20)
            context = "\n".join(lines[lo:hi])
            before = "\n".join(lines[lo:i])
            after = "\n".join(lines[i:hi])
            validation = tuple(
                sorted(set(m.group(0).lower() for m in _VALID.finditer(context)))
            )
            ownership = tuple(sorted(set(m.group(0) for m in _OWN.finditer(context))))
            exception = tuple(sorted(set(m.group(0) for m in _EXC.finditer(context))))
            reentry = tuple(sorted(set(m.group(0) for m in _REENTRY.finditer(context))))
            flow = []
            if _VALID.search(before):
                flow.append("validation-before-boundary")
            if _OWN.search(before) or _OWN.search(after):
                flow.append("ownership-context")
            if _REENTRY.search(after):
                flow.append("reentry-after-boundary")
            if _EXC.search(after):
                flow.append("exception-after-boundary")
            if op in {"raw-buffer", "element-access"} and re.search(
                r"\b(?:Py_DECREF|release\(|free\(|delete)\b", after
            ):
                flow.append("possible-lifetime-end-after-boundary")
            flags = []
            if (
                op in {"raw-buffer", "element-access"}
                and "validation-before-boundary" not in flow
            ):
                flags.append("validation-order-unresolved")
            if (
                op in {"raw-buffer", "element-access", "borrowed-item"}
                and "ownership-context" not in flow
            ):
                flags.append("ownership-order-unresolved")
            if (
                op in {"exception", "python-callback"}
                and "exception-after-boundary" not in flow
            ):
                flags.append("exception-path-unresolved")
            if op == "python-callback" and "reentry-after-boundary" not in flow:
                flags.append("reentry-path-unresolved")
            if "possible-lifetime-end-after-boundary" in flow:
                flags.append("lifetime-end-after-boundary")
            out.append(
                NativeEvidence(
                    rel(root, p),
                    i,
                    tech,
                    op,
                    _function_at(lines, i),
                    validation,
                    ownership,
                    exception,
                    reentry,
                    tuple(flow),
                    tuple(flags),
                )
            )
    return out
