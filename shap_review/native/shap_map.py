from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SHAPNativeComponent:
    name: str
    paths: tuple[str, ...]
    layer: str
    boundary: str
    risk_domains: tuple[str, ...]


COMPONENTS = (
    SHAPNativeComponent(
        "explainer-api",
        ("shap/explainers", "shap/Explainer.py"),
        "api",
        "Python -> explainer",
        "dispatch,output-space,state",
    ),
    SHAPNativeComponent(
        "cutils",
        ("shap/cutils", "shap/cutils/cutils.cpp"),
        "native-utility",
        "Python -> nanobind -> C++",
        "dtype,shape,buffer,ownership,reentry",
    ),
    SHAPNativeComponent(
        "cext",
        ("shap/cext", "shap/cext/_cext.cc", "shap/cext/_cext.cpp"),
        "tree-shap-native",
        "Python -> CPython/NumPy C API -> Tree SHAP",
        "lifetime,buffer,exception,numerical",
    ),
    SHAPNativeComponent(
        "cext-gpu",
        ("shap/cext", "shap/cext/_cext_gpu.cc", "shap/cext/_cext_gpu.cu"),
        "gpu",
        "Python -> CUDA",
        "device-memory,nan,parity,numerical",
    ),
)


def _symbols(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    patterns = [
        re.compile(
            r"\b(?:static\s+)?(?:inline\s+)?(?:[A-Za-z_][\w:<>,*&\s]+)\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
        ),
        re.compile(r"\b(?:NB_MODULE|PYBIND11_MODULE)\s*\(\s*([A-Za-z_]\w*)"),
    ]
    for lineno, line in enumerate(text.splitlines(), 1):
        for pat in patterns:
            for m in pat.finditer(line):
                out.append({"symbol": m.group(1), "file": str(path), "line": lineno})
    return out[:200]


def map_shap_native(root: str) -> dict:
    rootp = Path(root)
    components = []
    symbols = []
    for c in COMPONENTS:
        matches = []
        for rel in c.paths:
            p = rootp / rel
            if p.exists():
                matches.append(rel)
                if p.is_file() and p.suffix in {
                    ".cc",
                    ".cpp",
                    ".cu",
                    ".cxx",
                    ".h",
                    ".hpp",
                }:
                    symbols.extend([{**s, "component": c.name} for s in _symbols(p)])
                elif p.is_dir():
                    for src in (
                        list(p.rglob("*.cc"))
                        + list(p.rglob("*.cpp"))
                        + list(p.rglob("*.cu"))
                    ):
                        symbols.extend(
                            [{**s, "component": c.name} for s in _symbols(src)]
                        )
        components.append(
            {**asdict(c), "present": bool(matches), "matched_paths": matches}
        )
    return {
        "components": components,
        "symbols": symbols,
        "pipeline": [
            "Python API",
            "Explainer",
            "input normalization",
            "model adapter",
            "output transformation",
            "Tree/Masker logic",
            "_cutils/_cext",
            "NumPy/CPython/model-library boundary",
            "optional GPU backend",
        ],
        "mapping_rule": "findings should reference component/layer/risk_domain; symbol/file/line evidence is included when source parsing finds a native symbol.",
    }
