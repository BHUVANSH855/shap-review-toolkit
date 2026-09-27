from __future__ import annotations

import re
from pathlib import Path

from shap_review.utils import read_text


class BuildScanner:
    def scan(self, root: str | Path) -> dict:
        root = Path(root)
        cmake = read_text(root / "CMakeLists.txt")
        pyproject = read_text(root / "pyproject.toml")
        return {
            "backend": self._backend(pyproject),
            "python_min": self._python_min(pyproject),
            "cmake": (root / "CMakeLists.txt").exists(),
            "targets": self._targets(cmake),
            "native_technologies": self._technologies(cmake),
            "cuda_enabled_by_env": "SHAP_ENABLE_CUDA" in cmake,
            "stable_abi": "STABLE_ABI" in cmake or "WITH_SOABI" in cmake,
        }

    def _backend(self, text: str) -> str | None:
        m = re.search(r'build-backend\s*=\s*["\']([^"\']+)', text)
        return m.group(1) if m else None

    def _python_min(self, text: str) -> str | None:
        m = re.search(r'requires-python\s*=\s*["\']([^"\']+)', text)
        return m.group(1) if m else None

    def _targets(self, text: str) -> list[str]:
        return sorted(
            set(
                re.findall(
                    r"(?:nanobind_add_module|python_add_library)\(\s*([A-Za-z0-9_]+)",
                    text,
                )
            )
        )

    def _technologies(self, text: str) -> list[str]:
        out = []
        for name, token in [
            ("nanobind", "nanobind"),
            ("C++", "LANGUAGES CXX"),
            ("NumPy C API", "numpy.get_include"),
            ("CUDA", "enable_language(CUDA)"),
            ("stable ABI", "STABLE_ABI"),
        ]:
            if token in text:
                out.append(name)
        return out
