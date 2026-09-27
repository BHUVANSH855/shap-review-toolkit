from __future__ import annotations

import re
from pathlib import Path

from shap_review.utils import read_text


class DependencyScanner:
    def scan(self, root: str | Path) -> dict:
        t = read_text(Path(root) / "pyproject.toml")
        deps = []
        for line in t.splitlines():
            if (
                re.match(r'\s*["\']?[A-Za-z0-9_.-]+(?:[<>=!~].*)?["\']?,?\s*$', line)
                and "dependencies" not in line
            ):
                pass
        for name in [
            "numpy",
            "scipy",
            "scikit-learn",
            "pandas",
            "nanobind",
            "xgboost",
            "lightgbm",
            "catboost",
            "torch",
            "tensorflow",
        ]:
            if name.lower() in t.lower():
                deps.append(name)
        return {"detected": sorted(set(deps))}
