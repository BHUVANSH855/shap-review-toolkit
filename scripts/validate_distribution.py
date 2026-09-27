from __future__ import annotations

import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
archive = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if archive:
    bad = []
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():
            if any(
                x in n
                for x in (
                    "__pycache__/",
                    ".pytest_cache/",
                    ".ruff_cache/",
                    ".mypy_cache/",
                    ".pytype/",
                    ".egg-info/",
                    "/catboost_info/",
                    "/.out/",
                    "/.shap-review/",
                    "/htmlcov/",
                    "/build/",
                    "/dist/",
                )
            ) or n.endswith((".pyc", ".pyo", ".coverage", ".log", ".tmp")):
                bad.append(n)
    print("archive hygiene:", "PASS" if not bad else "FAIL")
    if bad:
        print("\n".join(bad))
        sys.exit(1)
else:
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")
    required = [
        "global-exclude __pycache__/*",
        "global-exclude *.py[cod]",
        "global-exclude *.egg-info/*",
        "global-exclude catboost_info/*",
        "global-exclude .pytest_cache/*",
        "global-exclude **/.out/*",
    ]
    missing = [x for x in required if x not in manifest]
    print("source hygiene rules:", "PASS" if not missing else "FAIL")
    if missing:
        print("\n".join(missing))
        sys.exit(1)
