from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from shap_review.utils import iter_source_files


@dataclass
class RepositoryInfo:
    root: str
    python_files: int
    native_files: int
    test_files: int
    docs_files: int
    has_pyproject: bool
    has_cmake: bool
    has_setup_py: bool


class RepositoryScanner:
    def scan(self, root: str | Path) -> RepositoryInfo:
        root = Path(root).resolve()
        files = list(root.rglob("*"))
        py = [p for p in files if p.is_file() and p.suffix == ".py"]
        native = list(iter_source_files(root))
        native = [p for p in native if p.suffix != ".py"]
        tests = [p for p in py if "test" in p.name.lower() or "tests" in p.parts]
        docs = [
            p
            for p in files
            if p.is_file() and ("docs" in p.parts or p.suffix in {".md", ".rst"})
        ]
        return RepositoryInfo(
            str(root),
            len(py),
            len(native),
            len(tests),
            len(docs),
            (root / "pyproject.toml").exists(),
            (root / "CMakeLists.txt").exists(),
            (root / "setup.py").exists(),
        )

    def write(self, root: str | Path, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self.scan(root)), indent=2), encoding="utf-8")
