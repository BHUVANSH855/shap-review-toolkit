from __future__ import annotations

from pathlib import Path

from shap_review.utils import iter_source_files, read_text, rel


class NativeScanner:
    def scan(self, root: str | Path) -> list[dict]:
        root = Path(root)
        out = []
        for p in iter_source_files(root):
            if p.suffix == ".py":
                continue
            text = read_text(p)
            out.append(
                {
                    "file": rel(root, p),
                    "language": self.lang(p),
                    "nanobind": "nanobind" in text,
                    "python_api": any(
                        x in text for x in ["PyObject", "PyArray", "Python.h"]
                    ),
                    "cuda": p.suffix in {".cu", ".cuh"},
                    "lines": len(text.splitlines()),
                }
            )
        return out

    @staticmethod
    def lang(p: Path) -> str:
        return {
            ".c": "c",
            ".cc": "cpp",
            ".cpp": "cpp",
            ".h": "c-header",
            ".hpp": "cpp-header",
            ".cu": "cuda",
            ".cuh": "cuda-header",
        }.get(p.suffix, p.suffix.lstrip("."))
