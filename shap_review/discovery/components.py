from __future__ import annotations

from pathlib import Path

from shap_review.utils import iter_source_files, read_text, rel


class ComponentScanner:
    def scan(self, root: str | Path) -> list[dict]:
        root = Path(root)
        rows = []
        for p in iter_source_files(root):
            t = read_text(p).lower()
            tags = []
            if "explainer" in p.name.lower() or "/explainers/" in p.as_posix():
                tags.append("explainer")
            if "masker" in p.name.lower() or "/maskers/" in p.as_posix():
                tags.append("masker")
            if "tree" in p.name.lower():
                tags.append("tree")
            if "cutils" in p.as_posix() or "cext" in p.as_posix():
                tags.append("native-boundary")
            if any(
                x in t
                for x in [
                    "model_output",
                    "expected_value",
                    "shap_values",
                    "interaction_values",
                ]
            ):
                tags.append("semantic-contract")
            if "nanobind" in t:
                tags.append("nanobind")
            if p.suffix in {".cu", ".cuh"}:
                tags.append("cuda")
            if tags:
                rows.append({"file": rel(root, p), "tags": sorted(set(tags))})
        return rows
