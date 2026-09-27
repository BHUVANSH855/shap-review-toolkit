from __future__ import annotations

from pathlib import Path

from shap_review.utils import iter_source_files, read_text, rel


class HotspotEngine:
    def run(self, root: str | Path) -> list[dict]:
        root = Path(root).resolve()
        rows = []
        for p in iter_source_files(root):
            t = read_text(p)
            score = 0
            reasons = []
            checks = [
                ("native", p.suffix in {".c", ".cc", ".cpp", ".cu"}, 3),
                (
                    "semantic",
                    any(
                        x in t
                        for x in [
                            "model_output",
                            "expected_value",
                            "shap_values",
                            "interaction_values",
                        ]
                    ),
                    3,
                ),
                ("exception", "PyErr_" in t or "raise " in t, 1),
                (
                    "conversion",
                    any(
                        x in t
                        for x in [".values", "to_numpy", "nb::ndarray", "PyArray_"]
                    ),
                    2,
                ),
                (
                    "state",
                    any(
                        x in t
                        for x in [
                            "self.expected_value",
                            "self.model_output",
                            "self.data",
                        ]
                    ),
                    2,
                ),
            ]
            for reason, ok, weight in checks:
                if ok:
                    score += weight
                    reasons.append(reason)
            if score:
                rows.append({"file": rel(root, p), "score": score, "reasons": reasons})
        return sorted(rows, key=lambda x: (-x["score"], x["file"]))
