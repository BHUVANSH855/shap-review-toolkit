from __future__ import annotations

import ast

from shap_review.analyzers.common.base import Analyzer
from shap_review.types import Candidate, EvidenceRef
from shap_review.utils import iter_source_files, rel

# Explicit contract mapping prevents method-name defaults from creating semantically
# inconsistent bug-class/invariant pairs (e.g. additivity + state).
RULES = {
    "expected_value": ("SHAP-07", "INV-STATE-001", "state"),
    "model_output": ("SHAP-02", "INV-OUT-001", "output-space"),
    "check_additivity": ("SHAP-01", "INV-ATTR-001", "additivity"),
    "interaction_values": ("SHAP-03", "INV-SHAPE-001", "shape"),
    "shap_interaction_values": ("SHAP-03", "INV-SHAPE-001", "shape"),
    "supports_model_with_masker": ("SHAP-04", "INV-OUT-001", "dispatch"),
}


class PythonContractAnalyzer(Analyzer):
    name = "python-contract"

    def analyze(self, root):
        out = []
        for p in iter_source_files(root):
            if p.suffix != ".py":
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                tags = []
                body = ast.get_source_segment(text, node) or ""
                for token in RULES:
                    if token in body:
                        tags.append(token)
                if not tags or node.name not in {
                    "shap_values",
                    "shap_interaction_values",
                    "interaction_values",
                    "supports_model_with_masker",
                    "__init__",
                }:
                    continue
                for token in sorted(tags):
                    bug, inv, label = RULES[token]
                    evidence = EvidenceRef(
                        "source",
                        "SHAP implementation",
                        f"AST-scoped `{token}` contract logic",
                        2,
                    )
                    out.append(
                        Candidate(
                            f"SHAP-CAND-PY-{bug}-{p.stem}-{node.lineno}-{token}",
                            bug,
                            inv,
                            rel(root, p),
                            node.lineno,
                            node.name,
                            f"Contract-sensitive `{node.name}` contains `{token}` logic; inspect its data/control-flow interaction against the {label} invariant.",
                            [evidence],
                            True,
                            "low",
                            ["ast", "semantic", label],
                        )
                    )
        return out
