from __future__ import annotations

from pathlib import Path

from shap_review.discovery import (
    BuildScanner,
    NativeScanner,
    RepositoryScanner,
    TestScanner,
)
from shap_review.invariants import InvariantRegistry
from shap_review.semantic import SHAPSemanticMapper


class HealthReport:
    def run(self, root: str | Path) -> dict:
        root = Path(root).resolve()
        repo = RepositoryScanner().scan(root)
        build = BuildScanner().scan(root)
        graph = SHAPSemanticMapper().map(root)
        inv = InvariantRegistry().load()
        tests = TestScanner().scan(root)
        return {
            "repository": repo.__dict__,
            "build": build,
            "native_files": len(NativeScanner().scan(root)),
            "semantic_nodes": len(graph.nodes),
            "semantic_edges": len(graph.edges),
            "invariants_loaded": len(inv),
            "tests": tests["count"],
        }
