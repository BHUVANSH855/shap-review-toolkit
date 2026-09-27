from __future__ import annotations

from pathlib import Path

from shap_review.semantic import SHAPSemanticMapper


class Explorer:
    def run(self, root: str | Path, query: str) -> dict:
        root = Path(root).resolve()
        g = SHAPSemanticMapper().map(root)
        q = query.lower()
        nodes = [
            n.__dict__
            for n in g.nodes
            if q in n.name.lower() or q in (n.file or "").lower() or q in n.kind.lower()
        ]
        ids = {n["id"] for n in nodes}
        edges = [e.__dict__ for e in g.edges if e.source in ids or e.target in ids]
        return {"query": query, "nodes": nodes, "edges": edges}
