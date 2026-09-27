from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from shap_review.utils import iter_source_files, read_text, rel


@dataclass
class SemanticNode:
    id: str
    kind: str
    name: str
    file: str | None = None
    tags: list[str] | None = None
    metadata: dict | None = None


@dataclass
class SemanticEdge:
    source: str
    relation: str
    target: str


class SemanticGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def to_dict(self):
        return {
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class SHAPSemanticMapper:
    EXPLAINERS = {
        "TreeExplainer": "tree",
        "ExactExplainer": "exact",
        "PermutationExplainer": "permutation",
        "PartitionExplainer": "partition",
        "LinearExplainer": "linear",
        "KernelExplainer": "kernel",
        "DeepExplainer": "deep",
        "GradientExplainer": "gradient",
        "AdditiveExplainer": "additive",
    }
    MASKERS = {
        "Tabular": "tabular",
        "Partition": "partition",
        "Image": "image",
        "Text": "text",
        "Composite": "composite",
    }

    def map(self, root: str | Path) -> SemanticGraph:
        root = Path(root)
        g = SemanticGraph()
        g.add_node(SemanticNode("repo", "repository", "SHAP", str(root)))
        found = {}
        for p in iter_source_files(root):
            if p.suffix not in {".py", ".pyx", ".cc", ".cpp", ".c", ".cu"}:
                continue
            text = read_text(p)
            relp = rel(root, p)
            for name, kind in {**self.EXPLAINERS, **self.MASKERS}.items():
                if name in text:
                    nid = f"{kind}:{name}"
                    found.setdefault(nid, (kind, name, relp))
            if "_cext" in text or "cext" in relp:
                found.setdefault("native:_cext", ("native", "_cext", relp))
            if "_cutils" in text or "cutils" in relp:
                found.setdefault("native:_cutils", ("native", "_cutils", relp))
            if p.suffix in {".cu"}:
                found.setdefault("native:_cext_gpu", ("native", "_cext_gpu", relp))
        for nid, (kind, name, file) in sorted(found.items()):
            g.add_node(SemanticNode(nid, kind, name, file, []))
            g.add_edge(SemanticEdge("repo", "contains", nid))
        # semantic relationships
        for nid in list(found):
            if nid.startswith("tree:"):
                if "native:_cext" in found:
                    g.add_edge(SemanticEdge(nid, "invokes", "native:_cext"))
                g.add_edge(SemanticEdge(nid, "produces", "concept:Explanation"))
            if nid.startswith("exact:") and "native:_cutils" in found:
                g.add_edge(SemanticEdge(nid, "invokes", "native:_cutils"))
            if nid.startswith("partition:") and "native:_cutils" in found:
                g.add_edge(SemanticEdge(nid, "may-invoke", "native:_cutils"))
        for cid in [
            "concept:Model",
            "concept:Masker",
            "concept:Explanation",
            "concept:OutputSpace",
            "concept:Additivity",
        ]:
            kind, name = cid.split(":", 1)
            g.add_node(SemanticNode(cid, kind, name))
        return g
