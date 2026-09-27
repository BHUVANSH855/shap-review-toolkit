from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Invariant:
    id: str
    name: str
    category: str
    description: str
    applies_to: list[str]
    evidence: list[str] = field(default_factory=list)
    oracle: str = "manual"
    validation: list[str] = field(default_factory=list)
    false_positive_notes: list[str] = field(default_factory=list)


class InvariantRegistry:
    def __init__(self, root: str | Path | None = None):
        self.root = (
            Path(root)
            if root
            else Path(__file__).resolve().parents[2] / "data" / "invariants"
        )

    def load(self) -> dict[str, Invariant]:
        out = {}
        if not self.root.exists():
            return out
        for p in self.root.rglob("*.json"):
            raw = json.loads(p.read_text(encoding="utf-8"))
            inv = Invariant(**raw)
            out[inv.id] = inv
        return out
