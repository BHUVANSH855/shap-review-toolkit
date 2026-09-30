from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ORACLE_MAP = {
    "numeric_additivity": "shap_review.contracts.oracles.AdditivityOracle",
    "interaction_oracle": "shap_review.contracts.oracles.InteractionOracle",
    "output_space": "shap_review.contracts.oracles.OutputSpaceOracle",
    "input_mutation": "shap_review.contracts.oracles.InputMutationOracle",
    "expected_value": "shap_review.contracts.oracles.ExpectedValueOracle",
    "semantic_trace": None,
    "manual": None,
}


@dataclass
class Invariant:
    id: str
    name: str
    category: str
    description: str
    applies_to: list[str]
    evidence: list[str] = field(default_factory=list)
    # oracle_name identifies the oracle implementation in contracts/oracles.py.
    # Use resolve_oracle() to resolve the implementation.
    oracle_name: str = "manual"
    validation: list[str] = field(default_factory=list)
    false_positive_notes: list[str] = field(default_factory=list)

    def resolve_oracle(self):
        """Return the configured oracle class, or None for non-runtime invariants.

        Unknown oracle names are rejected explicitly so a configuration typo
        cannot silently downgrade an invariant to a manual/static check.
        """
        if self.oracle_name not in ORACLE_MAP:
            raise ValueError(f"Unknown invariant oracle: {self.oracle_name!r}")

        qualified = ORACLE_MAP[self.oracle_name]
        if qualified is None:
            return None

        import importlib

        mod_name, cls_name = qualified.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)


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
            # Backwards compatibility: old files used "oracle", new ones use "oracle_name".
            if "oracle" in raw and "oracle_name" not in raw:
                raw["oracle_name"] = raw.pop("oracle")
            elif "oracle" in raw:
                raw.pop("oracle")  # discard stale field if both present
            try:
                inv = Invariant(**raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid invariant definition in {p}: {exc}") from exc

            if inv.id in out:
                raise ValueError(f"Duplicate invariant id {inv.id!r} in {p}")

            out[inv.id] = inv
        return out
