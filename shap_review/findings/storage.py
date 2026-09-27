from __future__ import annotations

import json
from pathlib import Path

from shap_review.types import Finding


class FindingStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save(self, finding: Finding):
        p = self.root / finding.finding_id
        p.mkdir(parents=True, exist_ok=True)
        (p / "meta.json").write_text(
            json.dumps(finding.to_dict(), indent=2), encoding="utf-8"
        )

    def load(self, finding_id: str):
        return json.loads(
            (self.root / finding_id / "meta.json").read_text(encoding="utf-8")
        )
