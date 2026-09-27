from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from shap_review.types import Candidate


class Analyzer(ABC):
    name = "base"

    @abstractmethod
    def analyze(self, root: Path) -> list[Candidate]: ...
