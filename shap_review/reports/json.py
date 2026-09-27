import json
from dataclasses import asdict

from shap_review.types import Candidate


def render(candidates: list[Candidate]) -> str:
    return json.dumps([asdict(c) for c in candidates], indent=2, default=str)
