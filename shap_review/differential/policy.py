from __future__ import annotations

FIELD_POLICY = {
    "values": "correctness-critical",
    "base_values": "correctness-critical",
    "model_output": "correctness-critical",
    "shape": "correctness-critical",
    "interaction_values": "correctness-critical",
    "data": "contextual",
    "error_std": "contextual",
    "output_indexes": "contextual",
    "feature_names": "metadata",
    "output_names": "metadata",
}
CRITICAL = {k for k, v in FIELD_POLICY.items() if v == "correctness-critical"}


def classify_field(name: str) -> str:
    return FIELD_POLICY.get(name, "contextual")
