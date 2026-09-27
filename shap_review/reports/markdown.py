from __future__ import annotations

from shap_review.types import Candidate


def render_candidates(candidates: list[Candidate]) -> str:
    lines = [
        "# SHAP Review Candidate Report",
        "",
        f"Candidates: **{len(candidates)}**",
        "",
        "> Candidates are hypotheses requiring validation; they are not confirmed findings.",
        "",
    ]
    for c in candidates:
        lines += [
            f"## {c.candidate_id}",
            f"- **Bug class:** `{c.bug_class}`",
            f"- **Invariant:** `{c.invariant}`",
            f"- **Location:** `{c.file}:{c.line or '?'}`",
            f"- **Symbol:** `{c.symbol or 'unknown'}`",
            f"- **Confidence:** `{c.confidence}`",
            f"- **Message:** {c.message}",
            f"- **Tags:** {', '.join(c.tags) or 'none'}",
            "- **Evidence:",
        ]
        lines += [f"  - `{e.kind}`: {e.source} — {e.note}" for e in c.evidence]
        lines.append("")
    return "\n".join(lines)
