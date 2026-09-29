from __future__ import annotations

from collections import defaultdict

from shap_review.evidence.chain import build_chain
from shap_review.types import Candidate

# Evidence tiers are deliberately explicit: static pattern matching is weaker than
# primary historical evidence or a runtime reproduction.
EVIDENCE_TIER = {
    "source": 1,
    "documentation": 2,
    "test-gap": 2,
    "issue": 4,
    "pull_request": 5,
    "dynamic": 7,
    "differential": 8,
    "sanitizer": 10,
}


class CandidateAggregator:
    def merge(self, candidates: list[Candidate]) -> list[Candidate]:
        groups = defaultdict(list)
        for c in candidates:
            key = (c.bug_class, c.file, c.line, c.invariant)
            groups[key].append(c)

        out = []
        for group in groups.values():
            base = group[0]
            evidence = []
            tags = []
            messages = []
            seen_e = set()

            for c in group:
                messages.append(c.message)
                tags.extend(c.tags)

                for e in c.evidence:
                    k = (e.kind, e.source, e.note)
                    if k not in seen_e:
                        seen_e.add(k)
                        evidence.append(e)

            score = sum(
                EVIDENCE_TIER.get(e.kind, max(1, e.strength))
                for e in evidence
            )
            independent_kinds = len({e.kind for e in evidence})

            # Multiple analyzers firing on the same location is NOT independent
            # evidence. They all derive from the same source code, so counting
            # them as corroboration would inflate confidence on pure pattern
            # matches.
            #
            # Genuinely different evidence types (for example, issue + dynamic)
            # do provide additional information, so their diversity contributes
            # a small bonus.
            score += max(0, independent_kinds - 1) * 2

            confidence = (
                "high"
                if score >= 14 and independent_kinds >= 2
                else "medium"
                if score >= 7
                else "low"
            )

            chain_entries = [
                {
                    "kind": e.kind,
                    "source": e.source,
                    "claim": e.note,
                    "passed": True,
                    "confidence": min(
                        1.0,
                        max(0.1, e.strength / 10 if e.strength else 1.0),
                    ),
                    # Historical corpus records are external facts, not derived
                    # observations from the current analysis run.
                    "origin": (
                        "external"
                        if e.kind in {"issue", "pull_request"}
                        else "source"
                    ),
                }
                for e in evidence
            ]

            chain = build_chain(entries=chain_entries)

            out.append(
                Candidate(
                    base.candidate_id,
                    base.bug_class,
                    base.invariant,
                    base.file,
                    base.line,
                    base.symbol,
                    " ".join(dict.fromkeys(messages)),
                    evidence,
                    any(c.validation_required for c in group),
                    confidence,
                    sorted(set(tags)),
                    evidence_chain=chain,
                )
            )

        return sorted(
            out,
            key=lambda c: (c.file, c.line or 0, c.bug_class),
        )