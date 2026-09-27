# Evidence Model v0.9

`EvidenceChain` is the canonical evidence model. Every item records:

- kind
- source
- claim
- pass/fail state
- confidence
- origin
- dependency IDs (`derived_from`)
- evidence ID
- details

Evidence is independent only when it is not derived from another evidence item and is not marked as derived. This prevents multiple observations generated from one historical issue from being misrepresented as independent confirmation.

## Verdict policy

```text
STATIC_CANDIDATE
HISTORICALLY_CORRELATED
DYNAMICALLY_VALIDATED
HIGH_CONFIDENCE
CONFIRMED
CONFIRMED_MEMORY_SAFETY
UNVALIDATED
```

A sanitizer-backed result can confirm memory-safety class. Other confirmation requires independent runtime evidence rather than static volume alone.

## v0.25 provenance rules

v0.25 makes provenance validation fail-closed. Dynamic, reproduction, differential,
and sanitizer evidence must carry an `execution_id` or explicit `fixture_id`; derived
evidence must declare `derived_from`. Missing provenance is invalid rather than being
silently interpreted as independent.

The same producer is **not** treated as correlated by itself. A validator may produce
independent observations across separate executions. Correlation is established by
shared execution/fixture, shared revision+environment where applicable, or explicit
ancestry.
