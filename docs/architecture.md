# Architecture

## Core pipeline

```text
Source → Discovery → Semantic IR → Deterministic Signals
                         ↓
              Evidence / Invariant Binding
                         ↓
                   Candidate IR
                         ↓
        AI investigation (optional adapter layer)
                         ↓
     Reproduction / Differential / Dynamic Fuzzing
                         ↓
                   Evidence Bundle
                         ↓
              Confirmed Finding / Regression
```

## Semantic IR

`shap_review.semantic_ir` is the shared representation used by analyzers. It intentionally normalizes only facts that can be extracted deterministically:

- `Symbol`: function/class definitions and source locations.
- `CallSite`: AST-resolved call names, enclosing symbol and arguments.
- `ValueFlow`: important representation transitions such as pandas-like input → NumPy-like input.
- `NativeBoundary`: C/C++/CUDA/nanobind/Python C-API crossings.
- `TestCoverage`: test symbols and coarse assertion/call signals.

This prevents every analyzer from independently re-parsing the repository and reduces the temptation to make a global token match look like a semantic fact.

## Signal aggregation

Each analyzer returns `Candidate` objects, but they are treated as signals until aggregation. The aggregator groups signals by bug class, file, line and invariant, deduplicates evidence, and raises confidence only when independent evidence accumulates.

A high-confidence static candidate still requires dynamic validation before it becomes a finding.

## Evidence

`EvidenceCorpus` provides structured records with:

- public source URL
- bug class
- invariant
- affected components
- observed and expected behavior
- fix/regression metadata when known

Evidence is a traceability mechanism, not proof that the current checkout still contains the historical defect.

## Dynamic validation

The TreeExplainer harness builds real scikit-learn tree models and invokes the installed SHAP implementation. Its oracles independently check:

1. execution completes without an unexpected exception;
2. SHAP values are finite;
3. the input array is unchanged;
4. the local-accuracy reconstruction agrees with the model output within a dtype-specific tolerance.

The harness supports regression and classification/probability paths. Cases are deterministic from a seed.
