# SHAP Review Toolkit v0.30.0

Evidence-driven, invariant-based, reproduction-first review tooling for [SHAP](https://github.com/shap/shap).

## v0.30.0 — Runtime integrity, evidence-state hardening & release gating

This release is a hardening release rather than a feature expansion. It restores the public import paths, aligns the fuzzing exports with the implemented TreeExplainer harness/oracle modules, tightens oracle-independence semantics, formalizes ambiguous regression outcomes, makes evidence validation explicit about scoring eligibility, hardens semantic alignment by role, and requires clean distribution archives.

The current release hardens evidence provenance semantics: required oracles cannot silently disappear, oracle results expose requirement provenance and enforce state invariants, backend execution is routed through concrete adapters, differential results separate agreement from correctness, API-era analysis tracks scope/reassignment/conditional provenance, and CPU/GPU fingerprints are emitted by the target process itself. The CatBoost interventional reconstruction case remains candidate-only until cross-version confirmation.


The toolkit deliberately prioritizes **depth and correctness over feature breadth**. The central object is an executable SHAP contract connected to runtime testing, differential analysis, provenance, fuzzing, reproduction, and maintainer-ready findings.

```text
SHAP repository
      ↓
Discovery + semantic IR
      ↓
┌─────────────────────────────────────────────┐
│ SHAP semantic contract / executable oracles │
└─────────────────────────────────────────────┘
      ↓
┌───────────────┬───────────────┬─────────────┐
│ static/native │ runtime fuzz  │ differential│
│ semantic      │ protocol/tree │ CPU/GPU     │
└───────────────┴───────────────┴─────────────┘
      ↓
Evidence provenance graph
      ↓
Reproduction / sanitizer evidence
      ↓
Finding promotion policy
      ↓
Maintainer-actionable report
```

## What the current release provides

- **Interaction-axis validation:** feature-pair axes are explicit and unequal feature dimensions are rejected instead of being compared heuristically.
- **Real TreeExplainer protocol probes:** protocol objects provide minimal array-like behavior so the real target can progress beyond initial shape validation; observed protocol coverage is reported separately from harness coverage.
- **Bounded re-entry evidence:** natural re-entry requires a protocol callback during an exact active TreeExplainer call; harness re-entry is reported separately and never promoted as natural SHAP re-entry.
- **Backend semantic smoke checks:** installed sklearn, XGBoost, LightGBM, and CatBoost campaigns now include a SHAP reconstruction/additivity check; execution accounting is separate from semantic certification.
- **Contract shape wildcards:** `-1` dimensions allow sample/output sizes to remain variable while feature dimensions remain constrained.
- **Release-integrity validation:** provider versions/schema and runtime-generated artifacts are checked before release packaging.
- **Executable SHAP semantic oracles:** shape, output-space, expected-value, additivity, interaction, and input-mutation checks.
- **EvidenceGraph:** provenance DAG with producer, parents, execution ID, revision, environment, and transformation metadata.
- **Evidence-gated lifecycle:** a finding cannot be promoted merely because static and historical signals agree.
- **Corrected protocol fuzzing:** every declared Python object protocol is explicitly triggerable, while coverage is reported separately for harness, target, and SHAP phases.
- **Fuzz targets:** `treeexplainer`, `protocol`, and backend discovery/matrix.
- **Backend matrix:** runtime discovery for scikit-learn, XGBoost, LightGBM, and CatBoost.
- **CPU/GPU differential API:** semantic comparison of independently executed backend results when CUDA-enabled SHAP is available.
- **SHAP native map:** explicit explainer → `_cutils` → `_cext` → NumPy/CPython/model-library → GPU layers.
- **API-era analysis:** legacy `shap_values()` vs modern `Explainer(...)(...) → Explanation` usage.
- **Differential field policy:** correctness-critical, contextual, and metadata fields are evaluated differently.

## Trust model

```text
STATIC CANDIDATE
      ↓
SEMANTICALLY CORRELATED
      ↓
RUNTIME VALIDATED
      ↓
REPRODUCED
      ↓
INDEPENDENTLY CONFIRMED
      ↓
MAINTAINER ACTIONABLE
```

A historical match is not itself a confirmation. Provenance and runtime evidence are recorded separately.

Independence is evidence-provenance independence, not merely a different producer label: shared execution ancestry, repository revision, fixture, environment, or producer lineage can make two observations correlated.

## Commands

```bash
python -m pip install -e .
python -m pytest

shap-review map /path/to/shap
shap-review analyze /path/to/shap
shap-review report /path/to/shap
shap-review history /path/to/shap
shap-review regressions
shap-review differential reference.py candidate.py
shap-review sanitizer reproduce.py --kind asan

# Runtime fuzzing
shap-review fuzz --target treeexplainer --iterations 100 --seed 42
shap-review fuzz --target protocol --iterations 100 --seed 42
shap-review fuzz --target backends

# Native/API analysis
shap-review native-map /path/to/shap
shap-review api-era /path/to/source.py

# CPU/GPU differential scripts
shap-review cpu-gpu-differential cpu.py gpu.py
```

For TreeExplainer execution:

```bash
python -m pip install shap scikit-learn numpy
```

Optional backend dependencies are detected at runtime and are reported as unavailable when not installed.

## Provider integrations

All providers use one canonical command/schema contract:

```text
                 SHAP Review Core
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       Claude        OpenAI       Gemini
       adapter       adapter      adapter
          │            │            │
          └────────────┼────────────┘
                       ↓
              same command contract
              same evidence semantics
```

Claude additionally ships as a native Claude Code plugin package. OpenAI and Gemini ship provider/tool integrations using the same canonical command contract; they should not be described as identical native plugin runtimes.

## Scope and limitations

The strongest current runtime surface is **TreeExplainer**, especially output-space/state semantics, input representations, historical regressions, protocol interaction, and differential correctness.

The backend matrix recognizes multiple TreeExplainer ecosystems, but deep model generation is only executed where the corresponding optional dependency and adapter are available.

Native C/C++/nanobind/CUDA analysis is explicitly **structured triage**, not a compiler-grade memory-safety proof. The native map is SHAP-specific; deeper ownership/alias/branch feasibility requires parser-backed analysis and backend-specific semantics.

CPU/GPU differential execution requires a CUDA-capable environment and GPU-enabled SHAP. The API reports unavailable execution rather than treating absence as a pass.

Static API-era analysis reports usage patterns but does not prove runtime semantics.

## Finding lifecycle

```text
DISCOVERED → CANDIDATE → INVESTIGATING → REPRO_PENDING
                                      ↓
                         FALSE_POSITIVE / REPRODUCED
                                      ↓
                               EVIDENCE_VALID
                                      ↓
                                  CONFIRMED
                                      ↓
                            REPORTED / FIXED
```

## Documentation

- `docs/v0.24-validation.md` — current release validation and evidence boundary.
- `docs/v0.24-traceability.md` — current review traceability.
- `docs/semantic-oracles.md` — executable SHAP semantic contracts.
- `docs/evidence-graph.md` — provenance and independence model.
- `docs/differential-v0.10.md` — differential field semantics.
- `docs/backend-matrix.md` — backend coverage model.
- `docs/native-map-v0.10.md` — SHAP native architecture map.
- `docs/api-era-v0.10.md` — legacy/modern API analysis.
- `docs/v0.9-32-point-traceability.md` — all 32 maintainer-review points with implementation, tests, evidence, and residual limitations.
- `CHANGELOG.md` — release history.

## Validation rule

> **A static signal is a candidate, not a confirmed bug.**

Every promoted finding should carry enough runtime/provenance evidence for the selected lifecycle state.


## v0.15 semantic execution integrity

The v0.15 release strengthens `SHAPSemanticTensor`, the canonical representation used by semantic reduction, backend smoke validation, and differential additivity checks. It carries explicit axis semantics rather than inferring meaning from equal dimensions. Backend results distinguish execution from semantic certification, and real TreeExplainer protocol campaigns report observed coverage by scenario.


## Historical v0.20 evidence boundaries

Protocol observations are tied to an exact active TreeExplainer call ID and runtime SHAP frame. Mutation reports distinguish protocol-visible metadata changes from storage-backed changes. Backend matrices report scheduled versus executed cells, and the CatBoost interventional mismatch is a candidate requiring cross-version confirmation.

## Historical v0.20 claim boundaries

`shap_observed` requires an active `shap_call_id`, runtime SHAP frame, and callback event provenance. Shape/dtype mutation can be protocol-visible metadata only; storage mutation is reported separately. The CatBoost interventional reconstruction mismatch is a reproducible candidate, not a confirmed SHAP defect, until cross-version validation.


## v0.24 evidence discipline

- Every policy-required oracle is executed or represented explicitly as unavailable; missing required evidence yields `INCONCLUSIVE`, never an accidental `PASS`.
- Failed semantic reconstruction is never reported as `CERTIFIED`; failed finite/shape-valid comparisons use `SEMANTIC_MISMATCH`.
- Protocol evidence records the nearest observed SHAP frame as `canonical_source`; the toolkit wrapper is not treated as the canonical SHAP source.
- Backend summaries distinguish scheduled, executed, and not-executed cells and expose execution reasons.
- CPU/GPU fingerprints are collected in the target process using the same Python executable and environment as the target run; optional deep mode adds runtime/library details.
- API-era callable findings require SHAP constructor provenance; arbitrary nested calls are not labeled as SHAP calls.
- The CatBoost reconstruction anomaly remains `candidate_only` until a distinct-version validation run supports confirmation.
- `catboost_candidate_reproducer()` emits environment, model/data, expected-vs-actual, and cross-version status metadata without mutating the current environment.
- Release archives are validated directly and exclude generated `.out`, cache, bytecode, and CatBoost artifacts.

## Installation for full validation

The core toolkit has no mandatory third-party runtime dependency. For the complete SHAP/backend integration campaign, install the optional integration stack:

```bash
python -m pip install -e '.[test,integration]'
pytest -q
```

If an integration dependency is unavailable, dependency-gated integration tests are reported as skipped rather than being misclassified as toolkit failures. Runtime results must still be validated in an environment containing the relevant SHAP/backend packages before making repository-level claims.

## Release hardening gate

Before publishing a release archive, run the source smoke gate and the complete test suite:

```text
python scripts/release_check.py
python -m pytest -q --disable-warnings
```

Then validate the exact archive that will be distributed:

```text
python scripts/validate_distribution.py <release.zip>
```

The archive validator must report `archive hygiene: PASS`. Generated `.out`, cache, bytecode, build, and local review artifacts are not release contents.
