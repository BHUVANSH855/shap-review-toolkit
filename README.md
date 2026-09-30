# SHAP Review Toolkit

Evidence-driven, invariant-based, reproduction-first tooling for reviewing [SHAP](https://github.com/shap/shap).

The toolkit combines deterministic source analysis, semantic contracts, runtime validation, differential testing, fuzzing, provenance tracking, and reproduction workflows to turn review signals into maintainable findings.

## Architecture

```text
SHAP repository
      ↓
Discovery + Semantic IR
      ↓
Semantic Contracts / Oracles
      ↓
┌───────────────┬───────────────┬──────────────┐
│ Static /      │ Runtime       │ Differential │
│ Native        │ Fuzzing       │ Analysis     │
└───────────────┴───────────────┴──────────────┘
      ↓
Evidence + Provenance
      ↓
Reproduction / Sanitizer Validation
      ↓
Finding
```

## Current scope

The strongest runtime surface is **TreeExplainer**, including:

- semantic and tensor invariant validation
- output-space and additivity checks
- expected-value and interaction validation
- input-mutation checks
- Python object protocol probing and bounded re-entry
- runtime fuzzing and testcase minimization
- historical regression reproduction
- CPU/GPU differential comparison
- backend validation for supported TreeExplainer ecosystems
- SHAP native-boundary analysis
- API-era analysis
- provenance-aware evidence and finding promotion

The supported SHAP ecosystem includes scikit-learn, XGBoost, LightGBM, CatBoost, NumPy, pandas, SciPy, and SHAP itself. These are first-class integration targets of the toolkit. Individual runtime campaigns require the corresponding packages to be available in the execution environment.

## Evidence model

The toolkit separates **candidate generation** from **confirmation**.

```text
STATIC SIGNAL
      ↓
CANDIDATE
      ↓
RUNTIME / DIFFERENTIAL EVIDENCE
      ↓
REPRODUCTION
      ↓
INDEPENDENT EVIDENCE
      ↓
CONFIRMED FINDING
```

Historical matches and static patterns are evidence for investigation, not proof of a current defect. Evidence records retain provenance so correlated observations are not treated as independent confirmation.

## Installation

```bash
python -m pip install -e .
python -m pytest
```

For TreeExplainer and backend integration testing:

```bash
python -m pip install -e '.[test,integration]'
```

## Common commands

```bash
shap-review map /path/to/shap
shap-review analyze /path/to/shap
shap-review report /path/to/shap
shap-review history /path/to/shap
shap-review regressions

shap-review differential reference.py candidate.py
shap-review sanitizer reproduce.py --kind asan

shap-review fuzz --target treeexplainer --iterations 100 --seed 42
shap-review fuzz --target protocol --iterations 100 --seed 42
shap-review fuzz --target backends

shap-review native-map /path/to/shap
shap-review api-era /path/to/source.py
```

## Limitations

The toolkit produces evidence and structured triage; static analysis alone does not prove a defect.

Native C/C++/nanobind/CUDA analysis is structured triage rather than a compiler-grade memory-safety proof.

Unavailable supported integrations or unsupported execution environments are reported explicitly rather than treated as successful validation. A supported integration being unavailable in the current environment does not make that integration unsupported by the toolkit.

## Documentation

See the `docs/` directory for:

- architecture
- backend validation
- differential semantics
- evidence and provenance
- findings
- fuzzing
- invariants
- native analysis
- regression states
- semantic oracles

## Development

Contributions should add evidence and tests for new SHAP-specific rules, keep invariants independent from detectors, prefer deterministic evidence over unsupported claims, and preserve the distinction between candidates and confirmed findings.

See [CONTRIBUTING.md](CONTRIBUTING.md).
