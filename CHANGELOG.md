# Changelog

## 0.30.0

### Correctness and release hardening

- Restored `shap_review.run_script` from the canonical reproduction runner.
- Restored the public fuzzing export from the implemented TreeExplainer oracle.
- Fixed CPU/GPU differential comparison to use the canonical differential semantic comparator.
- Added explicit oracle-independence state `DECLARED`; caller assertions no longer become `PROVEN` automatically.
- Expanded evidence validation with schema/provenance/scoring-eligibility state.
- Formalized regression outcomes including `AMBIGUOUS`, `TARGET_FAILURE`, `TOOLKIT_FAILURE`, and `UNSUPPORTED`.
- Made semantic singleton broadcasting role-aware and fail-closed for contribution tensors.
- Added release/import smoke checks and synchronized release metadata to 0.30.0 / schema 2.10.
- Removed generated `.out` artifacts from the source distribution.

# v0.28.0 — Semantic Consistency & Evidence Precision

## Focus

v0.28 is a hardening release. It deliberately avoids adding new feature families and instead strengthens the correctness of the existing semantic and evidence paths.

- Unified differential semantic alignment with the canonical contract-aware alignment engine.
- Removed differential-side generic broadcasting for semantic comparisons.
- Added explicit target-source and oracle-independence classification.
- Prevented model API calls from being automatically treated as independent oracles.
- Made API-era SHAP bindings lexical-scope local, including parameter shadowing and invalidation.
- Strengthened interaction validation with symmetry and row-wise reconstruction.
- Routed expected-value alignment through the canonical semantic alignment policy.
- Added v0.28 validation and traceability documentation.
- Bumped release metadata to `0.28.0` / schema `2.9`.

## v0.28.0 validation hardening

- Made regression fixtures portable on Windows and other platforms by removing hard-coded `/tmp` paths.
- Made integration tests explicitly dependency-aware instead of reporting missing third-party packages as toolkit failures.
- Added a dedicated `integration` optional-dependency group for full SHAP/backend validation.
- Made backend-matrix skipped results expose the same stable summary schema as executed runs.
- Made sanitizer signature testing platform-independent by exercising the current Python interpreter.
