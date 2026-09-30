---
name: SHAP bug report
about: Report a bug found using the SHAP review toolkit
title: '[BUG] '
labels: bug
assignees: ''
---

## Summary

One sentence: what goes wrong and in which SHAP component.

## Environment

- shap==
- numpy==
- pandas==
- scikit-learn==
- xgboost==
- lightgbm==
- catboost==
- Python==
- Platform==

## Steps to reproduce

Paste a minimal standalone reproducer here (no toolkit dependency).

## Expected behaviour

What SHAP should do according to its documentation or mathematical contract.

## Observed behaviour

What actually happens. Include full traceback or numerical values.

## Invariant violated

| Invariant | Description |
|---|---|
| INV-ID | e.g. Additivity: sum(shap_values) + expected_value == model.predict() |

## Evidence

| Kind | Source | Claim |
|---|---|---|
| dynamic | TreeExplainer fuzzer | e.g. additivity error = 0.003 > tolerance 5e-4 |
| historical | GitHub issue URL | e.g. same failure on shap==0.46.0 |

## Regression status

- Reproduced on shap==X.Y.Z
- Not reproduced on shap==A.B.C
- No fix version identified yet

## Additional context

Cross-version differential results, backend specifics, related issues.
