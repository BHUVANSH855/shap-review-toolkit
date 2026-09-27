# SHAP Semantic Oracles

The v0.11 oracle layer is intentionally contract-driven.

## Additivity

Canonical shapes are interpreted as:

- `(samples, features)` → reduce feature axis.
- `(samples, features, outputs)` → reduce feature axis and preserve output axis.
- `(samples, features, features)` → interaction reduction over both feature axes.
- `(samples, features, features, outputs)` → interaction reduction over both feature axes and preserve output axis.

This prevents multiclass output dimensions from being accidentally collapsed into the feature reduction.

## Expected values

`expected_value` and `Explanation.base_values` are not required to have identical raw shapes. Scalar and vector expected values may be broadcast across samples when the contract permits it.

## Output space

The output oracle reconstructs `base_values + SHAP contributions` and compares that reconstruction with an independently supplied/evaluated target in the contract's declared output space. Successful model evaluation alone is not sufficient.

## Applicability

Every required oracle must be either applicable and passing or the complete contract is `INCONCLUSIVE`. This prevents unavailable checks from silently becoming evidence of correctness.
