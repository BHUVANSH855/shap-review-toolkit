# Differential Validation

v0.6 supports two layers:

1. generic recursive JSON comparison;
2. SHAP-aware normalization for `Explanation`-style results and NumPy-like values.

The semantic layer converts common fields such as `values`, `base_values`, `data`, `feature_names`, and `output_names` into a stable JSON-compatible representation before comparison.

This is a semantic normalization layer, not yet a complete mathematical equivalence checker. Model-output space, additivity, and algorithm-specific contracts remain separate invariants.

## Role-aware semantic alignment

Differential comparisons use the canonical tensor alignment policy. Singleton expansion is role-aware: contribution tensors (`values`, `shap_values`, and interaction values) may only expand a singleton sample axis, while target/baseline fields may additionally use explicitly represented output/class axes. Feature and interaction axes are always protected. Ambiguous one-dimensional baselines are rejected unless the canonical tensor carries enough semantic axis information to select the output axis deterministically.
