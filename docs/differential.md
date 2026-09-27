# Differential Validation

v0.6 supports two layers:

1. generic recursive JSON comparison;
2. SHAP-aware normalization for `Explanation`-style results and NumPy-like values.

The semantic layer converts common fields such as `values`, `base_values`, `data`, `feature_names`, and `output_names` into a stable JSON-compatible representation before comparison.

This is a semantic normalization layer, not yet a complete mathematical equivalence checker. Model-output space, additivity, and algorithm-specific contracts remain separate invariants.
