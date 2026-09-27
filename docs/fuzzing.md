# TreeExplainer fuzzing

The fuzzer generates bounded model/input configurations and executes them against real SHAP when the runtime dependencies are available.

## Generated dimensions

- feature count
- tree depth
- number of trees
- sample count
- regression/classification
- raw/probability output space
- deterministic random seed

The current generator deliberately excludes unsupported or highly environment-dependent cases such as arbitrary third-party model plugins and GPU execution.

## Independent oracle

The fuzzer does not treat `check_additivity=True` as sufficient evidence. It reconstructs the output from:

```text
sum(SHAP feature contributions) + expected_value
```

and compares that result with the model's prediction/predicted probabilities. It also verifies finiteness and input immutability.

A runtime exception or oracle failure becomes a reproducibility candidate. The case is retained and passed through the minimizer.
