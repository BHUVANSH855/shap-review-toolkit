# Backend matrix

The runtime matrix recognizes the primary TreeExplainer ecosystems:

- scikit-learn
- XGBoost
- LightGBM
- CatBoost

Availability is detected at runtime. Missing optional dependencies are reported as unavailable rather than silently treated as tested.

The matrix dimensions are:

```text
backend × model_output × input_representation × missing_values × classification × interaction
```
