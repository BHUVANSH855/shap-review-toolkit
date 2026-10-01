# fuzz-backends

Run a fuzzing campaign across all available TreeExplainer backends.

## Usage

/fuzz-backends [--iterations N]

## What it does

Exercises sklearn, XGBoost, LightGBM, and CatBoost backends with the same
input space. Surfaces backend-specific additivity failures, shape mismatches,
and exception patterns.

## Output

Returns per-backend results with cross-backend comparison summary.
