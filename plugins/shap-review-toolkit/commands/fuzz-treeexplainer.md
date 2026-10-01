# fuzz-treeexplainer

Run a bounded TreeExplainer fuzzing campaign against the installed SHAP runtime.

## Usage

/fuzz-treeexplainer [--iterations N] [--seed S]

## What it does

Executes the TreeExplainerFuzzer with the specified number of iterations (default: 10).
Each iteration generates a random case covering different backends, input variants,
dtypes, and representations. Results include additivity checks, shape validation,
and input-mutation detection.

## Output

Returns a campaign summary with unique failures, coverage, and provenance.
