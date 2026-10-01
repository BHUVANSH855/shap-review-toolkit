# analyze

Run the full static analysis pipeline against the SHAP repository.

## Usage

/analyze [--path PATH]

## What it does

Executes all semantic analyzers: SHAP-specific semantic analysis,
Python contract analysis, native boundary detection, and API era
classification. Produces a ranked candidate list with evidence refs.

## Output

Returns candidates sorted by confidence with bug class and invariant tags.
