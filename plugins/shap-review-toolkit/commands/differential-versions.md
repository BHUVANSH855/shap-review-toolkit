# differential-versions

Run a cross-version differential test between two Python interpreters.

## Usage

/differential-versions --script PATH --reference PYTHON --candidate PYTHON

## What it does

Executes the target script under two explicit Python interpreters and compares
SHAP outputs. Never promotes a mismatch to confirmed without independent evidence.

## Output

Returns version-stamped differential result with reference_correctness=UNKNOWN.
