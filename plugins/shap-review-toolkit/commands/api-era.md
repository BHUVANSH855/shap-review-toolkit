# api-era

Analyze SHAP API era compatibility and deprecated usage patterns.

## Usage

/api-era

## What it does

Detects usage of deprecated SHAP APIs, identifies API era transitions
(legacy shap_values() list return vs array return), and flags callers
that may break across SHAP version boundaries.

## Output

Returns era-classified candidates with migration guidance.
