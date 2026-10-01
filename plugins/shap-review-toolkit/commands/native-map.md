# native-map

Map SHAP native extension boundaries and C/Cython entry points.

## Usage

/native-map

## What it does

Identifies all native extension modules in the SHAP installation, maps
Python-to-native call boundaries, and flags uninstrumented entry points
relevant to sanitizer coverage.

## Output

Returns a structured map of native targets with instrumentation status.
