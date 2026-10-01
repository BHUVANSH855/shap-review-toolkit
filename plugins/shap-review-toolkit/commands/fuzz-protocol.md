# fuzz-protocol

Run a protocol-level fuzzing campaign targeting SHAP Python object protocol paths.

## Usage

/fuzz-protocol [--iterations N]

## What it does

Exercises SHAP explainer construction and shap_values() calls with edge-case
inputs targeting the Python object protocol boundary: __array__, __len__,
__getitem__, and related dunder methods.

## Output

Returns classified anomalies with traceback-grounded SHAP frame detection.
