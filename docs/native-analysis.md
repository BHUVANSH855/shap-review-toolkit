# Native Boundary Analysis

The native analyzer covers SHAP's C/C++/NumPy/nanobind/CUDA boundaries.

For each detected boundary, v0.6 records nearby evidence for:

- dtype/shape/stride/contiguity validation;
- ownership and reference-count operations;
- exception/error handling;
- Python callbacks and re-entry/GIL context;
- risk flags when expected context is absent.

This is deliberately an evidence-producing analyzer. A missing nearby marker is **not** itself a proof of a defect; reviewers must follow the full control flow.
