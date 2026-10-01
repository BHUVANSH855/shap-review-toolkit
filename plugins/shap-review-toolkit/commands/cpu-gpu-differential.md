# cpu-gpu-differential

Run a CPU vs GPU differential test for TreeExplainer outputs.

## Usage

/cpu-gpu-differential --script PATH

## What it does

Executes the target script twice with different CUDA_VISIBLE_DEVICES settings
and compares SHAP outputs semantically. Requires GPU availability.

## Output

Returns differential agreement status with environment capture for both runs.
