# Gemini Adapter

Thin routing shim that maps Google Gemini function-calling conventions to
the `shap_review` core engine.  No SHAP analysis logic lives in this adapter.

## Usage

```python
from adapters import GeminiAdapter

adapter = GeminiAdapter()

# Route a function call from the Gemini function-calling API:
result = adapter.invoke("analyze", {"root": "/path/to/shap"})

# Check for errors:
if "error" in result:
    print(result["error"]["kind"], result["error"]["message"])
else:
    candidates = result["result"]
```

The response envelope always contains `provider`, `version`,
`schema_version`, and `command`.  On success it also contains `result`;
on error it contains `error` with `kind`, `message`, `command`, and `layer`.

## Security

> ⚠️ **The following commands execute arbitrary Python scripts via
> `subprocess`:**
> `differential`, `cpu-gpu-differential`, `differential-versions`,
> `reproduce`, `sanitizer`.
>
> Only invoke these commands with trusted, controlled input paths.
> Do **not** expose them to untrusted end-users without an additional
> authentication and path-restriction layer.

## Dynamic evidence note

The `analyze` command runs a bounded TreeExplainer campaign against the
*installed* SHAP runtime, not the repository under review.  The
`runtime-bridge.json` artifact records any anomalies found.  Those anomalies
do **not** currently enrich candidate evidence chains because the generic
bridge campaign does not set per-candidate fingerprints.  Treat bridge output
as diagnostic metadata only.

## Architecture

```
Gemini function call
        │
        ▼
  GeminiAdapter.invoke()
        │
        ▼
  BaseReviewAdapter.invoke()  ←  validates command, logs security warnings
        │
        ├── built-in: capabilities / version / evidence  → returns directly
        │
        └── all others → shap_review.cli.dispatch() → ReviewEngine
```

## Limitations

- `InvestigationPlan` from the `investigate` command returns a fixed set of
  generic steps regardless of candidate type.  It is a placeholder.
- Float32 SHAP values may trigger false FAIL results from `semantic-oracle`
  due to the default tolerance of `1e-6`.  Pass a relaxed `tolerance` in the
  contract arguments for float32 inputs.
- Native analysis findings are heuristic; treat them as triage signals
  requiring runtime validation, not confirmed bugs.
