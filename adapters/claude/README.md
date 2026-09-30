# Claude Adapter

Thin routing shim that maps Claude plugin tool-call conventions to the
`shap_review` core engine.  No SHAP analysis logic lives in this adapter.

## Usage

```python
from adapters import ClaudeAdapter

adapter = ClaudeAdapter()

# Route a tool call from the Claude plugin system:
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

## Available commands

| Command | Description |
|---|---|
| `map` | Repository structure and semantic graph |
| `health` | Repository health summary |
| `analyze` | Semantic analysis → candidate list |
| `report` | Markdown candidate report |
| `hotspots` | High-risk file ranking |
| `explore` | Interactive exploration |
| `investigate` | Investigation plan for a candidate |
| `reproduce` | Run a reproducer script |
| `fuzz` | TreeExplainer fuzz campaign |
| `fuzz-treeexplainer` | TreeExplainer-specific fuzz |
| `fuzz-protocol` | Adversarial protocol campaign |
| `fuzz-backends` | Backend matrix execution |
| `differential` | Script-level differential comparison |
| `cpu-gpu-differential` | CPU vs GPU differential |
| `regressions` | Historical regression suite |
| `history` | Git history correlation |
| `sanitizer` | Sanitizer-instrumented execution |
| `differential-versions` | Cross-interpreter differential |
| `evidence` | Evidence model metadata |
| `semantic-oracle` | SHAP contract oracle evaluation |
| `native-map` | SHAP native component map |
| `api-era` | API era detection on a source file |
| `capabilities` | Adapter capabilities |
| `version` | Toolkit version |

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
as diagnostic metadata only, not as evidence enrichment.

## Architecture

```
Claude plugin tool call
        │
        ▼
  ClaudeAdapter.invoke()
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
- Native analysis findings are heuristic (regex-based); treat them as triage
  signals requiring runtime validation, not confirmed bugs.
