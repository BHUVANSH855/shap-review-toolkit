# Evidence Policy

The toolkit separates **candidate generation** from **finding confirmation**.

```text
candidate
   ↓
evidence collection
   ↓
independent evidence kinds
   ↓
evidence chain
   ↓
verdict
```

Static source patterns can identify where to investigate, but should not alone be presented as a confirmed vulnerability.

For memory-safety findings, sanitizer-backed reproduction receives the strongest evidence weight.

For semantic regressions, differential execution plus a SHAP-specific mathematical oracle is preferred.

For historical claims, the toolkit should report the exact commit/path/symbol evidence rather than saying that a code pattern is merely "similar".
