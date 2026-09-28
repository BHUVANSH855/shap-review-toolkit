# Contributing

1. Add evidence before adding a new SHAP-specific rule.
2. Encode the invariant independently from the detector.
3. Prefer deterministic candidates over LLM-only suspicion.
4. Add a fixture for every detector regression.
5. Never label a candidate as confirmed without reproducible evidence.
6. Keep adapters thin; analysis belongs in `shap_review/`.