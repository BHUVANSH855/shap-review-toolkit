# Historical regression states

Regression runners use a conservative state vocabulary:

- `reproduced` — the expected behavior/anomaly was reproduced.
- `not_reproduced` — the target executed but the expected anomaly was absent.
- `blocked` — the environment or dependency prevented execution.
- `ambiguous` — execution produced a result that cannot be safely attributed to the historical defect.
- `target_failure` — the target under test failed outside the historical contract being checked.
- `toolkit_failure` — the review toolkit itself failed while attempting the regression.
- `unsupported` — the installed target version cannot exercise the relevant contract.
- `static_precondition` — source/API preconditions were observed, but runtime reproduction was not claimed.
- `candidate_reproduced` — a candidate anomaly reproduced, but it remains candidate-only until independent confirmation.

Only `reproduced` may directly support a historical reproduction claim. `ambiguous`, `blocked`, `target_failure`, `toolkit_failure`, and `unsupported` must never be promoted as confirmations.
