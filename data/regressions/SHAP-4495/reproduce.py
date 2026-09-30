"""Standalone reproducer for SHAP issue #4495.

expected_value must be stable across initialization and shap_values calls.

Expected: explainer.expected_value is identical before and after shap_values().
Observed: expected_value changes shape or value after shap_values() on XGBoost
          classifiers on affected versions.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys

import numpy as np
import shap
import xgboost as xgb
from sklearn.datasets import make_classification


def _versions() -> dict:
    out = {}
    for pkg in ("shap", "numpy", "xgboost", "scikit-learn"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


X, y = make_classification(n_samples=200, n_features=10, random_state=42)
model = xgb.XGBClassifier(n_estimators=50, random_state=42).fit(X[:150], y[:150])
explainer = shap.TreeExplainer(model)
before = np.asarray(explainer.expected_value).copy()
before_shape = list(before.shape)
explainer.shap_values(X[150:])
after = np.asarray(explainer.expected_value).copy()
after_shape = list(after.shape)
changed = not np.array_equal(before, after) or before_shape != after_shape

print(
    json.dumps(
        {
            "issue": "SHAP-4495",
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": _versions(),
            },
            "reproduced": changed,
            "status": "reproduced" if changed else "not_reproduced",
            "before": before.tolist(),
            "before_shape": before_shape,
            "after": after.tolist(),
            "after_shape": after_shape,
        },
        indent=2,
    )
)
