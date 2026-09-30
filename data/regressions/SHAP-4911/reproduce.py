"""Standalone reproducer for SHAP issue #4911.

Nullable pandas Int64 background reaches TreeExplainer native path as
object dtype and cannot be cast to float64.

Expected: SHAP converts nullable dtype or raises a clear, informative error.
Observed: TypeError from native C extension on affected versions.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys

import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

X = pd.DataFrame({"a": pd.Series([1, 2, pd.NA], dtype="Int64"), "b": [0.1, 0.2, 0.3]})
model = RandomForestRegressor(random_state=0).fit(X.fillna(0), [1, 2, 3])


def _versions() -> dict:
    out = {}
    for pkg in ("shap", "numpy", "pandas", "scikit-learn"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


result: dict = {
    "issue": "SHAP-4911",
    "environment": {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _versions(),
    },
    "reproduced": False,
    "exception_type": None,
    "exception_message": None,
    "status": "not_reproduced",
}

try:
    shap.TreeExplainer(model, data=X)
    result["status"] = "not_reproduced"
except Exception as exc:  # noqa: BLE001
    exc_type = type(exc).__name__
    exc_msg = str(exc).lower()
    nullable_dtype_markers = (
        "cannot cast array data from dtype('o') to dtype('float64')",
        "cannot cast array data from dtype('object') to dtype('float64')",
    )
    is_target = exc_type == "TypeError" and any(
        m in exc_msg for m in nullable_dtype_markers
    )
    result.update(
        {
            "reproduced": is_target,
            "exception_type": exc_type,
            "exception_message": str(exc),
            "status": "reproduced" if is_target else "ambiguous",
        }
    )

print(json.dumps(result, indent=2))
