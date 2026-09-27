"""Standalone SHAP/CatBoost reconstruction candidate reproducer.
Prints JSON only; no toolkit state is mutated.
"""

import json
import platform
import sys

import numpy as np
import shap
from catboost import CatBoostRegressor

X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=float)
y = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
probe = X[2:]
model = CatBoostRegressor(iterations=20, depth=3, verbose=False, random_seed=0).fit(
    X, y
)
explainer = shap.TreeExplainer(model, data=X[:2])
values = np.asarray(explainer.shap_values(probe), dtype=float)
base = np.asarray(explainer.expected_value, dtype=float)
prediction = np.asarray(model.predict(probe), dtype=float)
reconstructed = values.sum(axis=1) + base
error = float(np.max(np.abs(reconstructed - prediction)))
print(
    json.dumps(
        {
            "case": "SHAP-CATBOOST-INTERVENTIONAL-RECON",
            "shap_version": shap.__version__,
            "catboost_version": __import__("catboost").__version__,
            "python": sys.version,
            "platform": platform.platform(),
            "training_data": X.tolist(),
            "training_labels": y.tolist(),
            "probe_data": probe.tolist(),
            "shap_values": values.tolist(),
            "base_values": base.tolist(),
            "prediction": prediction.tolist(),
            "reconstructed": reconstructed.tolist(),
            "max_error": error,
            "reproduced": bool(
                not np.allclose(reconstructed, prediction, rtol=1e-5, atol=1e-8)
            ),
            "confirmation_status": "candidate_only",
        },
        sort_keys=True,
    )
)
