import numpy as np
import shap
import xgboost as xgb
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=10, random_state=42)
model = xgb.XGBClassifier(n_estimators=50, random_state=42).fit(X[:150], y[:150])
explainer = shap.TreeExplainer(model)
before = np.asarray(explainer.expected_value).copy()
before_shape = before.shape
explainer.shap_values(X[150:])
after = np.asarray(explainer.expected_value).copy()
after_shape = after.shape
print(
    {
        "before": before.tolist(),
        "before_shape": before_shape,
        "after": after.tolist(),
        "after_shape": after_shape,
    }
)
