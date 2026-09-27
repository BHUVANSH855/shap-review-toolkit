# Source-grounded reproducer from SHAP issue #5098.
# The exact behavior is version-dependent because it targets the pre-built
# TreeEnsemble path introduced on SHAP main.
import numpy as np
from shap.explainers._tree import TreeEnsemble, TreeExplainer


def make_tree_dict():
    return {
        "objective": "binary_crossentropy",
        "tree_output": "log_odds",
        "trees": [
            {
                "children_left": np.array([1, -1, -1], dtype=np.int32),
                "children_right": np.array([2, -1, -1], dtype=np.int32),
                "children_default": np.array([1, -1, -1], dtype=np.int32),
                "features": np.array([0, -2, -2], dtype=np.int32),
                "thresholds": np.array([0.5, -2, -2], dtype=np.float64),
                "values": np.array([[0.0], [0.2], [0.8]], dtype=np.float64),
                "node_sample_weight": np.array([100.0, 40.0, 60.0]),
            }
        ],
    }


prebuilt = TreeEnsemble(make_tree_dict())
explainer = TreeExplainer(model=prebuilt, model_output="probability")
print(explainer.model_output, getattr(explainer.model, "model_output", None))
