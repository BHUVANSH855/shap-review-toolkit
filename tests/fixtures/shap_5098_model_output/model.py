import shap


def build(model):
    return shap.TreeExplainer(model, model_output="probability")
