import shap


def explain(df):
    # Historical boundary shape: pandas nullable values are converted before native Tree SHAP.
    X = df.to_numpy()
    return shap.TreeExplainer(model).shap_values(X)
