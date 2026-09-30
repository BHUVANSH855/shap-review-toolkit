def test_multiclass_interactions(model, X):
    model.shap_interaction_values(X)
