def test_multiclass_interactions(model, X):
    values = model.shap_interaction_values(X)
    interaction_values = values
