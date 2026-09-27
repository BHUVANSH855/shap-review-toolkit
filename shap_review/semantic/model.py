MODEL_CONTRACTS = {
    "generic": {"required": ["predict"], "optional": ["predict_proba"]},
    "tree": {
        "families": ["sklearn", "xgboost", "lightgbm", "catboost", "pyspark"],
        "contracts": ["feature_count", "output_dimension", "prediction_space"],
    },
}
