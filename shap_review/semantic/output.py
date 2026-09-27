TREE_OUTPUTS = ["raw", "probability", "log_loss", "model_method"]


def output_contracts():
    return {
        "TreeExplainer": {
            "outputs": TREE_OUTPUTS,
            "probability_requires": "interventional",
            "log_loss_requires": "interventional",
        }
    }
