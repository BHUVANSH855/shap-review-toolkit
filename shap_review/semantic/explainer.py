EXPLAINER_PROFILES = {
    "TreeExplainer": {
        "native": ["_cext"],
        "contracts": [
            "additivity",
            "output_space",
            "shape",
            "interaction",
            "model_dispatch",
        ],
    },
    "ExactExplainer": {"native": ["_cutils"], "contracts": ["masking", "shape"]},
    "PermutationExplainer": {"contracts": ["masking", "shape", "additivity"]},
    "PartitionExplainer": {
        "native": ["_cutils"],
        "contracts": ["clustering", "masking", "shape"],
    },
    "LinearExplainer": {"contracts": ["model_output", "shape", "masking"]},
    "KernelExplainer": {
        "native": ["_cutils"],
        "contracts": ["masking", "shape", "numerical"],
    },
    "DeepExplainer": {"contracts": ["framework_callbacks", "shape", "output"]},
    "GradientExplainer": {"contracts": ["framework_callbacks", "shape", "output"]},
    "AdditiveExplainer": {"contracts": ["first_order", "additivity", "dispatch"]},
}
