from __future__ import annotations

import importlib.metadata
import os
import platform
import sys


def snapshot() -> dict:
    packages = {}
    for name in (
        "shap",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    env_keys = {
        k: os.getenv(k)
        for k in (
            "CUDA_VISIBLE_DEVICES",
            "CUDA_PATH",
            "OMP_NUM_THREADS",
            "OMP_RUNTIME",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "LD_LIBRARY_PATH",
        )
        if os.getenv(k) is not None
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "packages": packages,
        "environment": env_keys,
    }
