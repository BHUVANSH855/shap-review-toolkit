from __future__ import annotations

import os
import platform
import sys

from .runner import run_json_script
from .semantic import compare_shap_contract


def _fingerprint(backend):
    """Parent-process metadata retained for compatibility; subprocess metadata is authoritative."""
    fp = {"backend": backend, "python": sys.version, "platform": platform.platform()}

    try:
        import shap

        fp["shap_version"] = shap.__version__
    except (AttributeError, ImportError):
        fp["shap_version"] = None

    try:
        import numpy as np

        fp["numpy_version"] = np.__version__
    except (AttributeError, ImportError):
        fp["numpy_version"] = None

    fp["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return fp


def cpu_gpu_scripts(cpu_script, gpu_script, timeout=120, rtol=1e-5, atol=1e-8):
    cpu = run_json_script(
        cpu_script,
        timeout=timeout,
        env={"SHAP_REVIEW_BACKEND": "cpu"},
        capture_environment=True,
        environment_depth="deep",
    )
    gpu = run_json_script(
        gpu_script,
        timeout=timeout,
        env={"SHAP_REVIEW_BACKEND": "gpu"},
        capture_environment=True,
        environment_depth="deep",
    )
    out = {
        "applicable": bool(cpu.get("ok") and gpu.get("ok")),
        "backend_pair": "cpu-vs-gpu",
        "cpu": cpu,
        "gpu": gpu,
        "cpu_environment": cpu.get("subprocess_environment") or _fingerprint("cpu"),
        "gpu_environment": gpu.get("subprocess_environment") or _fingerprint("gpu"),
    }
    if not out["applicable"]:
        out.update(
            {"status": "SKIPPED", "reason": "one backend script did not execute"}
        )
        return out
    comparison = compare_shap_contract(cpu["value"], gpu["value"], rtol=rtol, atol=atol)
    out.update(
        {
            "status": "MATCH" if comparison.get("semantic_equal") else "MISMATCH",
            "comparison": comparison,
            "semantic_equal": comparison.get("semantic_equal", comparison.get("equal")),
        }
    )
    return out