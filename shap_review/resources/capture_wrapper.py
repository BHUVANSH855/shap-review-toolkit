"""Subprocess capture wrapper for differential testing.

Invoked as:
    python capture_wrapper.py <target_script> [environment_depth]

Runs <target_script> under runpy, captures its stdout JSON output, and
emits a single JSON object containing the result plus environment metadata.
This avoids passing a multi-line inline script to python -c, which breaks
on Windows paths containing backslashes.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import os
import platform
import sys

script_path = sys.argv[1] if len(sys.argv) > 1 else ""
environment_depth = sys.argv[2] if len(sys.argv) > 2 else "basic"

buf = io.StringIO()
rc = 0
err = None

try:
    with contextlib.redirect_stdout(buf):
        import runpy

        runpy.run_path(script_path, run_name="__main__")
except SystemExit as exc:
    rc = int(exc.code) if isinstance(exc.code, int) else 0
except Exception as exc:  # noqa: BLE001
    rc = 1
    err = f"{type(exc).__name__}: {exc}"

out = buf.getvalue()
value = None
parse_error = None

try:
    value = json.loads(out)
except Exception as exc:  # noqa: BLE001
    parse_error = str(exc)

envinfo: dict = {
    "python": sys.version,
    "platform": platform.platform(),
    "executable": sys.executable,
    "machine": platform.machine(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "environment_depth": environment_depth,
}

if environment_depth == "deep":
    try:
        import numpy as _np

        envinfo["blas_configuration"] = (
            _np.__config__.get_info("blas_opt_info")
            if hasattr(_np.__config__, "get_info")
            else None
        )
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        envinfo["openmp_runtime"] = os.environ.get("OMP_NUM_THREADS") or os.environ.get(
            "OMP_RUNTIME"
        )
    except Exception:  # noqa: BLE001, S110
        pass
    envinfo["cpu_model"] = platform.processor()
    envinfo["cuda_runtime"] = os.environ.get("CUDA_PATH")
    envinfo["gpu_model"] = os.environ.get("NVIDIA_VISIBLE_DEVICES")
    envinfo["environment_variables"] = {
        k: v
        for k, v in os.environ.items()
        if k
        in {
            "CUDA_VISIBLE_DEVICES",
            "CUDA_PATH",
            "OMP_NUM_THREADS",
            "OMP_RUNTIME",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "LD_LIBRARY_PATH",
        }
    }

for pkg in (
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
        envinfo[pkg + "_version"] = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        envinfo[pkg + "_version"] = None

print(
    json.dumps(
        {
            "value": value,
            "stdout": out[-20000:],
            "parse_error": parse_error,
            "returncode": rc,
            "error": err,
            "environment": envinfo,
        }
    )
)
