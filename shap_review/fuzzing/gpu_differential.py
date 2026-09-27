from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BackendRun:
    backend: str
    available: bool
    executed: bool
    result: object = None
    error: str | None = None


class CPUGPUDifferential:
    """Compare independently executed CPU/GPU TreeExplainer outputs when GPU support exists."""

    def __init__(self, rtol=1e-5, atol=1e-8):
        self.rtol = rtol
        self.atol = atol

    def availability(self):
        try:
            importlib.import_module("cupy")
            return True
        except Exception:
            return False

    def run(self, cpu_callable, gpu_callable, *args, **kwargs):
        cpu = self._run("cpu", cpu_callable, args, kwargs)
        gpu = self._run("gpu", gpu_callable, args, kwargs)
        if not cpu.executed or not gpu.executed:
            return {
                "applicable": False,
                "reason": "CPU/GPU execution unavailable",
                "cpu": asdict(cpu),
                "gpu": asdict(gpu),
            }
        comparison = self._compare(cpu.result, gpu.result)
        return {
            "applicable": True,
            "equal": comparison["equal"],
            "semantic_equal": comparison["semantic_equal"],
            "fields": comparison["fields"],
            "cpu": asdict(cpu),
            "gpu": asdict(gpu),
            "oracle": "semantic-field-policy",
        }

    def _run(self, name, fn, args, kwargs):
        try:
            return BackendRun(name, True, True, fn(*args, **kwargs))
        except Exception as exc:
            return BackendRun(name, True, False, None, f"{type(exc).__name__}: {exc}")

    def _compare(self, a, b):
        from .semantic import compare_shap_contract

        return compare_shap_contract(a, b, rtol=self.rtol, atol=self.atol)
