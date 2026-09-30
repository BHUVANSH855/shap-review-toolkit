from __future__ import annotations

import hashlib
import json
import multiprocessing
import platform
import random
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from .generators.input import generate_input_case  # noqa: F401 — kept for API compat
from .generators.tree import generate_tree_case
from .harnesses.treeexplainer import run_case
from .minimizers.basic import minimize_case
from .oracles.tree import evaluate_execution

# Default per-iteration timeout.
DEFAULT_CASE_TIMEOUT: int = 30


def _fuzz_worker(queue: multiprocessing.Queue, case: dict) -> None:
    """Module-level worker so it is picklable on Windows (spawn context)."""
    try:
        queue.put(run_case(case))
    except Exception as exc:  # noqa: BLE001
        import traceback as _tb
        queue.put({
            "executed": True,
            "failed": True,
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": _tb.format_exc(limit=10),
            "case": case,
        })

class TreeExplainerFuzzer:
    def __init__(self, seed: int = 0, case_timeout: int = DEFAULT_CASE_TIMEOUT):
        self.seed = seed
        self.rng = random.Random(seed)
        self.case_timeout = max(1, int(case_timeout))

    @staticmethod
    def _execution_trace(case: dict) -> dict:
        return {
            k: case[k]
            for k in (
                "representation",
                "background_representation",
                "classification",
                "model_output",
                "interaction",
                "dtype",
                "nan",
                "n_features",
                "n_samples",
                "n_trees",
                "depth",
            )
        }

    @staticmethod
    def _runtime_provenance() -> dict[str, Any]:
        """Describe the runtime target actually used by the fuzz campaign."""
        try:
            shap = import_module("shap")
        except ImportError:
            return {
                "available": False,
                "shap_version": None,
                "shap_source_path": None,
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            }

        source_path = getattr(shap, "__file__", None)

        return {
            "available": True,
            "shap_version": getattr(shap, "__version__", None),
            "shap_source_path": source_path,
            "shap_source_root": (
                str(Path(source_path).resolve().parent) if source_path else None
            ),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }

    @staticmethod
    def _provenance_fingerprint(provenance: dict[str, Any]) -> str:
        """Return a stable fingerprint for the runtime target."""
        encoded = json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _campaign_provenance(self, iterations: int) -> dict[str, Any]:
        runtime = self._runtime_provenance()
        payload = {
            **runtime,
            "seed": self.seed,
            "producer": "treeexplainer-fuzzer",
            "case_timeout": self.case_timeout,
            "iterations": iterations,
        }
        return {
            **payload,
            "target_fingerprint": self._provenance_fingerprint(payload),
        }

    def _run_case_with_timeout(self, case: dict) -> dict:
        """Run case in a child process with a hard OS-level timeout.

        Uses a module-level worker (_fuzz_worker) so it is picklable on
        Windows where multiprocessing uses the spawn start method.
        multiprocessing.Process.terminate() sends SIGTERM on POSIX and
        TerminateProcess on Windows — the worker is killed at OS level
        without blocking the parent, unlike ThreadPoolExecutor which blocks
        until the thread finishes regardless of the future timeout.
        """
        queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)
        proc = multiprocessing.Process(
            target=_fuzz_worker, args=(queue, case), daemon=True
        )
        proc.start()
        proc.join(timeout=self.case_timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
            return {
                "executed": True,
                "failed": True,
                "timeout": True,
                "exception": "TimeoutError",
                "message": (
                    f"case exceeded {self.case_timeout}s timeout"
                    " — worker process terminated"
                ),
                "traceback": "",
                "case": case,
            }

        if not queue.empty():
            return queue.get_nowait()

        return {
            "executed": True,
            "failed": True,
            "timeout": False,
            "exception": "WorkerError",
            "message": "worker process exited without returning a result",
            "traceback": "",
            "case": case,
        }

    @staticmethod
    def _failure_fingerprint(result: dict) -> str:
        """Fingerprint for crash deduplication.

        Uses exception type + full message + the innermost traceback frame
        (file + line number) so that two structurally different crashes that
        happen to share the same exception type and a similar message prefix
        are not incorrectly merged into a single deduplicated entry.
        """
        exc_type = result.get("exception", "")
        message = str(result.get("message", ""))
        # Extract innermost frame from traceback for structural identity.
        tb = result.get("traceback", "")
        innermost = ""
        if tb:
            for line in tb.splitlines():
                stripped = line.strip()
                if stripped.startswith("File "):
                    innermost = stripped
        payload = f"{exc_type}:{message}:{innermost}"
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def run(self, iterations: int = 10):
        campaign_provenance = self._campaign_provenance(iterations)
        execution_id = hashlib.sha256(
            (
                f"treeexplainer:{self.seed}:{iterations}:"
                f"{campaign_provenance['target_fingerprint']}"
            ).encode()
        ).hexdigest()

        results = []
        seen_failure_fingerprints: dict[str, int] = {}
        duplicate_failures: int = 0

        for i in range(iterations):
            case = generate_tree_case(self.rng)

            case["input"] = {
                "samples": case["n_samples"],
                "features": case["n_features"],
                "dtype": case["dtype"],
                "representation": case["representation"],
                "nan": case["nan"],
            }
            case["seed"] = self.rng.randrange(2**31)

            result = self._run_case_with_timeout(case)
            oracle = evaluate_execution(result)

            item = {
                "iteration": i,
                "case": case,
                "execution_trace": self._execution_trace(case),
                "execution": result,
                "oracle": oracle,
                "provenance": {
                    **campaign_provenance,
                    "execution_id": execution_id,
                    "input_fingerprint": self._case_fingerprint(case),
                },
            }

            is_duplicate = False
            if result.get("failed") and not oracle.get("valid"):
                fail_fp = self._failure_fingerprint(result)
                if fail_fp in seen_failure_fingerprints:
                    seen_failure_fingerprints[fail_fp] += 1
                    duplicate_failures += 1
                    is_duplicate = True
                else:
                    seen_failure_fingerprints[fail_fp] = 1
            item["duplicate"] = is_duplicate

            if not oracle.get("valid") and result.get("executed") and not is_duplicate:
                item["minimized_case"] = minimize_case(case)

            results.append(item)

        executed = [result for result in results if result["execution"].get("executed")]

        return {
            "iterations": iterations,
            "results": results,
            "executed_cases": len(executed),
            "validated_cases": sum(
                result["oracle"].get("valid", False) for result in results
            ),
            "failures": sum(
                not result["oracle"].get("valid", False) for result in results
            ),
            "unique_failures": len(seen_failure_fingerprints),
            "duplicate_failures": duplicate_failures,
            "timeout_cases": sum(
                bool(result["execution"].get("timeout")) for result in results
            ),
            "case_timeout_seconds": self.case_timeout,
            "skipped_cases": sum(
                bool(result["execution"].get("skipped")) for result in results
            ),
            "coverage": self.coverage(results),
            "provenance": {
                **campaign_provenance,
                "execution_id": execution_id,
                "iterations": iterations,
            },
        }

    @staticmethod
    def _case_fingerprint(case: dict[str, Any]) -> str:
        """Return a stable fingerprint for a generated fuzz case."""
        encoded = json.dumps(
            case,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(encoded.encode()).hexdigest()

    @staticmethod
    def coverage(results: list[dict]) -> dict:
        fields = [
            "representation",
            "background_representation",
            "classification",
            "model_output",
            "interaction",
            "dtype",
            "nan",
        ]
        coverage = {}

        for field in fields:
            values = sorted(
                {str(result["execution_trace"][field]) for result in results}
            )
            coverage[field] = {
                "observed": values,
                "count": len(values),
            }

        coverage["interaction_executed"] = sum(
            bool(result["execution"].get("interaction_executed"))
            for result in results
            if result["execution"].get("executed")
        )

        return coverage
