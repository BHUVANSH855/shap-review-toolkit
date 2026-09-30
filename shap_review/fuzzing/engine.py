from __future__ import annotations

import hashlib
import json
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
        """Run case with a per-iteration timeout using ThreadPoolExecutor."""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_case, case)
            try:
                return future.result(timeout=self.case_timeout)
            except concurrent.futures.TimeoutError:
                return {
                    "executed": True,
                    "failed": True,
                    "timeout": True,
                    "exception": "TimeoutError",
                    "message": f"case exceeded {self.case_timeout}s timeout",
                    "traceback": "",
                    "case": case,
                }

    @staticmethod
    def _failure_fingerprint(result: dict) -> str:
        """Coarse fingerprint for crash deduplication."""
        exc_type = result.get("exception", "")
        msg_prefix = str(result.get("message", ""))[:100]
        return hashlib.sha256(f"{exc_type}:{msg_prefix}".encode()).hexdigest()[:16]

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
