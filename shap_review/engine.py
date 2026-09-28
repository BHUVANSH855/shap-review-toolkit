from __future__ import annotations

import random

from .fuzzing.generators.input import generate_input_case
from .fuzzing.generators.tree import generate_tree_case
from .fuzzing.harnesses.treeexplainer import run_case
from .fuzzing.minimizers.basic import minimize_case
from .fuzzing.oracles.tree import evaluate_execution


class TreeExplainerFuzzer:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

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

    def run(self, iterations=10):
        results = []
        for i in range(iterations):
            case = generate_tree_case(self.rng)
            generate_input_case(
                self.rng
            )  # consume RNG, but don't create dead dimensions
            case["input"] = {
                "samples": case["n_samples"],
                "features": case["n_features"],
                "dtype": case["dtype"],
                "representation": case["representation"],
                "nan": case["nan"],
            }
            case["seed"] = self.rng.randrange(2**31)
            result = run_case(case)
            oracle = evaluate_execution(result)
            item = {
                "iteration": i,
                "case": case,
                "execution_trace": self._execution_trace(case),
                "execution": result,
                "oracle": oracle,
            }
            if not oracle.get("valid") and result.get("executed"):
                item["minimized_case"] = minimize_case(case)
            results.append(item)
        executed = [r for r in results if r["execution"].get("executed")]
        return {
            "iterations": iterations,
            "results": results,
            "executed_cases": len(executed),
            "validated_cases": sum(r["oracle"].get("valid", False) for r in results),
            "failures": sum(not r["oracle"].get("valid", False) for r in results),
            "skipped_cases": sum(bool(r["execution"].get("skipped")) for r in results),
            "coverage": self.coverage(results),
        }

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
            values = sorted({str(r["execution_trace"][field]) for r in results})
            coverage[field] = {"observed": values, "count": len(values)}
        coverage["interaction_executed"] = sum(
            bool(r["execution"].get("interaction_executed"))
            for r in results
            if r["execution"].get("executed")
        )
        return coverage
