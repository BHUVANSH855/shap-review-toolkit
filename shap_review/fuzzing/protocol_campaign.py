from __future__ import annotations

import random
import traceback

from .generators.protocol import PROTOCOLS, ProtocolObject, generate_protocol_case

REAL_SCENARIOS = ("shape-dtype", "array-coercion", "indexing")


class ProtocolCampaign:
    """Protocol campaign separating harness, target, SHAP-stack and scenario evidence."""

    def __init__(self, seed=0):
        self.seed = seed
        self.rng = random.Random(seed)

    def run(self, target_callable=None, iterations=20, scenarios=None):
        scenarios = tuple(scenarios or REAL_SCENARIOS)
        results = []
        for i in range(iterations):
            case = generate_protocol_case(self.rng)
            case["protocol"] = PROTOCOLS[i % len(PROTOCOLS)]
            case["scenario"] = scenarios[i % len(scenarios)]
            execution_id = f"protocol-{self.seed}-{i}"
            reentry_target = (
                None
                if target_callable is None
                else getattr(target_callable, "__shap_reentry__", None)
            )
            obj = ProtocolObject(
                self.rng,
                case["protocol"],
                mutation=case["mutation"],
                reentry=case["reentry"],
                exception=case["exception"],
                reentry_target=reentry_target,
                scenario=case["scenario"],
                execution_id=execution_id,
                allow_harness_reentry=target_callable is None,
            )
            executed = False
            raised = None
            try:
                if target_callable is not None:
                    obj.set_phase("target")
                    target_callable(obj, case)
                else:
                    obj.set_phase("harness")
                    if case.get("reentry"):
                        obj.reentry_target = lambda o: o.events.append(
                            "reentry-callback:harness"
                        )
                    self._trigger(obj, case["protocol"])
                executed = True
            except Exception as exc:  # noqa: BLE001 - campaign records arbitrary target failures
                raised = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }

            protocol_events = [e for e in obj.events if e.split(":", 1)[0] in PROTOCOLS]
            results.append(
                {
                    "iteration": i,
                    "case": case,
                    "executed": executed,
                    "exception": raised,
                    "protocol_calls": obj.calls,
                    "reentered": obj.reentered,
                    "reentry_depth": obj.reentry_depth,
                    "mutated": obj.mutated,
                    "mutation_events": list(obj.mutation_events),
                    "protocol_events": protocol_events,
                    "protocol_events_detail": [
                        e for e in getattr(obj, "_protocol_event_records", [])
                    ],
                    "all_events": list(obj.events),
                    "protocol_state": dict(obj.state),
                    "target_boundary": getattr(
                        target_callable, "__shap_boundary__", None
                    ),
                    "execution_id": execution_id,
                    "parent_execution_id": obj.state.get("outer_execution_id"),
                    "child_execution_id": obj.state.get("child_execution_id"),
                }
            )

        harness_observed = {
            e.split(":", 1)[0]
            for r in results
            for e in r["protocol_events"]
            if e.split(":", 1)[1] == "harness" and e.split(":", 1)[0] in PROTOCOLS
        }
        shap_observed = {
            e.split(":")[1]
            for r in results
            for e in r["all_events"]
            if e.startswith("provenance:")
            and e.endswith(":shap")
            and e.split(":")[1] in PROTOCOLS
        }
        target_observed = {
            e.split(":", 1)[0]
            for r in results
            for e in r["protocol_events"]
            if e.split(":", 1)[1] in {"target", "scenario", "shap"}
            and e.split(":", 1)[0] in PROTOCOLS
        }
        compatibility_observed = (
            harness_observed
            if not target_callable
            else (target_observed | shap_observed)
        )

        natural_reentry = sum(
            bool(
                r["reentered"]
                and any(
                    e.startswith("reentry:") and e.endswith(":shap")
                    for e in r["all_events"]
                )
            )
            for r in results
        )
        requested_mutations = sum(r["case"]["mutation"] != "none" for r in results)
        observed_mutations = sum(
            1 for r in results for m in r["mutation_events"] if m.get("changed")
        )

        scenario_stats = {}
        for scenario in scenarios:
            subset = [r for r in results if r["case"]["scenario"] == scenario]
            obs = {
                e.split(":", 1)[0]
                for r in subset
                for e in r["protocol_events"]
                if e.split(":", 1)[1] in {"target", "scenario", "shap"}
                and e.split(":", 1)[0] in PROTOCOLS
            }
            scenario_stats[scenario] = {
                "iterations": len(subset),
                "protocols_observed": sorted(obs),
                "coverage_percent": round(
                    100 * len(obs & set(PROTOCOLS)) / len(PROTOCOLS), 1
                ),
            }

        for r in results:
            r["scenario_boundary"] = (
                (getattr(target_callable, "__shap_scenario__", {}) or {}).get(
                    r["case"]["scenario"]
                )
                if target_callable
                else r["case"]["scenario"]
            )

        return {
            "mode": "real-target" if target_callable else "harness-self-test",
            "seed": self.seed,
            "iterations": iterations,
            "scenarios": list(scenarios),
            "executed": sum(bool(r["executed"] or r["exception"]) for r in results),
            "protocols_declared": list(PROTOCOLS),
            "harness_triggered_protocols": sorted(harness_observed),
            "target_observed_protocols": sorted(target_observed),
            "shap_observed_protocols": sorted(shap_observed),
            "protocols_observed": sorted(compatibility_observed),
            "protocol_coverage": {p: p in compatibility_observed for p in PROTOCOLS},
            "coverage_percent": round(
                100 * len(compatibility_observed & set(PROTOCOLS)) / len(PROTOCOLS),
                1,
            ),
            "harness_coverage_percent": round(
                100 * len(harness_observed & set(PROTOCOLS)) / len(PROTOCOLS),
                1,
            ),
            "target_observed_coverage_percent": round(
                100 * len(target_observed & set(PROTOCOLS)) / len(PROTOCOLS),
                1,
            ),
            "shap_observed_coverage_percent": round(
                100 * len(shap_observed & set(PROTOCOLS)) / len(PROTOCOLS),
                1,
            ),
            "reentry_requested": sum(bool(r["case"]["reentry"]) for r in results),
            "reentry_observed": sum(r["reentered"] for r in results),
            "harness_reentry_observed": sum(
                bool(
                    r["reentered"]
                    and any(
                        e.startswith("reentry:") and e.endswith(":harness")
                        for e in r["all_events"]
                    )
                )
                for r in results
            ),
            "natural_shap_reentry_observed": natural_reentry,
            "exact_shap_call_provenance": any(
                ev.get("shap_call_id") and ev.get("causal_to_active_shap_call")
                for r in results
                for ev in r.get("protocol_events_detail", [])
            ),
            "mutation_requested": requested_mutations,
            "mutation_observed": observed_mutations,
            "mutation_events": sum(len(r["mutation_events"]) for r in results),
            "scenario_stats": scenario_stats,
            "results": results,
            "evidence_policy": (
                "shap_observed requires a runtime stack containing SHAP while "
                "the protocol callback executes; harness phase alone cannot "
                "establish SHAP causality. Natural re-entry requires the "
                "protocol callback itself to invoke the re-entry target while "
                "a SHAP frame is active."
            ),
        }

    @staticmethod
    def _trigger(obj, protocol):
        return {
            "shape": lambda: obj.shape,
            "dtype": lambda: obj.dtype,
            "strides": lambda: obj.strides,
            "getitem": lambda: obj[0],
            "array": lambda: obj.__array__(),
            "len": lambda: len(obj),
            "iter": lambda: list(iter(obj)),
            "bool": lambda: bool(obj),
            "repr": lambda: repr(obj),
            "call": lambda: obj(),
        }[protocol]()


def make_treeexplainer_protocol_target(model, background=None):
    import numpy as np
    import shap

    explainer = (
        shap.TreeExplainer(model, data=background)
        if background is not None
        else shap.TreeExplainer(model)
    )
    execution_counter = {"n": 0}

    def next_execution(parent=None, depth=0):
        execution_counter["n"] += 1
        return {
            "execution_id": f"treeexplainer-{execution_counter['n']}",
            "parent_execution_id": parent,
            "reentry_depth": depth,
        }

    def invoke(obj, fn, call_id):
        obj.begin_shap_call(
            call_id,
            function="TreeExplainer.shap_values",
            source="shap_review.fuzzing.protocol_campaign",
        )
        try:
            return fn()
        finally:
            obj.end_shap_call()

    def target(obj, case):
        scenario = case.get("scenario", "shape-dtype")
        ctx = next_execution()
        obj.state["outer_execution_id"] = ctx["execution_id"]
        obj.events.append(f"execution:{ctx['execution_id']}")

        # No harness-side protocol touch here. The only path to natural re-entry is
        # a protocol callback invoked from inside SHAP itself.
        obj.set_phase("shap")
        if scenario == "array-coercion":
            obj.set_phase("scenario")
            obj.events.append("scenario-boundary:array-coercion")
            arr = np.asarray(obj)
            obj.events.append("boundary:np.asarray")
            obj.set_phase("shap")
            return invoke(
                obj,
                lambda: explainer.shap_values(arr),
                ctx["execution_id"],
            )
        if scenario == "indexing":
            obj.set_phase("scenario")
            obj.events.append("scenario-boundary:indexing")
            _ = obj[0]
            obj.events.append("boundary:__getitem__")
            obj.set_phase("shap")
            return invoke(
                obj,
                lambda: explainer.shap_values(obj),
                ctx["execution_id"],
            )
        return invoke(
            obj,
            lambda: explainer.shap_values(obj),
            ctx["execution_id"],
        )

    def reenter(obj):
        depth = int(obj.state.get("reentry_depth", 0))
        if depth >= 1:
            return None
        parent = obj.state.get("outer_execution_id")
        child = next_execution(parent=parent, depth=depth + 1)
        obj.state["child_execution_id"] = child["execution_id"]
        obj.state["reentry_depth"] = depth + 1
        obj.events.append(f"reentry-child:{child['execution_id']}")
        try:
            obj.set_phase("shap")
            return invoke(
                obj,
                lambda: explainer.shap_values(
                    np.array([[0.5, 0.5]], dtype=float),
                    check_additivity=False,
                ),
                child["execution_id"],
            )
        finally:
            obj.state["reentry_depth"] = depth

    target.__shap_boundary__ = "TreeExplainer.shap_values"
    target.__shap_reentry__ = reenter
    target.__shap_scenario__ = {
        "shape-dtype": (
            "TreeExplainer.shap_values directly consumes ProtocolObject; "
            "shape/dtype protocol boundary"
        ),
        "array-coercion": "np.asarray(obj) occurs before TreeExplainer",
        "indexing": "obj[0] occurs before TreeExplainer",
    }
    return target


def make_default_treeexplainer_protocol_target():
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor

    model = DecisionTreeRegressor(max_depth=2, random_state=0).fit(
        np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        np.array([0.0, 1.0, 2.0, 3.0]),
    )
    return make_treeexplainer_protocol_target(model)
