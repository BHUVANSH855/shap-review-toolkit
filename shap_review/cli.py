from __future__ import annotations

import argparse
import json
from pathlib import Path

from shap_review.differential import differential_scripts
from shap_review.differential.gpu_runner import cpu_gpu_scripts
from shap_review.differential.versions import differential_versions
from shap_review.discovery.history import HistoryScanner
from shap_review.engine import ReviewEngine
from shap_review.explore import Explorer
from shap_review.fuzzing.backend_matrix import (
    discover_backends,
    execute_installed_backend_matrix,
    matrix_dimensions,
)
from shap_review.fuzzing.engine import TreeExplainerFuzzer
from shap_review.fuzzing.gpu_differential import CPUGPUDifferential
from shap_review.fuzzing.protocol_campaign import ProtocolCampaign
from shap_review.health import HealthReport
from shap_review.hotspots import HotspotEngine
from shap_review.inspect import candidate_plan
from shap_review.native import map_shap_native
from shap_review.reports.markdown import render_candidates
from shap_review.reproduction.runner import run_script
from shap_review.reproduction.sanitizer import run_sanitized
from shap_review.semantic.api_era import scan_api_era
from shap_review.version import SCHEMA_VERSION


def dispatch(
    command: str,
    root: str = ".",
    arguments: dict | None = None,
    engine: ReviewEngine | None = None,
) -> dict:
    """Structured command dispatcher shared by Claude, OpenAI, Gemini, and the CLI."""
    arguments = arguments or {}
    eng = engine or ReviewEngine()
    if command == "capabilities":
        from shap_review.version import release_metadata

        return release_metadata("generic")
    if command == "version":
        from shap_review.version import VERSION

        return {
            "version": VERSION,
            "toolkit_version": VERSION,
            "schema_version": SCHEMA_VERSION,
        }
    if command == "evidence":
        return {
            "model": "EvidenceChain",
            "supported": True,
            "schema_version": SCHEMA_VERSION,
            "independence_tracking": True,
            "provenance_tracking": True,
        }
    if command == "semantic-oracle":
        import numpy as np

        from shap_review.contracts.shap_contract import SHAPContract, validate_contract

        args = arguments
        contract_data = dict(args.get("contract", {}))
        axis = contract_data.get("axis_spec")
        if isinstance(axis, dict):
            from shap_review.contracts.tensor import SHAPAxisSpec

            contract_data["axis_spec"] = SHAPAxisSpec(**axis)
        contract = SHAPContract(**contract_data)
        values = np.asarray(args["values"], dtype=float)
        base = (
            None
            if args.get("base_values") is None
            else np.asarray(args["base_values"], dtype=float)
        )
        target = (
            None
            if args.get("target") is None
            else np.asarray(args["target"], dtype=float)
        )
        expected = (
            None
            if args.get("expected_value") is None
            else np.asarray(args["expected_value"], dtype=float)
        )
        interactions = (
            None
            if args.get("interaction_values") is None
            else np.asarray(args["interaction_values"], dtype=float)
        )
        return validate_contract(
            contract,
            values,
            base,
            target,
            expected_value=expected,
            interaction_values=interactions,
        )
    if command == "map":
        return eng.discover(root, arguments.get("out"))
    if command == "health":
        return HealthReport().run(root)
    if command == "analyze":
        return {
            "candidates": [c.__dict__ for c in eng.analyze(root, arguments.get("out"))]
        }
    if command == "report":
        cs = eng.analyze(root)
        return {
            "format": arguments.get("format", "markdown"),
            "report": render_candidates(cs),
        }
    if command == "hotspots":
        return {"items": HotspotEngine().run(root)[: arguments.get("limit", 50)]}
    if command == "explore":
        return {"items": Explorer().run(root, arguments.get("query", ""))}
    if command == "investigate":
        return candidate_plan(arguments.get("candidate", {}))
    if command == "reproduce":
        return run_script(
            arguments["script"],
            python=arguments.get("python"),
            env=arguments.get("env"),
            cwd=arguments.get("cwd"),
        )
    if command == "fuzz-treeexplainer":
        return TreeExplainerFuzzer(arguments.get("seed", 0)).run(
            arguments.get("iterations", 10)
        )
    if command == "fuzz":
        target = arguments.get("target", "treeexplainer")
        if target == "protocol":
            return ProtocolCampaign(arguments.get("seed", 0)).run(
                iterations=arguments.get("iterations", 20)
            )
        if target == "backends":
            return {
                "backends": [b.__dict__ for b in discover_backends()],
                "matrix": matrix_dimensions(),
                "mode": "representative",
            }
        return TreeExplainerFuzzer(arguments.get("seed", 0)).run(
            arguments.get("iterations", 10)
        )
    if command == "fuzz-protocol":
        target = None
        if arguments.get("real_target"):
            from shap_review.fuzzing.protocol_campaign import (
                make_default_treeexplainer_protocol_target,
            )

            target = make_default_treeexplainer_protocol_target()
        return ProtocolCampaign(arguments.get("seed", 0)).run(
            target_callable=target, iterations=arguments.get("iterations", 20)
        )
    if command == "fuzz-backends":
        out = {
            "backends": [b.__dict__ for b in discover_backends()],
            "matrix": matrix_dimensions(),
        }
        if arguments.get("execute"):
            out["execution"] = execute_installed_backend_matrix(
                exhaustive=arguments.get("exhaustive", False)
            )
        return out
    if command == "cpu-gpu-differential":
        if arguments.get("reference") and arguments.get("candidate"):
            return cpu_gpu_scripts(
                arguments["reference"],
                arguments["candidate"],
                arguments.get("timeout", 120),
                arguments.get("rtol", 1e-5),
                arguments.get("atol", 1e-8),
            )
        return {
            "available": CPUGPUDifferential().availability(),
            "message": "Use the reference/candidate script arguments or the Python CPUGPUDifferential API for execution.",
        }
    if command == "native-map":
        return map_shap_native(root)
    if command == "api-era":
        text = Path(arguments.get("file", root)).read_text(
            encoding="utf-8", errors="replace"
        )
        return {"findings": scan_api_era(text)}
    if command == "regressions":
        return eng.regressions(arguments.get("out"))
    if command == "history":
        if arguments.get("issue") is not None:
            return HistoryScanner().analyze_issue(
                root, arguments["issue"], arguments.get("limit", 1000)
            )
        return HistoryScanner().scan(
            root, arguments.get("issue_numbers"), arguments.get("limit", 500)
        )
    if command == "sanitizer":
        return run_sanitized(
            arguments["script"],
            arguments.get("kind", "asan"),
            arguments.get("python"),
            arguments.get("timeout", 120),
            arguments.get("cwd"),
            arguments.get("env"),
        )
    if command == "differential":
        return differential_scripts(
            arguments["reference"],
            arguments["candidate"],
            arguments.get("timeout", 60),
            arguments.get("rtol", 1e-5),
            arguments.get("atol", 1e-8),
        )
    if command == "differential-versions":
        return differential_versions(
            arguments["reference"],
            arguments["candidate"],
            arguments["python_reference"],
            arguments["python_candidate"],
            arguments.get("timeout", 120),
            arguments.get("rtol", 1e-5),
            arguments.get("atol", 1e-8),
        )
    raise ValueError(f"unsupported command: {command}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="shap-review")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["map", "health", "analyze"]:
        sp = sub.add_parser(name)
        sp.add_argument("root")
    for name in ["capabilities", "version", "evidence"]:
        sub.add_parser(name)
    sp = sub.add_parser("semantic-oracle")
    sp.add_argument("input_json")
    sp = sub.add_parser("report")
    sp.add_argument("root")
    sp.add_argument("--format", choices=["markdown", "json"], default="markdown")
    sp = sub.add_parser("hotspots")
    sp.add_argument("root")
    sp = sub.add_parser("explore")
    sp.add_argument("root")
    sp.add_argument("query")
    sp = sub.add_parser("investigate")
    sp.add_argument("candidate_json")
    sp = sub.add_parser("reproduce")
    sp.add_argument("script")
    sp = sub.add_parser("fuzz")
    sp.add_argument(
        "--target",
        choices=["treeexplainer", "protocol", "backends"],
        default="treeexplainer",
    )
    sp.add_argument("--iterations", type=int, default=10)
    sp.add_argument("--seed", type=int, default=0)
    sp = sub.add_parser("fuzz-treeexplainer")
    sp.add_argument("--iterations", type=int, default=10)
    sp.add_argument("--seed", type=int, default=0)
    sp = sub.add_parser("fuzz-protocol")
    sp.add_argument("--iterations", type=int, default=20)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--real-target", action="store_true")
    sp = sub.add_parser("fuzz-backends")
    sp.add_argument("--execute", action="store_true")
    sp.add_argument("--exhaustive", action="store_true")
    sp = sub.add_parser("cpu-gpu-differential")
    sp.add_argument("reference", nargs="?", default=None)
    sp.add_argument("candidate", nargs="?", default=None)
    sp.add_argument("--timeout", type=int, default=120)
    sp.add_argument("--rtol", type=float, default=1e-5)
    sp.add_argument("--atol", type=float, default=1e-8)
    sp = sub.add_parser("native-map")
    sp.add_argument("root")
    sp = sub.add_parser("api-era")
    sp.add_argument("file")
    sp = sub.add_parser("regressions")
    sp.add_argument("--out", default=None)
    sp = sub.add_parser("history")
    sp.add_argument("root")
    sp.add_argument("--issue", default=None)
    sp = sub.add_parser("differential")
    sp.add_argument("reference")
    sp.add_argument("candidate")
    sp.add_argument("--timeout", type=int, default=60)
    sp.add_argument("--rtol", type=float, default=1e-5)
    sp.add_argument("--atol", type=float, default=1e-8)
    sp.add_argument("--no-semantic", action="store_true")
    sp = sub.add_parser("differential-versions")
    sp.add_argument("reference")
    sp.add_argument("candidate")
    sp.add_argument("--python-reference", required=True)
    sp.add_argument("--python-candidate", required=True)
    sp.add_argument("--timeout", type=int, default=120)
    sp.add_argument("--rtol", type=float, default=1e-5)
    sp.add_argument("--atol", type=float, default=1e-8)
    sp = sub.add_parser("sanitizer")
    sp.add_argument("script")
    sp.add_argument("--kind", choices=["asan", "ubsan"], default="asan")
    sp.add_argument("--python", default=None)
    sp.add_argument("--timeout", type=int, default=120)
    args = p.parse_args(argv)
    eng = ReviewEngine()
    if args.cmd == "capabilities":
        print(json.dumps(dispatch("capabilities"), indent=2))
    elif args.cmd == "version":
        print(json.dumps(dispatch("version"), indent=2))
    elif args.cmd == "evidence":
        print(json.dumps(dispatch("evidence"), indent=2))
    elif args.cmd == "semantic-oracle":
        print(
            json.dumps(
                dispatch(
                    "semantic-oracle",
                    arguments=json.loads(
                        Path(args.input_json).read_text(encoding="utf-8")
                    ),
                ),
                indent=2,
                default=str,
            )
        )
    elif args.cmd == "map":
        print(json.dumps(eng.discover(args.root), indent=2))
    elif args.cmd == "health":
        print(json.dumps(HealthReport().run(args.root), indent=2))
    elif args.cmd == "analyze":
        cs = eng.analyze(args.root)
        print(
            json.dumps(
                {"candidates": len(cs), "items": [c.__dict__ for c in cs]},
                default=lambda x: x.__dict__,
                indent=2,
            )
        )
    elif args.cmd == "report":
        cs = eng.analyze(args.root)
        if args.format == "json":
            print(
                json.dumps(
                    [c.__dict__ for c in cs], default=lambda x: x.__dict__, indent=2
                )
            )
        else:
            print(render_candidates(cs))
    elif args.cmd == "hotspots":
        print(json.dumps(HotspotEngine().run(args.root)[:50], indent=2))
    elif args.cmd == "explore":
        print(json.dumps(Explorer().run(args.root, args.query), indent=2))
    elif args.cmd == "investigate":
        print(
            json.dumps(
                candidate_plan(
                    json.loads(Path(args.candidate_json).read_text(encoding="utf-8"))
                ),
                indent=2,
                default=str,
            )
        )
    elif args.cmd == "reproduce":
        print(json.dumps(run_script(args.script), indent=2))
    elif args.cmd == "fuzz":
        print(
            json.dumps(
                dispatch(
                    "fuzz",
                    arguments={
                        "target": args.target,
                        "iterations": args.iterations,
                        "seed": args.seed,
                    },
                ),
                indent=2,
                default=str,
            )
        )
    elif args.cmd == "fuzz-treeexplainer":
        print(
            json.dumps(
                dispatch(
                    "fuzz",
                    arguments={
                        "target": "treeexplainer",
                        "iterations": args.iterations,
                        "seed": args.seed,
                    },
                ),
                indent=2,
                default=str,
            )
        )
    elif args.cmd == "fuzz-protocol":
        print(
            json.dumps(
                dispatch(
                    "fuzz-protocol",
                    arguments={
                        "iterations": args.iterations,
                        "seed": args.seed,
                        "real_target": args.real_target,
                    },
                ),
                indent=2,
                default=str,
            )
        )
    elif args.cmd == "fuzz-backends":
        print(
            json.dumps(
                dispatch(
                    "fuzz-backends",
                    arguments={"execute": args.execute, "exhaustive": args.exhaustive},
                ),
                indent=2,
                default=str,
            )
        )
    elif args.cmd == "cpu-gpu-differential":
        print(
            json.dumps(
                dispatch(
                    "cpu-gpu-differential",
                    arguments={
                        "reference": args.reference,
                        "candidate": args.candidate,
                        "timeout": args.timeout,
                        "rtol": args.rtol,
                        "atol": args.atol,
                    },
                ),
                indent=2,
            )
        )
    elif args.cmd == "native-map":
        print(json.dumps(map_shap_native(args.root), indent=2))
    elif args.cmd == "api-era":
        print(
            json.dumps(
                {
                    "findings": scan_api_era(
                        Path(args.file).read_text(encoding="utf-8", errors="replace")
                    )
                },
                indent=2,
            )
        )
    elif args.cmd == "regressions":
        print(json.dumps(eng.regressions(args.out), indent=2))
    elif args.cmd == "history":
        print(
            json.dumps(
                HistoryScanner().analyze_issue(args.root, args.issue)
                if args.issue
                else HistoryScanner().scan(args.root),
                indent=2,
            )
        )
    elif args.cmd == "sanitizer":
        print(
            json.dumps(
                run_sanitized(args.script, args.kind, args.python, args.timeout),
                indent=2,
            )
        )
    elif args.cmd == "differential":
        print(
            json.dumps(
                differential_scripts(
                    args.reference,
                    args.candidate,
                    args.timeout,
                    args.rtol,
                    args.atol,
                    not args.no_semantic,
                ),
                indent=2,
            )
        )
    elif args.cmd == "differential-versions":
        print(
            json.dumps(
                differential_versions(
                    args.reference,
                    args.candidate,
                    args.python_reference,
                    args.python_candidate,
                    args.timeout,
                    args.rtol,
                    args.atol,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
