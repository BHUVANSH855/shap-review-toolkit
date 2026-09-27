from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .comparator import compare
from .semantic import compare_shap_contract, normalize_shap_result


def run_json_script(
    script: str | Path,
    timeout: int = 60,
    python: str | None = None,
    env: dict | None = None,
    capture_environment: bool = False,
    environment_depth: str = "basic",
) -> dict:
    executable = python or sys.executable
    import os

    proc_env = os.environ.copy()
    proc_env.update(env or {})
    try:
        if capture_environment:
            wrapper = r"""import contextlib,io,json,runpy,sys,platform,os,importlib.metadata
p=sys.argv[1]; buf=io.StringIO(); rc=0; err=None
try:
 with contextlib.redirect_stdout(buf): runpy.run_path(p,run_name='__main__')
except SystemExit as exc: rc=int(exc.code) if isinstance(exc.code,int) else 0
except Exception as exc: rc=1; err=f'{type(exc).__name__}: {exc}'
out=buf.getvalue(); value=None; parse_error=None
try: value=json.loads(out)
except Exception as exc: parse_error=str(exc)
envinfo={'python':sys.version,'platform':platform.platform(),'executable':sys.executable,'machine':platform.machine(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'environment_depth':sys.argv[2] if len(sys.argv)>2 else 'basic'}
if envinfo['environment_depth']=='deep':
 try:
  import numpy as _np; envinfo['blas_configuration']=_np.__config__.get_info('blas_opt_info') if hasattr(_np.__config__,'get_info') else None
 except Exception: envinfo['blas_configuration']=None
 try: envinfo['openmp_runtime']=os.environ.get('OMP_NUM_THREADS') or os.environ.get('OMP_RUNTIME')
 except Exception: envinfo['openmp_runtime']=None
 envinfo['cpu_model']=platform.processor(); envinfo['cuda_runtime']=os.environ.get('CUDA_PATH'); envinfo['gpu_model']=os.environ.get('NVIDIA_VISIBLE_DEVICES')
 envinfo['environment_variables']={k:v for k,v in os.environ.items() if k in {'CUDA_VISIBLE_DEVICES','CUDA_PATH','OMP_NUM_THREADS','OMP_RUNTIME','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','LD_LIBRARY_PATH'}}
for m in ('shap','numpy','scipy','pandas','scikit-learn','xgboost','lightgbm','catboost'):
 try: envinfo[m+'_version']=importlib.metadata.version(m)
 except importlib.metadata.PackageNotFoundError: envinfo[m+'_version']=None
print(json.dumps({'value':value,'stdout':out[-20000:],'parse_error':parse_error,'returncode':rc,'error':err,'environment':envinfo}))
"""
            p = subprocess.run(
                [executable, "-c", wrapper, str(script), environment_depth],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=proc_env,
            )
            payload = json.loads(p.stdout) if p.stdout else {}
            return {
                "ok": p.returncode == 0
                and payload.get("returncode", 1) == 0
                and payload.get("parse_error") is None,
                "timeout": False,
                "returncode": payload.get("returncode", p.returncode),
                "stdout": payload.get("stdout", "")[-20000:],
                "stderr": p.stderr[-20000:],
                "value": payload.get("value"),
                "subprocess_environment": payload.get("environment"),
                "parse_error": payload.get("parse_error"),
                "execution_error": payload.get("error"),
            }
        p = subprocess.run(
            [executable, str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=proc_env,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "timeout": True,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "execution_status": "NOT_EXECUTED",
            "execution_reason": "TIMEOUT",
            "semantic_status": "NOT_EVALUATED",
        }
    r = {
        "ok": p.returncode == 0,
        "timeout": False,
        "returncode": p.returncode,
        "stdout": p.stdout[-20000:],
        "stderr": p.stderr[-20000:],
    }
    if p.returncode == 0:
        try:
            r["value"] = json.loads(p.stdout)
        except json.JSONDecodeError as exc:
            r.update({"ok": False, "parse_error": str(exc)})
    return r


def differential_scripts(
    reference: str | Path,
    candidate: str | Path,
    timeout: int = 60,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    semantic: bool = True,
) -> dict:
    left = run_json_script(reference, timeout, capture_environment=True)
    right = run_json_script(candidate, timeout, capture_environment=True)
    out = {"reference": left, "candidate": right, "equal": None}
    if left.get("timeout") or right.get("timeout"):
        out.update(
            {
                "equal": False,
                "status": "EXECUTION_FAILED",
                "comparison_status": "EXECUTION_FAILED",
                "comparison_reason": "TIMEOUT",
                "semantic_status": "NOT_EVALUATED",
                "differential_agreement": None,
                "reference_correctness": "UNKNOWN",
            }
        )
        return out
    if not left.get("ok") or not right.get("ok"):
        out.update(
            {
                "equal": False,
                "status": "EXECUTION_FAILED",
                "comparison_status": "EXECUTION_FAILED",
                "comparison_reason": "EXECUTION_FAILED",
                "semantic_status": "NOT_EVALUATED",
                "differential_agreement": None,
                "reference_correctness": "UNKNOWN",
            }
        )
        return out
    lv = normalize_shap_result(left["value"]) if semantic else left["value"]
    rv = normalize_shap_result(right["value"]) if semantic else right["value"]
    out["comparison"] = compare(lv, rv, rtol=rtol, atol=atol)
    out["contract_comparison"] = compare_shap_contract(
        left["value"], right["value"], rtol=rtol, atol=atol
    )
    agree = bool(out["comparison"]["equal"] and out["contract_comparison"]["equal"])
    out.update(
        {
            "equal": agree,
            "differential_agreement": agree,
            "reference_correctness": "UNKNOWN",
            "comparison_status": "MATCH" if agree else "MISMATCH",
            "status": "MATCH" if agree else "MISMATCH",
            "semantic_status": "NOT_EVALUATED" if agree else "INCONCLUSIVE",
            "semantic_disagreement": out["comparison"]["equal"]
            != out["contract_comparison"]["equal"],
            "comparison_reason": "differential agreement does not establish correctness",
            "semantic_note": "MATCH means reference and candidate agree; it is not a correctness proof",
        }
    )
    return out
