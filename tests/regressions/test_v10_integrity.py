import numpy as np
import pytest

from shap_review.contracts import SHAPContract, validate_contract
from shap_review.differential.semantic import compare_shap_contract
from shap_review.evidence.chain import build_chain
from shap_review.findings.lifecycle import evidence_transition
from shap_review.fuzzing.generators.protocol import PROTOCOLS
from shap_review.fuzzing.protocol_campaign import ProtocolCampaign
from shap_review.native import map_shap_native
from shap_review.semantic.api_era import scan_api_era
from shap_review.types import FindingStatus


def test_protocol_every_declared_protocol_is_triggerable():
    campaign = ProtocolCampaign(1)
    result = campaign.run(iterations=len(PROTOCOLS) * 3)
    assert result["coverage_percent"] == 100.0
    assert all(result["protocol_coverage"].values())


def test_evidence_graph_rejects_shared_ancestry_as_independent():
    chain = build_chain(
        entries=[
            {
                "kind": "historical",
                "source": "issue",
                "claim": "known",
                "passed": True,
                "evidence_id": "root",
            },
            {
                "kind": "static",
                "source": "rule-a",
                "claim": "hit",
                "passed": True,
                "evidence_id": "a",
                "derived_from": ["root"],
            },
            {
                "kind": "dynamic",
                "source": "rule-b",
                "claim": "hit",
                "passed": True,
                "evidence_id": "b",
                "derived_from": ["root"],
            },
        ]
    )
    assert not chain.graph.is_independent("a", "b")


def test_semantic_contract_executes_additivity_oracle():
    contract = SHAPContract(
        "TreeExplainer",
        "tree",
        values_shape=(2, 3),
        base_values_shape=(2,),
        target_shape=(2,),
        tolerance=1e-6,
    )
    result = validate_contract(contract, np.ones((2, 3)), np.zeros(2), np.full(2, 3.0))
    assert result["valid"]
    assert result["semantic_oracle"]["passed"]


def test_differential_marks_metadata_separately():
    a = {"values": [[1.0]], "feature_names": ["a"]}
    b = {"values": [[1.0]], "feature_names": ["b"]}
    result = compare_shap_contract(a, b)
    assert result["semantic_equal"]
    assert result["fields"]["feature_names"]["severity"] == "metadata"


def test_promotion_requires_runtime_evidence():
    chain = build_chain(
        entries=[
            {
                "kind": "static",
                "source": "s",
                "claim": "candidate",
                "passed": True,
                "evidence_id": "s",
            }
        ]
    )
    with pytest.raises(ValueError):
        evidence_transition(
            FindingStatus.REPRO_PENDING, FindingStatus.REPRODUCED, chain
        )


def test_api_era_scanner():
    findings = scan_api_era("explainer.shap_values(X)\nshap.Explainer(model)(X)")
    assert any(x["era"] == "legacy" for x in findings)
    assert any(x["api"] == "Explainer.__call__" for x in findings)


def test_native_map_has_expected_layers(tmp_path):
    (tmp_path / "shap/cutils").mkdir(parents=True)
    (tmp_path / "shap/cext").mkdir(parents=True)
    result = map_shap_native(str(tmp_path))
    names = {x["name"] for x in result["components"] if x["present"]}
    assert {"cutils", "cext"} <= names


def test_backend_matrix_has_primary_tree_backends():
    from shap_review.fuzzing.backend_matrix import DEFAULT_BACKENDS, matrix_dimensions

    names = {b.name for b in DEFAULT_BACKENDS}
    assert {"sklearn", "xgboost", "lightgbm", "catboost"} <= names
    assert "backend" in matrix_dimensions()


def test_cpu_gpu_script_differential_reports_semantic_fields(tmp_path):
    from shap_review.differential.gpu_runner import cpu_gpu_scripts

    cpu = tmp_path / "cpu.py"
    gpu = tmp_path / "gpu.py"
    cpu.write_text(
        "import json; print(json.dumps({'values': [[1.0]], 'base_values': [0.0]}))\n"
    )
    gpu.write_text(
        "import json; print(json.dumps({'values': [[1.0]], 'base_values': [0.0], 'feature_names':['x']}))\n"
    )
    result = cpu_gpu_scripts(cpu, gpu)
    assert result["applicable"]
    assert result["semantic_equal"]
