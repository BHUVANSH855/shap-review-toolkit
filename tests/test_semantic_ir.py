from pathlib import Path

from shap_review.semantic_ir import SemanticIRBuilder

FIXTURES = Path(__file__).parent / "fixtures"


def test_ir_extracts_treeexplainer_and_input_flow():
    ir = SemanticIRBuilder().build(FIXTURES / "shap_4911_nullable_dtype")

    assert any(call.callee.endswith("TreeExplainer") for call in ir.calls)
    assert any(flow.source == "pandas-like input" for flow in ir.flows)


def test_ir_extracts_model_output():
    ir = SemanticIRBuilder().build(FIXTURES / "shap_5098_model_output")

    assert any("model_output" in " ".join(call.arguments) for call in ir.calls)
