from pathlib import Path

from shap_review.semantic_ir import SemanticIRBuilder

FIX = Path(__file__).parent / "fixtures"


def test_ir_extracts_treeexplainer_and_input_flow():
    ir = SemanticIRBuilder().build(FIX / "shap_4911_nullable_dtype")
    assert any(c.callee.endswith("TreeExplainer") for c in ir.calls)
    assert any(f.source == "pandas-like input" for f in ir.flows)


def test_ir_extracts_model_output():
    ir = SemanticIRBuilder().build(FIX / "shap_5098_model_output")
    assert any("model_output" in " ".join(c.arguments) for c in ir.calls)
