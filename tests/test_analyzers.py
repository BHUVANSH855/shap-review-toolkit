from pathlib import Path

from shap_review.analyzers import SHAPSemanticAnalyzer


def test_shap_semantic_analyzer(tmp_path: Path):
    p = tmp_path / "_tree.py"
    p.write_text(
        "def shap_values(self):\n    self.expected_value = 1\n    return check_additivity\n"
    )
    cs = SHAPSemanticAnalyzer().analyze(tmp_path)
    assert cs
    assert any(c.invariant == "INV-ATTR-001" for c in cs)
