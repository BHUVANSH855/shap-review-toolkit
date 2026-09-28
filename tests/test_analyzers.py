from pathlib import Path

from shap_review.analyzers import SHAPSemanticAnalyzer


def test_shap_semantic_analyzer(tmp_path: Path):
    p = tmp_path / "_tree.py"
    # Must include a shap import so the fallback path activates (P1-C fix):
    # without it the fallback correctly stays silent on non-SHAP code.
    p.write_text(
        "import shap\n"
        "def shap_values(self):\n"
        "    self.expected_value = 1\n"
        "    return check_additivity\n"
    )
    cs = SHAPSemanticAnalyzer().analyze(tmp_path)
    assert cs
    assert any(c.invariant == "INV-ATTR-001" for c in cs)
