from pathlib import Path

from shap_review.analyzers import SHAPSemanticAnalyzer


def test_shap_semantic_analyzer(tmp_path: Path):
    source = tmp_path / "_tree.py"

    source.write_text(
        "import shap\n"
        "def shap_values(self):\n"
        "    self.expected_value = 1\n"
        "    return check_additivity\n"
    )

    findings = SHAPSemanticAnalyzer().analyze(tmp_path)

    assert findings
    assert any(finding.invariant == "INV-ATTR-001" for finding in findings)
