from pathlib import Path


def test_release_metadata_is_v29():
    root = Path(__file__).resolve().parents[2]
    assert 'VERSION = "0.29.0"' in (root / "shap_review/version.py").read_text()
    assert 'version = "0.29.0"' in (root / "pyproject.toml").read_text()
