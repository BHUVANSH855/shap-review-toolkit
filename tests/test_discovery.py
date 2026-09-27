from pathlib import Path

from shap_review.discovery import BuildScanner, RepositoryScanner


def test_repository_scanner(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[build-system]\nbuild-backend="x"')
    (tmp_path / "a.py").write_text("x=1")
    info = RepositoryScanner().scan(tmp_path)
    assert info.python_files == 1 and info.has_pyproject


def test_build_scanner(tmp_path: Path):
    (tmp_path / "CMakeLists.txt").write_text(
        "project(x LANGUAGES CXX)\nnanobind_add_module(_cutils)\npython_add_library(_cext MODULE x.cc)"
    )
    d = BuildScanner().scan(tmp_path)
    assert "_cutils" in d["targets"] and "_cext" in d["targets"]
