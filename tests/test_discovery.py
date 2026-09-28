from pathlib import Path

from shap_review.discovery import BuildScanner, RepositoryScanner


def test_repository_scanner(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nbuild-backend="x"'
    )
    (tmp_path / "a.py").write_text("x=1")

    info = RepositoryScanner().scan(tmp_path)

    assert info.python_files == 1
    assert info.has_pyproject


def test_build_scanner(tmp_path: Path):
    (tmp_path / "CMakeLists.txt").write_text(
        "project(x LANGUAGES CXX)\n"
        "nanobind_add_module(_cutils)\n"
        "python_add_library(_cext MODULE x.cc)"
    )

    discovery = BuildScanner().scan(tmp_path)

    assert "_cutils" in discovery["targets"]
    assert "_cext" in discovery["targets"]

def test_history_reports_changed_symbols(tmp_path: Path):
    import subprocess

    from shap_review.discovery.history import HistoryScanner

    subprocess.run(
        ["git", "init", str(tmp_path)],
        check=True,
        capture_output=True,
    )

    (tmp_path / "x.py").write_text(
        "def old_name():\n    return 1\n"
    )

    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "."],
        check=True,
    )

    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=T",
            "-c",
            "user.email=t@e",
            "commit",
            "-m",
            "initial",
        ],
        check=True,
        capture_output=True,
    )

    (tmp_path / "x.py").write_text(
        "def new_name():\n    return 2\n"
    )

    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "."],
        check=True,
    )

    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=T",
            "-c",
            "user.email=t@e",
            "commit",
            "-m",
            "fix #4911",
        ],
        check=True,
        capture_output=True,
    )

    result = HistoryScanner().analyze_issue(tmp_path, 4911)

    assert "new_name" in result["semantic_summary"]["symbols_changed"]
    assert "old_name" in result["semantic_summary"]["symbols_changed"]