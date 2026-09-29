import subprocess
from pathlib import Path

from shap_review.discovery.history import HistoryScanner


def test_history_scanner_correlates_issue_and_paths(tmp_path: Path):
    subprocess.run(
        ["git", "init", str(tmp_path)],
        check=True,
        capture_output=True,
    )

    (tmp_path / "x.py").write_text("print('x')")

    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "x.py"],
        check=True,
    )

    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "fix: resolve SHAP issue #4911",
        ],
        check=True,
        capture_output=True,
    )

    result = HistoryScanner().scan(
        tmp_path,
        issue_numbers=[4911],
    )

    assert result["git_repository"]
    assert result["issue_correlations"]
    assert "x.py" in result["changed_paths"]


def test_history_issue_before_after_snapshot(tmp_path: Path):
    subprocess.run(
        ["git", "init", str(tmp_path)],
        check=True,
        capture_output=True,
    )

    (tmp_path / "x.py").write_text("value = 1\n")

    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "x.py"],
        check=True,
    )

    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "initial",
        ],
        check=True,
        capture_output=True,
    )

    (tmp_path / "x.py").write_text("value = 2\n")

    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "x.py"],
        check=True,
    )

    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "fix: resolve #4911",
        ],
        check=True,
        capture_output=True,
    )

    result = HistoryScanner().analyze_issue(
        tmp_path,
        4911,
    )

    assert result["file_diffs"]
    assert result["file_diffs"][0]["changed"]
    assert result["file_diffs"][0]["before_lines"] == 1

def test_finding_classification():
    from shap_review.evidence.provenance import classify_finding

    assert (
        classify_finding(
            historical_issue="4911",
            reproduced=True,
            discovered_by_current_analysis=False,
        ).value
        == "historical_reproduction"
    )

    assert classify_finding(reproduced=True).value == "novel_reproduction"