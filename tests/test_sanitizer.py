from pathlib import Path

from shap_review.reproduction.sanitizer import run_sanitized


def test_sanitizer_runner_captures_clean_execution(tmp_path: Path):
    script = tmp_path / "ok.py"
    script.write_text("print('clean')")
    result = run_sanitized(script, "asan")
    assert result["ok"]
    assert not result["finding"]


def test_sanitizer_runner_detects_signature(tmp_path: Path, monkeypatch):
    # Use a fake Python executable to deterministically exercise signature parsing.
    # Use the current Python interpreter so the deterministic fixture works on Windows too.
    import sys

    script = tmp_path / "asan_signature.py"
    script.write_text(
        "import sys; print('AddressSanitizer: heap-use-after-free', file=sys.stderr); raise SystemExit(1)"
    )
    result = run_sanitized(script, "asan", python=sys.executable)
    assert result["finding"]
    assert not result["ok"]
