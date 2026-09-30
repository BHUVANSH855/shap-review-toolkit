from pathlib import Path

from shap_review.reproduction.sanitizer import run_sanitized


def test_sanitizer_runner_captures_clean_execution(tmp_path: Path):
    script = tmp_path / "ok.py"
    script.write_text("print('clean')")

    result = run_sanitized(script, "asan")

    assert result["ok"] is True
    assert result["finding"] is False


def test_sanitizer_runner_detects_signature(tmp_path: Path):
    # Use the current Python interpreter so the deterministic fixture
    # works consistently on Windows as well.
    import sys

    script = tmp_path / "asan_signature.py"
    script.write_text(
        "import sys; "
        "print('AddressSanitizer: heap-use-after-free', file=sys.stderr); "
        "raise SystemExit(1)"
    )

    result = run_sanitized(
        script,
        "asan",
        python=sys.executable,
    )

    assert result["finding"] is True
    assert result["ok"] is False


def test_sanitizer_clean_is_not_memory_safety_confirmation(tmp_path: Path):
    script = tmp_path / "ok.py"
    script.write_text('print("clean")')

    result = run_sanitized(script, "asan")

    assert result["verdict"] in {
        "SANITIZER_CLEAN",
        "SANITIZER_EXECUTION_FAILED",
    }
    assert result["memory_safety_confirmation"] is False


def test_sanitizer_finding_is_not_memory_safety_proof(tmp_path: Path):
    script = tmp_path / "bad.py"
    script.write_text('print("AddressSanitizer: heap-use-after-free")')

    result = run_sanitized(script, "asan")

    assert result["finding"] is True
    assert result["memory_safety_confirmation"] is False
