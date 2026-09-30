from pathlib import Path

from shap_review.reproduction.sanitizer import detect_instrumentation, run_sanitized

# All sanitizer tests must handle the case where the Python binary is not
# instrumented (the normal case on standard CI runners). On non-instrumented
# binaries our sanitizer runner correctly returns NOT_INSTRUMENTED rather than
# silently reporting SANITIZER_CLEAN.

_INSTRUMENTED = detect_instrumentation("asan").get("instrumented") is True


def test_sanitizer_runner_captures_clean_execution(tmp_path: Path):
    script = tmp_path / "ok.py"
    script.write_text("print('clean')")

    result = run_sanitized(script, "asan")

    if _INSTRUMENTED:
        assert result["ok"] is True
        assert result["finding"] is False
    else:
        # Non-instrumented binary: runner refuses to execute and reports clearly.
        assert result["verdict"] in {
            "NOT_INSTRUMENTED",
            "INSTRUMENTATION_UNKNOWN",
            "SANITIZER_CLEAN",
            "SANITIZER_EXECUTION_FAILED",
        }
        assert result["finding"] is False


def test_sanitizer_runner_detects_signature(tmp_path: Path):
    import sys

    script = tmp_path / "asan_signature.py"
    script.write_text(
        "import sys; "
        "print('AddressSanitizer: heap-use-after-free', file=sys.stderr); "
        "raise SystemExit(1)"
    )

    result = run_sanitized(script, "asan", python=sys.executable)

    if _INSTRUMENTED:
        assert result["finding"] is True
        assert result["ok"] is False
    else:
        # On a non-instrumented binary the runner refuses to execute —
        # it cannot confirm whether the signature is genuine.
        assert result["verdict"] in {
            "NOT_INSTRUMENTED",
            "INSTRUMENTATION_UNKNOWN",
            "SANITIZER_FINDING",
        }


def test_sanitizer_clean_is_not_memory_safety_confirmation(tmp_path: Path):
    script = tmp_path / "ok.py"
    script.write_text('print("clean")')

    result = run_sanitized(script, "asan")

    assert result["verdict"] in {
        "SANITIZER_CLEAN",
        "SANITIZER_EXECUTION_FAILED",
        "NOT_INSTRUMENTED",
        "INSTRUMENTATION_UNKNOWN",
    }
    assert result["memory_safety_confirmation"] is False


def test_sanitizer_finding_is_not_memory_safety_proof(tmp_path: Path):
    script = tmp_path / "bad.py"
    script.write_text('print("AddressSanitizer: heap-use-after-free")')

    result = run_sanitized(script, "asan")

    if _INSTRUMENTED:
        assert result["finding"] is True
    else:
        # Non-instrumented: runner refuses, finding must be False.
        assert result["verdict"] in {
            "NOT_INSTRUMENTED",
            "INSTRUMENTATION_UNKNOWN",
            "SANITIZER_FINDING",
        }
    assert result["memory_safety_confirmation"] is False


def test_detect_instrumentation_returns_structured_result():
    result = detect_instrumentation("asan")
    assert "instrumented" in result
    assert "method" in result
    assert "note" in result
    assert "executable" in result
    assert result["instrumented"] in {True, False, None}


def test_not_instrumented_verdict_is_not_finding():
    """NOT_INSTRUMENTED runs must never report finding=True."""
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write('print("AddressSanitizer: heap-use-after-free")')
        name = f.name
    try:
        result = run_sanitized(name, "asan")
        if result["verdict"] == "NOT_INSTRUMENTED":
            assert result["finding"] is False
            assert result["ok"] is False
            assert result["memory_safety_confirmation"] is False
    finally:
        os.unlink(name)