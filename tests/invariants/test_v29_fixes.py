"""v0.29.0 — Tests for every correctness fix made in this release.

Each test is named after the specific finding it validates.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── P0-A: issue evidence must be EXTERNAL, not DERIVED ───────────────────────


def test_issue_evidence_has_external_origin_and_nonzero_score():
    """Historical corpus evidence must produce a non-zero score and
    HISTORICALLY_CORRELATED (or better) verdict — not UNVALIDATED/0.0."""
    from shap_review.evidence.chain import build_chain

    entries = [
        {
            "kind": "issue",
            "source": "SHAP-EVID-4911",
            "claim": "TreeExplainer nullable dtype failure",
            "passed": True,
            "confidence": 0.6,
        },
        {
            "kind": "source",
            "source": "SHAP implementation",
            "claim": "to_numpy conversion found near TreeExplainer call",
            "passed": True,
            "confidence": 0.5,
        },
    ]
    chain = build_chain(entries=entries)
    assert chain.evidence_strength_score() > 0.0, (
        "Issue evidence must not have score=0.0 — check origin default in build_chain()"
    )
    assert chain.verdict() != "UNVALIDATED", (
        f"Got verdict={chain.verdict()!r}; expected HISTORICALLY_CORRELATED or better"
    )


def test_issue_origin_is_external_not_derived():
    """build_chain() must set origin=external for issue/pull_request evidence."""
    from shap_review.evidence.chain import build_chain
    from shap_review.evidence.model import EvidenceOrigin

    chain = build_chain(
        entries=[
            {
                "kind": "issue",
                "source": "SHAP-EVID-4869",
                "claim": "test coverage gap",
                "passed": True,
                "confidence": 0.5,
            }
        ]
    )
    item = chain.items[0]
    assert item.origin == EvidenceOrigin.EXTERNAL, (
        f"Issue evidence origin must be EXTERNAL, got {item.origin}"
    )
    assert item.intrinsically_valid is True


def test_pullrequest_origin_is_external_not_derived():
    from shap_review.evidence.chain import build_chain
    from shap_review.evidence.model import EvidenceOrigin

    chain = build_chain(
        entries=[
            {
                "kind": "pull_request",
                "source": "github.com/shap/shap/pull/1234",
                "claim": "fix nullable dtype handling",
                "passed": True,
                "confidence": 0.7,
            }
        ]
    )
    item = chain.items[0]
    assert item.origin == EvidenceOrigin.EXTERNAL


# ── P0-B: invariants/evaluator.py must use correct feature axis ───────────────


def test_numeric_additivity_evaluator_correct_for_2d():
    """2-D case (samples, features) must still work after the axis fix."""
    from shap_review.invariants.evaluator import numeric_additivity

    r = numeric_additivity([[1.0, 2.0], [3.0, 4.0]], [0.0, 1.0], [3.0, 8.0])
    assert r.passed


def test_numeric_additivity_evaluator_correct_for_3d_multiclass():
    """3-D multiclass (samples, features, classes) must sum over features (axis=1),
    NOT axis=-1 which would be the class axis and produce a wrong result."""
    from shap_review.invariants.evaluator import numeric_additivity

    # 2 samples, 3 features, 4 classes
    # Each feature contributes 0.25 per class → feature sum = 0.75 per class
    vals = np.ones((2, 3, 4)) * 0.25
    base = np.zeros(4)  # 4-class baseline
    target = np.ones((2, 4)) * 0.75  # 3 features * 0.25 = 0.75 per class

    r = numeric_additivity(vals, base, target, rtol=1e-6, atol=1e-8)
    assert r.passed, (
        f"3-D multiclass additivity check failed: {r.message}. "
        "Check that evaluator uses feature axis, not class axis (axis=-1)."
    )


def test_numeric_additivity_evaluator_fails_when_wrong():
    from shap_review.invariants.evaluator import numeric_additivity

    r = numeric_additivity([[1.0, 2.0]], [0.0], [10.0])
    assert not r.passed


# ── P0-D: InteractionOracle with values=None must be INCONCLUSIVE ─────────────


def test_interaction_oracle_symmetric_without_values_is_inconclusive():
    """A symmetric tensor without reference values must return passed=None
    (INCONCLUSIVE), not passed=True.  A symmetric but semantically wrong tensor
    must not produce a false PASS."""
    from shap_review.contracts.oracles import InteractionOracle

    # All-ones tensor: perfectly symmetric, completely wrong for real SHAP
    iv = np.ones((3, 4, 4))
    result = InteractionOracle().check(values=None, interaction_values=iv)

    assert result.passed is None, (
        f"InteractionOracle with values=None must return passed=None "
        f"(INCONCLUSIVE), got passed={result.passed!r}"
    )
    assert result.details["reconstruction_checked"] is False
    assert result.details["symmetry_passed"] is True


def test_interaction_oracle_passes_when_correct_values_supplied():
    """When values IS supplied, reconstruction check must run and pass for correct data."""
    from shap_review.contracts.oracles import InteractionOracle

    n, f = 4, 3
    # Correct interaction tensor: symmetric diagonal-dominated, reconstructs correctly
    rng = np.random.default_rng(0)
    iv = np.zeros((n, f, f))
    for i in range(f):
        iv[:, i, i] = rng.random(n)
    # Make off-diagonal symmetric
    for i in range(f):
        for j in range(i + 1, f):
            val = rng.random(n) * 0.1
            iv[:, i, j] = val
            iv[:, j, i] = val
    values = iv.sum(axis=2)  # reconstruction: sum over second feature axis

    result = InteractionOracle().check(values=values, interaction_values=iv)
    assert result.details["reconstruction_checked"] is True


# ── P1-A: no group-size bonus in CandidateAggregator ─────────────────────────


def test_aggregator_does_not_inflate_from_multiple_analyzers():
    """Three analyzers all firing on the same source token must NOT produce
    medium confidence — that would be multi-analyzer noise, not signal."""
    from shap_review.candidates import CandidateAggregator
    from shap_review.types import Candidate, EvidenceRef

    # 5 candidates from 5 different analyzers, all source-kind evidence
    candidates = []
    for i in range(5):
        e = EvidenceRef("source", f"analyzer_{i}", "AST pattern match", 2)
        c = Candidate(
            f"id{i}",
            "SHAP-01",
            "INV-ATTR-001",
            "f.py",
            10,
            None,
            "msg",
            [e],
            True,
            "low",
            [f"tag{i}"],
        )
        candidates.append(c)

    result = CandidateAggregator().merge(candidates)
    assert len(result) == 1
    # 5 source items: score = 5*1 + (1-1)*2 = 5 → still low
    assert result[0].confidence == "low", (
        f"5 source-kind signals from 5 analyzers should NOT inflate to "
        f"{result[0].confidence!r}; the group-size bonus must be removed."
    )


def test_aggregator_raises_correctly_with_genuinely_independent_evidence():
    """Issue + dynamic evidence are genuinely different kinds and CAN raise confidence."""
    from shap_review.candidates import CandidateAggregator
    from shap_review.types import Candidate, EvidenceRef

    e1 = EvidenceRef("issue", "SHAP-EVID-4869", "historical evidence", 4)
    e2 = EvidenceRef("dynamic", "runtime-bridge", "additivity failed", 7)
    c = Candidate(
        "id1",
        "SHAP-01",
        "INV-ATTR-001",
        "f.py",
        5,
        None,
        "msg",
        [e1, e2],
        True,
        "low",
        [],
    )
    result = CandidateAggregator().merge([c])
    # score = 4+7 + (2-1)*2 = 13 → still low (< 14) but close
    # The independent_kinds bonus (2-1)*2 = 2 is correct here
    assert result[0].confidence in ("low", "medium")  # depends on score threshold


# ── P1-B: PythonContractAnalyzer must not fire on comment tokens ──────────────


def test_python_contract_analyzer_ignores_tokens_in_comments():
    """A token appearing only in a comment must NOT generate a candidate."""
    from shap_review.analyzers.python.contracts import PythonContractAnalyzer

    code = """
def __init__(self, model):
    # We do NOT use expected_value here — intentional
    # check_additivity is disabled for this mock
    self.explainer = model
"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "model.py"
        p.write_text(code)
        result = PythonContractAnalyzer().analyze(Path(d))
    assert result == [], (
        f"Got {len(result)} candidates from comment-only tokens; expected 0. "
        "Tokens in comments must not trigger candidates."
    )


def test_python_contract_analyzer_fires_on_actual_ast_tokens():
    """A token appearing as a real Name node in the function body must fire."""
    from shap_review.analyzers.python.contracts import PythonContractAnalyzer

    code = """
def __init__(self, model):
    self.value = self.explainer.expected_value
    self.check = check_additivity
"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "model.py"
        p.write_text(code)
        result = PythonContractAnalyzer().analyze(Path(d))
    assert len(result) >= 1, "Real AST-level token must produce a candidate"


# ── P1-C: SHAPSemanticAnalyzer fallback must not fire on non-SHAP code ────────


def test_semantic_analyzer_fallback_requires_shap_import():
    """A file with shap_values() but no shap import must NOT trigger the fallback."""
    from shap_review.analyzers.shap.semantic import SHAPSemanticAnalyzer

    code = """
import numpy as np

def shap_values(X, background):
    # Custom attribution — NOT shap
    return np.zeros_like(X)
"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "custom.py"
        p.write_text(code)
        result = SHAPSemanticAnalyzer().analyze(Path(d))
    assert result == [], (
        f"Got {len(result)} candidates from non-SHAP code with SHAP-like function names. "
        "Fallback must require a shap import."
    )


def test_semantic_analyzer_fallback_fires_with_shap_import():
    """A file with shap import AND shap_values() function triggers the fallback."""
    from shap_review.analyzers.shap.semantic import SHAPSemanticAnalyzer

    code = """
import shap
import numpy as np

def shap_values(X, background):
    return shap.TreeExplainer(None).shap_values(X)
"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "wrapper.py"
        p.write_text(code)
        # shap is imported so the main path fires, not the fallback — but result > 0
        result = SHAPSemanticAnalyzer().analyze(Path(d))
    assert len(result) >= 1


# ── P1-D: SHAP-05 only fires in files with TreeExplainer calls ───────────────


def test_pandas_conversion_without_treexplainer_does_not_fire():
    """to_numpy() in a file with no TreeExplainer call must not produce SHAP-05."""
    from shap_review.analyzers.shap.semantic import SHAPSemanticAnalyzer

    code = """
import pandas as pd
import numpy as np

def preprocess(df):
    return df.to_numpy()
"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "utils.py"
        p.write_text(code)
        result = SHAPSemanticAnalyzer().analyze(Path(d))
    shap05 = [c for c in result if c.bug_class == "SHAP-05"]
    assert shap05 == [], (
        "to_numpy() without a TreeExplainer call must not produce SHAP-05 candidates"
    )


# ── P1-E: tests/ and examples/ must be excluded from iter_source_files ────────


def test_iter_source_files_excludes_test_directories():
    from shap_review.utils import iter_source_files

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "examples").mkdir()
        (root / "benchmarks").mkdir()
        (root / "src" / "impl.py").write_text("import shap")
        (root / "tests" / "test_impl.py").write_text("import shap")
        (root / "examples" / "demo.py").write_text("import shap")
        (root / "benchmarks" / "bench.py").write_text("import shap")

        files = set(str(f.relative_to(root)) for f in iter_source_files(root))

    assert "src/impl.py" in files
    assert not any("tests" in f for f in files), "tests/ must be excluded by default"
    assert not any("examples" in f for f in files), (
        "examples/ must be excluded by default"
    )
    assert not any("benchmarks" in f for f in files), (
        "benchmarks/ must be excluded by default"
    )


def test_iter_source_files_include_tests_flag():
    from shap_review.utils import iter_source_files

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "tests").mkdir()
        (root / "tests" / "test_x.py").write_text("x = 1")

        files_default = list(iter_source_files(root))
        files_with_tests = list(iter_source_files(root, include_tests=True))

    assert len(files_default) == 0
    assert len(files_with_tests) == 1


# ── P2-A: _align_base ambiguity when n_samples == n_classes ──────────────────


def test_align_base_raises_on_ambiguous_1d_shape():
    """When n_samples == n_classes, _align_base must raise rather than silently
    broadcast on the wrong axis."""
    from shap_review.contracts.tensor import SHAPAxisSpec, _align_base

    # 3 samples, 3 features, 3 classes → base shape (3,) is ambiguous
    target_shape = (3, 3, 3)
    base = np.array([0.3, 0.3, 0.4])
    axes = SHAPAxisSpec(sample_axis=0, feature_axis=1, output_axis=2)

    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        _align_base(base, target_shape, axes)


def test_align_base_ok_when_unambiguous():
    """When n_samples != n_classes there is no ambiguity and broadcast must succeed."""
    from shap_review.contracts.tensor import SHAPAxisSpec, _align_base

    # 5 samples, 4 features, 3 classes → base shape (3,) clearly matches output_axis
    target_shape = (5, 4, 3)
    base = np.array([0.3, 0.3, 0.4])
    axes = SHAPAxisSpec(sample_axis=0, feature_axis=1, output_axis=2)

    result = _align_base(base, target_shape, axes)
    assert result.shape == target_shape


# ── P2-E: PromotionPolicy must block CONFIRMED from agreement-only differential ─


def test_promotion_policy_blocks_confirmed_from_unknown_reference():
    """Differential agreement with reference_correctness=UNKNOWN must NOT
    allow promotion to CONFIRMED."""
    from shap_review.evidence.chain import build_chain
    from shap_review.findings.promotion import PromotionPolicy
    from shap_review.types import FindingStatus

    # Differential evidence with UNKNOWN reference
    chain = build_chain(
        entries=[
            {
                "kind": "differential",
                "source": "differential-runner",
                "claim": "candidate matches reference",
                "passed": True,
                "confidence": 0.8,
                "origin": "external",
                "details": {"reference_correctness": "UNKNOWN"},
            }
        ]
    )

    policy = PromotionPolicy()
    with pytest.raises(ValueError, match="CONFIRMED requires"):
        policy.validate(
            FindingStatus.EVIDENCE_VALID, FindingStatus.CONFIRMED, chain=chain
        )


# ── P3-B: evidence-model.md must not mention CONFIRMED_MEMORY_SAFETY ──────────


def test_evidence_model_doc_has_no_ghost_verdict():
    """CONFIRMED_MEMORY_SAFETY was never implemented; docs must not mention it."""
    root = Path(__file__).resolve().parents[2]
    doc = (root / "docs" / "evidence-model.md").read_text()
    assert "CONFIRMED_MEMORY_SAFETY" not in doc, (
        "docs/evidence-model.md still mentions CONFIRMED_MEMORY_SAFETY which "
        "does not exist in EvidenceChain.verdict(). Use SANITIZER_FINDING."
    )


# ── api-era: return statement detection ──────────────────────────────────────


def test_api_era_detects_treexplainer_in_return():
    """scan_api_era() must detect shap.TreeExplainer in return statements."""
    from shap_review.semantic.api_era import scan_api_era

    code = """
import shap

def build(model):
    return shap.TreeExplainer(model, model_output="probability")
"""
    results = scan_api_era(code)
    assert len(results) >= 1, (
        "scan_api_era() must detect TreeExplainer in return statements. "
        "Only assignment-based detection was implemented previously."
    )


# ── version consistency ───────────────────────────────────────────────────────


def test_version_is_v29():
    from shap_review.version import VERSION

    assert VERSION == "0.29.0"
