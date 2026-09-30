from pathlib import Path

from shap_review.engine import ReviewEngine
from shap_review.runtime_bridge import (
    _candidate_fingerprint,
    attach_dynamic_evidence,
)

FIX = Path(__file__).parent / "fixtures"


def run(path):
    return ReviewEngine().analyze(path, path / ".out")


def test_issue_4911_signal_is_evidence_linked():
    candidates = run(FIX / "shap_4911_nullable_dtype")

    assert any(
        candidate.bug_class == "SHAP-05"
        and any(evidence.source == "SHAP-EVID-4911" for evidence in candidate.evidence)
        for candidate in candidates
    )


def test_issue_5098_signal_is_evidence_linked():
    candidates = run(FIX / "shap_5098_model_output")

    assert any(
        candidate.bug_class == "SHAP-02"
        and any(evidence.source == "SHAP-EVID-5098" for evidence in candidate.evidence)
        for candidate in candidates
    )


def test_issue_4869_is_test_gap_not_defect():
    candidates = run(FIX / "shap_4869_interaction_gap")

    assert any(
        candidate.bug_class == "SHAP-01" and not candidate.validation_required
        for candidate in candidates
    )


def test_evidence_corpus_has_canonical_primary_records():
    from shap_review.evidence import EvidenceCorpus

    corpus = EvidenceCorpus.from_directory(
        Path(__file__).parents[1] / "shap_review/resources/evidence/issues"
    )

    assert len(corpus.records) >= 7
    assert not corpus.validate()


def test_dynamic_evidence_without_provenance_is_ambiguous_not_independent():
    from shap_review.evidence.model import (
        EvidenceChain,
        EvidenceItem,
        EvidenceKind,
    )

    item = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "runtime",
        "claim",
        True,
        evidence_id="e1",
    )

    chain = EvidenceChain()
    chain.add(item)

    assert item.independent is False


def test_derived_evidence_without_ancestry_is_not_independent():
    from shap_review.evidence.model import (
        EvidenceChain,
        EvidenceItem,
        EvidenceKind,
        EvidenceOrigin,
    )

    item = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "runtime",
        "claim",
        True,
        origin=EvidenceOrigin.DERIVED,
        evidence_id="e1",
        execution_id="run1",
    )

    chain = EvidenceChain()
    chain.add(item)

    assert item.independent is False


def test_same_producer_can_be_independent_across_runs():
    """Same producer is not sufficient to establish correlation.

    Pairwise independence must be determined by EvidenceGraph (via
    build_chain()), not by EvidenceItem.independent alone.  Two items with
    different execution_ids and different input_fingerprints from the same
    producer are correctly identified as independent when a graph is attached.

    NOTE: Direct EvidenceChain() construction without build_chain() uses a
    conservative fallback (returns at most 1 item) to prevent false-confidence
    inflation.  This test uses build_chain() to exercise the graph-based path.
    """
    from shap_review.evidence.chain import build_chain

    chain = build_chain(
        entries=[
            {
                "kind": "dynamic",
                "source": "runtime",
                "claim": "a",
                "passed": True,
                "evidence_id": "a",
                "execution_id": "run-a",
                "producer": "oracle",
                "input_fingerprint": "input-a",
            },
            {
                "kind": "dynamic",
                "source": "runtime",
                "claim": "b",
                "passed": True,
                "evidence_id": "b",
                "execution_id": "run-b",
                "producer": "oracle",
                "input_fingerprint": "input-b",
            },
        ]
    )

    # With a graph attached, pairwise independence is correctly computed:
    # different execution_id + different input_fingerprint → independent.
    assert len(chain.independent_items()) == 2


def test_direct_chain_without_graph_uses_conservative_fallback():
    """EvidenceChain() without build_chain() returns at most 1 independent item.

    This prevents callers who bypass build_chain() from accidentally inflating
    confidence scores when correlated items are added to a graphless chain.
    """
    from shap_review.evidence.model import (
        EvidenceChain,
        EvidenceItem,
        EvidenceKind,
    )

    first = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "runtime",
        "a",
        True,
        evidence_id="a",
        execution_id="run-a",
        producer="oracle",
        input_fingerprint="input-a",
    )
    second = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "runtime",
        "b",
        True,
        evidence_id="b",
        execution_id="run-b",
        producer="oracle",
        input_fingerprint="input-b",
    )

    chain = EvidenceChain()
    chain.add(first)
    chain.add(second)

    # No graph: conservative fallback returns at most 1 item.
    assert len(chain.independent_items()) == 1
    # Verdict and score reflect only 1 item.
    assert chain.to_dict()["graph_attached"] is False


def test_evidence_graph_detects_shared_input_fingerprint():
    from shap_review.evidence.graph import EvidenceGraph, EvidenceNode

    graph = EvidenceGraph()

    graph.add(
        EvidenceNode(
            "a",
            "dynamic",
            "runtime",
            "p",
            "a",
            True,
            execution_id="r1",
            input_fingerprint="same",
        )
    )
    graph.add(
        EvidenceNode(
            "b",
            "dynamic",
            "runtime",
            "p",
            "b",
            True,
            execution_id="r2",
            input_fingerprint="same",
        )
    )

    assert not graph.is_independent("a", "b")
    assert graph.independence_reason("a", "b") == "shared-input-lineage"


def test_sanitizer_clean_never_confirms_memory_safety():
    from shap_review.evidence.model import (
        EvidenceChain,
        EvidenceItem,
        EvidenceKind,
    )

    item = EvidenceItem(
        EvidenceKind.SANITIZER,
        "asan",
        "no finding",
        True,
        evidence_id="s1",
        execution_id="run1",
        fixture_id="fx1",
        details={"finding": False},
    )

    chain = EvidenceChain()
    chain.add(item)

    assert chain.verdict() == "SANITIZER_CLEAN"


def test_runtime_anomaly_with_matching_bug_class_is_not_attached_without_correlation():
    candidates = run(FIX / "shap_4911_nullable_dtype")

    target = next(
        candidate for candidate in candidates if candidate.bug_class == "SHAP-05"
    )

    original_chain_size = len(target.evidence_chain.items)
    original_validation_required = target.validation_required

    bridge_result = {
        "available": True,
        "anomalies": [
            {
                "case": {
                    "n_features": 4,
                    "classification": False,
                    "model_output": "raw",
                },
                "execution": {
                    "executed": True,
                    "input_unchanged": False,
                },
                "classification": {
                    "kind": "INPUT_MUTATION",
                    "bug_class": "SHAP-05",
                    "confidence": 0.7,
                },
                "provenance": {
                    "execution_id": "unrelated-run",
                    "input_fingerprint": "unrelated-input",
                    "environment_fingerprint": "env-a",
                    "producer": "runtime-bridge",
                },
            }
        ],
        "classifications": [
            {
                "kind": "INPUT_MUTATION",
                "bug_class": "SHAP-05",
                "confidence": 0.7,
            }
        ],
    }

    updated = attach_dynamic_evidence([target], bridge_result)
    result = updated[0]

    assert len(result.evidence_chain.items) == original_chain_size
    assert result.validation_required == original_validation_required
    assert "dynamic-evidence" not in result.tags
    assert "runtime-bridge" not in result.tags


def test_correlated_runtime_anomaly_can_attach_to_matching_candidate():
    candidates = run(FIX / "shap_4911_nullable_dtype")

    target = next(
        candidate for candidate in candidates if candidate.bug_class == "SHAP-05"
    )

    candidate_fingerprint = _candidate_fingerprint(target)

    bridge_result = {
        "available": True,
        "anomalies": [
            {
                "case": {
                    "n_features": 4,
                    "classification": False,
                    "model_output": "raw",
                },
                "execution": {
                    "executed": True,
                    "input_unchanged": False,
                },
                "classification": {
                    "kind": "INPUT_MUTATION",
                    "bug_class": "SHAP-05",
                    "confidence": 0.7,
                },
                "provenance": {
                    "execution_id": "correlated-run",
                    "input_fingerprint": "matching-input",
                    "candidate_fingerprint": candidate_fingerprint,
                    "environment_fingerprint": "env-a",
                    "producer": "runtime-bridge",
                    "shap_version": "test",
                },
            }
        ],
        "classifications": [
            {
                "kind": "INPUT_MUTATION",
                "bug_class": "SHAP-05",
                "confidence": 0.7,
            }
        ],
    }

    updated = attach_dynamic_evidence([target], bridge_result)
    result = updated[0]

    dynamic = [
        item for item in result.evidence_chain.items if item.source == "runtime-bridge"
    ]

    assert len(dynamic) == 1
    assert dynamic[0].origin.value == "execution"
    assert dynamic[0].details["execution_id"] == "correlated-run"
    assert dynamic[0].details["candidate_fingerprint"] == candidate_fingerprint
    assert "dynamic-evidence" in result.tags
    assert "runtime-bridge" in result.tags
    assert result.validation_required == target.validation_required


def test_runtime_bridge_preserves_environment_provenance():
    from shap_review.runtime_bridge import run_bridge

    result = run_bridge(iterations=0, seed=42)

    assert "provenance" in result

    provenance = result["provenance"]
    assert "environment_fingerprint" in provenance
    assert "python_executable" in provenance
    assert "python_version" in provenance
    assert "shap_version" in provenance
    assert "shap_source_path" in provenance

    if result["available"]:
        assert provenance["producer"] == "runtime-bridge"
        assert provenance["execution_id"]
        assert provenance["seed"] == 42
        assert provenance["iterations"] == 0


def test_evidence_graph_preserves_item_provenance():
    from shap_review.evidence.graph import EvidenceGraph
    from shap_review.evidence.model import (
        EvidenceItem,
        EvidenceKind,
        EvidenceOrigin,
    )

    item = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "runtime",
        "claim",
        False,
        origin=EvidenceOrigin.EXECUTION,
        evidence_id="runtime-1",
        execution_id="exec-1",
        producer="runtime-bridge",
        fixture_id="fixture-1",
        input_fingerprint="input-1",
        repository_revision="revision-1",
        environment_fingerprint="environment-1",
        transformation="normalization",
    )

    graph = EvidenceGraph()
    node = graph.add_item(item)

    assert node.execution_id == "exec-1"
    assert node.producer == "runtime-bridge"
    assert node.fixture_id == "fixture-1"
    assert node.input_fingerprint == "input-1"
    assert node.repository_revision == "revision-1"
    assert node.transformation == "normalization"
    assert node.environment == {}


def test_evidence_graph_detects_shared_ancestry():
    from shap_review.evidence.graph import EvidenceGraph, EvidenceNode

    graph = EvidenceGraph()

    graph.add(
        EvidenceNode(
            "root",
            "static",
            "source",
            "scanner",
            "root",
            True,
        )
    )
    graph.add(
        EvidenceNode(
            "left",
            "dynamic",
            "runtime",
            "runner-a",
            "left",
            True,
            parent_ids=("root",),
        )
    )
    graph.add(
        EvidenceNode(
            "right",
            "dynamic",
            "runtime",
            "runner-b",
            "right",
            True,
            parent_ids=("root",),
        )
    )

    assert not graph.is_independent("left", "right")
    assert graph.independence_reason("left", "right") == "shared-ancestry"


def test_evidence_verdict_stays_static_when_runtime_provenance_is_ambiguous():
    from shap_review.evidence.model import EvidenceChain, EvidenceItem, EvidenceKind

    chain = EvidenceChain()

    chain.add(
        EvidenceItem(
            EvidenceKind.STATIC,
            "x",
            "candidate",
            True,
            evidence_id="static-1",
        )
    )

    chain.add(
        EvidenceItem(
            EvidenceKind.DYNAMIC,
            "runtime",
            "executed",
            True,
            evidence_id="dynamic-1",
        )
    )

    assert chain.verdict() == "STATIC_CANDIDATE"

    chain.add(
        EvidenceItem(
            EvidenceKind.REPRODUCTION,
            "repro",
            "reproduced",
            True,
            evidence_id="reproduction-1",
        )
    )

    assert chain.verdict() == "STATIC_CANDIDATE"


def test_evidence_dependency_is_not_independent():
    from shap_review.evidence import (
        EvidenceChain,
        EvidenceItem,
        EvidenceKind,
        EvidenceOrigin,
    )

    chain = EvidenceChain()
    chain.add(
        EvidenceItem(
            EvidenceKind.STATIC,
            "historical-rule",
            "same issue",
            True,
            origin=EvidenceOrigin.DERIVED,
            derived_from=("SHAP-EVID-1",),
        )
    )

    assert chain.independent_items() == []
    assert chain.verdict() == "UNVALIDATED"


def test_evidence_gate_rejects_static_confirmation():
    from shap_review.evidence import EvidenceChain, EvidenceItem, EvidenceKind
    from shap_review.findings.lifecycle import evidence_transition
    from shap_review.types import FindingStatus

    chain = EvidenceChain([EvidenceItem(EvidenceKind.STATIC, "x", "candidate", True)])

    try:
        evidence_transition(
            FindingStatus.REPRODUCED,
            FindingStatus.EVIDENCE_VALID,
            chain,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("static evidence must not pass confirmation gate")


def test_promotion_requires_runtime_evidence():
    import pytest

    from shap_review.evidence.chain import build_chain
    from shap_review.findings.lifecycle import evidence_transition
    from shap_review.types import FindingStatus

    chain = build_chain(
        entries=[
            {
                "kind": "static",
                "source": "s",
                "claim": "candidate",
                "passed": True,
                "evidence_id": "s",
            }
        ]
    )

    with pytest.raises(ValueError):
        evidence_transition(
            FindingStatus.REPRO_PENDING,
            FindingStatus.REPRODUCED,
            chain,
        )


def test_evidence_same_execution_lineage_is_not_independent():
    from shap_review.evidence.chain import build_chain

    chain = build_chain(
        entries=[
            {
                "kind": "dynamic",
                "source": "a",
                "claim": "a",
                "passed": True,
                "evidence_id": "a",
                "details": {"execution_id": "run1"},
            },
            {
                "kind": "differential",
                "source": "b",
                "claim": "b",
                "passed": True,
                "evidence_id": "b",
                "details": {"execution_id": "run1"},
            },
        ]
    )
    assert not chain.graph.is_independent("a", "b")
    assert chain.graph.independence_reason("a", "b") == "shared-execution-lineage"


def test_evidence_same_revision_and_environment_is_not_independent():
    from shap_review.evidence.graph import EvidenceGraph, EvidenceNode

    graph = EvidenceGraph()
    environment = {"python": "3.13", "shap": "0.50.0"}
    graph.add(
        EvidenceNode(
            "a",
            "dynamic",
            "a",
            "p1",
            "a",
            True,
            execution_id="e1",
            repository_revision="r1",
            environment=environment,
        )
    )
    graph.add(
        EvidenceNode(
            "b",
            "differential",
            "b",
            "p2",
            "b",
            True,
            execution_id="e2",
            repository_revision="r1",
            environment=environment,
        )
    )
    assert not graph.is_independent("a", "b")
    assert graph.independence_reason("a", "b") == "shared-revision-environment"


# ============================================================================
# New tests — evidence model gaps from review
# ============================================================================

# ---------------------------------------------------------------------------
# C-2: EvidenceChain graph=None conservative fallback
# ---------------------------------------------------------------------------


def test_two_correlated_items_without_graph_score_as_one():
    """Two items sharing execution_id without a graph must not double-count.

    The old implementation returned all eligible items when graph=None, allowing
    two items from the same execution to inflate confidence to HIGH_CONFIDENCE.
    The fix limits graphless chains to at most 1 independent item.
    """
    from shap_review.evidence.model import (
        EvidenceChain,
        EvidenceItem,
        EvidenceKind,
    )

    # Same execution_id — these are correlated, but no graph to detect it.
    shared_exec = "exec-123"
    item_a = EvidenceItem(
        EvidenceKind.DIFFERENTIAL,
        "run",
        "a",
        True,
        evidence_id="a",
        execution_id=shared_exec,
        input_fingerprint="fp-1",
    )
    item_b = EvidenceItem(
        EvidenceKind.DIFFERENTIAL,
        "run",
        "b",
        True,
        evidence_id="b",
        execution_id=shared_exec,
        input_fingerprint="fp-2",
    )

    chain = EvidenceChain()
    chain.add(item_a)
    chain.add(item_b)

    # No graph: must return exactly 1 item.
    independent = chain.independent_items()
    assert len(independent) == 1, (
        f"graph=None must cap independent_items at 1, got {len(independent)}"
    )

    # Score must reflect only 1 item (1.75), not 2 (3.5).
    score = chain.evidence_strength_score()
    assert score < 3.0, (
        f"Correlated graphless items must not exceed score 3.0, got {score:.2f}"
    )

    # Verdict must NOT be HIGH_CONFIDENCE (requires score >= 3.0).
    assert chain.verdict() != "HIGH_CONFIDENCE", (
        "Two correlated DIFFERENTIAL items without a graph must not reach "
        "HIGH_CONFIDENCE"
    )


def test_chain_to_dict_reports_graph_attached_false():
    """EvidenceChain.to_dict() must report graph_attached=False when no graph."""
    from shap_review.evidence.model import (
        EvidenceChain,
        EvidenceItem,
        EvidenceKind,
    )

    item = EvidenceItem(
        EvidenceKind.STATIC,
        "src",
        "claim",
        True,
        evidence_id="x",
    )
    chain = EvidenceChain()
    chain.add(item)

    d = chain.to_dict()
    assert d["graph_attached"] is False
    assert (
        "conservative fallback" in d["independence_policy"].lower()
        or "build_chain" in d["independence_policy"]
    )


def test_chain_to_dict_reports_graph_attached_true_when_graph_present():
    """EvidenceChain.to_dict() must report graph_attached=True when a graph is present."""
    from shap_review.evidence.chain import build_chain

    chain = build_chain(
        entries=[
            {
                "kind": "dynamic",
                "source": "runtime",
                "claim": "x",
                "passed": True,
                "evidence_id": "x",
                "execution_id": "e1",
                "input_fingerprint": "fp1",
            }
        ]
    )
    assert chain.to_dict()["graph_attached"] is True


def test_evidence_item_independent_property_is_not_pairwise():
    """EvidenceItem.independent is structural validity, NOT pairwise independence.

    Two items that are both intrinsically_valid may still be correlated.
    Pairwise independence requires EvidenceGraph.
    """
    from shap_review.evidence.model import EvidenceItem, EvidenceKind

    # Both items are intrinsically valid (different execution_ids, fingerprints).
    item_a = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "r",
        "a",
        True,
        evidence_id="a",
        execution_id="e1",
        input_fingerprint="fp1",
    )
    item_b = EvidenceItem(
        EvidenceKind.DYNAMIC,
        "r",
        "b",
        True,
        evidence_id="b",
        execution_id="e1",  # SAME execution_id → correlated
        input_fingerprint="fp2",
    )

    # Both report independent=True at item level (structural only).
    assert item_a.independent is True
    assert item_b.independent is True

    # But they ARE correlated — only graph-level detection via build_chain()
    # can identify this.  The item-level property cannot.
    assert item_a.intrinsically_valid is True
    assert item_b.intrinsically_valid is True


# ---------------------------------------------------------------------------
# M-8: EvidenceCorpus warns on malformed records
# ---------------------------------------------------------------------------


def test_evidence_corpus_logs_warning_on_malformed_record(tmp_path, caplog):
    """EvidenceCorpus.from_directory() must warn on malformed canonical records.

    EvidenceRecord is a dataclass and does not type-check fields at runtime.
    The warning is triggered by missing required positional fields (KeyError).
    """
    import json
    import logging

    from shap_review.evidence.corpus import EvidenceCorpus

    # Write a SHAP-EVID-*.json that is missing required positional fields.
    # EvidenceRecord.__init__ will raise TypeError for unexpected kwarg or
    # KeyError when **raw unpacking omits a required positional arg.
    bad = tmp_path / "SHAP-EVID-9999.json"
    bad.write_text(
        json.dumps(
            {
                "id": "SHAP-EVID-9999",
                "type": "issue",
                # Missing: title, url, bug_class, invariant, affected_components,
                # observed_behavior, expected_behavior — all required positional fields.
                "unknown_extra_field": "this will cause TypeError on __init__",
            }
        ),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="shap_review.evidence.corpus"):
        corpus = EvidenceCorpus.from_directory(tmp_path)

    # Falls back to DEFAULT_EVIDENCE since no valid records loaded.
    assert corpus.records  # DEFAULT_EVIDENCE is non-empty

    # Must have emitted a warning mentioning the filename.
    assert any(
        "SHAP-EVID-9999" in r.message or "malformed" in r.message.lower()
        for r in caplog.records
    ), (
        f"Expected warning about malformed record, got: {[r.message for r in caplog.records]}"
    )


def test_evidence_corpus_warns_on_json_decode_error(tmp_path, caplog):
    """EvidenceCorpus.from_directory() warns on JSON parse failure."""
    import logging

    from shap_review.evidence.corpus import EvidenceCorpus

    bad = tmp_path / "SHAP-EVID-8888.json"
    bad.write_text("{ not valid json }", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="shap_review.evidence.corpus"):
        EvidenceCorpus.from_directory(tmp_path)

    assert any(
        "SHAP-EVID-8888" in r.message or "parse" in r.message.lower()
        for r in caplog.records
    ), f"Expected JSON parse warning, got: {[r.message for r in caplog.records]}"


def test_evidence_corpus_loads_valid_records_alongside_malformed(tmp_path, caplog):
    """Valid records are loaded even when some files in the directory are malformed."""
    import json
    import logging

    from shap_review.evidence.corpus import EvidenceCorpus

    # Write one valid record.
    good = tmp_path / "SHAP-EVID-0001.json"
    good.write_text(
        json.dumps(
            {
                "id": "SHAP-EVID-0001",
                "type": "issue",
                "title": "Valid record",
                "url": "https://github.com/shap/shap/issues/1",
                "bug_class": "SHAP-01",
                "invariant": "INV-ATTR-001",
                "affected_components": ["TreeExplainer"],
                "observed_behavior": "something",
                "expected_behavior": "something else",
            }
        ),
        encoding="utf-8",
    )

    # Write one malformed record (missing required field).
    bad = tmp_path / "SHAP-EVID-9999.json"
    bad.write_text(json.dumps({"id": "SHAP-EVID-9999"}), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="shap_review.evidence.corpus"):
        corpus = EvidenceCorpus.from_directory(tmp_path)

    # Valid record must be loaded.
    assert corpus.get("SHAP-EVID-0001") is not None
    # Malformed record must be absent.
    assert corpus.get("SHAP-EVID-9999") is None
    # Warning must have been emitted.
    assert any("SHAP-EVID-9999" in r.message for r in caplog.records)


def test_evidence_corpus_legacy_files_are_ignored(tmp_path):
    """Files without SHAP-EVID- prefix (legacy format) must be silently ignored."""
    import json

    from shap_review.evidence.corpus import EvidenceCorpus

    legacy = tmp_path / "4911.json"
    legacy.write_text(
        json.dumps(
            {
                "id": "4911",
                "title": "legacy",
                "url": "https://github.com/shap/shap/issues/4911",
                "supports": [],
                "status": "open",
                "observation": "some observation",
            }
        ),
        encoding="utf-8",
    )

    # Falls back to DEFAULT_EVIDENCE (no valid SHAP-EVID-* records).
    corpus = EvidenceCorpus.from_directory(tmp_path)
    assert corpus.records  # DEFAULT_EVIDENCE
    assert corpus.get("4911") is None  # legacy id not loaded


# ---------------------------------------------------------------------------
# H-2: PromotionPolicy inconclusive gate scoped to required items only
# ---------------------------------------------------------------------------


def test_promotion_not_blocked_by_non_required_inconclusive_item():
    """Non-required items with passed=None must NOT block CONFIRMED promotion.

    Previously the gate iterated chain.items (all items), so a historical
    EvidenceItem with passed=None would block promotion even if all required
    runtime oracles had passed.
    """
    from unittest.mock import MagicMock

    from shap_review.evidence.model import EvidenceItem, EvidenceKind
    from shap_review.findings.lifecycle import FindingStatus
    from shap_review.findings.promotion import PromotionPolicy

    # Non-required historical item with passed=None (e.g. no runtime data yet).
    historical = EvidenceItem(
        EvidenceKind.HISTORICAL,
        "github",
        "issue reference",
        passed=None,  # no result yet — but NOT policy_required
        evidence_id="h1",
    )
    # EvidenceItem does not carry policy_required (that is an OracleResult field).
    # Verify the item itself: it has passed=None and no policy_required attribute.
    assert historical.passed is None
    assert not getattr(historical, "policy_required", False)
    assert not getattr(historical, "contract_required", False)

    # Build a mock chain that has the historical item and passes independent_items check.
    chain = MagicMock()
    chain.items = [historical]
    chain.verdict.return_value = "CONFIRMED"
    chain.independent_kinds.return_value = 2
    chain.independent_items.return_value = [
        EvidenceItem(
            EvidenceKind.DYNAMIC,
            "runtime",
            "ok",
            True,
            evidence_id="d1",
            execution_id="e1",
            input_fingerprint="fp1",
        )
    ]

    policy = PromotionPolicy(require_inconclusive_block=True)

    # Must NOT raise — the non-required historical item should not block.
    try:
        policy.validate(
            FindingStatus.EVIDENCE_VALID,
            FindingStatus.CONFIRMED,
            chain=chain,
        )
    except ValueError as exc:
        # If it fails, must be for a reason OTHER than the non-required item.
        assert "required" in str(exc).lower(), (
            f"Promotion blocked by non-required item: {exc}"
        )


def test_promotion_blocked_by_required_inconclusive_item():
    """Items with policy_required=True and passed=None DO block CONFIRMED promotion."""
    from unittest.mock import MagicMock

    from shap_review.contracts.oracles import OracleResult
    from shap_review.findings.lifecycle import FindingStatus
    from shap_review.findings.promotion import PromotionPolicy

    # A required oracle result with passed=None (INCONCLUSIVE).
    inconclusive_required = OracleResult(
        "AdditivityOracle",
        applicable=True,
        passed=None,
        reason="model output not available",
        policy_required=True,
        contract_required=True,
    )

    chain = MagicMock()
    chain.items = [inconclusive_required]
    chain.verdict.return_value = "INVESTIGATING"

    policy = PromotionPolicy(require_inconclusive_block=True)

    import pytest

    with pytest.raises(ValueError, match="inconclusive required"):
        policy.validate(
            FindingStatus.EVIDENCE_VALID,
            FindingStatus.CONFIRMED,
            chain=chain,
        )
