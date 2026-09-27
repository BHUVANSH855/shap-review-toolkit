from pathlib import Path

from shap_review.engine import ReviewEngine

FIX = Path(__file__).parent / "fixtures"


def run(path):
    return ReviewEngine().analyze(path, path / ".out")


def test_issue_4911_signal_is_evidence_linked():
    cs = run(FIX / "shap_4911_nullable_dtype")
    assert any(
        c.bug_class == "SHAP-05"
        and any(e.source == "SHAP-EVID-4911" for e in c.evidence)
        for c in cs
    )


def test_issue_5098_signal_is_evidence_linked():
    cs = run(FIX / "shap_5098_model_output")
    assert any(
        c.bug_class == "SHAP-02"
        and any(e.source == "SHAP-EVID-5098" for e in c.evidence)
        for c in cs
    )


def test_issue_4869_is_test_gap_not_defect():
    cs = run(FIX / "shap_4869_interaction_gap")
    assert any(c.bug_class == "SHAP-01" and not c.validation_required for c in cs)


def test_evidence_corpus_has_canonical_primary_records():
    from pathlib import Path

    from shap_review.evidence import EvidenceCorpus

    corpus = EvidenceCorpus.from_directory(
        Path(__file__).parents[1] / "data/evidence/issues"
    )
    assert len(corpus.records) >= 7
    assert not corpus.validate()
