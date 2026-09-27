import pytest

from shap_review.findings.lifecycle import transition
from shap_review.types import FindingStatus


def test_valid_transition():
    assert (
        transition(FindingStatus.CANDIDATE, FindingStatus.INVESTIGATING)
        == FindingStatus.INVESTIGATING
    )


def test_invalid_transition():
    with pytest.raises(ValueError):
        transition(FindingStatus.CANDIDATE, FindingStatus.CONFIRMED)
