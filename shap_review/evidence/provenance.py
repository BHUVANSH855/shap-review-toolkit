from __future__ import annotations

from shap_review.types import FindingClassification


def classify_finding(
    *,
    historical_issue: str | None = None,
    reproduced: bool = False,
    discovered_by_current_analysis: bool = True,
    regression: bool = False,
):
    if historical_issue and reproduced and not discovered_by_current_analysis:
        return FindingClassification.HISTORICAL_REPRODUCTION
    if historical_issue and reproduced:
        return FindingClassification.KNOWN_ISSUE_REPRODUCED
    if historical_issue:
        return FindingClassification.HISTORICAL_CANDIDATE
    if regression:
        return FindingClassification.REGRESSION_CANDIDATE
    if reproduced and discovered_by_current_analysis:
        return FindingClassification.NOVEL_REPRODUCTION
    return FindingClassification.NOVEL_CANDIDATE
