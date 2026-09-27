import pytest

pytest.importorskip("shap")
pytest.importorskip("sklearn")
pytest.importorskip("pandas")
from shap_review.fuzzing.engine import TreeExplainerFuzzer


def test_fuzzer_executes_cases():
    result = TreeExplainerFuzzer(seed=11).run(iterations=3)
    assert result["executed_cases"] == 3
    assert result["validated_cases"] >= 1


def test_fuzzer_exercises_real_dimensions_without_false_runtime_failures():
    from shap_review.fuzzing.engine import TreeExplainerFuzzer

    r = TreeExplainerFuzzer(seed=123).run(iterations=25)
    assert r["executed_cases"] == 25
    assert r["validated_cases"] == 25
    assert r["failures"] == 0
    representations = {x["case"]["representation"] for x in r["results"]}
    assert {"ndarray", "dataframe"} <= representations
