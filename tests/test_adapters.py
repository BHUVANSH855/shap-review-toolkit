from adapters import ClaudeAdapter, GeminiAdapter, OpenAIAdapter


def test_all_frontends_share_capability_contract():
    adapters = [
        ClaudeAdapter(),
        OpenAIAdapter(),
        GeminiAdapter(),
    ]

    capabilities = [
        adapter.invoke("capabilities", {})
        for adapter in adapters
    ]

    assert {capability["provider"] for capability in capabilities} == {
        "claude",
        "openai",
        "gemini",
    }

    assert all(
        "differential" in capability["commands"]
        for capability in capabilities
    )
    assert all(
        "sanitizer" in capability["commands"]
        for capability in capabilities
    )

    assert (
        capabilities[0]["commands"]
        == capabilities[1]["commands"]
        == capabilities[2]["commands"]
    )


def test_backend_adapter_reports_actual_failure_stage():
    from shap_review.backends.adapter import MatrixBackendAdapter
    from shap_review.fuzzing.backend_matrix import BackendSpec

    class Model:
        pass

    adapter = MatrixBackendAdapter(
        BackendSpec("x", "x", (), True, "1"),
        Model(),
    )
    adapter.prepare = lambda **kwargs: {}

    def failing_fit(context):
        raise RuntimeError("fit broke")

    adapter.fit = failing_fit

    result = adapter.execute_case()

    assert result["stage"] == "fit"
    assert result["execution_reason"] == "BACKEND_ERROR"


def test_backend_rejects_invalid_regression_output_contract():
    from shap_review.backends.adapter import MatrixBackendAdapter
    from shap_review.fuzzing.backend_matrix import BackendSpec

    class Model:
        pass

    adapter = MatrixBackendAdapter(
        BackendSpec("x", "x", (), True, "1"),
        Model(),
    )

    supported, reason = adapter.supports_case(
        classification=False,
        model_output="log_loss",
        interaction=False,
    )

    assert supported is False
    assert "regression" in reason


def test_backend_failure_stage_is_preserved():
    from shap_review.backends.adapter import MatrixBackendAdapter
    from shap_review.fuzzing.backend_matrix import BackendSpec

    class Model:
        pass

    adapter = MatrixBackendAdapter(
        BackendSpec("x", "x", (), True, "1"),
        Model(),
    )
    adapter.prepare = lambda **kwargs: {}

    def failing_fit(context):
        raise RuntimeError("fit broke")

    adapter.fit = failing_fit

    result = adapter.execute_case()

    assert result["stage"] == "fit"
    assert result["execution_reason"] == "BACKEND_ERROR"
    assert result["status"] == "BACKEND_ERROR"