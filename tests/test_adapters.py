from adapters import ClaudeAdapter, GeminiAdapter, OpenAIAdapter


def test_all_frontends_share_capability_contract():
    adapters = [ClaudeAdapter(), OpenAIAdapter(), GeminiAdapter()]
    capabilities = [a.invoke("capabilities", {}) for a in adapters]
    assert {c["provider"] for c in capabilities} == {"claude", "openai", "gemini"}
    assert all("differential" in c["commands"] for c in capabilities)
    assert all("sanitizer" in c["commands"] for c in capabilities)
    assert (
        capabilities[0]["commands"]
        == capabilities[1]["commands"]
        == capabilities[2]["commands"]
    )
