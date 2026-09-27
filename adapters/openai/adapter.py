from adapters.base import BaseReviewAdapter
from adapters.interface import AdapterCapabilities
from shap_review.version import VERSION


class OpenAIAdapter(BaseReviewAdapter):
    provider = "openai"
    version = VERSION
    capabilities = AdapterCapabilities(provider=provider, version=VERSION)
