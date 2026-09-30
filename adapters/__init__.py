"""SHAP Review Toolkit — provider adapter layer.

Each adapter (Claude, OpenAI, Gemini) is a thin routing shim that maps
provider-specific tool-call conventions to the shared ``shap_review`` core.
No SHAP analysis logic lives here; all analysis is owned by the core engine.

Security note
-------------
Several toolkit commands (``differential``, ``reproduce``, ``sanitizer``,
``differential-versions``) execute arbitrary Python scripts via subprocess.
Adapters **must only** be called with trusted inputs from controlled
environments.  Do not expose these commands to untrusted end-users without
an additional authentication and path-restriction layer.

Dynamic evidence note
---------------------
``ReviewEngine.analyze()`` runs a bounded TreeExplainer campaign against the
*installed* SHAP runtime, not the repository under review.  The
``runtime-bridge.json`` artifact records campaign metadata; dynamic evidence
is attached to candidates only when explicit candidate fingerprint correlation
exists (which the generic bridge campaign does not provide).  Treat
``runtime-bridge`` output as diagnostic metadata, not as candidate enrichment,
until candidate-specific campaigns are implemented.
"""

from __future__ import annotations

from .base import BaseReviewAdapter
from .claude.adapter import ClaudeAdapter
from .gemini.adapter import GeminiAdapter
from .interface import CORE_COMMANDS, AdapterCapabilities, AdapterRequest, ReviewAdapter
from .openai.adapter import OpenAIAdapter

__all__ = [
    "CORE_COMMANDS",
    "AdapterCapabilities",
    "AdapterRequest",
    "BaseReviewAdapter",
    "ClaudeAdapter",
    "GeminiAdapter",
    "OpenAIAdapter",
    "ReviewAdapter",
]
