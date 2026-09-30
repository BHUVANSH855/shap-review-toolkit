"""Claude provider adapter for the SHAP Review Toolkit.

Maps Claude plugin tool-call conventions to the shared ``shap_review`` core.
No SHAP analysis logic lives here.

Usage
-----
::

    from adapters import ClaudeAdapter

    adapter = ClaudeAdapter()

    # Route a tool call received from the Claude plugin system:
    result = adapter.invoke("analyze", {"root": "/path/to/shap"})

    # The result envelope always contains provider/version/schema_version/command.
    # On success it also contains ``result``; on error it contains ``error``.

Security
--------
Commands ``differential``, ``reproduce``, ``sanitizer``, and
``differential-versions`` execute arbitrary Python scripts via subprocess.
Only invoke these commands with trusted, controlled input paths.  Never
expose them to untrusted end-users without an additional restriction layer.

Dynamic evidence note
---------------------
The ``analyze`` command runs a bounded TreeExplainer campaign against the
*installed* SHAP runtime, not the repository under review.  The
``runtime-bridge.json`` artifact records anomalies; those anomalies do **not**
currently enrich candidate evidence chains because the generic bridge campaign
does not set per-candidate fingerprints.  Treat bridge output as diagnostic
metadata only.
"""

from __future__ import annotations

from adapters.base import BaseReviewAdapter
from adapters.interface import AdapterCapabilities
from shap_review.version import CAPABILITIES as CORE_COMMANDS
from shap_review.version import VERSION


class ClaudeAdapter(BaseReviewAdapter):
    """Thin Claude adapter.  All routing logic is in ``BaseReviewAdapter``."""

    provider = "claude"

    def __init__(self, **kwargs) -> None:  # type: ignore[override]
        super().__init__(**kwargs)
        # Reconstruct capabilities with the correct provider string.
        # The base class already does this in __init__, but we re-assign here
        # to make the provider override explicit and visible to type checkers.
        self.capabilities = AdapterCapabilities(
            provider=self.provider,
            version=VERSION,
            commands=CORE_COMMANDS,
        )
