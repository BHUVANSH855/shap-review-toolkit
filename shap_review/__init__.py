"""Public package entry points for the SHAP Review Toolkit."""

from .reproduction.runner import run_script
from .version import CAPABILITIES, SCHEMA_VERSION, VERSION

__version__ = VERSION

__all__ = [
    "CAPABILITIES",
    "SCHEMA_VERSION",
    "VERSION",
    "__version__",
    "run_script",
]