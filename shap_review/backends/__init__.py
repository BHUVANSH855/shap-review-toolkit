from shap_review.fuzzing.backend_matrix import (
    DEFAULT_BACKENDS,
    BackendSpec,
    discover_backends,
    matrix_dimensions,
)

from .adapter import BackendAdapter, BackendExecution, MatrixBackendAdapter

__all__ = [
    "DEFAULT_BACKENDS",
    "BackendAdapter",
    "BackendExecution",
    "BackendSpec",
    "MatrixBackendAdapter",
    "discover_backends",
    "matrix_dimensions",
]
