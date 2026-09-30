"""SHAP Review Toolkit — public entry points.

Core classes and functions are exported here so callers do not need to
navigate sub-packages.

Security note
-------------
Commands ``differential``, ``reproduce``, ``sanitizer``, and
``differential-versions`` execute arbitrary Python scripts via subprocess.
Only use these with trusted, controlled input paths.

Dynamic evidence note
---------------------
``ReviewEngine.analyze()`` runs a bounded TreeExplainer campaign against the
*installed* SHAP runtime, not the repository under review.  The
``runtime-bridge.json`` artifact records campaign anomalies.  Those anomalies
do **not** currently enrich candidate evidence chains because the generic
bridge campaign does not set per-candidate fingerprints.
"""

from __future__ import annotations

# Contracts / oracles
from .contracts.oracles import (
    AdditivityOracle,
    ExpectedValueOracle,
    InputMutationOracle,
    InteractionOracle,
    OracleResult,
    OracleStatus,
    OutputSpaceOracle,
    ShapeOracle,
    SHAPSemanticOracle,
)
from .contracts.shap_contract import SHAPContract, dtype_tolerance, validate_contract
from .contracts.tensor import SHAPAxisSpec, SHAPSemanticTensor

# Evidence model
from .evidence.model import EvidenceChain, EvidenceItem, EvidenceKind, EvidenceOrigin

# Finding lifecycle
from .findings.lifecycle import transition
from .findings.promotion import PromotionPolicy

# Reproduction
from .reproduction.runner import run_script

# Types
from .types import Candidate, Finding, FindingStatus

# Version
from .version import CAPABILITIES, SCHEMA_VERSION, VERSION

__version__ = VERSION

__all__ = [
    # Contracts
    "AdditivityOracle",
    "ExpectedValueOracle",
    "InputMutationOracle",
    "InteractionOracle",
    "OracleResult",
    "OracleStatus",
    "OutputSpaceOracle",
    "SHAPAxisSpec",
    "SHAPContract",
    "SHAPSemanticOracle",
    "SHAPSemanticTensor",
    "ShapeOracle",
    "dtype_tolerance",
    "validate_contract",
    # Evidence
    "EvidenceChain",
    "EvidenceItem",
    "EvidenceKind",
    "EvidenceOrigin",
    # Findings
    "FindingStatus",
    "PromotionPolicy",
    "transition",
    # Types
    "Candidate",
    "Finding",
    # Reproduction
    "run_script",
    # Version
    "CAPABILITIES",
    "SCHEMA_VERSION",
    "VERSION",
    "__version__",
]
