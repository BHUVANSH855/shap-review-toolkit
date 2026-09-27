from .oracles import (
    ORACLE_REGISTRY,
    AdditivityOracle,
    ExpectedValueOracle,
    InputMutationOracle,
    InteractionOracle,
    OracleResult,
    OracleStatus,
    OutputSpaceOracle,
    ShapeOracle,
    SHAPSemanticOracle,
    validate_oracle_result,
)
from .shap_contract import SHAPContract, compare_contracts, validate_contract

__all__ = [
    "ORACLE_REGISTRY",
    "AdditivityOracle",
    "ExpectedValueOracle",
    "InputMutationOracle",
    "InteractionOracle",
    "OracleResult",
    "OracleStatus",
    "OutputSpaceOracle",
    "SHAPContract",
    "SHAPSemanticOracle",
    "ShapeOracle",
    "compare_contracts",
    "validate_contract",
    "validate_oracle_result",
]

from .tensor import SHAPAxisSpec, infer_axis_spec, reduce_contributions
