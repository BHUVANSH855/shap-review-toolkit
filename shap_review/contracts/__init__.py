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
from .tensor import SHAPAxisSpec, infer_axis_spec, reduce_contributions

__all__ = [
    "ORACLE_REGISTRY",
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
    "ShapeOracle",
    "compare_contracts",
    "infer_axis_spec",
    "reduce_contributions",
    "validate_contract",
    "validate_oracle_result",
]