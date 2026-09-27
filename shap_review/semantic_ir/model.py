from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    file: str
    line: int
    qualified_name: str = ""


@dataclass(frozen=True)
class CallSite:
    callee: str
    file: str
    line: int
    enclosing_symbol: str = ""
    arguments: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValueFlow:
    source: str
    target: str
    file: str
    line: int
    transform: str = ""
    properties: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValueProperty:
    value: str
    property: str
    file: str
    line: int
    source: str = ""


@dataclass(frozen=True)
class NativeBoundary:
    technology: str
    file: str
    line: int
    operation: str
    python_controlled: bool = False


@dataclass(frozen=True)
class TestCoverage:
    symbol: str
    file: str
    line: int
    assertions: tuple[str, ...] = ()


@dataclass
class AnalysisIR:
    root: str
    symbols: list[Symbol] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    flows: list[ValueFlow] = field(default_factory=list)
    properties: list[ValueProperty] = field(default_factory=list)
    boundaries: list[NativeBoundary] = field(default_factory=list)
    tests: list[TestCoverage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
