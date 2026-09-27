from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MinimizationResult:
    original: int
    minimized: int
    changed: bool


def shrink_int(value: int, predicate, minimum: int = 1) -> MinimizationResult:
    original = value
    cur = value
    while cur > minimum:
        trial = max(minimum, cur // 2)
        if predicate(trial):
            cur = trial
        else:
            break
    return MinimizationResult(original, cur, cur != original)
