from __future__ import annotations

from dataclasses import dataclass

from shap_review.types import FindingStatus


@dataclass(frozen=True)
class PromotionPolicy:
    require_dynamic_for_reproduced: bool = True
    require_independent_for_evidence_valid: bool = True
    require_runtime_or_differential_for_confirmed: bool = True
    require_reproducer_for_reported: bool = True
    require_inconclusive_block: bool = True

    def validate(
        self, current: FindingStatus, target: FindingStatus, chain=None, reproducer=None
    ) -> None:
        from .lifecycle import transition

        transition(current, target)

        if self.require_inconclusive_block and target in {
            FindingStatus.CONFIRMED,
            FindingStatus.REPORTED,
        }:
            if chain is None:
                raise ValueError("promotion blocked: no evidence chain supplied")

            # Only gate on items that are REQUIRED (policy_required or
            # contract_required). Non-required items (e.g. historical issue
            # refs with passed=None) must not block promotion when all required
            # runtime oracles have passed.
            inconclusive_required = [
                i
                for i in chain.items
                if getattr(i, "passed", None) is None
                and (
                    getattr(i, "policy_required", False)
                    or getattr(i, "contract_required", False)
                )
            ]
            if inconclusive_required:
                names = [getattr(i, "name", repr(i)) for i in inconclusive_required]
                raise ValueError(
                    f"promotion blocked by inconclusive required oracle results: {names}"
                )

        if (
            target == FindingStatus.REPRODUCED
            and self.require_dynamic_for_reproduced
            and (
                chain is None
                or not any(
                    i.passed is True and i.kind.value in {"dynamic", "reproduction"}
                    for i in chain.independent_items()
                )
            )
        ):
            raise ValueError(
                "REPRODUCED requires positive independent dynamic/reproduction evidence"
            )

        if (
            target == FindingStatus.EVIDENCE_VALID
            and self.require_independent_for_evidence_valid
            and (chain is None or chain.independent_kinds() < 2)
        ):
            raise ValueError(
                "EVIDENCE_VALID requires at least two independent evidence kinds"
            )

        if (
            target == FindingStatus.CONFIRMED
            and self.require_runtime_or_differential_for_confirmed
        ):
            qualifying = (
                [
                    i
                    for i in chain.independent_items()
                    if i.passed is True
                    and (
                        i.kind.value in {"dynamic", "reproduction"}
                        or (
                            i.kind.value == "sanitizer"
                            and i.details.get("finding") is True
                        )
                        or (
                            # Differential evidence qualifies ONLY when the reference
                            # correctness is established — not merely when two executions
                            # agree. reference_correctness=UNKNOWN means we only know
                            # they matched, not that either is correct.
                            i.kind.value == "differential"
                            and i.details.get("reference_correctness")
                            not in {None, "UNKNOWN"}
                        )
                    )
                ]
                if chain is not None
                else []
            )
            if not qualifying:
                raise ValueError(
                    "CONFIRMED requires positive runtime, reproduction, or sanitizer evidence, "
                    "or differential evidence with established reference correctness "
                    "(differential agreement with reference_correctness=UNKNOWN is not sufficient)"
                )

        if (
            target == FindingStatus.REPORTED
            and self.require_reproducer_for_reported
            and not reproducer
        ):
            raise ValueError("REPORTED requires a maintainer-usable reproducer")
