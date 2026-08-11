"""Explicit post-decision clearance with right-censoring and terminal mass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .actions import Action, Block
from .engine import KernelResult, ModelKernel
from .errors import ContractError
from .loss import TerminalMassCorrection
from .state import ModelState
from .transition import ExogenousRealization


class FrozenRecoveryRule(Protocol):
    def action(self, state: ModelState) -> Action: ...

    def realization(self, state: ModelState) -> ExogenousRealization: ...


@dataclass(frozen=True)
class ClearanceResult:
    final_state: ModelState
    weeks: int
    cleared: bool
    right_censored: bool
    operational_loss: float
    terminal_correction: float
    transitions: tuple[KernelResult, ...]

    @property
    def total_loss(self) -> float:
        return self.operational_loss + self.terminal_correction


class ClearanceRunner:
    def __init__(
        self,
        *,
        kernel: ModelKernel,
        recovery_rule: FrozenRecoveryRule,
        terminal_cost: TerminalMassCorrection,
        maximum_weeks: int,
        empty_tolerance: float,
    ) -> None:
        if maximum_weeks < 0 or empty_tolerance < 0:
            raise ContractError("Clearance cap and tolerance must be nonnegative")
        self.kernel = kernel
        self.recovery_rule = recovery_rule
        self.terminal_cost = terminal_cost
        self.maximum_weeks = maximum_weeks
        self.empty_tolerance = empty_tolerance

    def is_empty(self, state: ModelState) -> bool:
        return state.cargo_mass() <= self.empty_tolerance

    def run(self, state: ModelState) -> ClearanceResult:
        current = state.clone()
        records: list[KernelResult] = []
        loss = 0.0
        for _ in range(self.maximum_weeks):
            if self.is_empty(current):
                break
            action = self.recovery_rule.action(current)
            realization = self.recovery_rule.realization(current)
            optimized = {
                key: value
                for key, value in action.values.items()
                if key.block != Block.RELEASE and abs(float(value)) > 1e-12
            }
            if optimized:
                raise ContractError(
                    "Clearance permits only the frozen waiting-release schedule; "
                    f"new optimized actions were supplied: {optimized}"
                )
            blocked = {}
            for cargo_class, demand in realization.gulf_demand.items():
                if cargo_class not in realization.serviceable_share:
                    raise ContractError(
                        f"Clearance realization lacks serviceability for {cargo_class}"
                    )
                mass = (1.0 - realization.serviceable_share[cargo_class]) * demand
                if mass > 1e-12:
                    blocked[cargo_class] = mass
            if blocked:
                raise ContractError(
                    f"New blocked demand is prohibited during clearance: {blocked}"
                )
            result = self.kernel.execute(
                state=current,
                action=action,
                realization=realization,
            )
            records.append(result)
            loss += result.transition.loss.total
            current = result.transition.next_state
        cleared = self.is_empty(current)
        correction = 0.0 if cleared else self.terminal_cost.compute(current)
        return ClearanceResult(
            final_state=current,
            weeks=len(records),
            cleared=cleared,
            right_censored=not cleared,
            operational_loss=loss,
            terminal_correction=correction,
            transitions=tuple(records),
        )
