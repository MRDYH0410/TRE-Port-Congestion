"""Shared nested kernel used by MPC, training, execution, replay, and clearance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .actions import Action, ActionProjector, ProjectionResult
from .behavior import BehaviorProblem, EquilibriumResult, RCMSASolver
from .state import ModelState
from .transition import ExogenousRealization, TaggedTransition, TransitionResult


class BehaviorProblemFactory(Protocol):
    def __call__(
        self, state: ModelState, action: Action, realization: ExogenousRealization
    ) -> BehaviorProblem: ...


@dataclass(frozen=True)
class KernelResult:
    action: Action
    projection: ProjectionResult | None
    equilibrium: EquilibriumResult
    transition: TransitionResult
    input_state: ModelState | None = None


class ModelKernel:
    """One implementation boundary for Modules 4 and 5."""

    def __init__(
        self,
        *,
        behavior_factory: BehaviorProblemFactory,
        equilibrium_solver: RCMSASolver,
        transition: TaggedTransition,
        projector: ActionProjector | None = None,
    ) -> None:
        self.behavior_factory = behavior_factory
        self.equilibrium_solver = equilibrium_solver
        self.transition = transition
        self.projector = projector

    def execute(
        self,
        *,
        state: ModelState,
        action: Action,
        realization: ExogenousRealization,
        projection: ProjectionResult | None = None,
    ) -> KernelResult:
        input_state = state.clone()
        problem = self.behavior_factory(state, action, realization)
        equilibrium = self.equilibrium_solver.solve(
            problem, previous_shares=state.previous_shares
        )
        transition = self.transition.step(
            state=state,
            action=action,
            equilibrium=equilibrium,
            realization=realization,
        )
        return KernelResult(action, projection, equilibrium, transition, input_state)

    def execute_raw(
        self,
        *,
        state: ModelState,
        raw_action: Action,
        realization: ExogenousRealization,
    ) -> KernelResult:
        if self.projector is None:
            raise RuntimeError("Raw action execution requires the shared hard-feasibility projector")
        projection = self.projector.project(raw_action, state)
        return self.execute(
            state=state,
            action=projection.action,
            realization=realization,
            projection=projection,
        )
