"""Backend-neutral MG-PC-SAC contracts and integrated audit records.

This core module owns the Chapter 4 method contract: projected BC targets,
preprojection actor density and entropy, twin reward critics, a constraint
critic/dual, the shared hard-feasibility projector, and the exact nested
kernel.  A concrete tensor/autodiff implementation is intentionally injected
through the backend protocols.  That implementation boundary does not
authorize unregistered target-network or Polyak mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .actions import Action, ActionProjector
from .control import MPCCandidate, MPCResult, ProjectedStochasticMPC
from .engine import KernelResult, ModelKernel
from .errors import ContractError
from .scenarios import EventPath, RevealedEventHistory, ScenarioBundle
from .state import ModelState
from .transition import ExogenousRealization


@dataclass(frozen=True)
class TeacherExample:
    state: ModelState
    target_action: Action
    candidate_id: str
    objective: float


class BehaviorCloningBackend(Protocol):
    """A concrete backend may be PyTorch, JAX, or another differentiable stack."""

    def train_projected_batch(
        self,
        states: Sequence[ModelState],
        targets: Sequence[Action],
        projector: ActionProjector,
    ) -> float: ...

    def raw_action(self, state: ModelState) -> Action: ...

    def freeze(self) -> str: ...


class BehaviorCloningTrainer:
    def __init__(self, backend: BehaviorCloningBackend, projector: ActionProjector) -> None:
        self.backend = backend
        self.projector = projector

    def fit(
        self,
        examples: Sequence[TeacherExample],
        *,
        batch_size: int,
        epochs: int,
    ) -> tuple[float, ...]:
        if batch_size <= 0 or epochs <= 0 or not examples:
            raise ContractError("BC training needs examples and positive loop settings")
        losses: list[float] = []
        for _ in range(epochs):
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                losses.append(
                    float(
                        self.backend.train_projected_batch(
                            [item.state for item in batch],
                            [item.target_action for item in batch],
                            self.projector,
                        )
                    )
                )
        return tuple(losses)


@dataclass(frozen=True)
class SACActorLossInput:
    log_latent_probability: np.ndarray
    reward_q1: np.ndarray
    reward_q2: np.ndarray
    constraint_q: np.ndarray
    entropy_temperature: float
    constraint_dual: float


def constrained_sac_actor_loss(data: SACActorLossInput) -> float:
    """Equation (constrained-sac-loss) on preprojection latent log density."""

    arrays = (
        np.asarray(data.log_latent_probability, dtype=float),
        np.asarray(data.reward_q1, dtype=float),
        np.asarray(data.reward_q2, dtype=float),
        np.asarray(data.constraint_q, dtype=float),
    )
    if len({array.shape for array in arrays}) != 1:
        raise ContractError("All SAC actor-loss samples need the same shape")
    if data.entropy_temperature < 0 or data.constraint_dual < 0:
        raise ContractError("SAC entropy temperature and constraint dual must be nonnegative")
    terms = (
        data.entropy_temperature * arrays[0]
        - np.minimum(arrays[1], arrays[2])
        + data.constraint_dual * arrays[3]
    )
    return float(np.mean(terms))


@dataclass(frozen=True)
class ReplayTransition:
    state: ModelState
    raw_action: Action
    feasible_action: Action
    reward: float
    soft_constraint_cost: float
    next_state: ModelState
    terminal: bool
    audit: Any


class ConstrainedSACBackend(Protocol):
    def sample_raw_action(self, state: ModelState) -> Action: ...

    def update(self, replay: Sequence[ReplayTransition]) -> Mapping[str, float]: ...

    def freeze(self) -> str: ...


class TrainingPeriodPreparer(Protocol):
    """Run released-information/scenario preparation before each training decision."""

    def __call__(
        self, state: ModelState, offset: int, history: RevealedEventHistory
    ) -> ModelState: ...


@dataclass
class ReplayBuffer:
    capacity: int
    records: list[ReplayTransition] = field(default_factory=list)

    def append(self, record: ReplayTransition) -> None:
        if self.capacity <= 0:
            raise ContractError("Replay capacity must be positive")
        self.records.append(record)
        if len(self.records) > self.capacity:
            del self.records[0 : len(self.records) - self.capacity]


@dataclass(frozen=True)
class SACEpisodeResult:
    final_state: ModelState
    total_reward: float
    transitions: tuple[KernelResult, ...]
    update_metrics: tuple[Mapping[str, float], ...]


class SACEpisodeRunner:
    """Training paths use the same projector, equilibrium, and transition as execution."""

    def __init__(
        self,
        *,
        backend: ConstrainedSACBackend,
        kernel: ModelKernel,
        projector: ActionProjector,
        replay: ReplayBuffer,
        period_preparer: TrainingPeriodPreparer,
    ) -> None:
        self.backend = backend
        self.kernel = kernel
        self.projector = projector
        self.replay = replay
        self.period_preparer = period_preparer

    def run(
        self,
        initial_state: ModelState,
        path: EventPath,
        *,
        soft_constraint_cost: Callable[[KernelResult], float],
        update_after_each_step: bool = True,
    ) -> SACEpisodeResult:
        if not path.payload:
            raise ContractError("A SAC training path needs complete realization payloads")
        state = initial_state.clone()
        total_reward = 0.0
        transitions: list[KernelResult] = []
        metrics: list[Mapping[str, float]] = []
        for offset, payload in enumerate(path.payload):
            if not isinstance(payload, ExogenousRealization):
                raise ContractError("SAC path payloads must be ExogenousRealization objects")
            state = self.period_preparer(state, offset, path.revealed_before(offset))
            raw = self.backend.sample_raw_action(state)
            projection = self.projector.project(raw, state)
            result = self.kernel.execute(
                state=state,
                action=projection.action,
                realization=payload,
                projection=projection,
            )
            reward = -result.transition.loss.total
            next_state = result.transition.next_state
            record = ReplayTransition(
                state=state,
                raw_action=raw,
                feasible_action=projection.action,
                reward=reward,
                soft_constraint_cost=float(soft_constraint_cost(result)),
                next_state=next_state,
                terminal=offset == len(path.payload) - 1,
                audit=result.transition.audit,
            )
            self.replay.append(record)
            if update_after_each_step:
                metrics.append(dict(self.backend.update(tuple(self.replay.records))))
            transitions.append(result)
            total_reward += reward
            state = next_state
        return SACEpisodeResult(state, total_reward, tuple(transitions), tuple(metrics))


def generate_teacher_example(
    *,
    mpc: ProjectedStochasticMPC,
    state: ModelState,
    bundle: ScenarioBundle,
    candidates: Sequence[MPCCandidate],
) -> TeacherExample:
    result: MPCResult = mpc.solve(state=state, bundle=bundle, candidates=candidates)
    return TeacherExample(state.clone(), result.action, result.candidate_id, result.objective)
