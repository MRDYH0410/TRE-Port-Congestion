"""Projected stochastic MPC and two-proposal deployment selection (Module 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from .actions import Action, ActionProjector, ProjectionResult
from .behavior import EquilibriumResult
from .engine import KernelResult, ModelKernel
from .errors import AuditFailure, ContractError, NumericalFailure
from .loss import LossBreakdown
from .scenarios import RevealedEventHistory, ScenarioBundle
from .state import ModelState
from .transition import ExogenousRealization, TransitionAudit


class RawPolicy(Protocol):
    def __call__(self, state: ModelState) -> Action: ...


ContinuationPolicy = Callable[[ModelState, int, RevealedEventHistory], Action]
TerminalValue = Callable[[ModelState], float]


@dataclass(frozen=True)
class MPCCandidate:
    candidate_id: str
    first_raw_action: Action
    continuation: ContinuationPolicy


@dataclass(frozen=True)
class ModuleCertificate:
    """Auditable Modules 3--5 record for one candidate/scenario/period."""

    candidate_id: str
    scenario_id: str
    scenario_weight: float
    offset: int
    input_period: int
    output_period: int
    raw_action: Action
    projected_action: Action
    projection: ProjectionResult
    equilibrium: EquilibriumResult
    transition_audit: TransitionAudit
    period_loss: LossBreakdown


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    first_action: Action | None
    objective: float
    path_losses: tuple[float, ...]
    valid: bool
    failure: str | None
    scenario_ids: tuple[str, ...] = ()
    scenario_weights: tuple[float, ...] = ()
    terminal_losses: tuple[float, ...] = ()
    module_certificates: tuple[ModuleCertificate, ...] = ()
    failure_scenario_id: str | None = None
    failure_offset: int | None = None


@dataclass(frozen=True)
class SelectionLog:
    considered_candidate_ids: tuple[str, ...]
    valid_candidate_ids: tuple[str, ...]
    failed_candidate_ids: tuple[str, ...]
    objective_by_candidate: tuple[tuple[str, float], ...]
    failure_by_candidate: tuple[tuple[str, str | None], ...]
    selected_candidate_id: str
    used_fallback: bool


def _selection_log(
    evaluations: Sequence[CandidateEvaluation],
    *,
    selected_candidate_id: str,
    used_fallback: bool,
) -> SelectionLog:
    return SelectionLog(
        considered_candidate_ids=tuple(item.candidate_id for item in evaluations),
        valid_candidate_ids=tuple(item.candidate_id for item in evaluations if item.valid),
        failed_candidate_ids=tuple(item.candidate_id for item in evaluations if not item.valid),
        objective_by_candidate=tuple(
            (item.candidate_id, float(item.objective)) for item in evaluations
        ),
        failure_by_candidate=tuple(
            (item.candidate_id, item.failure) for item in evaluations
        ),
        selected_candidate_id=selected_candidate_id,
        used_fallback=used_fallback,
    )


@dataclass(frozen=True)
class MPCResult:
    action: Action
    candidate_id: str
    objective: float
    evaluations: tuple[CandidateEvaluation, ...]
    selection_log: SelectionLog


class ProjectedStochasticMPC:
    """Finite reproducible candidate lattice with common scenario paths."""

    def __init__(
        self,
        *,
        kernel: ModelKernel,
        projector: ActionProjector,
        lookahead: int,
        terminal_value: TerminalValue,
    ) -> None:
        if lookahead <= 0:
            raise ContractError("MPC lookahead must be positive")
        self.kernel = kernel
        self.projector = projector
        self.lookahead = lookahead
        self.terminal_value = terminal_value

    def evaluate(
        self,
        state: ModelState,
        bundle: ScenarioBundle,
        candidate: MPCCandidate,
    ) -> CandidateEvaluation:
        path_losses: list[float] = []
        terminal_losses: list[float] = []
        certificates: list[ModuleCertificate] = []
        first_action: Action | None = None
        active_scenario_id: str | None = None
        active_offset: int | None = None
        try:
            weights = tuple(float(value) for value in bundle.active_weights)
            for path_index, path in enumerate(bundle.paths):
                active_scenario_id = path.path_id
                if len(path.payload) < self.lookahead:
                    raise ContractError(
                        f"Scenario {path.path_id} has no realization payload for the MPC horizon"
                    )
                simulated = state.clone()
                cumulative = 0.0
                for offset in range(self.lookahead):
                    active_offset = offset
                    raw = (
                        candidate.first_raw_action
                        if offset == 0
                        else candidate.continuation(
                            simulated,
                            offset,
                            path.revealed_before(offset),
                        )
                    )
                    projection = self.projector.project(raw, simulated)
                    if offset == 0:
                        if first_action is None:
                            first_action = projection.action
                        elif first_action.values != projection.action.values:
                            raise ContractError(
                                "The current MPC action changed across common scenarios"
                            )
                    realization = path.payload[offset]
                    if not isinstance(realization, ExogenousRealization):
                        raise ContractError("MPC path payloads must be ExogenousRealization objects")
                    result = self.kernel.execute(
                        state=simulated,
                        action=projection.action,
                        realization=realization,
                        projection=projection,
                    )
                    cumulative += result.transition.loss.total
                    certificates.append(
                        ModuleCertificate(
                            candidate_id=candidate.candidate_id,
                            scenario_id=path.path_id,
                            scenario_weight=weights[path_index],
                            offset=offset,
                            input_period=int(simulated.period),
                            output_period=int(result.transition.next_state.period),
                            raw_action=raw,
                            projected_action=projection.action,
                            projection=projection,
                            equilibrium=result.equilibrium,
                            transition_audit=result.transition.audit,
                            period_loss=result.transition.loss,
                        )
                    )
                    simulated = result.transition.next_state
                terminal_loss = float(self.terminal_value(simulated))
                terminal_losses.append(terminal_loss)
                cumulative += terminal_loss
                path_losses.append(cumulative)
            objective = float(np.dot(bundle.active_weights, np.asarray(path_losses)))
            return CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                first_action=first_action,
                objective=objective,
                path_losses=tuple(path_losses),
                valid=True,
                failure=None,
                scenario_ids=tuple(path.path_id for path in bundle.paths),
                scenario_weights=weights,
                terminal_losses=tuple(terminal_losses),
                module_certificates=tuple(certificates),
            )
        except (ContractError, NumericalFailure, AuditFailure) as exc:
            return CandidateEvaluation(
                candidate_id=candidate.candidate_id,
                first_action=first_action,
                objective=float("inf"),
                path_losses=tuple(path_losses),
                valid=False,
                failure=str(exc),
                scenario_ids=tuple(path.path_id for path in bundle.paths),
                scenario_weights=tuple(float(value) for value in bundle.active_weights),
                terminal_losses=tuple(terminal_losses),
                module_certificates=tuple(certificates),
                failure_scenario_id=active_scenario_id,
                failure_offset=active_offset,
            )

    def solve(
        self,
        *,
        state: ModelState,
        bundle: ScenarioBundle,
        candidates: Sequence[MPCCandidate],
    ) -> MPCResult:
        if not candidates:
            raise ContractError("The finite MPC lattice cannot be empty")
        evaluations = tuple(self.evaluate(state, bundle, candidate) for candidate in candidates)
        valid = [evaluation for evaluation in evaluations if evaluation.valid]
        if not valid:
            raise NumericalFailure("Every projected MPC candidate failed an inner certificate")
        selected = min(
            valid,
            key=lambda evaluation: (
                evaluation.objective,
                sum(abs(value) for value in evaluation.first_action.values.values())
                if evaluation.first_action is not None
                else float("inf"),
            ),
        )
        assert selected.first_action is not None
        return MPCResult(
            selected.first_action,
            selected.candidate_id,
            selected.objective,
            evaluations,
            _selection_log(
                evaluations,
                selected_candidate_id=selected.candidate_id,
                used_fallback=False,
            ),
        )


@dataclass(frozen=True)
class ProposalSelection:
    action: Action
    source: str
    evaluations: Mapping[str, CandidateEvaluation]
    used_fallback: bool
    selection_log: SelectionLog


class TwoProposalSelector:
    """Compare frozen BC/SAC proposals with common nested evaluations."""

    def __init__(
        self,
        *,
        mpc_evaluator: ProjectedStochasticMPC,
        fallback_raw_action: RawPolicy,
        continuation: ContinuationPolicy,
    ) -> None:
        self.mpc_evaluator = mpc_evaluator
        self.fallback_raw_action = fallback_raw_action
        self.continuation = continuation

    def select(
        self,
        *,
        state: ModelState,
        bundle: ScenarioBundle,
        bc_raw_action: Action,
        sac_raw_action: Action,
    ) -> ProposalSelection:
        candidates = (
            MPCCandidate("BC", bc_raw_action, self.continuation),
            MPCCandidate("SAC", sac_raw_action, self.continuation),
        )
        evaluations = {
            candidate.candidate_id: self.mpc_evaluator.evaluate(state, bundle, candidate)
            for candidate in candidates
        }
        valid = [evaluation for evaluation in evaluations.values() if evaluation.valid]
        if valid:
            selected = min(valid, key=lambda item: (item.objective, item.candidate_id))
            assert selected.first_action is not None
            return ProposalSelection(
                selected.first_action,
                selected.candidate_id,
                evaluations,
                False,
                _selection_log(
                    tuple(evaluations.values()),
                    selected_candidate_id=selected.candidate_id,
                    used_fallback=False,
                ),
            )

        fallback = MPCCandidate(
            "passive_fallback", self.fallback_raw_action(state), self.continuation
        )
        fallback_evaluation = self.mpc_evaluator.evaluate(state, bundle, fallback)
        evaluations[fallback.candidate_id] = fallback_evaluation
        if not fallback_evaluation.valid or fallback_evaluation.first_action is None:
            raise NumericalFailure(
                "BC, SAC, and the prespecified projected passive fallback all failed"
            )
        return ProposalSelection(
            fallback_evaluation.first_action,
            fallback.candidate_id,
            evaluations,
            True,
            _selection_log(
                tuple(evaluations.values()),
                selected_candidate_id=fallback.candidate_id,
                used_fallback=True,
            ),
        )
