"""Common Section 5.1 trajectory statistics derived from model transitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .clearance import ClearanceResult
from .engine import KernelResult
from .errors import ContractError
from .keys import Network, ResourceKey, Stage
from .state import ModelState
from .transition import TransitionAudit, TransitionResult


LOSS_COMPONENTS = ("queue", "waiting", "exit", "overflow", "route_resource", "action")


@dataclass(frozen=True)
class TrajectoryStatistics:
    """One policy/path/seed outcome using the paper's common reporting basis."""

    decision_periods: int
    clearance_followup_periods: int
    physical_scope: str
    decision_operational_loss: float
    clearance_operational_loss: float
    terminal_correction: float
    total_operational_objective: float
    loss_queue: float
    loss_waiting: float
    loss_exit: float
    loss_overflow: float
    loss_route_resource: float
    loss_action: float
    peak_pressure: float
    overload: float
    waiting_model_unit_weeks: float
    direct_sue_exit: float
    duration_attrition: float
    total_exit: float
    delivered_landbridge: float
    ending_outstanding_mass: float
    clearance_status: str
    clearance_weeks_observed: int | None
    right_censored: bool
    numerical_failure: bool
    transition_audits_passed: bool
    maximum_transition_residual: float
    accepted: bool

    def as_record(self) -> dict[str, Any]:
        """Return a flat, CSV-ready record without replacing censored values."""

        return asdict(self)


def _unwrap(result: KernelResult | TransitionResult) -> TransitionResult:
    return result.transition if isinstance(result, KernelResult) else result


def _required_resources(network: Network) -> tuple[ResourceKey, ...]:
    resources = [
        ResourceKey(stage, gateway)
        for gateway in network.gateways()
        for stage in (Stage.BERTH, Stage.YARD, Stage.GATE)
    ]
    resources.extend(ResourceKey(Stage.CORRIDOR, edge) for edge in network.corridors())
    return tuple(sorted(resources))


def aggregate_stage_queues(state: ModelState, network: Network) -> dict[ResourceKey, float]:
    """Aggregate permanent route tags to physical gateway/corridor queues."""

    aggregates = {resource: 0.0 for resource in _required_resources(network)}
    for stage, queue in (
        (Stage.BERTH, state.berth),
        (Stage.YARD, state.yard),
        (Stage.GATE, state.gate),
        (Stage.CORRIDOR, state.corridor),
    ):
        for tag, mass in queue.items():
            route = network.route(tag.route)
            location = route.corridor if stage == Stage.CORRIDOR else route.gateway
            resource = ResourceKey(stage, location)
            aggregates[resource] += float(mass)
    return aggregates


def stage_pressures(
    state: ModelState,
    network: Network,
    thresholds: Mapping[ResourceKey, float],
) -> dict[ResourceKey, float]:
    """Compute B/barB, Y/barY, G/barG, and M/barM for one state."""

    resources = _required_resources(network)
    missing = sorted(set(resources) - set(thresholds))
    if missing:
        raise ContractError(f"Pressure thresholds are missing for resources: {missing}")
    invalid = {
        resource: thresholds[resource]
        for resource in resources
        if thresholds[resource] <= 0 or not np.isfinite(thresholds[resource])
    }
    if invalid:
        raise ContractError(f"Pressure thresholds must be finite and positive: {invalid}")
    queues = aggregate_stage_queues(state, network)
    return {resource: queues[resource] / thresholds[resource] for resource in resources}


def _audit_residual(audit: TransitionAudit) -> float:
    return float(
        max(
            abs(audit.adaptive_mass_residual),
            abs(audit.committed_mass_residual),
            abs(audit.pipeline_mass_residual),
            abs(audit.tagged_balance_residual),
            abs(audit.action_feasibility_violation),
            abs(audit.equilibrium_residual),
            max(-audit.minimum_state_component, 0.0),
            abs(audit.unavailable_route_service_violation),
            abs(audit.readiness_order_balance_residual),
            abs(audit.readiness_stock_balance_residual),
            abs(audit.direct_order_balance_residual),
            abs(audit.direct_stock_balance_residual),
            abs(audit.readiness_stock_feasibility_violation),
            abs(audit.budget_balance_residual),
            abs(audit.corridor_history_residual),
            abs(audit.serviceability_history_residual),
            abs(audit.previous_share_history_residual),
        )
    )


def _trace_states(
    initial_state: ModelState,
    results: Sequence[KernelResult | TransitionResult],
) -> tuple[list[ModelState], ModelState, list[TransitionResult]]:
    states: list[ModelState] = []
    transitions: list[TransitionResult] = []
    current = initial_state
    for item in results:
        transition = _unwrap(item)
        if transition.next_state.period != current.period + 1:
            raise ContractError(
                "Trajectory transitions must be ordered and advance the state by one period"
            )
        states.append(current)
        transitions.append(transition)
        current = transition.next_state
    return states, current, transitions


def _sum_loss_components(transitions: Iterable[TransitionResult]) -> dict[str, float]:
    return {
        component: float(sum(getattr(transition.loss, component) for transition in transitions))
        for component in LOSS_COMPONENTS
    }


def compute_trajectory_statistics(
    *,
    initial_state: ModelState,
    decision_results: Sequence[KernelResult | TransitionResult],
    network: Network,
    thresholds: Mapping[ResourceKey, float],
    clearance: ClearanceResult | None = None,
    include_clearance_in_physical_metrics: bool = True,
    numerical_failure: bool = False,
    tolerance: float = 1e-9,
) -> TrajectoryStatistics:
    """Compute the common loss, pressure, flow, exit, and clearance outcomes.

    The formal objective always equals decision operational loss plus clearance
    operational loss plus the terminal correction. By default, physical sums
    use every realised transition, including the frozen-recovery clearance
    tail. Set ``include_clearance_in_physical_metrics=False`` only when an
    experiment explicitly reports decision-window physical outcomes.

    A right-censored run records the realised follow-up length and terminal
    outstanding mass, but ``clearance_weeks_observed`` remains ``None``; the cap
    is never fabricated as an observed clearing time.
    """

    if tolerance <= 0:
        raise ContractError("Trajectory-statistics tolerance must be positive")
    decision_states, decision_final, decision_transitions = _trace_states(
        initial_state, decision_results
    )
    clearance_states: list[ModelState] = []
    clearance_transitions: list[TransitionResult] = []
    final_state = decision_final
    if clearance is not None:
        if clearance.weeks != len(clearance.transitions):
            raise ContractError("Clearance weeks must equal its realised transition count")
        clearance_states, traced_final, clearance_transitions = _trace_states(
            decision_final, clearance.transitions
        )
        if traced_final.period != clearance.final_state.period:
            raise ContractError("Clearance transition trace and final-state period disagree")
        if abs(traced_final.cargo_mass() - clearance.final_state.cargo_mass()) > tolerance:
            raise ContractError("Clearance transition trace and final outstanding mass disagree")
        final_state = clearance.final_state

    decision_loss = float(sum(transition.loss.total for transition in decision_transitions))
    traced_clearance_loss = float(
        sum(transition.loss.total for transition in clearance_transitions)
    )
    if clearance is None:
        clearance_loss = 0.0
        terminal_correction = 0.0
        clearance_status = "not_run"
        clearance_observed: int | None = None
        right_censored = False
        followup_periods = 0
    else:
        if abs(traced_clearance_loss - clearance.operational_loss) > tolerance:
            raise ContractError(
                "Clearance operational loss must equal the sum of its transition losses"
            )
        if clearance.cleared == clearance.right_censored:
            raise ContractError("Clearance must be either cleared or right-censored")
        if (
            clearance.operational_loss < 0
            or clearance.terminal_correction < 0
            or not np.isfinite(clearance.operational_loss)
            or not np.isfinite(clearance.terminal_correction)
        ):
            raise ContractError("Clearance loss and terminal correction must be finite and nonnegative")
        if clearance.cleared and clearance.terminal_correction > tolerance:
            raise ContractError("A cleared trajectory cannot carry a terminal-mass correction")
        clearance_loss = float(clearance.operational_loss)
        terminal_correction = float(clearance.terminal_correction)
        clearance_status = "cleared" if clearance.cleared else "right_censored"
        clearance_observed = clearance.weeks if clearance.cleared else None
        right_censored = clearance.right_censored
        followup_periods = clearance.weeks

    physical_states = decision_states + (
        clearance_states if include_clearance_in_physical_metrics else []
    )
    physical_transitions = decision_transitions + (
        clearance_transitions if include_clearance_in_physical_metrics else []
    )
    pressure_vectors = [stage_pressures(state, network, thresholds) for state in physical_states]
    pressure_values = [value for vector in pressure_vectors for value in vector.values()]
    peak = max(pressure_values, default=0.0)
    overload = float(sum(max(value - 1.0, 0.0) for value in pressure_values))
    waiting = float(sum(state.waiting_mass() for state in physical_states))
    direct_exit = float(
        sum(sum(transition.direct_exit.values()) for transition in physical_transitions)
    )
    attrition = float(
        sum(sum(transition.duration_attrition.values()) for transition in physical_transitions)
    )
    delivered = float(
        sum(sum(transition.delivered.values()) for transition in physical_transitions)
    )

    all_transitions = decision_transitions + clearance_transitions
    components = _sum_loss_components(all_transitions)
    component_total = float(sum(components.values()))
    if any(value < -tolerance or not np.isfinite(value) for value in components.values()):
        raise ContractError("Trajectory loss components must be finite and nonnegative")
    if abs(component_total - decision_loss - clearance_loss) > tolerance:
        raise ContractError("Loss components do not reconstruct the operational objective")
    audits_passed = all(transition.audit.passed for transition in all_transitions)
    maximum_residual = max(
        (_audit_residual(transition.audit) for transition in all_transitions), default=0.0
    )
    accepted = audits_passed and not numerical_failure

    return TrajectoryStatistics(
        decision_periods=len(decision_transitions),
        clearance_followup_periods=followup_periods,
        physical_scope=(
            "decision_plus_clearance"
            if include_clearance_in_physical_metrics
            else "decision_only"
        ),
        decision_operational_loss=decision_loss,
        clearance_operational_loss=clearance_loss,
        terminal_correction=terminal_correction,
        total_operational_objective=decision_loss + clearance_loss + terminal_correction,
        loss_queue=components["queue"],
        loss_waiting=components["waiting"],
        loss_exit=components["exit"],
        loss_overflow=components["overflow"],
        loss_route_resource=components["route_resource"],
        loss_action=components["action"],
        peak_pressure=float(peak),
        overload=overload,
        waiting_model_unit_weeks=waiting,
        direct_sue_exit=direct_exit,
        duration_attrition=attrition,
        total_exit=direct_exit + attrition,
        delivered_landbridge=delivered,
        ending_outstanding_mass=float(final_state.cargo_mass()),
        clearance_status=clearance_status,
        clearance_weeks_observed=clearance_observed,
        right_censored=right_censored,
        numerical_failure=numerical_failure,
        transition_audits_passed=audits_passed,
        maximum_transition_residual=maximum_residual,
        accepted=accepted,
    )
