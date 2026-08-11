"""Module 5: event-ordered tagged conservation transition and period audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .actions import Action, ActionDomain, ActionKey
from .behavior import EquilibriumResult
from .capacity import CapacityDynamics, CurrentCapacity
from .errors import AuditFailure, ContractError, NumericalFailure
from .keys import Network, Provenance, ResourceKey, SourceKey, Stage, Tag
from .loss import LossBreakdown, OperationalLoss
from .state import ModelState, PipelineLot, RiskInformation


@dataclass(frozen=True)
class ExogenousRealization:
    gulf_demand: Mapping[str, float]
    serviceable_share: Mapping[str, float]
    committed_fraction: Mapping[str, float]
    committed_route_share: Mapping[Tag, float]
    base_arrivals: Mapping[Tag, float]
    choice_route_available: frozenset[str]
    physical_route_available: frozenset[str]
    serviceability_observation: float
    next_disruption_seen: bool
    next_disruption_active: bool
    next_disruption_duration: int
    next_risk: RiskInformation | None = None
    next_observed_covariates: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(value < 0 or not np.isfinite(value) for value in self.gulf_demand.values()):
            raise ContractError("Gulf-bound demand must be finite and nonnegative")
        if any(not 0 <= value <= 1 for value in self.serviceable_share.values()):
            raise ContractError("Serviceable shares must lie in [0, 1]")
        if any(not 0 <= value <= 1 for value in self.committed_fraction.values()):
            raise ContractError("Committed fractions must lie in [0, 1]")
        if any(value < 0 or not np.isfinite(value) for value in self.committed_route_share.values()):
            raise ContractError("Committed route shares must be finite and nonnegative")
        if any(value < 0 or not np.isfinite(value) for value in self.base_arrivals.values()):
            raise ContractError("Baseline arrivals must be finite and nonnegative")
        if self.next_disruption_duration < 0:
            raise ContractError("Event duration cannot be negative")


@dataclass(frozen=True)
class DemandSplit:
    blocked: Mapping[str, float]
    committed: Mapping[str, float]
    decision_eligible: Mapping[str, float]
    committed_by_tag: Mapping[Tag, float]


def construct_demand_split(
    network: Network, realization: ExogenousRealization, *, tolerance: float
) -> DemandSplit:
    blocked: dict[str, float] = {}
    committed: dict[str, float] = {}
    decision: dict[str, float] = {}
    committed_by_tag: dict[Tag, float] = {}
    for cargo_class, demand in realization.gulf_demand.items():
        q_blocked = (1.0 - realization.serviceable_share[cargo_class]) * demand
        q_committed = realization.committed_fraction[cargo_class] * q_blocked
        blocked[cargo_class] = q_blocked
        committed[cargo_class] = q_committed
        decision[cargo_class] = q_blocked - q_committed
        tags = [
            Tag(cargo_class, route_id)
            for route_id in network.routes_for_class(cargo_class)
            if Tag(cargo_class, route_id) in realization.committed_route_share
        ]
        shares = sum(realization.committed_route_share[tag] for tag in tags)
        if q_committed > tolerance and not np.isclose(shares, 1.0, atol=tolerance):
            raise ContractError(
                f"Committed itinerary shares for {cargo_class} must sum to one"
            )
        for tag in tags:
            committed_by_tag[tag] = q_committed * realization.committed_route_share[tag]
    return DemandSplit(blocked, committed, decision, committed_by_tag)


@dataclass(frozen=True)
class TransitionAudit:
    adaptive_mass_residual: float
    committed_mass_residual: float
    pipeline_mass_residual: float
    tagged_balance_residual: float
    action_feasibility_violation: float
    equilibrium_residual: float
    minimum_state_component: float
    state_complete: bool
    passed: bool
    unavailable_route_service_violation: float = 0.0
    readiness_order_balance_residual: float = 0.0
    readiness_stock_balance_residual: float = 0.0
    direct_order_balance_residual: float = 0.0
    direct_stock_balance_residual: float = 0.0
    readiness_stock_feasibility_violation: float = 0.0
    budget_balance_residual: float = 0.0
    corridor_history_residual: float = 0.0
    serviceability_history_residual: float = 0.0
    previous_share_history_residual: float = 0.0
    waiting_vintage_balance_residuals: tuple[tuple[str, int, float], ...] = ()
    waiting_vintage_no_reset_residuals: tuple[tuple[str, float], ...] = ()
    waiting_vintage_expected_balance_count: int = 0
    waiting_vintage_expected_no_reset_count: int = 0
    waiting_vintage_certificate_complete: bool = False


@dataclass(frozen=True)
class TransitionResult:
    next_state: ModelState
    loss: LossBreakdown
    delivered: Mapping[Tag, float]
    direct_exit: Mapping[SourceKey, float]
    duration_attrition: Mapping[tuple[str, int], float]
    demand_split: DemandSplit
    capacity: CurrentCapacity
    audit: TransitionAudit


def _proportional_service(workload: Mapping[Tag, float], total_service: float) -> dict[Tag, float]:
    total = float(sum(workload.values()))
    if total <= 0 or total_service <= 0:
        return {tag: 0.0 for tag in workload}
    return {tag: float(mass / total * total_service) for tag, mass in workload.items()}


def _clean(mapping: Mapping[Tag, float], tolerance: float) -> dict[Tag, float]:
    """Retain every positive physical mass; tolerance is only for validation.

    Dropping several individually small positive tag masses can make their
    aggregate conservation residual exceed the audit tolerance and, more
    importantly, lets physical mass disappear before terminal accounting.
    """
    return {
        tag: float(value)
        for tag, value in mapping.items()
        if value > 0.0
    }


class TaggedTransition:
    def __init__(
        self,
        *,
        network: Network,
        action_domain: ActionDomain,
        capacity: CapacityDynamics,
        loss: OperationalLoss,
        waiting_hazard: Mapping[str, np.ndarray],
        release_action_map: Mapping[str, ActionKey],
        corridor_history_window: int,
        audit_tolerance: float,
    ) -> None:
        if corridor_history_window <= 0 or audit_tolerance <= 0:
            raise ContractError("Transition audit settings must be positive")
        self.network = network
        self.action_domain = action_domain
        self.capacity_model = capacity
        self.loss_model = loss
        self.waiting_hazard = {
            key: np.asarray(value, dtype=float) for key, value in waiting_hazard.items()
        }
        self.release_action_map = dict(release_action_map)
        self.corridor_history_window = corridor_history_window
        self.tolerance = audit_tolerance

    def _inject_and_advance_pipeline(
        self,
        state: ModelState,
        committed: Mapping[Tag, float],
        adaptive: Mapping[Tag, float],
        realization: ExogenousRealization,
    ) -> tuple[dict[Tag, float], list[PipelineLot], float]:
        arrivals: dict[Tag, float] = dict(realization.base_arrivals)
        carried: list[PipelineLot] = []
        model_arrivals = 0.0
        for lot in state.maritime_pipeline:
            available = lot.route in realization.physical_route_available
            if not available:
                carried.append(lot)
            elif lot.remaining_lag == 0:
                tag = Tag(lot.cargo_class, lot.route)
                arrivals[tag] = arrivals.get(tag, 0.0) + lot.mass
                model_arrivals += lot.mass
            else:
                carried.append(
                    PipelineLot(
                        lot.cargo_class,
                        lot.route,
                        lot.remaining_lag - 1,
                        lot.provenance,
                        lot.mass,
                    )
                )

        for provenance, dispatch in (
            (Provenance.COMMITTED, committed),
            (Provenance.ADAPTIVE, adaptive),
        ):
            for tag, mass in dispatch.items():
                if mass <= 0.0:
                    continue
                route = self.network.route(tag.route)
                available = tag.route in realization.physical_route_available
                for lag, share in enumerate(route.maritime_lag_kernel):
                    amount = mass * share
                    if amount <= 0.0:
                        continue
                    if lag == 0 and available:
                        arrivals[tag] = arrivals.get(tag, 0.0) + amount
                        model_arrivals += amount
                    else:
                        remaining = lag - 1 if available and lag > 0 else lag
                        carried.append(
                            PipelineLot(
                                tag.cargo_class,
                                tag.route,
                                remaining,
                                provenance,
                                amount,
                            )
                        )
        return arrivals, carried, model_arrivals

    def step(
        self,
        *,
        state: ModelState,
        action: Action,
        equilibrium: EquilibriumResult,
        realization: ExogenousRealization,
    ) -> TransitionResult:
        state.validate(tolerance=self.tolerance)
        self.action_domain.assert_feasible(action, state, tolerance=self.tolerance)
        if equilibrium.status != "converged":
            raise NumericalFailure("A nonconverged SUE cannot be used to fabricate a state transition")
        split = construct_demand_split(self.network, realization, tolerance=self.tolerance)

        unavailable_adaptive = {
            route: mass
            for (_, route), mass in equilibrium.route_dispatch.items()
            if mass > self.tolerance and route not in realization.choice_route_available
        }
        if unavailable_adaptive:
            raise AuditFailure(
                f"Adaptive flow used routes outside the current choice set: {unavailable_adaptive}"
            )
        unavailable_base_arrivals = {
            tag: mass
            for tag, mass in realization.base_arrivals.items()
            if mass > self.tolerance and tag.route not in realization.physical_route_available
        }
        if unavailable_base_arrivals:
            raise AuditFailure(
                "Baseline arrivals cannot materialize downstream of an unavailable "
                f"physical route: {unavailable_base_arrivals}"
            )

        for cargo_class, decision_mass in split.decision_eligible.items():
            source = SourceKey(cargo_class, None)
            flow_mass = sum(equilibrium.flows.get(source, {}).values())
            if abs(flow_mass - decision_mass) > self.tolerance:
                raise AuditFailure(
                    f"New source mass for {cargo_class} does not match the commitment split"
                )
            release = np.asarray(equilibrium.releases[cargo_class], dtype=float)
            expected_release = action.value(self.release_action_map[cargo_class]) * float(
                np.asarray(state.waiting[cargo_class]).sum()
            )
            if abs(float(release.sum()) - expected_release) > self.tolerance:
                raise AuditFailure(f"Equilibrium release for {cargo_class} is inconsistent with rho")
            for vintage, released_mass in enumerate(release):
                source_mass = sum(
                    equilibrium.flows.get(SourceKey(cargo_class, vintage), {}).values()
                )
                if abs(source_mass - released_mass) > self.tolerance:
                    raise AuditFailure(
                        f"Released vintage {cargo_class}/{vintage} left its source simplex"
                    )

        adaptive_dispatch = {
            Tag(cargo_class, route): mass
            for (cargo_class, route), mass in equilibrium.route_dispatch.items()
        }
        arrivals, next_pipeline, model_arrivals = self._inject_and_advance_pipeline(
            state, split.committed_by_tag, adaptive_dispatch, realization
        )
        capacity = self.capacity_model.transition(state, action)

        all_tags = set(state.berth) | set(state.yard) | set(state.gate) | set(state.corridor) | set(arrivals)
        berth_pre = {tag: state.berth.get(tag, 0.0) + arrivals.get(tag, 0.0) for tag in all_tags}
        yard_pre = {tag: state.yard.get(tag, 0.0) for tag in all_tags}
        gate_pre = {tag: state.gate.get(tag, 0.0) for tag in all_tags}
        corridor_pre = {tag: state.corridor.get(tag, 0.0) for tag in all_tags}

        def service_by_location(
            workload: Mapping[Tag, float], stage: Stage
        ) -> dict[Tag, float]:
            service: dict[Tag, float] = {tag: 0.0 for tag in workload}
            locations = self.network.corridors() if stage == Stage.CORRIDOR else self.network.gateways()
            for location in locations:
                subset = {
                    tag: mass
                    for tag, mass in workload.items()
                    if tag.route in realization.physical_route_available
                    and (
                            self.network.route(tag.route).corridor
                            if stage == Stage.CORRIDOR
                            else self.network.route(tag.route).gateway
                        )
                        == location
                }
                amount = min(
                    float(sum(subset.values())),
                    capacity.current.effective[ResourceKey(stage, location)],
                )
                service.update(_proportional_service(subset, amount))
            return service

        service_berth = service_by_location(berth_pre, Stage.BERTH)
        service_yard = service_by_location(yard_pre, Stage.YARD)
        service_gate = service_by_location(gate_pre, Stage.GATE)
        service_corridor = service_by_location(corridor_pre, Stage.CORRIDOR)

        next_berth = {
            tag: berth_pre.get(tag, 0.0) - service_berth.get(tag, 0.0) for tag in all_tags
        }
        next_yard = {
            tag: yard_pre.get(tag, 0.0)
            + service_berth.get(tag, 0.0)
            - service_yard.get(tag, 0.0)
            for tag in all_tags
        }
        next_gate = {
            tag: gate_pre.get(tag, 0.0)
            + service_yard.get(tag, 0.0)
            - service_gate.get(tag, 0.0)
            for tag in all_tags
        }
        next_corridor = {
            tag: corridor_pre.get(tag, 0.0)
            + service_gate.get(tag, 0.0)
            - service_corridor.get(tag, 0.0)
            for tag in all_tags
        }

        next_waiting: dict[str, np.ndarray] = {}
        attrition: dict[tuple[str, int], float] = {}
        waiting_vintage_balance_residuals: list[tuple[str, int, float]] = []
        waiting_vintage_no_reset_residuals: list[tuple[str, float]] = []
        for cargo_class, stock_value in state.waiting.items():
            stock = np.asarray(stock_value, dtype=float)
            release = np.asarray(equilibrium.releases[cargo_class], dtype=float)
            hazard = self.waiting_hazard[cargo_class]
            if stock.shape != release.shape or stock.shape != hazard.shape:
                raise ContractError("Waiting state, release, and hazard shapes must agree")
            following = np.zeros_like(stock)
            new_waiting = equilibrium.renewed_waiting.get(
                SourceKey(cargo_class, None), 0.0
            )
            following[0] = new_waiting
            waiting_vintage_no_reset_residuals.append(
                (cargo_class, float(following[0] - new_waiting))
            )
            for vintage in range(stock.size):
                renewed = equilibrium.renewed_waiting.get(
                    SourceKey(cargo_class, vintage), 0.0
                )
                exposed = stock[vintage] - release[vintage] + renewed
                lost = hazard[vintage] * exposed
                attrition[(cargo_class, vintage)] = float(lost)
                survivor = (1.0 - hazard[vintage]) * exposed
                if vintage + 1 < stock.size:
                    following[vintage + 1] = survivor
                    recorded_survivor = following[vintage + 1]
                else:
                    # The registered final-vintage hazard is one.  Recording a
                    # zero carried survivor makes any otherwise silent terminal
                    # vintage loss fail closed instead of resetting/disappearing.
                    recorded_survivor = 0.0
                waiting_vintage_balance_residuals.append(
                    (
                        cargo_class,
                        vintage,
                        float(exposed - lost - recorded_survivor),
                    )
                )
            next_waiting[cargo_class] = following

        gate_release_history: dict[tuple[str, str], float] = {}
        for tag, mass in service_gate.items():
            route = self.network.route(tag.route)
            key = (route.gateway, route.corridor)
            gate_release_history[key] = gate_release_history.get(key, 0.0) + mass
        next_history: dict[tuple[str, str], tuple[float, ...]] = {}
        history_keys = set(state.corridor_history) | set(gate_release_history)
        for key in history_keys:
            values = (float(gate_release_history.get(key, 0.0)),) + tuple(
                state.corridor_history.get(key, ())
            )
            next_history[key] = values[: self.corridor_history_window]

        action_cost = self.action_domain.action_cost(action)
        remaining_budget = state.budget - action_cost
        if remaining_budget < -self.tolerance:
            raise AuditFailure(
                f"Projected action exceeded remaining budget by {-remaining_budget:.6g}"
            )
        next_state = ModelState(
            period=state.period + 1,
            horizon=state.horizon,
            risk=realization.next_risk or state.risk,
            # D_seen records whether the disruption has ever been officially
            # announced.  Recovery can end active disruption but cannot reset
            # the observable authority phase to its preannouncement value.
            disruption_seen=(state.disruption_seen or realization.next_disruption_seen),
            disruption_active=realization.next_disruption_active,
            disruption_duration=realization.next_disruption_duration,
            waiting=next_waiting,
            berth=_clean(next_berth, self.tolerance),
            yard=_clean(next_yard, self.tolerance),
            gate=_clean(next_gate, self.tolerance),
            corridor=_clean(next_corridor, self.tolerance),
            maritime_pipeline=next_pipeline,
            previous_shares={source: dict(shares) for source, shares in equilibrium.normalized_shares.items()},
            corridor_history=next_history,
            serviceability_history=state.serviceability_history
            + (float(realization.serviceability_observation),),
            readiness=capacity.next_readiness,
            direct_capacity=capacity.next_direct,
            budget=max(float(remaining_budget), 0.0),
            observed_covariates=dict(realization.next_observed_covariates),
        )

        loss = self.loss_model.compute(
            state=state,
            committed_dispatch=split.committed_by_tag,
            adaptive_dispatch=adaptive_dispatch,
            direct_exit=equilibrium.direct_exit,
            duration_attrition=attrition,
            action_cost=action_cost,
        )

        adaptive_residuals = []
        for cargo_class, q_decision in split.decision_eligible.items():
            lhs = q_decision + float(np.asarray(state.waiting[cargo_class]).sum())
            route = sum(
                mass for tag, mass in adaptive_dispatch.items() if tag.cargo_class == cargo_class
            )
            direct = sum(
                mass
                for source, mass in equilibrium.direct_exit.items()
                if source.cargo_class == cargo_class
            )
            attr = sum(
                mass for (klass, _), mass in attrition.items() if klass == cargo_class
            )
            rhs = route + direct + attr + float(next_waiting[cargo_class].sum())
            adaptive_residuals.append(abs(lhs - rhs))
        committed_residual = max(
            (
                abs(
                    split.committed[cargo_class]
                    - sum(
                        mass
                        for tag, mass in split.committed_by_tag.items()
                        if tag.cargo_class == cargo_class
                    )
                )
                for cargo_class in split.committed
            ),
            default=0.0,
        )
        pipeline_residual = abs(
            state.pipeline_mass()
            + sum(split.committed_by_tag.values())
            + sum(adaptive_dispatch.values())
            - model_arrivals
            - sum(lot.mass for lot in next_pipeline)
        )

        tagged_residual = 0.0
        for tag in all_tags:
            residuals = (
                next_berth[tag] - (state.berth.get(tag, 0.0) + arrivals.get(tag, 0.0) - service_berth[tag]),
                next_yard[tag] - (state.yard.get(tag, 0.0) + service_berth[tag] - service_yard[tag]),
                next_gate[tag] - (state.gate.get(tag, 0.0) + service_yard[tag] - service_gate[tag]),
                next_corridor[tag]
                - (state.corridor.get(tag, 0.0) + service_gate[tag] - service_corridor[tag]),
            )
            tagged_residual = max(tagged_residual, *(abs(value) for value in residuals))
        components = [
            *next_berth.values(),
            *next_yard.values(),
            *next_gate.values(),
            *next_corridor.values(),
            *(float(value) for vintages in next_waiting.values() for value in vintages),
            *(lot.mass for lot in next_pipeline),
            next_state.budget,
        ]
        minimum = min(components, default=0.0)
        complete = True
        try:
            next_state.validate(tolerance=self.tolerance)
        except ContractError:
            complete = False
        action_violation = self.action_domain.violation(action, state)
        unavailable_service_violation = float(
            sum(
                service.get(tag, 0.0)
                for service in (
                    service_berth,
                    service_yard,
                    service_gate,
                    service_corridor,
                )
                for tag in all_tags
                if tag.route not in realization.physical_route_available
            )
        )
        budget_residual = abs(
            next_state.budget - max(float(state.budget - action_cost), 0.0)
        )
        expected_serviceability_history = state.serviceability_history + (
            float(realization.serviceability_observation),
        )
        serviceability_history_residual = (
            0.0
            if next_state.serviceability_history == expected_serviceability_history
            else float("inf")
        )
        corridor_history_residual = 0.0
        for key in set(next_state.corridor_history) | set(next_history):
            observed = next_state.corridor_history.get(key, ())
            expected = next_history.get(key, ())
            if len(observed) != len(expected):
                corridor_history_residual = float("inf")
                break
            corridor_history_residual = max(
                corridor_history_residual,
                max(
                    (abs(left - right) for left, right in zip(observed, expected)),
                    default=0.0,
                ),
            )
        previous_share_residual = 0.0
        for source in set(next_state.previous_shares) | set(equilibrium.normalized_shares):
            observed = next_state.previous_shares.get(source, {})
            expected = equilibrium.normalized_shares.get(source, {})
            for choice in set(observed) | set(expected):
                previous_share_residual = max(
                    previous_share_residual,
                    abs(observed.get(choice, 0.0) - expected.get(choice, 0.0)),
                )
        capacity_audit = capacity.audit
        maximum_residual = max(
            max(adaptive_residuals, default=0.0),
            committed_residual,
            pipeline_residual,
            tagged_residual,
            action_violation,
            equilibrium.residual,
            max(-minimum, 0.0),
            unavailable_service_violation,
            capacity_audit.maximum_residual,
            budget_residual,
            corridor_history_residual,
            serviceability_history_residual,
            previous_share_residual,
            max(
                (abs(item[2]) for item in waiting_vintage_balance_residuals),
                default=0.0,
            ),
            max(
                (abs(item[1]) for item in waiting_vintage_no_reset_residuals),
                default=0.0,
            ),
        )
        audit = TransitionAudit(
            adaptive_mass_residual=max(adaptive_residuals, default=0.0),
            committed_mass_residual=committed_residual,
            pipeline_mass_residual=pipeline_residual,
            tagged_balance_residual=tagged_residual,
            action_feasibility_violation=action_violation,
            equilibrium_residual=equilibrium.residual,
            minimum_state_component=minimum,
            state_complete=complete,
            passed=complete and maximum_residual <= self.tolerance,
            unavailable_route_service_violation=unavailable_service_violation,
            readiness_order_balance_residual=(
                capacity_audit.readiness_order_balance_residual
            ),
            readiness_stock_balance_residual=(
                capacity_audit.readiness_stock_balance_residual
            ),
            direct_order_balance_residual=capacity_audit.direct_order_balance_residual,
            direct_stock_balance_residual=capacity_audit.direct_stock_balance_residual,
            readiness_stock_feasibility_violation=(
                capacity_audit.readiness_stock_feasibility_violation
            ),
            budget_balance_residual=budget_residual,
            corridor_history_residual=corridor_history_residual,
            serviceability_history_residual=serviceability_history_residual,
            previous_share_history_residual=previous_share_residual,
            waiting_vintage_balance_residuals=tuple(
                waiting_vintage_balance_residuals
            ),
            waiting_vintage_no_reset_residuals=tuple(
                waiting_vintage_no_reset_residuals
            ),
            waiting_vintage_expected_balance_count=sum(
                np.asarray(stock, dtype=float).size for stock in state.waiting.values()
            ),
            waiting_vintage_expected_no_reset_count=len(state.waiting),
            waiting_vintage_certificate_complete=True,
        )
        if not audit.passed:
            raise AuditFailure(f"Tagged transition audit failed: {audit}")
        return TransitionResult(
            next_state=next_state,
            loss=loss,
            delivered={tag: mass for tag, mass in service_corridor.items() if mass > 0.0},
            direct_exit=equilibrium.direct_exit,
            duration_attrition=attrition,
            demand_split=split,
            capacity=capacity.current,
            audit=audit,
        )
