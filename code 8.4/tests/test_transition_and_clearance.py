from __future__ import annotations

import numpy as np
import pytest

from tre84.actions import (
    Action,
    ActionDomain,
    ActionKey,
    Block,
    ConvexPiecewiseLinearCurve,
)
from tre84.behavior import EquilibriumResult, SourceKey, StartRecord
from tre84.capacity import (
    CapacityActionMap,
    CapacityDynamics,
    CapacityTechnology,
    ServiceParameters,
)
from tre84.clearance import ClearanceRunner
from tre84.errors import ContractError
from tre84.keys import Network, ResourceKey, Route, Stage, Tag
from tre84.loss import (
    LossParameters,
    OperationalLoss,
    TerminalCostParameters,
    TerminalMassCorrection,
)
from tre84.state import CapacityState, ModelState, RiskInformation
from tre84.transition import ExogenousRealization, TaggedTransition


def build_transition() -> tuple[Network, ActionDomain, TaggedTransition, tuple[ActionKey, ...]]:
    network = Network(
        {
            "r1": Route("r1", "c", "g1", "e", (1.0,)),
            "r2": Route("r2", "c", "g2", "e", (1.0,)),
        }
    )
    resources = {
        ResourceKey(stage, location)
        for stage in (Stage.BERTH, Stage.YARD, Stage.GATE)
        for location in ("g1", "g2")
    } | {ResourceKey(Stage.CORRIDOR, "e")}
    release = ActionKey.one(Block.RELEASE, "c")
    disclosure = ActionKey(Block.DISCLOSURE, ("c", "r1"))
    readiness_order = ActionKey.one(Block.READINESS_ORDER, "B:g1")
    direct_order = ActionKey.one(Block.DIRECT_ORDER, "B:g1")
    exercise = ActionKey.one(Block.READINESS_EXERCISE, "B:g1")
    keys = (readiness_order, direct_order, exercise, release, disclosure)
    free = ConvexPiecewiseLinearCurve((0.0,), (0.0,))
    domain = ActionDomain(
        keys=keys,
        phase_upper={0: {key: 1.0 for key in keys}, 1: {key: 1.0 for key in keys}},
        cost_curves={key: free for key in keys},
        period_budget_cap=lambda state: state.budget,
    )
    tech = CapacityTechnology(
        readiness_lead={resource: 2 for resource in resources},
        readiness_maturity_yield={resource: 1.0 for resource in resources},
        readiness_consumption={resource: 1.0 for resource in resources},
        readiness_capacity_yield={resource: 1.0 for resource in resources},
        readiness_decay={resource: 0.0 for resource in resources},
        direct_lead={(phase, resource): 1 for phase in (0, 1) for resource in resources},
        direct_maturity_yield={resource: 1.0 for resource in resources},
        direct_decay={resource: 0.0 for resource in resources},
    )
    thresholds = {resource: 10.0 for resource in resources}
    service = ServiceParameters(
        base_capacity={resource: 1.0 for resource in resources},
        thresholds=thresholds,
        yard_feedback={"g1": lambda ratio: 1.0, "g2": lambda ratio: 1.0},
        corridor_feedback={"g1": lambda ratio: 1.0, "g2": lambda ratio: 1.0},
        fallback_corridor_share={("g1", "e"): 1.0, ("g2", "e"): 1.0},
    )
    capacity = CapacityDynamics(
        network,
        tech,
        service,
        CapacityActionMap(
            readiness_order={readiness_order: ResourceKey(Stage.BERTH, "g1")},
            direct_order={direct_order: ResourceKey(Stage.BERTH, "g1")},
            readiness_exercise={exercise: ResourceKey(Stage.BERTH, "g1")},
        ),
    )
    loss_parameters = LossParameters(
        queue_cost={resource: 0.0 for resource in resources},
        waiting_cost={("c", 0): 0.0, ("c", 1): 0.0},
        exit_failure_cost={"c": 1.0},
        overflow_cost={resource: 0.0 for resource in resources},
        thresholds=thresholds,
        route_resource_increment={Tag("c", "r1"): 0.0, Tag("c", "r2"): 0.0},
    )
    transition = TaggedTransition(
        network=network,
        action_domain=domain,
        capacity=capacity,
        loss=OperationalLoss(network, loss_parameters),
        waiting_hazard={"c": np.array([0.2, 1.0])},
        release_action_map={"c": release},
        corridor_history_window=2,
        audit_tolerance=1e-8,
    )
    return network, domain, transition, keys


def build_state() -> ModelState:
    return ModelState(
        period=0,
        horizon=8,
        risk=RiskInformation(np.array([1.0]), np.array([1.0])),
        disruption_seen=True,
        disruption_active=True,
        disruption_duration=1,
        waiting={"c": np.array([0.0, 2.0])},
        berth={Tag("c", "r1"): 1.0},
        yard={},
        gate={},
        corridor={},
        maritime_pipeline=[],
        previous_shares={},
        corridor_history={("g1", "e"): (1.0,), ("g2", "e"): (1.0,)},
        serviceability_history=(0.0,),
        readiness=CapacityState(),
        direct_capacity=CapacityState(),
        budget=10.0,
    )


def manual_equilibrium() -> EquilibriumResult:
    new = SourceKey("c", None)
    young = SourceKey("c", 0)
    old = SourceKey("c", 1)
    flows = {
        new: {"r1": 3.0, "r2": 0.0, "__WAIT__": 0.0, "__EXIT__": 0.0},
        young: {"r1": 0.0, "r2": 0.0, "__WAIT__": 0.0, "__EXIT__": 0.0},
        old: {"r1": 0.0, "r2": 1.0, "__WAIT__": 0.0, "__EXIT__": 0.0},
    }
    return EquilibriumResult(
        flows=flows,
        releases={"c": np.array([0.0, 1.0])},
        route_dispatch={("c", "r1"): 3.0, ("c", "r2"): 1.0},
        renewed_waiting={new: 0.0, young: 0.0, old: 0.0},
        direct_exit={new: 0.0, young: 0.0, old: 0.0},
        normalized_shares={new: {"r1": 1.0}, old: {"r2": 1.0}},
        residual=0.0,
        kl_discrepancy=0.0,
        multi_start_dispersion=0.0,
        iterations=1,
        status="converged",
        starts=(StartRecord("manual", 0.0, 1, True),),
    )


def test_tagged_transition_preserves_provenance_clock_and_shared_corridor() -> None:
    network, _, transition, keys = build_transition()
    assert network.shared_corridors() == {"e": ("g1", "g2")}
    action = Action({keys[3]: 0.5})
    realization = ExogenousRealization(
        gulf_demand={"c": 4.0},
        serviceable_share={"c": 0.0},
        committed_fraction={"c": 0.25},
        committed_route_share={Tag("c", "r1"): 1.0},
        base_arrivals={},
        choice_route_available=frozenset({"r1", "r2"}),
        physical_route_available=frozenset({"r1", "r2"}),
        serviceability_observation=0.0,
        next_disruption_seen=True,
        next_disruption_active=True,
        next_disruption_duration=2,
    )
    result = transition.step(
        state=build_state(), action=action, equilibrium=manual_equilibrium(), realization=realization
    )
    assert result.audit.passed
    assert np.isclose(sum(result.next_state.yard.values()), 2.0)
    assert np.isclose(sum(result.next_state.gate.values()), 0.0)  # no double internal service
    assert sum(result.delivered.values()) == 0.0  # gate release is not delivery
    assert np.isclose(result.duration_attrition[("c", 1)], 1.0)
    assert np.isclose(result.next_state.waiting["c"].sum(), 0.0)
    assert result.audit.pipeline_mass_residual <= 1e-8
    assert result.audit.waiting_vintage_expected_balance_count == 2
    assert result.audit.waiting_vintage_expected_no_reset_count == 1
    assert len(result.audit.waiting_vintage_balance_residuals) == 2
    assert len(result.audit.waiting_vintage_no_reset_residuals) == 1
    assert result.audit.waiting_vintage_certificate_complete
    assert max(abs(item[2]) for item in result.audit.waiting_vintage_balance_residuals) <= 1e-8
    assert max(abs(item[1]) for item in result.audit.waiting_vintage_no_reset_residuals) <= 1e-8


def test_physically_unavailable_route_tag_is_held_at_its_current_stage() -> None:
    _, _, transition, keys = build_transition()
    state = build_state()
    action = Action({keys[3]: 0.5})
    realization = ExogenousRealization(
        gulf_demand={"c": 4.0},
        serviceable_share={"c": 0.0},
        committed_fraction={"c": 0.25},
        committed_route_share={Tag("c", "r1"): 1.0},
        base_arrivals={},
        choice_route_available=frozenset({"r1", "r2"}),
        # r1 remains in the choice/provenance system but its physical link is
        # unavailable.  Its existing berth mass and new pipeline lots must hold.
        physical_route_available=frozenset({"r2"}),
        serviceability_observation=0.0,
        next_disruption_seen=True,
        next_disruption_active=True,
        next_disruption_duration=2,
    )
    result = transition.step(
        state=state,
        action=action,
        equilibrium=manual_equilibrium(),
        realization=realization,
    )
    unavailable = Tag("c", "r1")
    assert np.isclose(result.next_state.berth[unavailable], state.berth[unavailable])
    assert result.next_state.yard.get(unavailable, 0.0) == 0.0
    assert result.delivered.get(unavailable, 0.0) == 0.0
    assert result.audit.unavailable_route_service_violation == 0.0
    assert sum(
        lot.mass for lot in result.next_state.maritime_pipeline if lot.route == "r1"
    ) > 0.0


def test_disruption_seen_is_monotone_after_announcement() -> None:
    _, _, transition, keys = build_transition()
    state = build_state()
    realization = ExogenousRealization(
        gulf_demand={"c": 4.0},
        serviceable_share={"c": 0.0},
        committed_fraction={"c": 0.25},
        committed_route_share={Tag("c", "r1"): 1.0},
        base_arrivals={},
        choice_route_available=frozenset({"r1", "r2"}),
        physical_route_available=frozenset({"r1", "r2"}),
        serviceability_observation=0.0,
        next_disruption_seen=False,
        next_disruption_active=False,
        next_disruption_duration=0,
    )
    result = transition.step(
        state=state,
        action=Action({keys[3]: 0.5}),
        equilibrium=manual_equilibrium(),
        realization=realization,
    )
    assert state.phase == 1
    assert result.next_state.phase == 1
    assert result.audit.passed


class _NoCallKernel:
    def execute(self, **kwargs):  # pragma: no cover - maximum_weeks=0 must prevent this call
        raise AssertionError("clearance transition should not execute")


class _NoCallRecovery:
    def action(self, state):
        raise AssertionError

    def realization(self, state):
        raise AssertionError


class _NoCallTerminal:
    def compute(self, state):
        raise AssertionError


def _clearance_realization(*, blocked_demand: float = 0.0) -> ExogenousRealization:
    return ExogenousRealization(
        gulf_demand={"c": blocked_demand},
        serviceable_share={"c": 0.0},
        committed_fraction={"c": 0.0},
        committed_route_share={},
        base_arrivals={},
        choice_route_available=frozenset({"r1", "r2"}),
        physical_route_available=frozenset({"r1", "r2"}),
        serviceability_observation=1.0,
        next_disruption_seen=True,
        next_disruption_active=False,
        next_disruption_duration=0,
    )


def test_clearance_rejects_new_optimized_actions_and_blocked_demand() -> None:
    _, _, _, keys = build_transition()

    class _OptimizingRecovery:
        def action(self, state):
            return Action({keys[1]: 1.0})

        def realization(self, state):
            return _clearance_realization()

    optimizing = ClearanceRunner(
        kernel=_NoCallKernel(),
        recovery_rule=_OptimizingRecovery(),
        terminal_cost=_NoCallTerminal(),
        maximum_weeks=1,
        empty_tolerance=1e-12,
    )
    with pytest.raises(ContractError, match="new optimized actions"):
        optimizing.run(build_state())

    class _NewDemandRecovery:
        def action(self, state):
            return Action()

        def realization(self, state):
            return _clearance_realization(blocked_demand=1.0)

    new_demand = ClearanceRunner(
        kernel=_NoCallKernel(),
        recovery_rule=_NewDemandRecovery(),
        terminal_cost=_NoCallTerminal(),
        maximum_weeks=1,
        empty_tolerance=1e-12,
    )
    with pytest.raises(ContractError, match="New blocked demand"):
        new_demand.run(build_state())


def test_clearance_cap_retains_terminal_mass_and_right_censoring() -> None:
    network, _, _, _ = build_transition()
    resources = {
        ResourceKey(stage, location)
        for stage in (Stage.BERTH, Stage.YARD, Stage.GATE)
        for location in ("g1", "g2")
    } | {ResourceKey(Stage.CORRIDOR, "e")}
    terminal = TerminalMassCorrection(
        network,
        TerminalCostParameters(
            waiting_unit_cost={"c": 2.0},
            pipeline_unit_cost={"c": 2.0},
            tagged_unit_cost={resource: 2.0 for resource in resources},
        ),
    )
    result = ClearanceRunner(
        kernel=_NoCallKernel(),
        recovery_rule=_NoCallRecovery(),
        terminal_cost=terminal,
        maximum_weeks=0,
        empty_tolerance=1e-12,
    ).run(build_state())
    assert result.right_censored and not result.cleared
    assert np.isclose(result.terminal_correction, 6.0)  # waiting 2 plus berth 1, charged once
