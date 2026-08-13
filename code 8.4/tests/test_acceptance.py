from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tre84.acceptance import (
    evaluate_acceptance,
    evaluate_trajectory_acceptance,
    source_masses_for_acceptance,
)
from tre84.actions import (
    Action,
    ActionDomain,
    ActionKey,
    Block,
    ConvexPiecewiseLinearCurve,
)
from tre84.behavior import EquilibriumResult
from tre84.clearance import ClearanceResult
from tre84.keys import Network, Route, SourceKey
from tre84.loss import LossBreakdown, TerminalCostParameters, TerminalMassCorrection
from tre84.state import CapacityState, ModelState, RiskInformation
from tre84.transition import TransitionAudit


def test_acceptance_source_masses_use_release_ledger_not_underflowed_flow_sum() -> None:
    tiny = float(np.nextafter(0.0, 1.0))
    result = SimpleNamespace(
        transition=SimpleNamespace(
            demand_split=SimpleNamespace(decision_eligible={"c": 0.0})
        ),
        equilibrium=SimpleNamespace(
            releases={"c": np.asarray([0.0, tiny])},
            # Summing alternative-level subnormal flows can underflow to zero;
            # this must not erase the strictly positive formal source.
            flows={SourceKey("c", 1): {"r1": 0.0, "r2": 0.0}},
        ),
    )
    masses = source_masses_for_acceptance(result)
    assert masses[SourceKey("c", None)] == 0.0
    assert masses[SourceKey("c", 1)] == tiny


def _state(waiting_mass: float = 0.0) -> ModelState:
    return ModelState(
        period=0,
        horizon=4,
        risk=RiskInformation(np.array([1.0]), np.array([1.0])),
        disruption_seen=False,
        disruption_active=False,
        disruption_duration=0,
        waiting={"c": np.array([waiting_mass])},
        berth={},
        yard={},
        gate={},
        corridor={},
        maritime_pipeline=[],
        previous_shares={},
        corridor_history={},
        serviceability_history=(),
        readiness=CapacityState(),
        direct_capacity=CapacityState(),
        budget=10.0,
    )


def _domain() -> ActionDomain:
    keys = tuple(ActionKey.one(block, block.value) for block in Block)
    free = ConvexPiecewiseLinearCurve((0.0,), (0.0,))
    return ActionDomain(
        keys=keys,
        phase_upper={phase: {key: 1.0 for key in keys} for phase in (0, 1)},
        cost_curves={key: free for key in keys},
        period_budget_cap=lambda state: state.budget,
    )


def _zero_equilibrium() -> EquilibriumResult:
    source = SourceKey("c", None)
    return EquilibriumResult(
        flows={source: {"__WAIT__": 0.0, "__EXIT__": 0.0}},
        releases={"c": np.array([0.0])},
        route_dispatch={},
        renewed_waiting={source: 0.0},
        direct_exit={source: 0.0},
        normalized_shares={},
        residual=0.0,
        kl_discrepancy=0.0,
        multi_start_dispersion=0.0,
        iterations=0,
        status="converged",
        starts=(),
    )


def _terminal_cost() -> TerminalMassCorrection:
    network = Network({"r": Route("r", "c", "g", "e", (1.0,))})
    return TerminalMassCorrection(
        network,
        TerminalCostParameters(
            waiting_unit_cost={"c": 2.0},
            pipeline_unit_cost={"c": 2.0},
            tagged_unit_cost={},
        ),
    )


def test_period_acceptance_reads_the_expanded_physical_certificate() -> None:
    audit = TransitionAudit(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        True,
        True,
        budget_balance_residual=1.0,
    )
    report = evaluate_acceptance(
        decision_time=0,
        information_timestamps=(),
        state=_state(),
        action=Action(),
        action_domain=_domain(),
        equilibrium=_zero_equilibrium(),
        source_masses={SourceKey("c", None): 0.0},
        transition_audit=audit,
        loss=LossBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        tolerance=1e-9,
    )
    assert not report.physical_closure
    assert not report.passed


def test_period_acceptance_checks_each_waiting_vintage_and_no_reset_entry() -> None:
    audit = TransitionAudit(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        True,
        True,
        waiting_vintage_balance_residuals=(("c", 0, 1.0e-4),),
        waiting_vintage_no_reset_residuals=(("c", 0.0),),
        waiting_vintage_expected_balance_count=1,
        waiting_vintage_expected_no_reset_count=1,
        waiting_vintage_certificate_complete=True,
    )
    report = evaluate_acceptance(
        decision_time=0,
        information_timestamps=(),
        state=_state(),
        action=Action(),
        action_domain=_domain(),
        equilibrium=_zero_equilibrium(),
        source_masses={SourceKey("c", None): 0.0},
        transition_audit=audit,
        loss=LossBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        tolerance=1e-9,
    )
    assert not report.physical_closure
    assert "per-vintage no-reset" in report.messages[-1]


def test_trajectory_acceptance_reconciles_terminal_correction_exactly_once() -> None:
    state = _state(waiting_mass=1.0)
    clearance = ClearanceResult(
        final_state=state,
        weeks=0,
        cleared=False,
        right_censored=True,
        operational_loss=0.0,
        terminal_correction=2.0,
        transitions=(),
    )
    accepted = evaluate_trajectory_acceptance(
        initial_state=state,
        decision_results=(),
        decision_information=(),
        action_domain=_domain(),
        clearance=clearance,
        terminal_cost=_terminal_cost(),
        reported_total_objective=2.0,
        tolerance=1e-9,
    )
    duplicated = evaluate_trajectory_acceptance(
        initial_state=state,
        decision_results=(),
        decision_information=(),
        action_domain=_domain(),
        clearance=clearance,
        terminal_cost=_terminal_cost(),
        reported_total_objective=4.0,
        tolerance=1e-9,
    )
    assert accepted.passed
    assert not duplicated.objective_closure
    assert not duplicated.passed


def test_trajectory_acceptance_rejects_arbitrary_terminal_and_substituted_state() -> None:
    initial = _state(waiting_mass=1.0)
    substituted = initial.clone()
    substituted.budget = 0.0
    substituted.serviceability_history = (0.5,)
    substituted.observed_covariates["serviceability_timestamps"] = (0,)
    clearance = ClearanceResult(
        final_state=substituted,
        weeks=0,
        cleared=False,
        right_censored=True,
        operational_loss=0.0,
        terminal_correction=999.0,
        transitions=(),
    )
    report = evaluate_trajectory_acceptance(
        initial_state=initial,
        decision_results=(),
        decision_information=(),
        action_domain=_domain(),
        clearance=clearance,
        terminal_cost=_terminal_cost(),
        reported_total_objective=999.0,
        tolerance=1e-9,
    )
    assert not report.physical_closure
    assert not report.objective_closure
    assert not report.passed
