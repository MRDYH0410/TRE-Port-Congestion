from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tre84.clearance import ClearanceResult
from tre84.engine import KernelResult
from tre84.errors import ContractError
from tre84.keys import Network, ResourceKey, Route, SourceKey, Stage, Tag
from tre84.loss import LossBreakdown
from tre84.metrics import compute_trajectory_statistics
from tre84.state import CapacityState, ModelState, RiskInformation
from tre84.transition import TransitionAudit


def _state(
    period: int,
    *,
    waiting: float,
    berth: float = 0.0,
    yard: float = 0.0,
    gate: float = 0.0,
    corridor: float = 0.0,
) -> ModelState:
    tag = Tag("c", "r")
    return ModelState(
        period=period,
        horizon=8,
        risk=RiskInformation(np.array([1.0]), np.array([1.0])),
        disruption_seen=True,
        disruption_active=True,
        disruption_duration=period,
        waiting={"c": np.array([waiting])},
        berth={tag: berth} if berth else {},
        yard={tag: yard} if yard else {},
        gate={tag: gate} if gate else {},
        corridor={tag: corridor} if corridor else {},
        maritime_pipeline=[],
        previous_shares={},
        corridor_history={("g", "e"): (1.0,)},
        serviceability_history=(0.0,),
        readiness=CapacityState(),
        direct_capacity=CapacityState(),
        budget=10.0,
    )


def _audit(residual: float) -> TransitionAudit:
    return TransitionAudit(
        adaptive_mass_residual=residual,
        committed_mass_residual=0.0,
        pipeline_mass_residual=0.0,
        tagged_balance_residual=0.0,
        action_feasibility_violation=0.0,
        equilibrium_residual=0.0,
        minimum_state_component=0.0,
        state_complete=True,
        passed=True,
    )


def _result(
    next_state: ModelState,
    loss: LossBreakdown,
    *,
    delivered: float,
    direct_exit: float,
    attrition: float,
    residual: float,
) -> KernelResult:
    transition = SimpleNamespace(
        next_state=next_state,
        loss=loss,
        delivered={Tag("c", "r"): delivered},
        direct_exit={SourceKey("c", None): direct_exit},
        duration_attrition={("c", 0): attrition},
        audit=_audit(residual),
    )
    return KernelResult(None, None, None, transition)  # type: ignore[arg-type]


def _fixture():
    network = Network({"r": Route("r", "c", "g", "e", (1.0,))})
    thresholds = {
        ResourceKey(Stage.BERTH, "g"): 10.0,
        ResourceKey(Stage.YARD, "g"): 10.0,
        ResourceKey(Stage.GATE, "g"): 10.0,
        ResourceKey(Stage.CORRIDOR, "e"): 10.0,
    }
    initial = _state(0, waiting=2.0, berth=12.0)
    decision_final = _state(1, waiting=1.0, yard=15.0)
    clearance_final = _state(2, waiting=0.0, corridor=1.0)
    decision = _result(
        decision_final,
        LossBreakdown(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        delivered=3.0,
        direct_exit=1.0,
        attrition=0.25,
        residual=0.01,
    )
    clearance_step = _result(
        clearance_final,
        LossBreakdown(0.5, 0.5, 0.25, 0.25, 0.25, 0.25),
        delivered=4.0,
        direct_exit=0.5,
        attrition=0.25,
        residual=0.02,
    )
    clearance = ClearanceResult(
        final_state=clearance_final,
        weeks=1,
        cleared=False,
        right_censored=True,
        operational_loss=2.0,
        terminal_correction=7.0,
        transitions=(clearance_step,),
    )
    return network, thresholds, initial, decision, clearance


def test_common_statistics_include_complete_trace_and_preserve_censoring() -> None:
    network, thresholds, initial, decision, clearance = _fixture()
    metrics = compute_trajectory_statistics(
        initial_state=initial,
        decision_results=(decision,),
        network=network,
        thresholds=thresholds,
        clearance=clearance,
    )
    assert metrics.physical_scope == "decision_plus_clearance"
    assert np.isclose(metrics.total_operational_objective, 15.0)
    assert np.isclose(metrics.peak_pressure, 1.5)
    assert np.isclose(metrics.overload, 0.7)
    assert np.isclose(metrics.waiting_model_unit_weeks, 3.0)
    assert np.isclose(metrics.direct_sue_exit, 1.5)
    assert np.isclose(metrics.duration_attrition, 0.5)
    assert np.isclose(metrics.total_exit, 2.0)
    assert np.isclose(metrics.delivered_landbridge, 7.0)
    assert np.isclose(metrics.ending_outstanding_mass, 1.0)
    assert metrics.clearance_status == "right_censored"
    assert metrics.clearance_weeks_observed is None
    assert metrics.clearance_followup_periods == 1
    assert np.isclose(metrics.maximum_transition_residual, 0.02)
    assert metrics.accepted


def test_decision_only_scope_changes_physical_metrics_not_formal_loss() -> None:
    network, thresholds, initial, decision, clearance = _fixture()
    metrics = compute_trajectory_statistics(
        initial_state=initial,
        decision_results=(decision,),
        network=network,
        thresholds=thresholds,
        clearance=clearance,
        include_clearance_in_physical_metrics=False,
    )
    assert metrics.physical_scope == "decision_only"
    assert np.isclose(metrics.total_operational_objective, 15.0)
    assert np.isclose(metrics.peak_pressure, 1.2)
    assert np.isclose(metrics.overload, 0.2)
    assert np.isclose(metrics.waiting_model_unit_weeks, 2.0)
    assert np.isclose(metrics.total_exit, 1.25)
    assert np.isclose(metrics.delivered_landbridge, 3.0)


def test_clearance_metadata_cannot_fabricate_a_cleared_outcome() -> None:
    network, thresholds, initial, decision, clearance = _fixture()
    inconsistent = replace(
        clearance,
        cleared=True,
        right_censored=False,
        terminal_correction=7.0,
    )
    with pytest.raises(ContractError, match="terminal-mass correction"):
        compute_trajectory_statistics(
            initial_state=initial,
            decision_results=(decision,),
            network=network,
            thresholds=thresholds,
            clearance=inconsistent,
        )
