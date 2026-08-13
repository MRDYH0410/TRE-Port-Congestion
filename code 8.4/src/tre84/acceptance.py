"""Layered acceptance checks; compilation or SUE convergence alone is insufficient."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from numbers import Number
from typing import Any, Mapping, Sequence

import numpy as np

from .actions import Action, ActionDomain, Block
from .behavior import EquilibriumResult, oldest_first
from .clearance import ClearanceResult
from .engine import KernelResult
from .errors import ContractError
from .loss import LossBreakdown, TerminalMassCorrection
from .keys import SourceKey
from .state import ModelState
from .transition import TransitionAudit


@dataclass(frozen=True)
class AcceptanceReport:
    information_timing: bool
    action_feasibility: bool
    behavioral_closure: bool
    physical_closure: bool
    objective_closure: bool
    messages: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all(
            (
                self.information_timing,
                self.action_feasibility,
                self.behavioral_closure,
                self.physical_closure,
                self.objective_closure,
            )
        )


@dataclass(frozen=True)
class PeriodInformation:
    """Complete information ledger for one executed decision period."""

    decision_time: Any
    information_timestamps: tuple[Any, ...]


def _complete_value_match(left: Any, right: Any, tolerance: float) -> bool:
    """Tolerance-aware equality for every component of the augmented state."""

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(
                np.allclose(
                    np.asarray(left, dtype=float),
                    np.asarray(right, dtype=float),
                    atol=tolerance,
                    rtol=0.0,
                    equal_nan=True,
                )
            )
        except (TypeError, ValueError):
            return False
    if isinstance(left, Number) and isinstance(right, Number):
        return bool(np.isclose(left, right, atol=tolerance, rtol=0.0, equal_nan=True))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _complete_value_match(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _complete_value_match(a, b, tolerance) for a, b in zip(left, right)
        )
    if is_dataclass(left) and is_dataclass(right) and type(left) is type(right):
        return all(
            _complete_value_match(
                getattr(left, field.name), getattr(right, field.name), tolerance
            )
            for field in fields(left)
        )
    try:
        equal = left == right
        return bool(np.all(equal)) if isinstance(equal, np.ndarray) else bool(equal)
    except (TypeError, ValueError):
        return False


def _behavior_certificate(
    equilibrium: EquilibriumResult,
    source_masses: Mapping[SourceKey, float],
    state: ModelState,
    tolerance: float,
) -> bool:
    release_ok = True
    for cargo_class, release in equilibrium.releases.items():
        if cargo_class not in state.waiting:
            return False
        expected = oldest_first(float(np.asarray(release).sum()), state.waiting[cargo_class])
        release_ok = release_ok and np.allclose(release, expected, atol=tolerance)
    source_conservation_ok = (
        set(equilibrium.flows) == set(source_masses)
        and all(
            abs(sum(equilibrium.flows.get(source, {}).values()) - mass) <= tolerance
            for source, mass in source_masses.items()
        )
        and all(
            value >= -tolerance and np.isfinite(value)
            for flows in equilibrium.flows.values()
            for value in flows.values()
        )
    )
    # A numerically small but strictly positive released vintage still has a
    # well-defined SUE simplex and therefore appears in normalized_shares.
    # The numerical tolerance belongs to balance residuals; using it to erase
    # a positive source makes the source-set certificate internally
    # inconsistent whenever 0 < mass <= tolerance.  Shares are checked against
    # the represented source-flow slice.  At ordinary scales its sum is the
    # source mass; for IEEE-754 subnormal vintages this preserves the simplex
    # even when individual alternatives underflow independently.
    positive_sources = {
        source for source, mass in source_masses.items() if mass > 0.0
    }
    simplex_ok = set(equilibrium.normalized_shares) == positive_sources and all(
        shares
        and set(shares) == set(equilibrium.flows[source])
        and all(value >= -tolerance and np.isfinite(value) for value in shares.values())
        and abs(sum(shares.values()) - 1.0) <= tolerance
        and all(
            abs(
                shares[choice]
                - (
                    equilibrium.flows[source][choice]
                    / sum(equilibrium.flows[source].values())
                    if sum(equilibrium.flows[source].values()) > 0.0
                    else 1.0 / len(equilibrium.flows[source])
                )
            )
            <= tolerance
            for choice in shares
        )
        for source, shares in equilibrium.normalized_shares.items()
    )
    positive_mass = sum(source_masses.values()) > tolerance
    if positive_mass:
        names = tuple(record.name for record in equilibrium.starts)
        selected_records = [
            record for record in equilibrium.starts if record.name == equilibrium.selected_start
        ]
        start_trace_ok = (
            set(names) == {"previous", "free_flow", "dispersed"}
            and len(names) == 3
            and len(selected_records) == 1
            and selected_records[0].converged
            and abs(selected_records[0].residual - equilibrium.residual) <= tolerance
            and selected_records[0].iterations == equilibrium.iterations
            and selected_records[0].selected_step_multipliers
            == equilibrium.selected_step_multipliers
        )
    else:
        start_trace_ok = equilibrium.residual <= tolerance
    return (
        equilibrium.status == "converged"
        and equilibrium.residual <= tolerance
        and release_ok
        and source_conservation_ok
        and simplex_ok
        and start_trace_ok
    )


def _physical_certificate(audit: TransitionAudit, tolerance: float) -> bool:
    residuals = (
        audit.adaptive_mass_residual,
        audit.committed_mass_residual,
        audit.pipeline_mass_residual,
        audit.tagged_balance_residual,
        audit.action_feasibility_violation,
        audit.equilibrium_residual,
        max(-audit.minimum_state_component, 0.0),
        audit.unavailable_route_service_violation,
        audit.readiness_order_balance_residual,
        audit.readiness_stock_balance_residual,
        audit.direct_order_balance_residual,
        audit.direct_stock_balance_residual,
        audit.readiness_stock_feasibility_violation,
        audit.budget_balance_residual,
        audit.corridor_history_residual,
        audit.serviceability_history_residual,
        audit.previous_share_history_residual,
    )
    vintage_balance_keys = [
        (cargo_class, vintage)
        for cargo_class, vintage, _ in audit.waiting_vintage_balance_residuals
    ]
    no_reset_keys = [
        cargo_class
        for cargo_class, _ in audit.waiting_vintage_no_reset_residuals
    ]
    vintage_coverage_ok = (
        audit.waiting_vintage_certificate_complete
        and
        len(vintage_balance_keys) == audit.waiting_vintage_expected_balance_count
        and len(set(vintage_balance_keys)) == len(vintage_balance_keys)
        and len(no_reset_keys) == audit.waiting_vintage_expected_no_reset_count
        and len(set(no_reset_keys)) == len(no_reset_keys)
    )
    vintage_residuals_ok = all(
        np.isfinite(value) and abs(value) <= tolerance
        for _, _, value in audit.waiting_vintage_balance_residuals
    ) and all(
        np.isfinite(value) and abs(value) <= tolerance
        for _, value in audit.waiting_vintage_no_reset_residuals
    )
    return (
        audit.passed
        and audit.state_complete
        and all(np.isfinite(value) and abs(value) <= tolerance for value in residuals)
        and vintage_coverage_ok
        and vintage_residuals_ok
    )


def evaluate_acceptance(
    *,
    decision_time: Any,
    information_timestamps: Sequence[Any],
    state: ModelState,
    action: Action,
    action_domain: ActionDomain,
    equilibrium: EquilibriumResult,
    source_masses: Mapping[SourceKey, float],
    transition_audit: TransitionAudit,
    loss: LossBreakdown,
    tolerance: float,
) -> AcceptanceReport:
    messages: list[str] = []
    information_ok = all(timestamp <= decision_time for timestamp in information_timestamps)
    if not information_ok:
        messages.append("At least one feature or scenario weight used unreleased information")
    action_layout_complete = set(Block).issubset(
        {key.block for key in action_domain.keys}
    )
    action_ok = (
        action_layout_complete
        and action_domain.violation(action, state) <= tolerance
    )
    if not action_ok:
        messages.append(
            "The five-block projected action layout or a phase, stock, component, or budget bound failed"
        )
    behavior_ok = _behavior_certificate(equilibrium, source_masses, state, tolerance)
    if not behavior_ok:
        messages.append("Oldest-first traceability or the selected SUE certificate failed")
    physical_ok = _physical_certificate(transition_audit, tolerance)
    if not physical_ok:
        messages.append(
            "Tagged conservation, per-vintage no-reset, pipeline, capacity, budget, or state audit failed"
        )
    components = np.asarray(
        [loss.queue, loss.waiting, loss.exit, loss.overflow, loss.route_resource, loss.action],
        dtype=float,
    )
    objective_ok = np.all(np.isfinite(components)) and np.all(components >= -tolerance)
    if not objective_ok:
        messages.append("The operational loss breakdown is incomplete or invalid")
    return AcceptanceReport(
        information_ok,
        action_ok,
        behavior_ok,
        physical_ok,
        objective_ok,
        tuple(messages),
    )


def source_masses_for_acceptance(result: KernelResult) -> dict[SourceKey, float]:
    """Recover formal decision masses without summing possibly underflowed flows.

    The demand split and oldest-first release ledger are the authoritative
    source-mass records.  Reconstructing masses from route-flow sums can erase
    an IEEE-754 subnormal but strictly positive released vintage while its
    normalized SUE simplex remains well defined.
    """
    masses = {
        SourceKey(cargo_class, None): float(mass)
        for cargo_class, mass in result.transition.demand_split.decision_eligible.items()
    }
    for cargo_class, release in result.equilibrium.releases.items():
        for vintage, mass in enumerate(np.asarray(release, dtype=float)):
            if mass > 0.0:
                masses[SourceKey(cargo_class, vintage)] = float(mass)
    return masses


# Backward-compatible private alias for frozen downstream audit scripts.
_source_masses = source_masses_for_acceptance


def evaluate_trajectory_acceptance(
    *,
    initial_state: ModelState,
    decision_results: Sequence[KernelResult],
    decision_information: Sequence[PeriodInformation],
    action_domain: ActionDomain,
    clearance: ClearanceResult,
    terminal_cost: TerminalMassCorrection,
    reported_total_objective: float,
    tolerance: float,
) -> AcceptanceReport:
    """Evaluate the complete Chapter 4 trajectory contract.

    Unlike :func:`evaluate_acceptance`, this routine cannot pass on a single
    period.  It requires the explicit frozen-recovery tail and reconciles the
    reported objective to decision loss + clearance loss + one terminal
    correction.
    """

    if tolerance <= 0:
        raise ContractError("Trajectory-acceptance tolerance must be positive")
    messages: list[str] = []
    if len(decision_results) != len(decision_information):
        raise ContractError("Every decision result needs its own information ledger")
    all_results = (*decision_results, *clearance.transitions)
    unique_trace = len({id(result) for result in all_results}) == len(all_results)
    if not unique_trace:
        messages.append("A transition was counted more than once in the trajectory")

    current = initial_state
    period_reports: list[AcceptanceReport] = []
    trace_ok = True
    for result, information in zip(decision_results, decision_information):
        if result.input_state is None or not _complete_value_match(
            current, result.input_state, tolerance
        ):
            trace_ok = False
        report = evaluate_acceptance(
            decision_time=information.decision_time,
            information_timestamps=information.information_timestamps,
            state=current,
            action=result.action,
            action_domain=action_domain,
            equilibrium=result.equilibrium,
            source_masses=source_masses_for_acceptance(result),
            transition_audit=result.transition.audit,
            loss=result.transition.loss,
            tolerance=tolerance,
        )
        period_reports.append(report)
        if result.transition.next_state.period != current.period + 1:
            trace_ok = False
        current = result.transition.next_state

    for result in clearance.transitions:
        if result.input_state is None or not _complete_value_match(
            current, result.input_state, tolerance
        ):
            trace_ok = False
        report = evaluate_acceptance(
            decision_time=current.period,
            information_timestamps=(),
            state=current,
            action=result.action,
            action_domain=action_domain,
            equilibrium=result.equilibrium,
            source_masses=source_masses_for_acceptance(result),
            transition_audit=result.transition.audit,
            loss=result.transition.loss,
            tolerance=tolerance,
        )
        period_reports.append(report)
        if result.transition.next_state.period != current.period + 1:
            trace_ok = False
        current = result.transition.next_state

    clearance_trace_loss = float(
        sum(result.transition.loss.total for result in clearance.transitions)
    )
    try:
        clearance.final_state.validate(tolerance=tolerance)
        final_state_complete = True
    except ContractError:
        final_state_complete = False
    expected_terminal = (
        0.0 if clearance.cleared else terminal_cost.compute(clearance.final_state)
    )
    clearance_ok = (
        clearance.weeks == len(clearance.transitions)
        and clearance.cleared != clearance.right_censored
        and abs(clearance_trace_loss - clearance.operational_loss) <= tolerance
        and final_state_complete
        and _complete_value_match(current, clearance.final_state, tolerance)
        and abs(clearance.terminal_correction - expected_terminal) <= tolerance
        and clearance.operational_loss >= -tolerance
        and clearance.terminal_correction >= -tolerance
        and np.isfinite(clearance.operational_loss)
        and np.isfinite(clearance.terminal_correction)
    )
    if not trace_ok or not clearance_ok:
        messages.append("The ordered decision/clearance state trace is incomplete or inconsistent")

    reconstructed_objective = float(
        sum(result.transition.loss.total for result in decision_results)
        + clearance.operational_loss
        + clearance.terminal_correction
    )
    objective_ok = (
        unique_trace
        and clearance_ok
        and np.isfinite(reported_total_objective)
        and abs(reconstructed_objective - reported_total_objective) <= tolerance
        and all(report.objective_closure for report in period_reports)
    )
    if not objective_ok:
        messages.append(
            "Reported objective does not equal decision loss, clearance loss, and one terminal correction"
        )

    information_ok = all(report.information_timing for report in period_reports)
    action_ok = set(Block).issubset({key.block for key in action_domain.keys}) and all(
        report.action_feasibility for report in period_reports
    )
    if not action_ok:
        messages.append("The trajectory did not register and certify all five action blocks")
    behavior_ok = all(report.behavioral_closure for report in period_reports)
    physical_ok = (
        trace_ok
        and clearance_ok
        and all(report.physical_closure for report in period_reports)
    )
    for report in period_reports:
        messages.extend(report.messages)
    return AcceptanceReport(
        information_timing=information_ok,
        action_feasibility=action_ok,
        behavioral_closure=behavior_ok,
        physical_closure=physical_ok,
        objective_closure=objective_ok,
        messages=tuple(dict.fromkeys(messages)),
    )
