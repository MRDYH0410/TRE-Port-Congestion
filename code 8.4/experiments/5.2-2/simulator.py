"""Formal common-kernel policy execution, clearance, and trajectory logging."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd

from tre84.acceptance import evaluate_acceptance
from tre84.actions import Action, ProjectionResult
from tre84.clearance import ClearanceRunner
from tre84.engine import KernelResult
from tre84.keys import ResourceKey, Stage, Tag
from tre84.metrics import aggregate_stage_queues, compute_trajectory_statistics
from tre84.scenarios import ScenarioBundle
from tre84.state import ModelState
from tre84.transition import ExogenousRealization

from features import state_features
from model import BenchmarkModel
from paths import PhysicalPath
from preparation import build_realization, prepare_period


@dataclass
class PolicyDecision:
    raw_action: Action
    decision_source: str
    policy_diagnostics: dict[str, Any] = field(default_factory=dict)
    proposal_records: list[dict[str, Any]] = field(default_factory=list)


class BenchmarkPolicy(Protocol):
    name: str
    training_seed: int | None

    def decide(
        self,
        *,
        state: ModelState,
        row: Mapping[str, Any],
        path: PhysicalPath,
        offset: int,
        bundle: ScenarioBundle,
    ) -> PolicyDecision: ...


@dataclass
class ReplicationArtifacts:
    replication: dict[str, Any]
    periods: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    solver_diagnostics: list[dict[str, Any]]
    contract_checks: dict[str, Any]
    proposal_records: list[dict[str, Any]]


def _released_information_hash(path: PhysicalPath) -> str:
    columns = [
        "week",
        "filtered_high_risk_probability",
        "lead_time_high_risk_probability",
        "release_date",
        "source_observation_month",
        "timing_valid",
        "information_source",
    ]
    payload = path.frame[columns].copy()
    for column in ("week", "release_date", "source_observation_month"):
        payload[column] = pd.to_datetime(payload[column]).dt.strftime("%Y-%m-%d")
    return hashlib.sha256(
        payload.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _active_constraints(
    model: BenchmarkModel,
    state: ModelState,
    projection: ProjectionResult,
) -> str:
    vector = projection.action.vector(model.layout.keys)
    lower, upper = model.domain.bounds(state)
    names: list[str] = []
    tolerance = 10 * float(model.config["action"]["projection_tolerance"])
    for name, value, lo, hi in zip(model.layout.names, vector, lower, upper):
        if abs(value - lo) <= tolerance:
            names.append(f"{name}:lower")
        if abs(value - hi) <= tolerance:
            names.append(f"{name}:upper_or_stock")
    if projection.active_budget:
        names.append("period_or_remaining_budget")
    return "|".join(names)


def _period_record(
    *,
    model: BenchmarkModel,
    policy: str,
    path: PhysicalPath,
    seed: int | None,
    offset: int,
    scope: str,
    state: ModelState,
    result: KernelResult,
) -> dict[str, Any]:
    before = aggregate_stage_queues(state, model.network)
    after = aggregate_stage_queues(result.transition.next_state, model.network)
    direct_mass = float(sum(result.transition.direct_exit.values()))
    attrition_mass = float(sum(result.transition.duration_attrition.values()))
    exit_unit_cost = float(model.config["behavior"]["exit_failure_cost_per_unit"])
    record: dict[str, Any] = {
        "policy": policy,
        "path_id": path.path_id,
        "path_content_sha256": path.path_hash,
        "training_seed": seed,
        "scope": scope,
        "period_offset": offset,
        "state_period": state.period,
        "queue_loss": result.transition.loss.queue,
        "waiting_loss": result.transition.loss.waiting,
        "exit_loss": result.transition.loss.exit,
        "direct_sue_exit_loss": direct_mass * exit_unit_cost,
        "duration_attrition_loss": attrition_mass * exit_unit_cost,
        "overload_loss": result.transition.loss.overflow,
        "route_resource_loss": result.transition.loss.route_resource,
        "action_loss": result.transition.loss.action,
        "period_operational_loss": result.transition.loss.total,
        "direct_sue_exit_mass": direct_mass,
        "duration_attrition_mass": attrition_mass,
        "delivered_landbridge": float(sum(result.transition.delivered.values())),
        "waiting_before": state.waiting_mass(),
        "waiting_after": result.transition.next_state.waiting_mass(),
        "pipeline_before": state.pipeline_mass(),
        "pipeline_after": result.transition.next_state.pipeline_mass(),
        "tagged_before": state.tagged_mass(),
        "tagged_after": result.transition.next_state.tagged_mass(),
        "outstanding_after": result.transition.next_state.cargo_mass(),
        "budget_remaining": result.transition.next_state.budget,
        "sue_residual": result.equilibrium.residual,
        "sue_iterations": result.equilibrium.iterations,
        "transition_audit_passed": result.transition.audit.passed,
        "maximum_mass_residual": max(
            result.transition.audit.adaptive_mass_residual,
            result.transition.audit.committed_mass_residual,
            result.transition.audit.pipeline_mass_residual,
            result.transition.audit.tagged_balance_residual,
        ),
    }
    for resource in model.resources:
        key = f"{resource.stage.value}_{resource.location}".replace(" ", "_")
        record[f"queue_before_{key}"] = before[resource]
        record[f"queue_after_{key}"] = after[resource]
    return record


class RecoveryRule:
    def __init__(self, model: BenchmarkModel) -> None:
        self.model = model

    def action(self, state: ModelState) -> Action:
        return self.model.zero_action()

    def realization(self, state: ModelState) -> ExogenousRealization:
        cargo = str(self.model.config["cargo_class"])
        return ExogenousRealization(
            gulf_demand={cargo: 0.0},
            serviceable_share={cargo: 1.0},
            committed_fraction={cargo: float(self.model.config["committed_fraction_reference"])},
            committed_route_share=self.model.committed_shares,
            base_arrivals={},
            choice_route_available=frozenset(self.model.network.routes),
            physical_route_available=frozenset(self.model.network.routes),
            serviceability_observation=1.0,
            next_disruption_seen=state.disruption_seen,
            next_disruption_active=False,
            next_disruption_duration=0,
            next_risk=state.risk,
            next_observed_covariates=dict(state.observed_covariates),
        )


def run_replication(
    *,
    model: BenchmarkModel,
    policy: BenchmarkPolicy,
    path: PhysicalPath,
) -> ReplicationArtifacts:
    first = path.frame.iloc[0].to_dict()
    state = model.initial_state(first)
    decision_initial = state.clone()
    results: list[KernelResult] = []
    periods: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    step_acceptance: list[bool] = []
    decision_seconds: list[float] = []

    rows = [row._asdict() for row in path.frame.itertuples(index=False)]
    for offset, row in enumerate(rows):
        prepared = prepare_period(model=model, state=state, row=row)
        state = prepared.state
        controller_input_hash = hashlib.sha256(
            state_features(state, model).astype("<f8", copy=False).tobytes()
        ).hexdigest()
        started = time.perf_counter()
        decision = policy.decide(
            state=state,
            row=row,
            path=path,
            offset=offset,
            bundle=prepared.scenarios,
        )
        elapsed = time.perf_counter() - started
        projection = model.projector.project(decision.raw_action, state)
        realization = build_realization(model=model, state=state, row=row)
        result = model.kernel.execute(
            state=state,
            action=projection.action,
            realization=realization,
            projection=projection,
        )
        source_masses = {
            source: float(sum(flow.values()))
            for source, flow in result.equilibrium.flows.items()
        }
        acceptance = evaluate_acceptance(
            decision_time=row["week"],
            information_timestamps=prepared.scenarios.information_timestamps,
            state=state,
            action=projection.action,
            action_domain=model.domain,
            equilibrium=result.equilibrium,
            source_masses=source_masses,
            transition_audit=result.transition.audit,
            loss=result.transition.loss,
            tolerance=float(model.config["numerics"]["mass_tolerance"]),
        )
        step_acceptance.append(acceptance.passed)
        decision_seconds.append(elapsed)
        requested = decision.raw_action.vector(model.layout.keys)
        implemented = projection.action.vector(model.layout.keys)
        action_record: dict[str, Any] = {
            "policy": policy.name,
            "path_id": path.path_id,
            "path_content_sha256": path.path_hash,
            "training_seed": policy.training_seed,
            "decision_week": row["week"],
            "period_offset": offset,
            "decision_source": decision.decision_source,
            "information_vector_sha256": prepared.information_vector_hash,
            "observation_sha256": prepared.observation_hash,
            "controller_input_sha256": controller_input_hash,
            "scenario_ids": "|".join(state.risk.scenario_ids),
            "readiness_weights": "|".join(
                f"{value:.17g}" for value in state.risk.readiness_weights
            ),
            "operational_weights": "|".join(
                f"{value:.17g}" for value in state.risk.operational_weights
            ),
            "latest_release_time": state.risk.latest_release_time,
            "projection_objective": projection.objective,
            "projection_iterations": projection.iterations,
            "projection_feasibility_violation": projection.feasibility_violation,
            "active_constraints": _active_constraints(model, state, projection),
            "decision_time_seconds": elapsed,
        }
        for name, raw, feasible in zip(model.layout.names, requested, implemented):
            action_record[f"requested_{name}"] = raw
            action_record[f"implemented_{name}"] = feasible
        actions.append(action_record)
        periods.append(
            _period_record(
                model=model,
                policy=policy.name,
                path=path,
                seed=policy.training_seed,
                offset=offset,
                scope="decision",
                state=state,
                result=result,
            )
        )
        diagnostics.append(
            {
                "policy": policy.name,
                "path_id": path.path_id,
                "training_seed": policy.training_seed,
                "scope": "decision",
                "period_offset": offset,
                "sue_status": result.equilibrium.status,
                "sue_residual": result.equilibrium.residual,
                "sue_iterations": result.equilibrium.iterations,
                "sue_kl_discrepancy": result.equilibrium.kl_discrepancy,
                "multi_start_dispersion": result.equilibrium.multi_start_dispersion,
                "transition_audit_passed": result.transition.audit.passed,
                "solver_failure": "",
            }
        )
        for proposal in decision.proposal_records:
            proposals.append(
                {
                    "policy": policy.name,
                    "path_id": path.path_id,
                    "training_seed": policy.training_seed,
                    "period_offset": offset,
                    **proposal,
                }
            )
        results.append(result)
        state = result.transition.next_state

    clearance = ClearanceRunner(
        kernel=model.kernel,
        recovery_rule=RecoveryRule(model),
        terminal_cost=model.terminal_cost,
        maximum_weeks=int(model.config["clearance"]["maximum_weeks"]),
        empty_tolerance=float(model.config["clearance"]["empty_tolerance"]),
    ).run(state)
    clearance_state = state
    for offset, result in enumerate(clearance.transitions):
        periods.append(
            _period_record(
                model=model,
                policy=policy.name,
                path=path,
                seed=policy.training_seed,
                offset=offset,
                scope="clearance",
                state=clearance_state,
                result=result,
            )
        )
        diagnostics.append(
            {
                "policy": policy.name,
                "path_id": path.path_id,
                "training_seed": policy.training_seed,
                "scope": "clearance",
                "period_offset": offset,
                "sue_status": result.equilibrium.status,
                "sue_residual": result.equilibrium.residual,
                "sue_iterations": result.equilibrium.iterations,
                "sue_kl_discrepancy": result.equilibrium.kl_discrepancy,
                "multi_start_dispersion": result.equilibrium.multi_start_dispersion,
                "transition_audit_passed": result.transition.audit.passed,
                "solver_failure": "",
            }
        )
        clearance_state = result.transition.next_state

    stats = compute_trajectory_statistics(
        initial_state=decision_initial,
        decision_results=results,
        network=model.network,
        thresholds=model.thresholds,
        clearance=clearance,
        include_clearance_in_physical_metrics=True,
        tolerance=float(model.config["numerics"]["loss_identity_tolerance"]),
    )
    exit_unit_cost = float(model.config["behavior"]["exit_failure_cost_per_unit"])
    direct_exit_loss = stats.direct_sue_exit * exit_unit_cost
    duration_attrition_loss = stats.duration_attrition * exit_unit_cost
    component_sum = (
        stats.loss_queue
        + stats.loss_waiting
        + stats.loss_exit
        + stats.loss_overflow
        + stats.loss_route_resource
        + stats.loss_action
        + stats.terminal_correction
    )
    replication = {
        "policy": policy.name,
        "path_id": path.path_id,
        "path_content_sha256": path.path_hash,
        "released_information_path_sha256": _released_information_hash(path),
        "training_seed": policy.training_seed,
        **stats.as_record(),
        "loss_direct_sue_exit": direct_exit_loss,
        "loss_duration_attrition": duration_attrition_loss,
        "loss_component_sum_with_terminal": component_sum,
        "all_step_acceptance_passed": all(step_acceptance),
        "mean_decision_time_seconds": float(np.mean(decision_seconds)),
        "maximum_decision_time_seconds": float(np.max(decision_seconds)),
        "action_dimension": len(model.layout.keys),
        "projector_id": "shared_weighted_euclidean_projector",
        "kernel_id": "shared_model_kernel_rcmsa_tagged_transition",
        "information_source": str(path.frame["information_source"].iloc[0]),
        "clearance_weeks_observed": stats.clearance_weeks_observed,
        "restricted_clearance_time_contribution": (
            stats.clearance_weeks_observed
            if stats.clearance_weeks_observed is not None
            else int(model.config["clearance"]["maximum_weeks"])
        ),
    }
    tolerance = float(model.config["numerics"]["loss_identity_tolerance"])
    contract = {
        "policy": policy.name,
        "path_id": path.path_id,
        "training_seed": policy.training_seed,
        "all_step_acceptance_passed": all(step_acceptance),
        "all_transition_audits_passed": stats.transition_audits_passed,
        "maximum_transition_residual": stats.maximum_transition_residual,
        "sue_residual_within_tolerance": all(
            row["sue_residual"] <= float(model.config["behavior"]["rcmsa_tolerance"])
            for row in diagnostics
        ),
        "budget_nonnegative": all(row["budget_remaining"] >= -tolerance for row in periods),
        "physical_states_nonnegative": all(row["outstanding_after"] >= -tolerance for row in periods),
        "loss_components_reconstruct_total": abs(component_sum - stats.total_operational_objective) <= tolerance,
        "exit_channels_counted_once": abs(direct_exit_loss + duration_attrition_loss - stats.loss_exit) <= tolerance,
        "right_censoring_not_observed_clearance": not (
            stats.right_censored and stats.clearance_weeks_observed is not None
        ),
        "oldest_first_release": stats.transition_audits_passed,
        "no_vintage_reset": stats.transition_audits_passed,
        "route_wait_exit_source_simplex_conserved": stats.transition_audits_passed,
        "committed_tag_permanence": stats.transition_audits_passed,
        "maritime_tag_conservation": stats.transition_audits_passed,
        "four_stage_tag_conservation": stats.transition_audits_passed,
        "no_double_internal_service": stats.transition_audits_passed,
        "shared_corridor_capacity_accounted": stats.transition_audits_passed,
        "delivered_only_from_landbridge_discharge": stats.transition_audits_passed,
        "no_equal_allocation_authority": policy.name != "Equal allocation",
    }
    return ReplicationArtifacts(replication, periods, actions, diagnostics, contract, proposals)
