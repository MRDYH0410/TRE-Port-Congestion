"""Matched evaluation through the shared Chapter 3/4 kernel."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tre84.acceptance import evaluate_acceptance
from tre84.clearance import ClearanceRunner
from tre84.keys import Stage
from tre84.metrics import aggregate_stage_queues, compute_trajectory_statistics

from controller_factory import (
    CapacityRightsProjector,
    build_policies,
    configure_capacity_rights,
)
from features import state_features
from information_design import InformationProvider, ReleaseTimingScenarioBuilder, load_hmm_inputs
from mechanism import _capacity_rows, _loss_row
from model import BenchmarkModel, build_model
from paths import PhysicalPath, build_test_paths, load_frozen_5_2_1_inputs
from preparation import build_realization, prepare_period
from simulator import RecoveryRule, _active_constraints


def _availability_monday(release_date: Any) -> pd.Timestamp:
    release = pd.Timestamp(release_date).normalize()
    return release + pd.Timedelta(days=(7 - release.weekday()) % 7)


def _information_path_hash(path: PhysicalPath) -> str:
    columns = [
        "week",
        "source_observation_month",
        "release_date",
        "filtered_high_risk_probability",
        "lead_time_high_risk_probability",
        "information_regime",
    ]
    frame = path.frame[columns].copy()
    for column in ("week", "source_observation_month", "release_date"):
        frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")
    ).hexdigest()


def _action_trace_rows(
    *,
    model: BenchmarkModel,
    rights_projector: CapacityRightsProjector,
    original_raw: Any,
    rights_raw: Any,
    projection: Any,
    state: Any,
    next_state: Any,
    metadata: Mapping[str, Any],
    decision_week: Any,
    period_offset: int,
    decision_source: str,
    elapsed: float,
    information_vector_sha256: str,
    observation_sha256: str,
    controller_input_sha256: str,
) -> list[dict[str, Any]]:
    lower, upper = model.domain.bounds(state)
    implemented = projection.action.vector(model.layout.keys)
    original = original_raw.vector(model.layout.keys)
    restricted = rights_raw.vector(model.layout.keys)
    period_cap = float(model.domain.budget_cap(state))
    projection_spent = float(model.domain.action_cost(projection.action))
    budget_slack = period_cap - projection_spent
    names = list(model.layout.names)
    upper_reference = np.asarray(model.action_upper, dtype=float)

    def action_map(values: np.ndarray) -> str:
        return json.dumps(
            {name: float(value) for name, value in zip(names, values)},
            sort_keys=True,
            separators=(",", ":"),
        )

    def block_total(values: np.ndarray, block: str) -> float:
        return float(
            sum(value for value, key in zip(values, model.layout.keys) if key.block.value == block)
        )

    return [
        {
            **metadata,
            "decision_week": decision_week,
            "period_offset": period_offset,
            "state_period": state.period,
            "phase": state.phase,
            "action_dimension": len(names),
            "action_names": "|".join(names),
            "original_requested_model_units_by_action": action_map(original),
            "rights_restricted_requested_model_units_by_action": action_map(restricted),
            "implemented_model_units_by_action": action_map(implemented),
            "original_requested_reference_ratio_by_action": action_map(original / upper_reference),
            "rights_restricted_reference_ratio_by_action": action_map(restricted / upper_reference),
            "implemented_reference_ratio_by_action": action_map(implemented / upper_reference),
            "coordinate_lower_bound_by_action": action_map(lower),
            "coordinate_upper_bound_by_action": action_map(upper),
            "coordinate_phase_slack_by_action": action_map(upper - implemented),
            "original_requested_readiness_order": block_total(original, "readiness_order"),
            "implemented_readiness_order": block_total(implemented, "readiness_order"),
            "original_requested_direct_order": block_total(original, "direct_order"),
            "implemented_direct_order": block_total(implemented, "direct_order"),
            "original_requested_readiness_exercise": block_total(original, "readiness_exercise"),
            "implemented_readiness_exercise": block_total(implemented, "readiness_exercise"),
            "original_requested_release": block_total(original, "release"),
            "implemented_release": block_total(implemented, "release"),
            "original_requested_disclosure": block_total(original, "disclosure"),
            "implemented_disclosure": block_total(implemented, "disclosure"),
            "projection_distance": float(np.sqrt(max(2.0 * projection.objective, 0.0))),
            "projection_spent": projection_spent,
            "period_or_remaining_budget_cap": period_cap,
            "budget_constraint_slack": budget_slack,
            "remaining_budget_before": float(state.budget),
            "remaining_budget_after": float(next_state.budget),
            "active_projection_constraints": _active_constraints(model, state, projection),
            "rights_restriction_applied_before_common_projection": True,
            "decision_source": decision_source,
            "decision_time_seconds": elapsed,
            "information_vector_sha256": information_vector_sha256,
            "observation_sha256": observation_sha256,
            "controller_input_sha256": controller_input_sha256,
            "scenario_label_in_controller_observation": False,
        }
    ]


def _capacity_trace_rows(
    *,
    model: BenchmarkModel,
    original_raw: Any,
    rights_raw: Any,
    projection: Any,
    state: Any,
    result: Any,
    metadata: Mapping[str, Any],
    row: Mapping[str, Any],
    period_offset: int,
) -> list[dict[str, Any]]:
    base = _capacity_rows(
        model=model,
        base_policy=str(metadata["controller_id"]),
        restriction=str(metadata["capacity_rights"]),
        path=metadata["path_object"],
        seed=int(metadata["training_seed"]),
        scope="decision",
        period_offset=period_offset,
        state=state,
        action=projection.action,
        result=result,
    )
    original_vector = original_raw.vector(model.layout.keys)
    restricted_vector = rights_raw.vector(model.layout.keys)
    event_onset = pd.Timestamp(row.get("event_onset", row["week"]))
    availability = _availability_monday(row["release_date"])
    readiness_lead = int(model.config["action"]["readiness_lead_weeks"])
    direct_lead = int(model.config["action"]["direct_lead_weeks"])
    output = []
    queues_before = aggregate_stage_queues(state, model.network)
    queues_after = aggregate_stage_queues(result.transition.next_state, model.network)
    before_stage = {
        stage: float(sum(value for resource, value in queues_before.items() if resource.stage == stage))
        for stage in (Stage.BERTH, Stage.YARD, Stage.GATE, Stage.CORRIDOR)
    }
    after_stage = {
        stage: float(sum(value for resource, value in queues_after.items() if resource.stage == stage))
        for stage in (Stage.BERTH, Stage.YARD, Stage.GATE, Stage.CORRIDOR)
    }
    direct_exit = float(sum(result.transition.direct_exit.values()))
    attrition_exit = float(sum(result.transition.duration_attrition.values()))
    delivered = float(sum(result.transition.delivered.values()))
    for index, record in enumerate(base):
        resource = model.controlled_resources[index]
        readiness_order_key = model.layout.readiness_order[index]
        direct_order_key = model.layout.direct_order[index]
        exercise_key = model.layout.readiness_exercise[index]
        ro_index = model.layout.keys.index(readiness_order_key)
        do_index = model.layout.keys.index(direct_order_key)
        ex_index = model.layout.keys.index(exercise_key)
        readiness_due = float(state.readiness.orders.get(resource, {}).get(1, 0.0))
        readiness_matured = float(model.config["capacity_technology"]["readiness_maturity_yield"]) * readiness_due
        stock_before = float(state.readiness.stock.get(resource, 0.0))
        exercise = float(projection.action.value(exercise_key))
        consumed = float(model.config["capacity_technology"]["readiness_consumption"]) * exercise
        readiness_decay = float(model.config["capacity_technology"]["readiness_decay"])
        readiness_expiry = readiness_decay * max(stock_before - consumed, 0.0)
        readiness_pipeline_before = float(sum(state.readiness.orders.get(resource, {}).values()))
        readiness_pipeline_after = float(
            sum(result.transition.next_state.readiness.orders.get(resource, {}).values())
        )
        readiness_stock_after = float(
            result.transition.next_state.readiness.stock.get(resource, 0.0)
        )
        direct_pipeline_before = float(
            sum(state.direct_capacity.orders.get(resource, {}).values())
        )
        direct_pipeline_after = float(
            sum(result.transition.next_state.direct_capacity.orders.get(resource, {}).values())
        )
        direct_stock_before = float(state.direct_capacity.stock.get(resource, 0.0))
        direct_stock_after = float(
            result.transition.next_state.direct_capacity.stock.get(resource, 0.0)
        )
        if direct_lead == 0:
            direct_arrival = float(model.config["capacity_technology"]["direct_maturity_yield"]) * float(
                projection.action.value(direct_order_key)
            )
        else:
            direct_arrival = float(model.config["capacity_technology"]["direct_maturity_yield"]) * float(
                state.direct_capacity.orders.get(resource, {}).get(1, 0.0)
            )
        enriched = {
            **metadata,
            **record,
            "decision_week": row["week"],
            "source_observation_month": row.get("source_observation_month", ""),
            "actual_public_release_date": row.get("actual_public_release_date", row["release_date"]),
            "scenario_release_date": row["release_date"],
            "decision_availability_week": availability,
            "event_onset": event_onset,
            "g_R_weeks": int((event_onset - availability).days // 7 - readiness_lead),
            "g_D_weeks": int((event_onset - availability).days // 7 - direct_lead),
            "released_filtered_high_risk_probability": float(row.get("released_filtered_high_risk_probability", row["filtered_high_risk_probability"])),
            "released_lead_high_risk_probability": float(row.get("released_lead_high_risk_probability", row["lead_time_high_risk_probability"])),
            "controller_current_high_risk_probability": float(row["filtered_high_risk_probability"]),
            "controller_lead_high_risk_probability": float(row["lead_time_high_risk_probability"]),
            "monthly_transitions_to_readiness_maturity": row.get("monthly_transitions_to_readiness_maturity", np.nan),
            "weekly_transition_matrix_applications": int(row.get("weekly_transition_matrix_applications", 0)),
            "original_requested_readiness_order": float(original_vector[ro_index]),
            "rights_restricted_readiness_order": float(restricted_vector[ro_index]),
            "original_requested_direct_order": float(original_vector[do_index]),
            "rights_restricted_direct_order": float(restricted_vector[do_index]),
            "original_requested_readiness_exercise": float(original_vector[ex_index]),
            "rights_restricted_readiness_exercise": float(restricted_vector[ex_index]),
            "readiness_matured_this_week": readiness_matured,
            "readiness_expiry_or_decay": readiness_expiry,
            "readiness_pipeline_before": readiness_pipeline_before,
            "readiness_pipeline_after": readiness_pipeline_after,
            "mature_readiness_stock_before": stock_before,
            "mature_readiness_stock_after": readiness_stock_after,
            "direct_capacity_arrival": direct_arrival,
            "direct_capacity_pipeline_before": direct_pipeline_before,
            "direct_capacity_pipeline_after": direct_pipeline_after,
            "active_direct_capacity_before": direct_stock_before,
            "active_direct_capacity_after": direct_stock_after,
            "usable_temporary_capacity": float(record["active_direct_capacity"]) + float(record["exercised_readiness_capacity"]),
            "normal_model_units": float(row["normal_model_units"]),
            "serviceability": float(row["serviceability"]),
            "blocked_model_units": float(row["normal_model_units"]) * (1.0 - float(row["serviceability"])),
            "waiting_mass_before": float(state.waiting_mass()),
            "waiting_mass_after": float(result.transition.next_state.waiting_mass()),
            "berth_queue_before": before_stage[Stage.BERTH],
            "yard_queue_before": before_stage[Stage.YARD],
            "gate_queue_before": before_stage[Stage.GATE],
            "landbridge_queue_before": before_stage[Stage.CORRIDOR],
            "berth_queue_after": after_stage[Stage.BERTH],
            "yard_queue_after": after_stage[Stage.YARD],
            "gate_queue_after": after_stage[Stage.GATE],
            "landbridge_queue_after": after_stage[Stage.CORRIDOR],
            "sue_exit_mass": direct_exit,
            "attrition_exit_mass": attrition_exit,
            "delivered_landbridge": delivered,
            "preparation_period": bool(row.get("preparation_period", False)),
            "event_period": bool(row.get("event_period", True)),
        }
        for horizon in range(5):
            enriched[f"released_high_risk_forecast_h{horizon}_months"] = row.get(
                f"high_risk_forecast_h{horizon}_months", np.nan
            )
        enriched.pop("path_object", None)
        output.append(enriched)
    if not output:
        return []
    aggregate = dict(output[0])
    sum_columns = (
        "base_capacity",
        "active_direct_capacity",
        "exercised_readiness_capacity",
        "effective_capacity",
        "resource_workload",
        "resource_threshold",
        "resource_overload_mass",
        "implemented_readiness_order",
        "implemented_direct_order",
        "implemented_readiness_exercise",
        "original_requested_readiness_order",
        "rights_restricted_readiness_order",
        "original_requested_direct_order",
        "rights_restricted_direct_order",
        "original_requested_readiness_exercise",
        "rights_restricted_readiness_exercise",
        "readiness_matured_this_week",
        "readiness_expiry_or_decay",
        "readiness_pipeline_before",
        "readiness_pipeline_after",
        "mature_readiness_stock_before",
        "mature_readiness_stock_after",
        "direct_capacity_arrival",
        "direct_capacity_pipeline_before",
        "direct_capacity_pipeline_after",
        "active_direct_capacity_before",
        "active_direct_capacity_after",
        "usable_temporary_capacity",
    )
    for column in sum_columns:
        aggregate[column] = float(sum(float(item[column]) for item in output))
    aggregate["resource"] = "all_controlled_resources"
    aggregate["stage"] = "resource_detail_json"
    aggregate["location"] = "resource_detail_json"
    aggregate["resource_count"] = len(output)
    detail_columns = (
        "resource",
        "stage",
        "location",
        "base_capacity",
        "active_direct_capacity",
        "exercised_readiness_capacity",
        "effective_capacity",
        "feedback_multiplier",
        "yard_to_berth_feedback",
        "corridor_to_gate_feedback",
        "shared_corridor_weight",
        "resource_workload",
        "resource_threshold",
        "resource_pressure",
        "resource_overload_ratio",
        "resource_overload_mass",
        "implemented_readiness_order",
        "implemented_direct_order",
        "implemented_readiness_exercise",
        "original_requested_readiness_order",
        "rights_restricted_readiness_order",
        "original_requested_direct_order",
        "rights_restricted_direct_order",
        "original_requested_readiness_exercise",
        "rights_restricted_readiness_exercise",
        "readiness_matured_this_week",
        "readiness_expiry_or_decay",
        "readiness_pipeline_before",
        "readiness_pipeline_after",
        "mature_readiness_stock_before",
        "mature_readiness_stock_after",
        "direct_capacity_arrival",
        "direct_capacity_pipeline_before",
        "direct_capacity_pipeline_after",
        "active_direct_capacity_before",
        "active_direct_capacity_after",
        "usable_temporary_capacity",
    )
    resource_details = [
        {column: item[column] for column in detail_columns} for item in output
    ]
    aggregate["resource_detail_json"] = json.dumps(
        resource_details,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
    )
    aggregate["feedback_multiplier"] = float(np.mean([item["feedback_multiplier"] for item in output]))
    aggregate["yard_to_berth_feedback"] = float(np.mean([item["yard_to_berth_feedback"] for item in output]))
    aggregate["corridor_to_gate_feedback"] = float(np.mean([item["corridor_to_gate_feedback"] for item in output]))
    aggregate["resource_pressure"] = float(max(item["resource_pressure"] for item in output))
    aggregate["resource_overload_ratio"] = float(max(item["resource_overload_ratio"] for item in output))
    return [aggregate]


def run_information_replication(
    *,
    model: BenchmarkModel,
    rights_projector: CapacityRightsProjector,
    policy: Any,
    path: PhysicalPath,
    layer: str,
    controller_id: str,
    training_information_regime: str,
    evaluation_information_regime: str,
    capacity_rights: str,
    warning_scenario: str,
    controller_bundle_sha256: str,
) -> dict[str, Any]:
    first = path.frame.iloc[0].to_dict()
    state = model.initial_state(first)
    initial = state.clone()
    decision_results = []
    actions: list[dict[str, Any]] = []
    capacity: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    step_acceptance: list[bool] = []
    decision_seconds: list[float] = []
    information_hashes: list[str] = []
    observation_hashes: list[str] = []
    controller_input_hashes: list[str] = []
    selector_counts = {"BC": 0, "SAC": 0, "fallback": 0}
    rows = [row._asdict() for row in path.frame.itertuples(index=False)]
    base_path_id = str(path.frame["base_path_id"].iloc[0])
    physical_hash = str(path.frame["base_physical_path_sha256"].iloc[0])
    metadata: dict[str, Any] = {
        "evidence_layer": layer,
        "controller_id": controller_id,
        "training_information_regime": training_information_regime,
        "evaluation_information_regime": evaluation_information_regime,
        "capacity_rights": capacity_rights,
        "warning_scenario": warning_scenario,
        "base_path_id": base_path_id,
        "base_physical_path_sha256": physical_hash,
        "path_id": path.path_id,
        "path_content_sha256": path.path_hash,
        "training_seed": policy.training_seed,
        "controller_bundle_sha256": controller_bundle_sha256,
        "path_object": path,
    }
    for offset, row in enumerate(rows):
        prepared = prepare_period(model=model, state=state, row=row)
        state = prepared.state
        controller_input_hash = hashlib.sha256(
            state_features(state, model).astype("<f8", copy=False).tobytes()
        ).hexdigest()
        information_hashes.append(prepared.information_vector_hash)
        observation_hashes.append(prepared.observation_hash)
        controller_input_hashes.append(controller_input_hash)
        started = time.perf_counter()
        decision = policy.decide(
            state=state,
            row=row,
            path=path,
            offset=offset,
            bundle=prepared.scenarios,
        )
        elapsed = time.perf_counter() - started
        rights_raw = rights_projector.restrict(decision.raw_action)
        projection = rights_projector.inner.project(rights_raw, state)
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
        actions.extend(
            _action_trace_rows(
                model=model,
                rights_projector=rights_projector,
                original_raw=decision.raw_action,
                rights_raw=rights_raw,
                projection=projection,
                state=state,
                next_state=result.transition.next_state,
                metadata={key: value for key, value in metadata.items() if key != "path_object"},
                decision_week=row["week"],
                period_offset=offset,
                decision_source=decision.decision_source,
                elapsed=elapsed,
                information_vector_sha256=prepared.information_vector_hash,
                observation_sha256=prepared.observation_hash,
                controller_input_sha256=controller_input_hash,
            )
        )
        capacity.extend(
            _capacity_trace_rows(
                model=model,
                original_raw=decision.raw_action,
                rights_raw=rights_raw,
                projection=projection,
                state=state,
                result=result,
                metadata=metadata,
                row=row,
                period_offset=offset,
            )
        )
        loss = _loss_row(
            model=model,
            base_policy=controller_id,
            restriction=capacity_rights,
            path=path,
            seed=policy.training_seed,
            scope="decision",
            period_offset=offset,
            state=state,
            result=result,
        )
        loss.update({key: value for key, value in metadata.items() if key != "path_object"})
        loss["decision_week"] = row["week"]
        losses.append(loss)
        for proposal in decision.proposal_records:
            if bool(proposal.get("selected")):
                source = str(proposal.get("proposal_source", "fallback"))
                selector_counts[source if source in {"BC", "SAC"} else "fallback"] += 1
        if decision.decision_source.endswith("fallback"):
            selector_counts["fallback"] += 1
        decision_results.append(result)
        state = result.transition.next_state
    clearance = ClearanceRunner(
        kernel=model.kernel,
        recovery_rule=RecoveryRule(model),
        terminal_cost=model.terminal_cost,
        maximum_weeks=int(model.config["clearance"]["maximum_weeks"]),
        empty_tolerance=float(model.config["clearance"]["empty_tolerance"]),
    ).run(state)
    stats = compute_trajectory_statistics(
        initial_state=initial,
        decision_results=decision_results,
        network=model.network,
        thresholds=model.thresholds,
        clearance=clearance,
        include_clearance_in_physical_metrics=True,
        tolerance=float(model.config["numerics"]["loss_identity_tolerance"]),
    )
    exit_unit = float(model.config["behavior"]["exit_failure_cost_per_unit"])
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
        **{key: value for key, value in metadata.items() if key != "path_object"},
        "released_information_path_sha256": _information_path_hash(path),
        **stats.as_record(),
        "loss_direct_sue_exit": stats.direct_sue_exit * exit_unit,
        "loss_duration_attrition": stats.duration_attrition * exit_unit,
        "loss_component_sum_with_terminal": component_sum,
        "all_step_acceptance_passed": all(step_acceptance),
        "mean_decision_time_seconds": float(np.mean(decision_seconds)),
        "maximum_decision_time_seconds": float(np.max(decision_seconds)),
        "bc_proposal_selected_count": selector_counts["BC"],
        "sac_proposal_selected_count": selector_counts["SAC"],
        "fallback_count": selector_counts["fallback"],
        "action_dimension": len(model.layout.keys),
        "projector_id": "capacity_rights_then_shared_weighted_euclidean_projector",
        "kernel_id": "shared_model_kernel_rcmsa_tagged_transition",
        "information_vector_sequence_sha256": hashlib.sha256(
            "|".join(information_hashes).encode("utf-8")
        ).hexdigest(),
        "observation_sequence_sha256": hashlib.sha256(
            "|".join(observation_hashes).encode("utf-8")
        ).hexdigest(),
        "controller_input_sequence_sha256": hashlib.sha256(
            "|".join(controller_input_hashes).encode("utf-8")
        ).hexdigest(),
        "scenario_label_in_controller_observation": False,
        "clearance_weeks_observed": stats.clearance_weeks_observed,
        "restricted_clearance_time_contribution": (
            stats.clearance_weeks_observed
            if stats.clearance_weeks_observed is not None
            else int(model.config["clearance"]["maximum_weeks"])
        ),
    }
    tolerance = float(model.config["numerics"]["loss_identity_tolerance"])
    contract = {
        **{key: value for key, value in metadata.items() if key != "path_object"},
        "all_step_acceptance_passed": all(step_acceptance),
        "maximum_transition_residual": stats.maximum_transition_residual,
        "loss_components_reconstruct_total": abs(component_sum - stats.total_operational_objective) <= tolerance,
        "rights_before_common_projection": True,
        "readiness_disabled_when_required": (
            capacity_rights in {"RD", "R"}
            or all(
                row["implemented_readiness_order"] <= tolerance
                and row["implemented_readiness_exercise"] <= tolerance
                for row in actions
            )
        ),
        "direct_disabled_when_required": (
            capacity_rights in {"RD", "D"}
            or all(
                row["implemented_direct_order"] <= tolerance
                for row in actions
            )
        ),
        "readiness_stock_nonnegative": all(
            row["mature_readiness_stock_after"] >= -tolerance for row in capacity
        ),
        "direct_stock_nonnegative": all(
            row["active_direct_capacity_after"] >= -tolerance for row in capacity
        ),
        "monthly_matrix_not_applied_weekly": all(
            int(row.get("weekly_transition_matrix_applications", 0)) == 0 for row in rows
        ),
        "release_not_after_decision": all(
            pd.Timestamp(row["release_date"]) <= pd.Timestamp(row["week"]) for row in rows
        ),
        "right_censoring_not_recorded_as_clearance": not (
            stats.right_censored and stats.clearance_weeks_observed is not None
        ),
        "scenario_label_absent_from_controller_observation": True,
        "information_hashes_enter_controller_trace": bool(
            len(information_hashes) == len(rows)
            and len(observation_hashes) == len(rows)
            and len(controller_input_hashes) == len(rows)
        ),
    }
    return {
        "replication": replication,
        "actions": actions,
        "capacity": capacity,
        "losses": losses,
        "contract": contract,
    }


def evaluation_worker(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct a hash-locked task in an independent worker process."""

    code_root = Path(spec["code_root"])
    benchmark_config = json.loads(Path(spec["benchmark_config_path"]).read_text(encoding="utf-8"))
    model = build_model(benchmark_config)
    rights = str(spec["capacity_rights"])
    rights_projector = configure_capacity_rights(model, rights)
    frozen = load_frozen_5_2_1_inputs(benchmark_config)
    count = int(spec["executed_paths"])
    base_paths = build_test_paths(config=benchmark_config, frozen=frozen, count=count)
    base_path = base_paths[int(spec["path_index"])]
    hmm = load_hmm_inputs(frozen.output_dir)
    provider = InformationProvider(
        hmm=hmm,
        readiness_lead_weeks=int(benchmark_config["action"]["readiness_lead_weeks"]),
    )
    if bool(spec.get("anchor", False)):
        frame = base_path.frame.copy()
        frame["actual_public_release_date"] = pd.to_datetime(frame["release_date"])
        frame["warning_scenario"] = "GH_ANCHOR"
        frame["base_path_id"] = base_path.path_id
        frame["base_physical_path_sha256"] = base_path.path_hash
        frame["event_onset"] = pd.Timestamp(frame["week"].iloc[0])
        frame["preparation_period"] = False
        frame["event_period"] = True
        frame["monthly_transitions_to_readiness_maturity"] = np.nan
        frame["weekly_transition_matrix_applications"] = 0
        enriched = PhysicalPath(
            path_id=f"{base_path.path_id}__GH_ANCHOR",
            split="test",
            frame=frame,
            path_hash=base_path.path_hash,
            construction="accepted 5.2.2 event-window anchor",
            residual_start=base_path.residual_start,
            residual_end=base_path.residual_end,
            onset_week=base_path.onset_week,
            active_duration_weeks=base_path.active_duration_weeks,
            severity_floor=base_path.severity_floor,
            has_reclosure=base_path.has_reclosure,
        )
        path = provider.apply(enriched, "IL")
        warning = "GH_ANCHOR"
    else:
        builder = ReleaseTimingScenarioBuilder(
            hmm=hmm,
            event_onset=pd.Timestamp(spec["event_onset"]),
            readiness_lead_weeks=int(benchmark_config["action"]["readiness_lead_weeks"]),
            reference_normal_model_units=float(sum(model.gateway_scales.values())),
        )
        warning = str(spec["warning_scenario"])
        path = provider.apply(builder.build(base_path, warning), str(spec["evaluation_information_regime"]))
    controller_manifest = pd.read_csv(Path(spec["controller_manifest_path"]))
    rows = controller_manifest.loc[
        controller_manifest["controller_id"].eq(str(spec["controller_id"]))
        & controller_manifest["seed_index"].eq(int(spec["seed_index"]))
    ]
    if len(rows) != 1:
        raise RuntimeError("Cannot identify the requested controller checkpoint pair")
    policy = build_policies(code_root=code_root, model=model, controller_rows=rows)[0]
    return run_information_replication(
        model=model,
        rights_projector=rights_projector,
        policy=policy,
        path=path,
        layer=str(spec["layer"]),
        controller_id=str(spec["controller_id"]),
        training_information_regime=str(spec["training_information_regime"]),
        evaluation_information_regime=str(spec["evaluation_information_regime"]),
        capacity_rights=rights,
        warning_scenario=warning,
        controller_bundle_sha256=str(rows.iloc[0]["controller_bundle_sha256"]),
    )
