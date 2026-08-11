"""Mechanism-level replay and fixed-policy restricted-action diagnostics."""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from tre84.acceptance import evaluate_acceptance
from tre84.actions import Action, Block
from tre84.behavior import EXIT, WAIT
from tre84.clearance import ClearanceRunner
from tre84.keys import Provenance, ResourceKey, SourceKey, Stage, Tag
from tre84.metrics import compute_trajectory_statistics
from tre84.state import ModelState

from model import BenchmarkModel
from paths import PhysicalPath
from features import state_features
from preparation import build_realization, prepare_period
from simulator import (
    BenchmarkPolicy,
    PolicyDecision,
    RecoveryRule,
    _active_constraints,
    _released_information_hash,
)


RESTRICTIONS = (
    "full_action",
    "no_readiness",
    "no_direct_capacity",
    "no_release_pacing_authority",
    "no_disclosure",
)


@dataclass
class MechanismArtifacts:
    replication: dict[str, Any]
    actions: list[dict[str, Any]]
    behavior: list[dict[str, Any]]
    physical: list[dict[str, Any]]
    capacity: list[dict[str, Any]]
    losses: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    contract: dict[str, Any]


def _block_slices(model: BenchmarkModel) -> dict[str, slice]:
    sizes = {
        "readiness_order": len(model.layout.readiness_order),
        "direct_order": len(model.layout.direct_order),
        "readiness_exercise": len(model.layout.readiness_exercise),
        "release": len(model.layout.release),
        "disclosure": len(model.layout.disclosure),
    }
    result: dict[str, slice] = {}
    cursor = 0
    for name, size in sizes.items():
        result[name] = slice(cursor, cursor + size)
        cursor += size
    return result


def apply_restriction(
    *,
    model: BenchmarkModel,
    raw_action: Action,
    restriction: str,
    no_release_pacing_baseline: float,
) -> Action:
    """Apply a frozen diagnostic restriction before the shared projection."""

    if restriction not in RESTRICTIONS:
        raise ValueError(f"Unknown restricted-action diagnostic: {restriction}")
    vector = raw_action.vector(model.layout.keys).copy()
    blocks = _block_slices(model)
    if restriction == "no_readiness":
        vector[blocks["readiness_order"]] = 0.0
        vector[blocks["readiness_exercise"]] = 0.0
    elif restriction == "no_direct_capacity":
        vector[blocks["direct_order"]] = 0.0
    elif restriction == "no_release_pacing_authority":
        if not np.isclose(no_release_pacing_baseline, 1.0):
            raise ValueError("The frozen no-pacing baseline must be rho_base=1")
        vector[blocks["release"]] = (
            no_release_pacing_baseline * model.action_upper[blocks["release"]]
        )
    elif restriction == "no_disclosure":
        vector[blocks["disclosure"]] = 0.0
    return Action.from_vector(model.layout.keys, vector)


def _capacity_pipeline(capacity: Any, resource: ResourceKey) -> float:
    return float(sum(capacity.orders.get(resource, {}).values()))


def _action_rows(
    *,
    model: BenchmarkModel,
    base_policy: str,
    restriction: str,
    path: PhysicalPath,
    seed: int | None,
    scope: str,
    period_offset: int,
    decision_week: Any,
    decision_source: str,
    state: ModelState,
    next_state: ModelState,
    original_raw: Action,
    restricted_raw: Action,
    implemented: Action,
    projection: Any | None,
    decision_seconds: float,
    information_vector_hash: str = "",
    observation_hash: str = "",
    controller_input_hash: str = "",
) -> list[dict[str, Any]]:
    original = original_raw.vector(model.layout.keys)
    restricted = restricted_raw.vector(model.layout.keys)
    feasible = implemented.vector(model.layout.keys)
    lower, upper = model.domain.bounds(state)
    action_cost = model.domain.action_cost(implemented)
    resource_lookup: dict[int, ResourceKey] = {}
    for block_keys in (
        model.layout.readiness_order,
        model.layout.direct_order,
        model.layout.readiness_exercise,
    ):
        for key, resource in zip(block_keys, model.controlled_resources):
            resource_lookup[model.layout.keys.index(key)] = resource
    rows: list[dict[str, Any]] = []
    active = "" if projection is None else _active_constraints(model, state, projection)
    for index, (key, name) in enumerate(zip(model.layout.keys, model.layout.names)):
        resource = resource_lookup.get(index)
        if resource is None:
            readiness_pipeline_before = readiness_stock_before = np.nan
            readiness_pipeline_after = readiness_stock_after = np.nan
            direct_pipeline_before = direct_stock_before = np.nan
            direct_pipeline_after = direct_stock_after = np.nan
            resource_name = ""
        else:
            resource_name = f"{resource.stage.value}:{resource.location}"
            readiness_pipeline_before = _capacity_pipeline(state.readiness, resource)
            readiness_stock_before = float(state.readiness.stock.get(resource, 0.0))
            readiness_pipeline_after = _capacity_pipeline(next_state.readiness, resource)
            readiness_stock_after = float(next_state.readiness.stock.get(resource, 0.0))
            direct_pipeline_before = _capacity_pipeline(state.direct_capacity, resource)
            direct_stock_before = float(state.direct_capacity.stock.get(resource, 0.0))
            direct_pipeline_after = _capacity_pipeline(next_state.direct_capacity, resource)
            direct_stock_after = float(next_state.direct_capacity.stock.get(resource, 0.0))
        scale = max(float(model.action_upper[index]), 1e-12)
        rows.append(
            {
                "base_policy": base_policy,
                "restriction": restriction,
                "path_id": path.path_id,
                "path_content_sha256": path.path_hash,
                "training_seed": seed,
                "scope": scope,
                "period_offset": period_offset,
                "decision_week": decision_week,
                "decision_source": decision_source,
                "information_vector_sha256": information_vector_hash,
                "observation_sha256": observation_hash,
                "controller_input_sha256": controller_input_hash,
                "scenario_ids": "|".join(state.risk.scenario_ids),
                "readiness_weights": "|".join(
                    f"{value:.17g}" for value in state.risk.readiness_weights
                ),
                "operational_weights": "|".join(
                    f"{value:.17g}" for value in state.risk.operational_weights
                ),
                "latest_release_time": state.risk.latest_release_time,
                "action_index": index,
                "action_block": key.block.value,
                "action_coordinate": "|".join(key.coordinate),
                "action_name": name,
                "resource": resource_name,
                "original_requested_model_units": original[index],
                "restricted_requested_model_units": restricted[index],
                "implemented_model_units": feasible[index],
                "formal_upper_bound_model_units": model.action_upper[index],
                "original_requested_reference_ratio": original[index] / scale,
                "restricted_requested_reference_ratio": restricted[index] / scale,
                "implemented_reference_ratio": feasible[index] / scale,
                "state_lower_bound": lower[index],
                "state_upper_bound": upper[index],
                "readiness_pipeline_before": readiness_pipeline_before,
                "mature_readiness_stock_before": readiness_stock_before,
                "readiness_pipeline_after": readiness_pipeline_after,
                "mature_readiness_stock_after": readiness_stock_after,
                "direct_capacity_pipeline_before": direct_pipeline_before,
                "active_direct_capacity_before": direct_stock_before,
                "direct_capacity_pipeline_after": direct_pipeline_after,
                "active_direct_capacity_after": direct_stock_after,
                "period_action_cost": action_cost,
                "budget_before": state.budget,
                "budget_after": next_state.budget,
                "active_projection_constraints": active,
                "projection_objective": np.nan if projection is None else projection.objective,
                "projection_feasibility_violation": (
                    0.0 if projection is None else projection.feasibility_violation
                ),
                "decision_time_seconds": decision_seconds,
            }
        )
    return rows


def _behavior_rows(
    *,
    model: BenchmarkModel,
    base_policy: str,
    restriction: str,
    path: PhysicalPath,
    seed: int | None,
    scope: str,
    period_offset: int,
    state: ModelState,
    action: Action,
    realization: Any,
    result: Any,
) -> list[dict[str, Any]]:
    problem = model.kernel.behavior_factory(state, action, realization)
    equilibrium = result.equilibrium
    private_wait = problem.private_waiting_oracle(equilibrium.flows)
    costs = problem.costs(equilibrium.flows)
    cargo = str(model.config["cargo_class"])
    releases = np.asarray(equilibrium.releases[cargo], dtype=float)
    stock = np.asarray(state.waiting[cargo], dtype=float)
    next_waiting = np.asarray(result.transition.next_state.waiting[cargo], dtype=float)
    active_vintages = {
        vintage
        for vintage in range(len(stock))
        if stock[vintage] > 0.0
        or releases[vintage] > 0.0
        or result.transition.duration_attrition.get((cargo, vintage), 0.0) > 0.0
    }
    sources = [SourceKey(cargo, None)] + [SourceKey(cargo, age) for age in sorted(active_vintages)]
    start_text = "|".join(
        f"{record.name}:{record.residual:.12g}:{record.iterations}:{int(record.converged)}"
        for record in equilibrium.starts
    )
    step_text = "|".join(f"{value:.12g}" for value in equilibrium.selected_step_multipliers)
    rows: list[dict[str, Any]] = []
    for source in sources:
        flows = equilibrium.flows.get(source, {})
        decision_mass = float(sum(flows.values()))
        if source.is_new:
            release_mass = 0.0
            stock_mass = 0.0
            renewed = float(flows.get(WAIT, 0.0))
            attrition = 0.0
            continuation = float(next_waiting[0])
            waiting_residual = renewed - continuation
            age = 0
            hazard = 0.0
        else:
            assert source.vintage is not None
            vintage = source.vintage
            release_mass = float(releases[vintage])
            stock_mass = float(stock[vintage])
            renewed = float(flows.get(WAIT, 0.0))
            attrition = float(result.transition.duration_attrition.get((cargo, vintage), 0.0))
            continuation = float(next_waiting[vintage + 1]) if vintage + 1 < len(stock) else 0.0
            waiting_residual = stock_mass - release_mass + renewed - attrition - continuation
            age = vintage + 1
            hazard = float(model.waiting_hazard[vintage])
        direct_exit = float(flows.get(EXIT, 0.0))
        route_flows = {route: float(flows.get(route, 0.0)) for route in sorted(model.network.routes)}
        simplex_residual = decision_mass - sum(route_flows.values()) - renewed - direct_exit
        row: dict[str, Any] = {
            "base_policy": base_policy,
            "restriction": restriction,
            "path_id": path.path_id,
            "path_content_sha256": path.path_hash,
            "training_seed": seed,
            "scope": scope,
            "period_offset": period_offset,
            "cargo_class": cargo,
            "source_type": "new" if source.is_new else "released_waiting_vintage",
            "source_vintage": -1 if source.is_new else int(source.vintage),
            "source_age": age,
            "waiting_stock_at_source_age": stock_mass,
            "released_waiting_mass": release_mass,
            "decision_mass": decision_mass,
            "choose_waiting": renewed,
            "choose_sue_exit": direct_exit,
            "attrition_hazard": hazard,
            "duration_attrition": attrition,
            "waiting_continuation_next_state": continuation,
            "source_simplex_residual": simplex_residual,
            "waiting_identity_residual": waiting_residual,
            "selected_sue_start": equilibrium.selected_start,
            "selected_rcmsa_step_multiplier_trace": step_text,
            "sue_start_diagnostics": start_text,
            "sue_iterations": equilibrium.iterations,
            "sue_residual": equilibrium.residual,
            "sue_status": equilibrium.status,
            "sue_kl_discrepancy": equilibrium.kl_discrepancy,
            "sue_multistart_dispersion": equilibrium.multi_start_dispersion,
        }
        for route in sorted(model.network.routes):
            key = (cargo, route)
            intensity = float(problem.disclosure.intensity.get(key, 0.0))
            public = float(problem.disclosure.public_signal[key])
            private = float(private_wait[key])
            row[f"route_choice__{route}"] = route_flows[route]
            row[f"route_share__{route}"] = (
                route_flows[route] / decision_mass if decision_mass > 0 else 0.0
            )
            row[f"private_wait_signal__{route}"] = private
            row[f"public_wait_signal__{route}"] = public
            row[f"reference_wait_forecast__{route}"] = float(
                problem.disclosure.reference_forecast[key]
            )
            row[f"disclosure_intensity__{route}"] = intensity
            row[f"perceived_wait__{route}"] = (1.0 - intensity) * private + intensity * public
            row[f"generalised_route_cost__{route}"] = float(costs.get(source, {}).get(route, np.nan))
        rows.append(row)
    return rows


def _behavior_residuals(
    *, model: BenchmarkModel, state: ModelState, result: Any
) -> tuple[float, float]:
    """Compute the two blocking behavior identities without materialising rows."""

    cargo = str(model.config["cargo_class"])
    equilibrium = result.equilibrium
    releases = np.asarray(equilibrium.releases[cargo], dtype=float)
    stock = np.asarray(state.waiting[cargo], dtype=float)
    next_waiting = np.asarray(result.transition.next_state.waiting[cargo], dtype=float)
    active_vintages = {
        vintage
        for vintage in range(len(stock))
        if stock[vintage] > 0.0
        or releases[vintage] > 0.0
        or result.transition.duration_attrition.get((cargo, vintage), 0.0) > 0.0
    }
    sources = [SourceKey(cargo, None)] + [
        SourceKey(cargo, age) for age in sorted(active_vintages)
    ]
    maximum_simplex = 0.0
    maximum_waiting = 0.0
    for source in sources:
        flows = equilibrium.flows.get(source, {})
        decision_mass = float(sum(flows.values()))
        renewed = float(flows.get(WAIT, 0.0))
        direct_exit = float(flows.get(EXIT, 0.0))
        route_mass = sum(
            float(flows.get(route, 0.0)) for route in sorted(model.network.routes)
        )
        maximum_simplex = max(
            maximum_simplex,
            abs(decision_mass - route_mass - renewed - direct_exit),
        )
        if source.is_new:
            continuation = float(next_waiting[0])
            waiting_residual = renewed - continuation
        else:
            assert source.vintage is not None
            vintage = source.vintage
            continuation = (
                float(next_waiting[vintage + 1])
                if vintage + 1 < len(stock)
                else 0.0
            )
            waiting_residual = (
                float(stock[vintage])
                - float(releases[vintage])
                + renewed
                - float(result.transition.duration_attrition.get((cargo, vintage), 0.0))
                - continuation
            )
        maximum_waiting = max(maximum_waiting, abs(waiting_residual))
    return maximum_simplex, maximum_waiting


def _pipeline_due_by_provenance(
    *,
    model: BenchmarkModel,
    state: ModelState,
    realization: Any,
    committed: Mapping[Tag, float],
    adaptive: Mapping[Tag, float],
) -> dict[tuple[Provenance, str], float]:
    due: dict[tuple[Provenance, str], float] = {}
    if realization.base_arrivals:
        raise ValueError("The 5.2.3 provenance audit requires zero unprovenanced base arrivals")
    for lot in state.maritime_pipeline:
        if lot.route in realization.physical_route_available and lot.remaining_lag == 0:
            key = (lot.provenance, lot.route)
            due[key] = due.get(key, 0.0) + lot.mass
    for provenance, dispatch in (
        (Provenance.COMMITTED, committed),
        (Provenance.ADAPTIVE, adaptive),
    ):
        for tag, mass in dispatch.items():
            if tag.route not in realization.physical_route_available:
                continue
            route = model.network.route(tag.route)
            zero_share = float(route.maritime_lag_kernel[0])
            if zero_share > 0:
                key = (provenance, tag.route)
                due[key] = due.get(key, 0.0) + float(mass) * zero_share
    return due


def _allocate_proportionally(
    values: Mapping[Provenance, float], total_service: float
) -> dict[Provenance, float]:
    total = float(sum(values.values()))
    if total <= 0 or total_service <= 0:
        return {provenance: 0.0 for provenance in Provenance}
    return {
        provenance: float(values.get(provenance, 0.0) / total * total_service)
        for provenance in Provenance
    }


def _physical_rows(
    *,
    model: BenchmarkModel,
    base_policy: str,
    restriction: str,
    path: PhysicalPath,
    seed: int | None,
    scope: str,
    period_offset: int,
    state: ModelState,
    realization: Any,
    result: Any,
    shadow: dict[tuple[Provenance, str, Stage], float],
    store_rows: bool = True,
) -> tuple[list[dict[str, Any]], dict[tuple[Provenance, str, Stage], float], float]:
    cargo = str(model.config["cargo_class"])
    committed = result.transition.demand_split.committed_by_tag
    adaptive = {
        Tag(cargo_class, route): float(mass)
        for (cargo_class, route), mass in result.equilibrium.route_dispatch.items()
    }
    due = _pipeline_due_by_provenance(
        model=model,
        state=state,
        realization=realization,
        committed=committed,
        adaptive=adaptive,
    )
    state_maps = {
        Stage.BERTH: state.berth,
        Stage.YARD: state.yard,
        Stage.GATE: state.gate,
        Stage.CORRIDOR: state.corridor,
    }
    next_maps = {
        Stage.BERTH: result.transition.next_state.berth,
        Stage.YARD: result.transition.next_state.yard,
        Stage.GATE: result.transition.next_state.gate,
        Stage.CORRIDOR: result.transition.next_state.corridor,
    }
    service_total: dict[tuple[Stage, str], float] = {}
    for route in sorted(model.network.routes):
        tag = Tag(cargo, route)
        arrivals = sum(due.get((provenance, route), 0.0) for provenance in Provenance)
        berth_service = state.berth.get(tag, 0.0) + arrivals - result.transition.next_state.berth.get(tag, 0.0)
        yard_service = state.yard.get(tag, 0.0) + berth_service - result.transition.next_state.yard.get(tag, 0.0)
        gate_service = state.gate.get(tag, 0.0) + yard_service - result.transition.next_state.gate.get(tag, 0.0)
        corridor_service = state.corridor.get(tag, 0.0) + gate_service - result.transition.next_state.corridor.get(tag, 0.0)
        for stage, amount in zip(
            (Stage.BERTH, Stage.YARD, Stage.GATE, Stage.CORRIDOR),
            (berth_service, yard_service, gate_service, corridor_service),
        ):
            service_total[(stage, route)] = max(float(amount), 0.0)

    service_by_provenance: dict[tuple[Provenance, str, Stage], float] = {}
    next_shadow: dict[tuple[Provenance, str, Stage], float] = {}
    rows: list[dict[str, Any]] = []
    previous_stage = {
        Stage.BERTH: None,
        Stage.YARD: Stage.BERTH,
        Stage.GATE: Stage.YARD,
        Stage.CORRIDOR: Stage.GATE,
    }
    for route in sorted(model.network.routes):
        for stage in (Stage.BERTH, Stage.YARD, Stage.GATE, Stage.CORRIDOR):
            current_by_provenance = {
                provenance: float(shadow.get((provenance, route, stage), 0.0))
                for provenance in Provenance
            }
            pre_by_provenance = dict(current_by_provenance)
            if stage == Stage.BERTH:
                for provenance in Provenance:
                    pre_by_provenance[provenance] += due.get((provenance, route), 0.0)
            allocated = _allocate_proportionally(
                pre_by_provenance, service_total[(stage, route)]
            )
            for provenance in Provenance:
                service_by_provenance[(provenance, route, stage)] = allocated[provenance]
                upstream_stage = previous_stage[stage]
                upstream = (
                    0.0
                    if upstream_stage is None
                    else service_by_provenance[(provenance, route, upstream_stage)]
                )
                post_local = pre_by_provenance[provenance] - allocated[provenance]
                next_mass = post_local + upstream
                next_shadow[(provenance, route, stage)] = max(float(next_mass), 0.0)
                route_definition = model.network.route(route)
                location = (
                    route_definition.corridor
                    if stage == Stage.CORRIDOR
                    else route_definition.gateway
                )
                if store_rows:
                    rows.append(
                    {
                        "base_policy": base_policy,
                        "restriction": restriction,
                        "path_id": path.path_id,
                        "path_content_sha256": path.path_hash,
                        "training_seed": seed,
                        "scope": scope,
                        "period_offset": period_offset,
                        "cargo_class": cargo,
                        "provenance": provenance.value,
                        "route": route,
                        "gateway": route_definition.gateway,
                        "corridor": route_definition.corridor,
                        "stage": stage.value,
                        "location": location,
                        "route_lag_weeks": int(np.argmax(route_definition.maritime_lag_kernel)),
                        "route_physically_available": route in realization.physical_route_available,
                        "queue_state_before": current_by_provenance[provenance],
                        "external_maritime_due": (
                            due.get((provenance, route), 0.0) if stage == Stage.BERTH else 0.0
                        ),
                        "preservice_workload": pre_by_provenance[provenance],
                        "service_discharge": allocated[provenance],
                        "after_service_local": post_local,
                        "upstream_release_entering_next_state": upstream,
                        "queue_next_state": next_shadow[(provenance, route, stage)],
                        "delivered_cargo": (
                            allocated[provenance] if stage == Stage.CORRIDOR else 0.0
                        ),
                    }
                    )

    maximum_shadow_residual = 0.0
    for stage in (Stage.BERTH, Stage.YARD, Stage.GATE, Stage.CORRIDOR):
        mapping = next_maps[stage]
        for route in sorted(model.network.routes):
            tag = Tag(cargo, route)
            diagnostic = sum(
                next_shadow.get((provenance, route, stage), 0.0)
                for provenance in Provenance
            )
            maximum_shadow_residual = max(
                maximum_shadow_residual,
                abs(diagnostic - float(mapping.get(tag, 0.0))),
            )

    for route in sorted(model.network.routes):
        route_definition = model.network.route(route)
        for provenance in Provenance:
            pipeline_before = sum(
                lot.mass
                for lot in state.maritime_pipeline
                if lot.route == route and lot.provenance == provenance
            )
            pipeline_after = sum(
                lot.mass
                for lot in result.transition.next_state.maritime_pipeline
                if lot.route == route and lot.provenance == provenance
            )
            dispatch = (
                float(committed.get(Tag(cargo, route), 0.0))
                if provenance == Provenance.COMMITTED
                else float(adaptive.get(Tag(cargo, route), 0.0))
            )
            held = sum(
                lot.mass
                for lot in state.maritime_pipeline
                if lot.route == route
                and lot.provenance == provenance
                and lot.route not in realization.physical_route_available
            )
            if store_rows:
                rows.append(
                {
                    "base_policy": base_policy,
                    "restriction": restriction,
                    "path_id": path.path_id,
                    "path_content_sha256": path.path_hash,
                    "training_seed": seed,
                    "scope": scope,
                    "period_offset": period_offset,
                    "cargo_class": cargo,
                    "provenance": provenance.value,
                    "route": route,
                    "gateway": route_definition.gateway,
                    "corridor": route_definition.corridor,
                    "stage": "maritime_pipeline",
                    "location": route,
                    "route_lag_weeks": int(np.argmax(route_definition.maritime_lag_kernel)),
                    "route_physically_available": route in realization.physical_route_available,
                    "queue_state_before": pipeline_before,
                    "external_maritime_due": due.get((provenance, route), 0.0),
                    "preservice_workload": pipeline_before,
                    "service_discharge": due.get((provenance, route), 0.0),
                    "after_service_local": pipeline_after,
                    "upstream_release_entering_next_state": dispatch,
                    "queue_next_state": pipeline_after,
                    "delivered_cargo": 0.0,
                    "new_dispatch": dispatch,
                    "unavailable_route_holding_mass": held,
                }
                )
    return rows, next_shadow, maximum_shadow_residual


def _capacity_rows(
    *,
    model: BenchmarkModel,
    base_policy: str,
    restriction: str,
    path: PhysicalPath,
    seed: int | None,
    scope: str,
    period_offset: int,
    state: ModelState,
    action: Action,
    result: Any,
) -> list[dict[str, Any]]:
    aggregate = {
        Stage.BERTH: {},
        Stage.YARD: {},
        Stage.GATE: {},
        Stage.CORRIDOR: {},
    }
    for stage, mapping in (
        (Stage.BERTH, state.berth),
        (Stage.YARD, state.yard),
        (Stage.GATE, state.gate),
        (Stage.CORRIDOR, state.corridor),
    ):
        for tag, mass in mapping.items():
            route = model.network.route(tag.route)
            location = route.corridor if stage == Stage.CORRIDOR else route.gateway
            aggregate[stage][location] = aggregate[stage].get(location, 0.0) + mass
    rows: list[dict[str, Any]] = []
    for index, resource in enumerate(model.controlled_resources):
        direct_spot = float(result.transition.capacity.direct_spot.get(resource, 0.0))
        active_direct = float(state.direct_capacity.stock.get(resource, 0.0)) + direct_spot
        readiness = float(result.transition.capacity.readiness_capacity.get(resource, 0.0))
        base = float(model.base_capacity[resource])
        prefeedback = base + active_direct + readiness
        effective = float(result.transition.capacity.effective[resource])
        feedback = effective / prefeedback if prefeedback > 0 else 1.0
        workload = float(aggregate[resource.stage].get(resource.location, 0.0))
        threshold = float(model.thresholds[resource])
        rows.append(
            {
                "base_policy": base_policy,
                "restriction": restriction,
                "path_id": path.path_id,
                "path_content_sha256": path.path_hash,
                "training_seed": seed,
                "scope": scope,
                "period_offset": period_offset,
                "resource": f"{resource.stage.value}:{resource.location}",
                "stage": resource.stage.value,
                "location": resource.location,
                "base_capacity": base,
                "active_direct_capacity": active_direct,
                "exercised_readiness_capacity": readiness,
                "effective_capacity": effective,
                "feedback_multiplier": feedback,
                "yard_to_berth_feedback": feedback if resource.stage == Stage.BERTH else 1.0,
                "corridor_to_gate_feedback": feedback if resource.stage == Stage.GATE else 1.0,
                "shared_corridor_weight": (
                    sum(
                        result.transition.capacity.corridor_weights.get(
                            (resource.location, corridor), 0.0
                        )
                        for corridor in model.network.corridors()
                    )
                    if resource.stage == Stage.GATE
                    else np.nan
                ),
                "resource_workload": workload,
                "resource_threshold": threshold,
                "resource_pressure": workload / threshold,
                "resource_overload_ratio": max(workload / threshold - 1.0, 0.0),
                "resource_overload_mass": max(workload - threshold, 0.0),
                "implemented_readiness_order": action.value(model.layout.readiness_order[index]),
                "implemented_direct_order": action.value(model.layout.direct_order[index]),
                "implemented_readiness_exercise": action.value(model.layout.readiness_exercise[index]),
            }
        )
    return rows


def _loss_row(
    *,
    model: BenchmarkModel,
    base_policy: str,
    restriction: str,
    path: PhysicalPath,
    seed: int | None,
    scope: str,
    period_offset: int,
    state: ModelState,
    result: Any,
) -> dict[str, Any]:
    stage_mass = {
        "berth": float(sum(state.berth.values())),
        "yard": float(sum(state.yard.values())),
        "gate": float(sum(state.gate.values())),
        "landbridge": float(sum(state.corridor.values())),
    }
    queue_unit = float(model.config["loss"]["queue_cost_per_model_unit_week"])
    direct_mass = float(sum(result.transition.direct_exit.values()))
    attrition_mass = float(sum(result.transition.duration_attrition.values()))
    exit_unit = float(model.config["behavior"]["exit_failure_cost_per_unit"])
    queue_components = {name: queue_unit * value for name, value in stage_mass.items()}
    component_sum = (
        sum(queue_components.values())
        + result.transition.loss.waiting
        + direct_mass * exit_unit
        + attrition_mass * exit_unit
        + result.transition.loss.overflow
        + result.transition.loss.route_resource
        + result.transition.loss.action
    )
    return {
        "base_policy": base_policy,
        "restriction": restriction,
        "path_id": path.path_id,
        "path_content_sha256": path.path_hash,
        "training_seed": seed,
        "scope": scope,
        "period_offset": period_offset,
        "queue_loss_berth": queue_components["berth"],
        "queue_loss_yard": queue_components["yard"],
        "queue_loss_gate": queue_components["gate"],
        "queue_loss_landbridge": queue_components["landbridge"],
        "queue_loss": result.transition.loss.queue,
        "waiting_loss": result.transition.loss.waiting,
        "sue_exit_loss": direct_mass * exit_unit,
        "attrition_exit_loss": attrition_mass * exit_unit,
        "exit_loss": result.transition.loss.exit,
        "overload_loss": result.transition.loss.overflow,
        "route_resource_loss": result.transition.loss.route_resource,
        "action_loss": result.transition.loss.action,
        "terminal_loss": 0.0,
        "period_operational_loss": result.transition.loss.total,
        "period_loss_identity_residual": component_sum - result.transition.loss.total,
        "waiting_mass_before": state.waiting_mass(),
        "waiting_mass_after": result.transition.next_state.waiting_mass(),
        "sue_exit_mass": direct_mass,
        "attrition_exit_mass": attrition_mass,
        "delivered_landbridge": float(sum(result.transition.delivered.values())),
        "outstanding_mass_after": result.transition.next_state.cargo_mass(),
    }


def run_mechanism_replication(
    *,
    model: BenchmarkModel,
    base_policy: BenchmarkPolicy,
    path: PhysicalPath,
    restriction: str,
    no_release_pacing_baseline: float,
    store_detail: bool = True,
) -> MechanismArtifacts:
    first = path.frame.iloc[0].to_dict()
    state = model.initial_state(first)
    decision_initial = state.clone()
    decision_results: list[Any] = []
    actions: list[dict[str, Any]] = []
    behavior: list[dict[str, Any]] = []
    physical: list[dict[str, Any]] = []
    capacity: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    shadow: dict[tuple[Provenance, str, Stage], float] = {}
    step_acceptance: list[bool] = []
    decision_seconds: list[float] = []
    shadow_residuals: list[float] = []
    rows = [row._asdict() for row in path.frame.itertuples(index=False)]
    maximum_behavior_source = 0.0
    maximum_waiting_identity = 0.0
    maximum_loss_identity = 0.0
    allowed_changed_blocks = {
        "full_action": frozenset(),
        "no_readiness": frozenset(
            {Block.READINESS_ORDER.value, Block.READINESS_EXERCISE.value}
        ),
        "no_direct_capacity": frozenset({Block.DIRECT_ORDER.value}),
        "no_release_pacing_authority": frozenset({Block.RELEASE.value}),
        "no_disclosure": frozenset({Block.DISCLOSURE.value}),
    }[restriction]
    restriction_changes_only_declared_block = True
    direct_procurement_retained = True
    no_pacing_uses_immediate_release = True
    no_disclosure_action_is_zero = True

    for offset, row in enumerate(rows):
        prepared = prepare_period(model=model, state=state, row=row)
        state = prepared.state
        controller_input_hash = (
            hashlib.sha256(
                state_features(state, model).astype("<f8", copy=False).tobytes()
            ).hexdigest()
            if store_detail
            else ""
        )
        started = time.perf_counter()
        decision: PolicyDecision = base_policy.decide(
            state=state,
            row=row,
            path=path,
            offset=offset,
            bundle=prepared.scenarios,
        )
        elapsed = time.perf_counter() - started
        restricted_raw = apply_restriction(
            model=model,
            raw_action=decision.raw_action,
            restriction=restriction,
            no_release_pacing_baseline=no_release_pacing_baseline,
        )
        original_vector = decision.raw_action.vector(model.layout.keys)
        restricted_vector = restricted_raw.vector(model.layout.keys)
        for key, original_value, restricted_value in zip(
            model.layout.keys, original_vector, restricted_vector
        ):
            if (
                key.block.value not in allowed_changed_blocks
                and not np.isclose(original_value, restricted_value)
            ):
                restriction_changes_only_declared_block = False
            if (
                restriction == "no_readiness"
                and key.block == Block.DIRECT_ORDER
                and not np.isclose(original_value, restricted_value)
            ):
                direct_procurement_retained = False
            if (
                restriction == "no_release_pacing_authority"
                and key.block == Block.RELEASE
                and not np.isclose(
                    restricted_value,
                    model.action_upper[model.layout.keys.index(key)],
                )
            ):
                no_pacing_uses_immediate_release = False
            if (
                restriction == "no_disclosure"
                and key.block == Block.DISCLOSURE
                and not np.isclose(restricted_value, 0.0)
            ):
                no_disclosure_action_is_zero = False
        projection = model.projector.project(restricted_raw, state)
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
        source_residual, waiting_residual = _behavior_residuals(
            model=model, state=state, result=result
        )
        maximum_behavior_source = max(maximum_behavior_source, source_residual)
        maximum_waiting_identity = max(maximum_waiting_identity, waiting_residual)
        period_loss_identity = (
            result.transition.loss.queue
            + result.transition.loss.waiting
            + result.transition.loss.exit
            + result.transition.loss.overflow
            + result.transition.loss.route_resource
            + result.transition.loss.action
            - result.transition.loss.total
        )
        maximum_loss_identity = max(maximum_loss_identity, abs(period_loss_identity))
        actions.extend(
            _action_rows(
                model=model,
                base_policy=base_policy.name,
                restriction=restriction,
                path=path,
                seed=base_policy.training_seed,
                scope="decision",
                period_offset=offset,
                decision_week=row["week"],
                decision_source=decision.decision_source,
                state=state,
                next_state=result.transition.next_state,
                original_raw=decision.raw_action,
                restricted_raw=restricted_raw,
                implemented=projection.action,
                projection=projection,
                decision_seconds=elapsed,
                information_vector_hash=prepared.information_vector_hash,
                observation_hash=prepared.observation_hash,
                controller_input_hash=controller_input_hash,
            )
            if store_detail
            else []
        )
        behavior.extend(
            _behavior_rows(
                model=model,
                base_policy=base_policy.name,
                restriction=restriction,
                path=path,
                seed=base_policy.training_seed,
                scope="decision",
                period_offset=offset,
                state=state,
                action=projection.action,
                realization=realization,
                result=result,
            )
            if store_detail
            else []
        )
        physical_rows, shadow, shadow_residual = _physical_rows(
            model=model,
            base_policy=base_policy.name,
            restriction=restriction,
            path=path,
            seed=base_policy.training_seed,
            scope="decision",
            period_offset=offset,
            state=state,
            realization=realization,
            result=result,
            shadow=shadow,
            store_rows=store_detail,
        )
        physical.extend(physical_rows)
        shadow_residuals.append(shadow_residual)
        capacity.extend(
            _capacity_rows(
                model=model,
                base_policy=base_policy.name,
                restriction=restriction,
                path=path,
                seed=base_policy.training_seed,
                scope="decision",
                period_offset=offset,
                state=state,
                action=projection.action,
                result=result,
            )
            if store_detail
            else []
        )
        losses.append(
            _loss_row(
                model=model,
                base_policy=base_policy.name,
                restriction=restriction,
                path=path,
                seed=base_policy.training_seed,
                scope="decision",
                period_offset=offset,
                state=state,
                result=result,
            )
        ) if store_detail else None
        if store_detail:
            proposals.extend(
                {
                    "base_policy": base_policy.name,
                    "restriction": restriction,
                    "path_id": path.path_id,
                    "training_seed": base_policy.training_seed,
                    "period_offset": offset,
                    **proposal,
                }
                for proposal in decision.proposal_records
            )
        decision_results.append(result)
        state = result.transition.next_state

    clearance = ClearanceRunner(
        kernel=model.kernel,
        recovery_rule=RecoveryRule(model),
        terminal_cost=model.terminal_cost,
        maximum_weeks=int(model.config["clearance"]["maximum_weeks"]),
        empty_tolerance=float(model.config["clearance"]["empty_tolerance"]),
    ).run(state)
    recovery = RecoveryRule(model)
    clearance_state = state
    zero = model.zero_action()
    for offset, result in enumerate(clearance.transitions):
        realization = recovery.realization(clearance_state)
        source_residual, waiting_residual = _behavior_residuals(
            model=model, state=clearance_state, result=result
        )
        maximum_behavior_source = max(maximum_behavior_source, source_residual)
        maximum_waiting_identity = max(maximum_waiting_identity, waiting_residual)
        period_loss_identity = (
            result.transition.loss.queue
            + result.transition.loss.waiting
            + result.transition.loss.exit
            + result.transition.loss.overflow
            + result.transition.loss.route_resource
            + result.transition.loss.action
            - result.transition.loss.total
        )
        maximum_loss_identity = max(maximum_loss_identity, abs(period_loss_identity))
        actions.extend(
            _action_rows(
                model=model,
                base_policy=base_policy.name,
                restriction=restriction,
                path=path,
                seed=base_policy.training_seed,
                scope="clearance",
                period_offset=offset,
                decision_week="",
                decision_source="frozen_zero_action_recovery_rule",
                state=clearance_state,
                next_state=result.transition.next_state,
                original_raw=zero,
                restricted_raw=zero,
                implemented=zero,
                projection=None,
                decision_seconds=0.0,
            )
            if store_detail
            else []
        )
        behavior.extend(
            _behavior_rows(
                model=model,
                base_policy=base_policy.name,
                restriction=restriction,
                path=path,
                seed=base_policy.training_seed,
                scope="clearance",
                period_offset=offset,
                state=clearance_state,
                action=zero,
                realization=realization,
                result=result,
            )
            if store_detail
            else []
        )
        physical_rows, shadow, shadow_residual = _physical_rows(
            model=model,
            base_policy=base_policy.name,
            restriction=restriction,
            path=path,
            seed=base_policy.training_seed,
            scope="clearance",
            period_offset=offset,
            state=clearance_state,
            realization=realization,
            result=result,
            shadow=shadow,
            store_rows=store_detail,
        )
        physical.extend(physical_rows)
        shadow_residuals.append(shadow_residual)
        capacity.extend(
            _capacity_rows(
                model=model,
                base_policy=base_policy.name,
                restriction=restriction,
                path=path,
                seed=base_policy.training_seed,
                scope="clearance",
                period_offset=offset,
                state=clearance_state,
                action=zero,
                result=result,
            )
            if store_detail
            else []
        )
        losses.append(
            _loss_row(
                model=model,
                base_policy=base_policy.name,
                restriction=restriction,
                path=path,
                seed=base_policy.training_seed,
                scope="clearance",
                period_offset=offset,
                state=clearance_state,
                result=result,
            )
        ) if store_detail else None
        clearance_state = result.transition.next_state

    stats = compute_trajectory_statistics(
        initial_state=decision_initial,
        decision_results=decision_results,
        network=model.network,
        thresholds=model.thresholds,
        clearance=clearance,
        include_clearance_in_physical_metrics=True,
        tolerance=float(model.config["numerics"]["loss_identity_tolerance"]),
    )
    exit_unit = float(model.config["behavior"]["exit_failure_cost_per_unit"])
    direct_exit_loss = stats.direct_sue_exit * exit_unit
    attrition_loss = stats.duration_attrition * exit_unit
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
        "base_policy": base_policy.name,
        "restriction": restriction,
        "path_id": path.path_id,
        "path_content_sha256": path.path_hash,
        "released_information_path_sha256": _released_information_hash(path),
        "training_seed": base_policy.training_seed,
        **stats.as_record(),
        "loss_direct_sue_exit": direct_exit_loss,
        "loss_duration_attrition": attrition_loss,
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
    terminal_row = {
        "base_policy": base_policy.name,
        "restriction": restriction,
        "path_id": path.path_id,
        "path_content_sha256": path.path_hash,
        "training_seed": base_policy.training_seed,
        "scope": "terminal",
        "period_offset": clearance.weeks,
        "queue_loss_berth": 0.0,
        "queue_loss_yard": 0.0,
        "queue_loss_gate": 0.0,
        "queue_loss_landbridge": 0.0,
        "queue_loss": 0.0,
        "waiting_loss": 0.0,
        "sue_exit_loss": 0.0,
        "attrition_exit_loss": 0.0,
        "exit_loss": 0.0,
        "overload_loss": 0.0,
        "route_resource_loss": 0.0,
        "action_loss": 0.0,
        "terminal_loss": clearance.terminal_correction,
        "period_operational_loss": clearance.terminal_correction,
        "period_loss_identity_residual": 0.0,
        "waiting_mass_before": clearance.final_state.waiting_mass(),
        "waiting_mass_after": clearance.final_state.waiting_mass(),
        "sue_exit_mass": 0.0,
        "attrition_exit_mass": 0.0,
        "delivered_landbridge": 0.0,
        "outstanding_mass_after": clearance.final_state.cargo_mass(),
    }
    if store_detail:
        losses.append(terminal_row)
    behavior_source = maximum_behavior_source
    waiting_identity = maximum_waiting_identity
    loss_identity = maximum_loss_identity
    tolerance = float(model.config["numerics"]["mass_tolerance"])
    contract = {
        "base_policy": base_policy.name,
        "restriction": restriction,
        "path_id": path.path_id,
        "training_seed": base_policy.training_seed,
        "all_step_acceptance_passed": all(step_acceptance),
        "maximum_transition_residual": stats.maximum_transition_residual,
        "maximum_source_simplex_residual": behavior_source,
        "maximum_waiting_identity_residual": waiting_identity,
        "maximum_provenance_shadow_residual": max(shadow_residuals, default=0.0),
        "maximum_period_loss_identity_residual": loss_identity,
        "loss_components_reconstruct_total": abs(component_sum - stats.total_operational_objective) <= tolerance,
        "restriction_changes_only_declared_action_block": restriction_changes_only_declared_block,
        "no_readiness_starts_without_readiness_stock": (
            restriction != "no_readiness"
            or (
                decision_initial.readiness.total_orders() <= tolerance
                and sum(decision_initial.readiness.stock.values()) <= tolerance
            )
        ),
        "direct_procurement_retained_under_no_readiness": (
            restriction != "no_readiness"
            or direct_procurement_retained
        ),
        "no_release_pacing_uses_immediate_release_baseline": (
            restriction != "no_release_pacing_authority"
            or no_pacing_uses_immediate_release
        ),
        "no_disclosure_preserves_information_system": (
            restriction != "no_disclosure"
            or no_disclosure_action_is_zero
        ),
    }
    return MechanismArtifacts(
        replication,
        actions if store_detail else [],
        behavior if store_detail else [],
        physical if store_detail else [],
        capacity if store_detail else [],
        losses if store_detail else [],
        proposals if store_detail else [],
        contract,
    )
