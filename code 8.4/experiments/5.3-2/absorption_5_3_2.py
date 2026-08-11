"""Corrected route-lag-aware optimistic absorption certificate for 5.3.2."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from tre84.diagnostics import AbsorptionBoundaryLP, AbsorptionInput
from tre84.keys import ResourceKey, Stage, Tag

from paths import PhysicalPath
from reclosure_worker import GridCell, build_cell_path


def _capacity_envelope(model: Any, resource: ResourceKey) -> float:
    index = model.controlled_resources.index(resource)
    count = len(model.controlled_resources)
    direct_upper = float(model.action_upper[count + index])
    exercise_upper = float(model.action_upper[2 * count + index])
    direct_yield = float(model.config["capacity_technology"]["direct_maturity_yield"])
    readiness_yield = float(model.config["capacity_technology"]["readiness_capacity_yield"])
    return float(model.base_capacity[resource]) + direct_yield * direct_upper + readiness_yield * exercise_upper


def _gateway_capacity_curves(model: Any, maximum_horizon: int) -> dict[tuple[str, int], float]:
    corridor = next(iter(model.network.corridors()))
    corridor_resource = ResourceKey(Stage.CORRIDOR, corridor)
    result: dict[tuple[str, int], float] = {}
    for gateway in sorted(model.network.gateways()):
        resources = [ResourceKey(stage, gateway) for stage in (Stage.BERTH, Stage.YARD, Stage.GATE)]
        capacities = [_capacity_envelope(model, resource) for resource in resources]
        corridor_capacity = _capacity_envelope(model, corridor_resource)
        thresholds = [float(model.thresholds[resource]) for resource in resources]
        corridor_threshold = float(model.thresholds[corridor_resource])
        for horizon in range(1, maximum_horizon + 1):
            data = AbsorptionInput(
                initial_berth=0.0,
                initial_yard=0.0,
                initial_gate=0.0,
                initial_corridor=0.0,
                berth_threshold=thresholds[0],
                yard_threshold=thresholds[1],
                gate_threshold=thresholds[2],
                corridor_threshold=corridor_threshold,
                berth_capacity=np.full(horizon, capacities[0]),
                yard_capacity=np.full(horizon, capacities[1]),
                gate_capacity=np.full(horizon, capacities[2]),
                optimistic_corridor_capacity=np.full(horizon, corridor_capacity),
            )
            result[(gateway, horizon)] = AbsorptionBoundaryLP.solve(data).maximum_absorbable
    return result


def _committed_berth_arrivals(model: Any, path: PhysicalPath) -> dict[str, np.ndarray]:
    horizon = len(path.frame)
    arrivals = {gateway: np.zeros(horizon, dtype=float) for gateway in model.network.gateways()}
    chi = float(model.config["committed_fraction_reference"])
    normal = path.frame["normal_model_units"].to_numpy(dtype=float)
    service = path.frame["serviceability"].to_numpy(dtype=float)
    committed = chi * normal * (1.0 - service)
    for route_id, route in model.network.routes.items():
        share = float(model.committed_shares.get(Tag(str(model.config["cargo_class"]), route_id), 0.0))
        dispatch = committed * share
        kernel = np.asarray(route.maritime_lag_kernel, dtype=float)
        for origin, mass in enumerate(dispatch):
            for lag, probability in enumerate(kernel):
                arrival = origin + lag
                if arrival < horizon:
                    arrivals[route.gateway][arrival] += float(mass) * float(probability)
    return arrivals


def absorption_certificate(
    *,
    model: Any,
    base_paths: Sequence[PhysicalPath],
    cells: Sequence[GridCell],
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maximum_horizon = max(
        len(path.frame) - int(path.onset_week)
        for cell in cells
        for path in (build_cell_path(base_paths[0], cell),)
    )
    capacities = _gateway_capacity_curves(model, maximum_horizon)
    path_rows: list[dict[str, Any]] = []
    for cell in cells:
        for base_path in base_paths:
            path = build_cell_path(base_path, cell)
            arrivals = _committed_berth_arrivals(model, path)
            certificate_start = int(path.onset_week)
            gateway_violations = []
            for gateway, series in arrivals.items():
                # The certificate is anchored at reclosure onset.  Convolving
                # the complete dispatch history before slicing retains every
                # committed lot that was dispatched before reclosure but is
                # still in the maritime pipeline at the certificate start.
                # Existing port queues are set to zero as an additional
                # optimistic relaxation; future base/adaptive arrivals remain
                # excluded.  A violation is therefore still sufficient, while
                # nonviolation proves neither feasibility nor policy existence.
                certificate_arrivals = series[certificate_start:]
                cumulative = np.cumsum(certificate_arrivals)
                first = None
                maximum_excess = -np.inf
                capacity_at_first = committed_at_first = np.nan
                for horizon, committed in enumerate(cumulative, start=1):
                    capacity = capacities[(gateway, horizon)]
                    excess = float(committed - capacity)
                    maximum_excess = max(maximum_excess, excess)
                    if first is None and excess > tolerance:
                        first = horizon
                        capacity_at_first = capacity
                        committed_at_first = float(committed)
                violated = first is not None
                gateway_violations.append(violated)
                path_rows.append({
                    "cell_id": cell.cell_id,
                    "open_interval_weeks": cell.open_weeks,
                    "reclosure_intensity": cell.intensity,
                    "reclosure_duration_weeks": cell.duration_weeks,
                    "path_id": base_path.path_id,
                    "gateway": gateway,
                    "certificate_start_week_index": certificate_start,
                    "certificate_horizon_weeks": len(certificate_arrivals),
                    "pre_reclosure_committed_pipeline_retained": True,
                    "initial_port_queue_relaxed_to_zero": True,
                    "route_lag_kernel_preserved": True,
                    "route_tag_preserved_to_gateway_program": True,
                    "shared_corridor_allocated_optimistically": True,
                    "future_base_arrivals_removed": True,
                    "future_adaptive_arrivals_removed": True,
                    "policy_action_used": False,
                    "first_violating_horizon": first,
                    "certificate_violated": violated,
                    "committed_arrival_at_first_violation": committed_at_first,
                    "maximum_absorbable_at_first_violation": capacity_at_first,
                    "maximum_committed_minus_capacity": maximum_excess,
                })
            # A path is certified only when at least one gateway program violates.
            path_rows.append({
                "cell_id": cell.cell_id,
                "open_interval_weeks": cell.open_weeks,
                "reclosure_intensity": cell.intensity,
                "reclosure_duration_weeks": cell.duration_weeks,
                "path_id": base_path.path_id,
                "gateway": "ANY_GATEWAY",
                "certificate_start_week_index": certificate_start,
                "certificate_horizon_weeks": len(path.frame) - certificate_start,
                "pre_reclosure_committed_pipeline_retained": True,
                "initial_port_queue_relaxed_to_zero": True,
                "route_lag_kernel_preserved": True,
                "route_tag_preserved_to_gateway_program": True,
                "shared_corridor_allocated_optimistically": True,
                "future_base_arrivals_removed": True,
                "future_adaptive_arrivals_removed": True,
                "policy_action_used": False,
                "first_violating_horizon": np.nan,
                "certificate_violated": any(gateway_violations),
                "committed_arrival_at_first_violation": np.nan,
                "maximum_absorbable_at_first_violation": np.nan,
                "maximum_committed_minus_capacity": np.nan,
            })
    path_frame = pd.DataFrame(path_rows)
    any_gateway = path_frame.loc[path_frame["gateway"] == "ANY_GATEWAY"]
    summary_rows = []
    for cell_id, group in any_gateway.groupby("cell_id"):
        first = group.iloc[0]
        share = float(group["certificate_violated"].astype(bool).mean())
        summary_rows.append({
            "cell_id": cell_id,
            "open_interval_weeks": int(first["open_interval_weeks"]),
            "reclosure_intensity": float(first["reclosure_intensity"]),
            "reclosure_duration_weeks": int(first["reclosure_duration_weeks"]),
            "physical_paths": int(group["path_id"].nunique()),
            "path_certificate_violation_share": share,
            "any_matched_path_certified": bool(share > 0.0),
            "all_matched_paths_certified": bool(np.isclose(share, 1.0)),
            "hatching_unavoidable_cell": bool(np.isclose(share, 1.0)),
            "nonviolation_proves_feasibility": False,
        })
    summary = pd.DataFrame(summary_rows)
    envelope_rows = []
    for (gateway, horizon), capacity in capacities.items():
        envelope_rows.append({
            "gateway": gateway,
            "horizon_weeks": horizon,
            "maximum_absorbable_committed_arrival": capacity,
            "shared_corridor_capacity_reused_across_gateway_programs": True,
            "gateway_capacities_additive": False,
        })
    return path_frame, summary, pd.DataFrame(envelope_rows)
