"""Declared semi-synthetic gateway cells for Experiment 5.3.3."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


OBSERVED_ROUTES = ("Khor_Fakkan", "Fujairah", "Sohar")
OBSERVED_GATEWAYS = ("Khor Fakkan", "Fujairah", "Sohar")


@dataclass(frozen=True)
class NetworkCell:
    cell_id: str
    gateway_count: int
    architecture: str
    eligibility: str
    is_common_baseline: bool = False


def declared_cells() -> list[NetworkCell]:
    cells = [NetworkCell("n03_reference", 3, "observed_reference", "observed_reference", True)]
    for n in (4, 5, 7, 9):
        for architecture in ("capacity_neutral", "port_only", "end_to_end"):
            for eligibility in ("emergency_only", "precontracted"):
                cells.append(
                    NetworkCell(
                        f"n{n:02d}_{architecture}_{eligibility}",
                        n,
                        architecture,
                        eligibility,
                    )
                )
    if len(cells) != 25 or len({cell.cell_id for cell in cells}) != 25:
        raise RuntimeError("The declared gateway design must contain 25 unique cells")
    return cells


def training_cell(n: int) -> NetworkCell:
    if n == 3:
        return declared_cells()[0]
    return NetworkCell(f"n{n:02d}_training_reference", n, "capacity_neutral", "emergency_only")


def is_full_policy_anchor(cell: NetworkCell) -> bool:
    return cell.is_common_baseline or cell.gateway_count == 9


def _anchor_maps(base: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    scales = {
        "Khor Fakkan": 30.796000000000003,
        "Fujairah": 1.4997000000000007,
        "Sohar": 70.0008,
    }
    shares = {
        "Khor_Fakkan": 0.5952380952380952,
        "Fujairah": 0.1071428571428571,
        "Sohar": 0.2976190476190476,
    }
    sigma = {str(key): float(value) for key, value in base["information"]["waiting_error_scale_weeks_by_route"].items()}
    return scales, shares, sigma


def _e2e_capacity(
    stage: Mapping[str, float], corridor_total: float
) -> dict[str, float]:
    total_gate = float(sum(stage.values()))
    return {
        gateway: min(
            float(capacity),
            corridor_total * float(capacity) / total_gate,
        )
        for gateway, capacity in stage.items()
    }


def build_cell_config(
    base: Mapping[str, Any], experiment: Mapping[str, Any], cell: NetworkCell
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    config["experiment_id"] = str(experiment["experiment_id"])
    anchor_scales, official_shares, anchor_sigma = _anchor_maps(base)
    baseline_total = float(sum(anchor_scales.values()))
    template_scale = float(np.median(list(anchor_scales.values())))
    observed_lags = [int(item["maritime_lag_weeks"]) for item in base["routes"]]
    template_lag = int(np.median(observed_lags))
    template_sigma = float(np.median(list(anchor_sigma.values())))

    new_count = cell.gateway_count - 3
    new_gateways = [f"SemiSynthetic_Gateway_{index:02d}" for index in range(1, new_count + 1)]
    raw_stage = {**anchor_scales, **{gateway: template_scale for gateway in new_gateways}}
    if cell.architecture == "capacity_neutral":
        factor = baseline_total / float(sum(raw_stage.values()))
        stage = {gateway: value * factor for gateway, value in raw_stage.items()}
        corridor_total = baseline_total
    elif cell.architecture in {"port_only", "end_to_end"}:
        stage = raw_stage
        corridor_total = (
            baseline_total
            if cell.architecture == "port_only"
            else baseline_total + new_count * template_scale
        )
    elif cell.is_common_baseline:
        stage = dict(anchor_scales)
        corridor_total = baseline_total
    else:
        raise ValueError(cell.architecture)

    routes = copy.deepcopy(list(base["routes"]))
    for index, gateway in enumerate(new_gateways, start=1):
        routes.append(
            {
                "route_id": f"SemiSynthetic_Route_{index:02d}",
                "gateway": gateway,
                "corridor": "landbridge_shared",
                "maritime_lag_weeks": template_lag,
                "evidence_status": "semi-synthetic median template",
            }
        )
    route_gateway = {str(item["route_id"]): str(item["gateway"]) for item in routes}
    e2e_gateway = _e2e_capacity(stage, corridor_total)
    new_e2e = sum(e2e_gateway[gateway] for gateway in new_gateways)
    total_e2e = sum(e2e_gateway.values())
    new_total_share = new_e2e / total_e2e if total_e2e > 0.0 else 0.0
    reference_loading: dict[str, float] = {}
    for route in OBSERVED_ROUTES:
        reference_loading[route] = (1.0 - new_total_share) * official_shares[route]
    if new_gateways:
        for route, gateway in route_gateway.items():
            if gateway in new_gateways:
                reference_loading[route] = new_total_share * e2e_gateway[gateway] / new_e2e

    if cell.eligibility in {"emergency_only", "observed_reference"}:
        committed = {**official_shares}
        for route, gateway in route_gateway.items():
            if gateway in new_gateways:
                committed[route] = 0.0
    elif cell.eligibility == "precontracted":
        committed = dict(reference_loading)
    else:
        raise ValueError(cell.eligibility)

    waiting_sigma = dict(anchor_sigma)
    for route, gateway in route_gateway.items():
        if gateway in new_gateways:
            waiting_sigma[route] = template_sigma

    config["routes"] = routes
    config["information"]["waiting_error_scale_weeks_by_route"] = waiting_sigma
    config["information"]["waiting_error_scale_calibration_source_routes"] = list(OBSERVED_ROUTES)
    config["network_design"] = {
        "cell_id": cell.cell_id,
        "gateway_count": cell.gateway_count,
        "architecture": cell.architecture,
        "eligibility": cell.eligibility,
        "observed_route_ids": list(OBSERVED_ROUTES),
        "semi_synthetic_gateway_ids": new_gateways,
        "gateway_scales_model_units": stage,
        "shared_corridor_capacity_model_units": corridor_total,
        "committed_shares_by_route": committed,
        "reference_loading_shares_by_route": reference_loading,
        "end_to_end_reference_capacity_by_gateway": e2e_gateway,
        "new_gateway_committed_share": new_total_share if cell.eligibility == "precontracted" else 0.0,
        "new_gateway_reference_loading_share": new_total_share,
        "template_scale_model_units": template_scale,
        "template_maritime_lag_weeks": template_lag,
        "template_waiting_error_scale_weeks": template_sigma,
        "evidence_status": "semi-synthetic network structural stress; not a named-port forecast",
    }
    config["main_policies"] = list(experiment["policy_design"]["full_policy_anchors"])
    config["learning_policies"] = ["Behaviour cloning", "Constrained SAC", "Model-guided constrained SAC"]
    config["computation"]["parallel_evaluation_workers"] = int(experiment["execution"]["parallel_workers"])
    expected_action_dimension = 3 * (3 * cell.gateway_count + 1) + 1 + cell.gateway_count
    config["network_design"]["expected_action_dimension"] = expected_action_dimension
    return config


def network_register(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    design = config["network_design"]
    route_lookup = {str(item["gateway"]): str(item["route_id"]) for item in config["routes"]}
    rows = []
    for gateway, capacity in design["gateway_scales_model_units"].items():
        route = route_lookup[gateway]
        rows.append(
            {
                "cell_id": design["cell_id"],
                "gateway_count": design["gateway_count"],
                "architecture": design["architecture"],
                "eligibility": design["eligibility"],
                "gateway_id": gateway,
                "route_id": route,
                "is_semi_synthetic": gateway in set(design["semi_synthetic_gateway_ids"]),
                "stage_capacity_model_units": capacity,
                "shared_corridor_capacity_model_units": design["shared_corridor_capacity_model_units"],
                "end_to_end_reference_capacity": design["end_to_end_reference_capacity_by_gateway"][gateway],
                "committed_share": design["committed_shares_by_route"][route],
                "reference_loading_share": design["reference_loading_shares_by_route"][route],
                "maritime_lag_weeks": next(int(x["maritime_lag_weeks"]) for x in config["routes"] if x["route_id"] == route),
                "sigma_W_weeks": config["information"]["waiting_error_scale_weeks_by_route"][route],
                "evidence_status": "semi-synthetic median template" if gateway in set(design["semi_synthetic_gateway_ids"]) else "observed reference gateway",
            }
        )
    return rows
