"""Frozen parameter cells and model/path transformations for Experiment 5.3.4."""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from paths import PhysicalPath, _canonical_path_hash


CODE_ROOT = Path(__file__).resolve().parents[2]
NETWORK_DIR = CODE_ROOT / "experiments" / "5.3-3"
if str(NETWORK_DIR) not in sys.path:
    sys.path.insert(0, str(NETWORK_DIR))

from network_5_3_3 import NetworkCell, build_cell_config  # noqa: E402

DIAGNOSTIC_ONLY_FACTORS = {"clearance_tolerance"}


def _level_tag(value: float) -> str:
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class RobustnessCell:
    cell_id: str
    cell_type: str
    family: str
    display_factor: str
    display_level: str
    factors: Mapping[str, float]
    basis: str
    path_stress: str = "historical"
    network_stress: str = "reference"

    @property
    def is_reference(self) -> bool:
        return self.cell_type == "reference"

    @property
    def is_interaction(self) -> bool:
        return self.cell_type == "interaction"


def build_cells(experiment: Mapping[str, Any]) -> tuple[list[RobustnessCell], list[dict[str, Any]]]:
    cells = [
        RobustnessCell(
            cell_id="reference",
            cell_type="reference",
            family="reference",
            display_factor="reference",
            display_level="reference",
            factors={},
            basis="accepted common-authority reference parameter register",
        )
    ]
    diagnostics: list[dict[str, Any]] = []
    for item in experiment["one_factor_screen"]:
        factor = str(item["factor"])
        reference = float(item["reference"])
        if factor in DIAGNOSTIC_ONLY_FACTORS:
            for level in item["levels"]:
                diagnostics.append(
                    {
                        "factor": factor,
                        "level": float(level),
                        "reference": reference,
                        "operator": item["operator"],
                        "execution_mode": "post_trajectory_classification_diagnostic",
                    }
                )
            continue
        for level in item["levels"]:
            value = float(level)
            if np.isclose(value, reference):
                continue
            cells.append(
                RobustnessCell(
                    cell_id=f"{factor}__{_level_tag(value)}",
                    cell_type="one_factor",
                    family=str(item["family"]),
                    display_factor=factor,
                    display_level=f"{value:g}",
                    factors={factor: value},
                    basis=str(item["operator"]),
                )
            )
    for item in experiment["interactions"]:
        factors = {str(key): float(value) for key, value in item["factors"].items()}
        cells.append(
            RobustnessCell(
                cell_id=str(item["cell_id"]),
                cell_type="interaction",
                family="interaction",
                display_factor=" + ".join(factors),
                display_level=" + ".join(f"{key}={value:g}" for key, value in factors.items()),
                factors=factors,
                basis=str(item["basis"]),
                path_stress=str(item.get("path_stress", "historical")),
                network_stress=str(item.get("network_stress", "reference")),
            )
        )
    expected = int(experiment["cell_contract"]["total_unique_cells"])
    if len(cells) != expected or len({cell.cell_id for cell in cells}) != expected:
        raise RuntimeError(f"Expected {expected} unique simulated cells, obtained {len(cells)}")
    if len(diagnostics) != 3:
        raise RuntimeError("Clearance classification requires exactly three declared tolerances")
    return cells, diagnostics


def full_policy_anchor(cell: RobustnessCell) -> bool:
    return cell.is_reference or cell.cell_id in {
        "interaction__long_lag__severe_reclosure",
        "interaction__convex_hazard__low_exit",
        "interaction__n09__low_corridor",
    }


def dimension_changed_cell(cell: RobustnessCell) -> bool:
    return "maritime_lag" in cell.factors or "readiness_lead" in cell.factors


def policy_family(cell: RobustnessCell, experiment: Mapping[str, Any]) -> list[str]:
    if full_policy_anchor(cell):
        return list(experiment["policy_design"]["full_policy_anchors"])
    if dimension_changed_cell(cell):
        return list(experiment["policy_design"]["dimension_changed_policies"])
    return list(experiment["policy_design"]["screening_policies"])


def _observed_gateway_scales() -> dict[str, float]:
    frame = pd.read_csv(
        CODE_ROOT
        / "experiments"
        / "data"
        / "processed"
        / "anchors"
        / "gateway_reference_scales.csv"
    )
    result = {
        str(row.gateway): float(row.activity_scale_model_units)
        for row in frame.itertuples(index=False)
    }
    if len(result) != 3 or any(value <= 0.0 for value in result.values()):
        raise RuntimeError("The frozen observed gateway scale register is invalid")
    return result


def _half_up_positive_week(value: float) -> int:
    return max(1, int(math.floor(float(value) + 0.5)))


def model_config(
    base: Mapping[str, Any], experiment: Mapping[str, Any], cell: RobustnessCell
) -> dict[str, Any]:
    """Apply only declared transformations to a deep copy of the accepted model."""

    if cell.network_stress == "n09_end_to_end_precontracted":
        config = build_cell_config(
            base,
            experiment,
            NetworkCell(
                "n09_end_to_end_precontracted",
                9,
                "end_to_end",
                "precontracted",
            ),
        )
    else:
        config = copy.deepcopy(dict(base))
    config["experiment_id"] = str(experiment["experiment_id"])
    config["committed_fraction_reference"] = float(
        experiment["commitment_contract"]["reference_fraction"]
    )
    config["main_policies"] = list(experiment["policy_design"]["full_policy_anchors"])
    config["learning_policies"] = ["Behaviour cloning", "Model-guided constrained SAC"]
    factors = dict(cell.factors)

    if "route_sensitivity" in factors:
        config["behavior"]["logit_theta"] = float(base["behavior"]["logit_theta"]) * factors["route_sensitivity"]
    if "waiting_hazard_power" in factors:
        config["behavior"]["hazard_power"] = factors["waiting_hazard_power"]
    if "exit_consequence" in factors:
        multiplier = factors["exit_consequence"]
        config["behavior"]["exit_failure_cost_per_unit"] = float(base["behavior"]["exit_failure_cost_per_unit"]) * multiplier
    if "maritime_lag" in factors:
        multiplier = factors["maritime_lag"]
        for route in config["routes"]:
            route["maritime_lag_weeks"] = _half_up_positive_week(
                float(route["maritime_lag_weeks"]) * multiplier
            )
    if "physical_feedback" in factors:
        config["physical_feedback"]["strength"] = factors["physical_feedback"]
    if "route_resource_cost" in factors:
        config["route_resource_cost"]["robustness_multiplier"] = factors["route_resource_cost"]
    if "action_cost" in factors:
        multiplier = factors["action_cost"]
        for key in (
            "readiness_order_cost_per_unit",
            "direct_order_cost_per_unit",
            "readiness_exercise_cost_per_unit",
            "release_cost_per_unit",
            "publication_cost_per_unit",
        ):
            config["action"][key] = float(base["action"][key]) * multiplier
    if "authority_budget" in factors:
        config["action"]["period_budget_fraction"] = factors["authority_budget"]
        config["action"]["cumulative_budget_fraction"] = factors["authority_budget"]
    if "readiness_lead" in factors:
        config["action"]["readiness_lead_weeks"] = _half_up_positive_week(factors["readiness_lead"])
    if "waiting_error_scale" in factors:
        multiplier = factors["waiting_error_scale"]
        config["information"]["waiting_error_scale_robustness_multiplier"] = multiplier
        config["information"]["waiting_error_scale_weeks_by_route"] = {
            route: float(value) * multiplier
            for route, value in config["information"]["waiting_error_scale_weeks_by_route"].items()
        }
    if "information_credibility" in factors:
        config["information"]["gamma_I"] = factors["information_credibility"]

    if "port_service" in factors or "corridor_capacity" in factors:
        observed = _observed_gateway_scales()
        port_multiplier = factors.get("port_service", 1.0)
        corridor_multiplier = factors.get("corridor_capacity", 1.0)
        if cell.network_stress == "reference":
            config["network_design"] = {
                "observed_route_ids": [str(route["route_id"]) for route in config["routes"]],
                "gateway_scales_model_units": {
                    gateway: value * port_multiplier for gateway, value in observed.items()
                },
                "shared_corridor_capacity_model_units": sum(observed.values()) * corridor_multiplier,
            }
        else:
            design = config["network_design"]
            design["gateway_scales_model_units"] = {
                gateway: float(value) * port_multiplier
                for gateway, value in design["gateway_scales_model_units"].items()
            }
            design["shared_corridor_capacity_model_units"] = (
                float(design["shared_corridor_capacity_model_units"])
                * corridor_multiplier
            )

    if not 0.0 <= float(config["information"]["gamma_I"]) <= 1.0:
        raise RuntimeError("Information credibility left its formal [0,1] domain")
    if float(config["behavior"]["hazard_power"]) <= 0.0:
        raise RuntimeError("Waiting hazard power must be positive")
    if float(config["behavior"]["exit_failure_cost_per_unit"]) <= 0.0:
        raise RuntimeError("Exit opportunity cost must be positive")
    return config


def _extension_row(
    base_path: PhysicalPath,
    extension_offset: int,
    serviceability: float,
) -> dict[str, Any]:
    last = base_path.frame.iloc[-1]
    row = last.to_dict()
    row["week"] = pd.Timestamp(last["week"]) + pd.Timedelta(weeks=extension_offset + 1)
    row["serviceability"] = float(serviceability)
    row["filtered_high_risk_probability"] = float(last["filtered_high_risk_probability"])
    row["lead_time_high_risk_probability"] = float(last["lead_time_high_risk_probability"])
    row["release_date"] = pd.Timestamp(last["release_date"])
    row["source_observation_month"] = pd.Timestamp(last["source_observation_month"])
    row["timing_valid"] = True
    row["information_source"] = "released_hmm_filter_last_available_carry_forward"
    return row


def _severe_reclosure_path(
    path: PhysicalPath, experiment: Mapping[str, Any]
) -> PhysicalPath:
    stress = experiment["anchor_contract"]["severe_reclosure"]
    open_weeks = int(stress["open_weeks"])
    duration = int(stress["reclosure_duration_weeks"])
    recovery = int(stress["recovery_weeks"])
    serviceability = 1.0 - float(stress["reclosure_intensity"])
    rows = [row._asdict() for row in path.frame.itertuples(index=False)]
    extension = 0
    for _ in range(open_weeks):
        rows.append(_extension_row(path, extension, 1.0))
        extension += 1
    for _ in range(duration):
        rows.append(_extension_row(path, extension, serviceability))
        extension += 1
    for step in range(1, recovery + 1):
        value = serviceability + (1.0 - serviceability) * step / recovery
        rows.append(_extension_row(path, extension, value))
        extension += 1
    frame = pd.DataFrame(rows)
    return PhysicalPath(
        path_id=path.path_id,
        split="parameter_robustness_severe_reclosure",
        frame=frame,
        path_hash=_canonical_path_hash(frame),
        construction=(path.construction + "; preregistered open=1, intensity=0.95, duration=32, recovery=8"),
        residual_start=path.residual_start,
        residual_end=path.residual_end,
        onset_week=len(path.frame) + open_weeks,
        active_duration_weeks=duration,
        severity_floor=serviceability,
        has_reclosure=True,
    )


def transform_paths(
    paths: Sequence[PhysicalPath],
    cell: RobustnessCell,
    experiment: Mapping[str, Any],
) -> list[PhysicalPath]:
    multiplier = float(cell.factors.get("network_exposure", 1.0))
    transformed = []
    for path in paths:
        current = path
        if not np.isclose(multiplier, 1.0):
            frame = path.frame.copy()
            frame["normal_model_units"] = frame["normal_model_units"].astype(float) * multiplier
            current = PhysicalPath(
                path_id=path.path_id,
                split=path.split,
                frame=frame,
                path_hash=_canonical_path_hash(frame),
                construction=path.construction + f"; normal demand multiplied by {multiplier:g}",
                residual_start=path.residual_start,
                residual_end=path.residual_end,
                onset_week=path.onset_week,
                active_duration_weeks=path.active_duration_weeks,
                severity_floor=path.severity_floor,
                has_reclosure=path.has_reclosure,
            )
        if cell.path_stress == "severe_reclosure":
            current = _severe_reclosure_path(current, experiment)
        transformed.append(current)
    return transformed


def cell_registry(
    cells: Sequence[RobustnessCell], experiment: Mapping[str, Any]
) -> pd.DataFrame:
    full = list(experiment["policy_design"]["full_policy_anchors"])
    rows = []
    for cell in cells:
        family = set(policy_family(cell, experiment))
        for policy in full:
            evaluated = policy in family
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "cell_type": cell.cell_type,
                    "family": cell.family,
                    "display_factor": cell.display_factor,
                    "display_level": cell.display_level,
                    "factor_values": "|".join(f"{key}={value:g}" for key, value in cell.factors.items()),
                    "basis": cell.basis,
                    "path_stress": cell.path_stress,
                    "network_stress": cell.network_stress,
                    "policy": policy,
                    "policy_evaluated": evaluated,
                    "evaluation_status": "EVALUATED" if evaluated else experiment["policy_design"]["not_evaluated_status"],
                    "comparison_family": (
                        "five_policy_anchor" if full_policy_anchor(cell)
                        else "dimension_changed_rule_mpc" if dimension_changed_cell(cell)
                        else "three_policy_screen"
                    ),
                }
            )
    return pd.DataFrame(rows)
