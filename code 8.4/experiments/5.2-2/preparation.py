"""Beginning-of-week information, scenario, and realization preparation.

This module is the experiment-layer adapter to the current ``src/tre84``
contract.  It deliberately separates what is known at the decision time from
the physical outcome subsequently realised during the week.  Every policy,
teacher, training episode, validation run, and test run calls the same
``prepare_period`` function before producing an action.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tre84.errors import ContractError
from tre84.scenarios import EventPath, ScenarioBundle
from tre84.state import ModelState, RiskInformation
from tre84.transition import ExogenousRealization

from model import BenchmarkModel


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: pd.Timestamp(value).isoformat()
        if isinstance(value, (pd.Timestamp, np.datetime64))
        else np.asarray(value).tolist()
        if isinstance(value, np.ndarray)
        else str(value),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PreparedBenchmarkPeriod:
    state: ModelState
    scenarios: ScenarioBundle
    decision_time: pd.Timestamp
    information_vector_hash: str
    observation_hash: str


def _last_revealed_serviceability(state: ModelState) -> float:
    if not state.serviceability_history:
        return 1.0
    return float(state.serviceability_history[-1])


def _next_observed_covariates(
    *,
    model: BenchmarkModel,
    state: ModelState,
    row: Mapping[str, Any],
    serviceability: float,
    availability_time: pd.Timestamp,
) -> dict[str, Any]:
    covariates = dict(state.observed_covariates)
    covariates.update(
        {
            "estimated_normal_demand": float(row["normal_model_units"]),
            "observed_serviceability": float(serviceability),
            "last_realized_week": pd.Timestamp(row["week"]),
            "operational_information_available_at": availability_time,
            "released_high_risk_belief": float(state.risk.belief[-1]),
            "lead_high_risk_forecast": float(state.risk.lead_time_forecast[-1]),
            "release_date": state.risk.latest_release_time,
            "serviceability_timestamps": tuple(
                state.observed_covariates.get("serviceability_timestamps", ())
            )
            + (availability_time,),
        }
    )
    for route_id in model.network.routes:
        covariates.setdefault(
            f"market_cost_{route_id}",
            float(model.config["behavior"]["route_market_cost_default"]),
        )
    return covariates


def build_realization(
    *,
    model: BenchmarkModel,
    state: ModelState,
    row: Mapping[str, Any],
    information_available_at: Any | None = None,
) -> ExogenousRealization:
    """Build the physical outcome realised *after* the current action.

    No next-row value is accepted by this API.  Consequently a caller cannot
    accidentally insert the following week's serviceability or risk payload
    into the current state.
    """

    cargo = str(model.config["cargo_class"])
    decision_time = pd.Timestamp(row["week"])
    availability_time = pd.Timestamp(
        information_available_at
        if information_available_at is not None
        else decision_time + pd.Timedelta(weeks=1)
    )
    serviceability = float(row["serviceability"])
    if not 0.0 <= serviceability <= 1.0:
        raise ContractError("Serviceability must lie in [0, 1]")
    active = serviceability < 1.0 - 1e-12
    return ExogenousRealization(
        gulf_demand={cargo: float(row["normal_model_units"])},
        serviceable_share={cargo: serviceability},
        committed_fraction={
            cargo: float(model.config["committed_fraction_reference"])
        },
        committed_route_share=model.committed_shares,
        base_arrivals={},
        choice_route_available=frozenset(model.network.routes),
        physical_route_available=frozenset(model.network.routes),
        serviceability_observation=serviceability,
        next_disruption_seen=bool(state.disruption_seen or active),
        next_disruption_active=active,
        next_disruption_duration=(state.disruption_duration + 1 if active else 0),
        next_risk=state.risk,
        next_observed_covariates=_next_observed_covariates(
            model=model,
            state=state,
            row=row,
            serviceability=serviceability,
            availability_time=availability_time,
        ),
    )


def _scenario_bundle(
    *,
    model: BenchmarkModel,
    state: ModelState,
    row: Mapping[str, Any],
) -> ScenarioBundle:
    horizon = int(model.config["mpc"]["control_horizon_weeks"])
    # Only a completed, timestamped service observation can affect current
    # operational weights.  The current path row is the subsequent outcome.
    current_service = _last_revealed_serviceability(state)
    normal = float(
        state.observed_covariates.get(
            "estimated_normal_demand", sum(model.gateway_scales.values())
        )
    )
    decision_time = pd.Timestamp(row["week"])
    release_time = pd.Timestamp(row["release_date"])
    scenario_ids = tuple(str(value) for value in model.config["mpc"]["scenario_ids"])
    recovery_speeds = tuple(
        float(value)
        for value in model.config["mpc"]["scenario_recovery_speed_multipliers"]
    )
    if len(scenario_ids) != len(recovery_speeds):
        raise ContractError("MPC scenario ids and recovery speeds must align")

    paths: list[EventPath] = []
    for scenario_id, recovery_speed in zip(scenario_ids, recovery_speeds):
        services = np.asarray(
            [
                np.clip(
                    current_service
                    + (1.0 - current_service)
                    * min(1.0, recovery_speed * (offset + 1) / horizon),
                    0.0,
                    1.0,
                )
                for offset in range(horizon)
            ],
            dtype=float,
        )
        active = tuple(bool(value < 1.0 - 1e-12) for value in services)
        onset = tuple(
            flag and (offset == 0 or not active[offset - 1])
            for offset, flag in enumerate(active)
        )
        simulated = state.clone()
        payload: list[ExogenousRealization] = []
        for offset, service in enumerate(services):
            predicted_week = decision_time + pd.Timedelta(weeks=offset)
            predicted = {
                "week": predicted_week,
                "normal_model_units": normal,
                "serviceability": float(service),
            }
            realization = build_realization(
                model=model,
                state=simulated,
                row=predicted,
                information_available_at=predicted_week + pd.Timedelta(weeks=1),
            )
            payload.append(realization)
            simulated.disruption_seen = realization.next_disruption_seen
            simulated.disruption_active = realization.next_disruption_active
            simulated.disruption_duration = realization.next_disruption_duration
            simulated.serviceability_history = simulated.serviceability_history + (
                float(service),
            )
            simulated.observed_covariates = dict(realization.next_observed_covariates)
        paths.append(
            EventPath(
                path_id=f"mpc_{scenario_id}",
                onset=onset,
                serviceability={str(model.config["cargo_class"]): services},
                active=active,
                payload=tuple(payload),
            )
        )

    lead = float(state.risk.lead_time_forecast[-1])
    central_mass = float(model.config["mpc"]["scenario_weight_central_mass"])
    tail_mass = float(model.config["mpc"]["scenario_weight_tail_mass"])
    readiness = np.asarray(
        [tail_mass * lead, central_mass, tail_mass * (1.0 - lead)], dtype=float
    )
    readiness /= readiness.sum()
    operational = np.asarray(
        [
            tail_mass * (1.0 - current_service),
            central_mass,
            tail_mass * current_service,
        ],
        dtype=float,
    )
    operational /= operational.sum()
    active_weights = readiness if state.phase == 0 else operational
    reclosure = float(active_weights[0])
    weighted = sum(
        weight * np.asarray(path.serviceability[str(model.config["cargo_class"])])
        for weight, path in zip(active_weights, paths)
    )
    return ScenarioBundle(
        paths=tuple(paths),
        readiness_weights=readiness,
        operational_weights=operational,
        active_weights=active_weights,
        reclosure_probability=reclosure,
        weighted_serviceability={str(model.config["cargo_class"]): weighted},
        information_timestamps=(release_time,),
        seed_manifest={scenario_id: index for index, scenario_id in enumerate(scenario_ids)},
        decision_time=decision_time,
    )


def prepare_period(
    *,
    model: BenchmarkModel,
    state: ModelState,
    row: Mapping[str, Any],
) -> PreparedBenchmarkPeriod:
    """Prepare released information and one common scenario bundle."""

    decision_time = pd.Timestamp(row["week"])
    release_time = pd.Timestamp(row["release_date"])
    if release_time > decision_time or not bool(row["timing_valid"]):
        raise ContractError("A decision attempted to use unreleased GPR information")
    source = str(row.get("information_source", ""))
    if "ramp" in source.lower():
        raise ContractError("Artificial risk ramps are forbidden")
    current = float(row["filtered_high_risk_probability"])
    lead = float(row["lead_time_high_risk_probability"])
    if not (0.0 <= current <= 1.0 and 0.0 <= lead <= 1.0):
        raise ContractError("Released risk probabilities must lie in [0, 1]")

    prepared = state.clone()
    preliminary = RiskInformation(
        belief=np.asarray([1.0 - current, current], dtype=float),
        lead_time_forecast=np.asarray([1.0 - lead, lead], dtype=float),
        scenario_ids=tuple(str(value) for value in model.config["mpc"]["scenario_ids"]),
        readiness_weights=np.full(int(model.config["mpc"]["scenario_count"]), 1.0 / int(model.config["mpc"]["scenario_count"])),
        operational_weights=np.full(int(model.config["mpc"]["scenario_count"]), 1.0 / int(model.config["mpc"]["scenario_count"])),
        latest_release_period=state.period,
        latest_release_time=release_time,
    )
    prepared.risk = preliminary
    prepared.observed_covariates.update(
        {
            "released_high_risk_belief": current,
            "lead_high_risk_forecast": lead,
            "release_date": release_time,
            "decision_week": decision_time,
            "information_source": source,
        }
    )
    timestamps = tuple(prepared.observed_covariates.get("serviceability_timestamps", ()))
    if len(timestamps) != len(prepared.serviceability_history):
        raise ContractError("Every revealed serviceability value needs one timestamp")
    if any(pd.Timestamp(timestamp) > decision_time for timestamp in timestamps):
        raise ContractError("Future operational information reached the current decision")

    bundle = _scenario_bundle(model=model, state=prepared, row=row)
    prepared.risk = RiskInformation(
        belief=preliminary.belief,
        lead_time_forecast=preliminary.lead_time_forecast,
        scenario_ids=tuple(path.path_id for path in bundle.paths),
        readiness_weights=bundle.readiness_weights.copy(),
        operational_weights=bundle.operational_weights.copy(),
        reclosure_probability=bundle.reclosure_probability,
        latest_release_period=state.period,
        latest_release_time=release_time,
    )
    prepared.validate(tolerance=float(model.config["numerics"]["mass_tolerance"]))
    information_payload = {
        "decision_time": decision_time,
        "release_time": release_time,
        "belief": prepared.risk.belief,
        "lead": prepared.risk.lead_time_forecast,
        "scenario_ids": prepared.risk.scenario_ids,
        "readiness_weights": prepared.risk.readiness_weights,
        "operational_weights": prepared.risk.operational_weights,
    }
    observation_payload = {
        **information_payload,
        "period": prepared.period,
        "phase": prepared.phase,
        "budget": prepared.budget,
        "cargo_mass": prepared.cargo_mass(),
        "serviceability_history": prepared.serviceability_history,
        "serviceability_timestamps": timestamps,
    }
    return PreparedBenchmarkPeriod(
        state=prepared,
        scenarios=bundle,
        decision_time=decision_time,
        information_vector_hash=_sha256_payload(information_payload),
        observation_hash=_sha256_payload(observation_payload),
    )

