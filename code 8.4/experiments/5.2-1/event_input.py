"""Formula-derived historical event input; no committed-share construction."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from counterfactual import fit_forecast_model


def construct_historical_event(
    weekly: pd.DataFrame,
    *,
    selected_model: str,
    counterfactual_config: Mapping[str, Any],
    selection_cutoff: pd.Timestamp,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    network_exposure_reference: float,
    model_unit_tonnes: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the paper's serviceability and positive-shortfall identities."""

    frame = weekly.sort_values("week_start").reset_index(drop=True).copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    training = frame.loc[frame["week_start"].le(selection_cutoff)].copy()
    event = frame.loc[frame["week_start"].between(event_start, event_end)].copy()
    expected_weeks = int((event_end - event_start).days // 7 + 1)
    if len(event) != expected_weeks:
        raise ValueError("The historical event window is not a complete weekly sequence")
    fitted = fit_forecast_model(
        selected_model,
        training["observed_activity"].to_numpy(dtype=float),
        len(event),
        counterfactual_config,
    )
    estimated = np.asarray(fitted.predictions, dtype=float)
    if np.any(estimated <= 0) or np.any(~np.isfinite(estimated)):
        raise ValueError("The selected no-disruption event forecast must be finite and positive")
    observed = event["observed_activity"].to_numpy(dtype=float)
    serviceability = np.minimum(1.0, observed / estimated)
    blocked = np.maximum(estimated - observed, 0.0)
    identity = (1.0 - serviceability) * estimated
    model_units = network_exposure_reference * blocked / model_unit_tonnes
    result = pd.DataFrame(
        {
            "week": event["week_start"].to_numpy(),
            "observed_activity": observed,
            "estimated_no_disruption_activity": estimated,
            "serviceability": serviceability,
            "blocked_activity_proxy": blocked,
            "blocked_identity_value": identity,
            "blocked_identity_residual": blocked - identity,
            "network_exposure_reference": network_exposure_reference,
            "model_unit_tonnes": model_unit_tonnes,
            "model_blocked_units": model_units,
            "model_unit_conversion_residual": (
                model_units - network_exposure_reference * blocked / model_unit_tonnes
            ),
            "selected_counterfactual_model": selected_model,
            "counterfactual_training_cutoff": selection_cutoff,
            "counterfactual_event_observations_used_in_fit": False,
            "activity_unit": "metric tonnes of AIS-derived carrying-capacity proxy per week",
            "blocked_interpretation": "estimated positive activity shortfall; not observed diversion",
        }
    )
    parameters = dict(fitted.parameters)
    parameters["selected_model"] = selected_model
    parameters["training_start"] = str(training["week_start"].min().date())
    parameters["training_cutoff"] = str(selection_cutoff.date())
    parameters["event_horizon_weeks"] = len(event)
    parameters["event_observations_used"] = False
    return result, parameters


def build_frozen_information_event_path(
    event: pd.DataFrame, release_clock: pd.DataFrame
) -> pd.DataFrame:
    """Join the formula-derived event with the one released-information clock."""

    clock = release_clock.copy()
    clock["decision_week"] = pd.to_datetime(clock["decision_week"])
    result = event.merge(
        clock,
        left_on="week",
        right_on="decision_week",
        how="left",
        validate="one_to_one",
    )
    if result["source_observation_month"].isna().any():
        raise ValueError("Every historical event week needs one released HMM source month")
    result["interface_version"] = "5.2.1-v1"
    result["event_path_source"] = "formula_derived_5.2.1"
    result["contains_committed_share"] = False
    return result


def event_parameter_json(parameters: Mapping[str, Any]) -> str:
    return json.dumps(parameters, indent=2, sort_keys=True)

