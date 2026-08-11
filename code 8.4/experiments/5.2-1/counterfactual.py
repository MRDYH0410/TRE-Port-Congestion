"""Event-free counterfactual models and exhaustive weekly rolling origins."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class FittedForecast:
    predictions: np.ndarray
    parameters: dict[str, Any]


def _json(parameters: Mapping[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def _harmonic_design(
    indices: np.ndarray, *, annual_period: float, order: int, trend_scale: float
) -> np.ndarray:
    columns = [np.ones(indices.size), indices / trend_scale]
    for harmonic in range(1, order + 1):
        angle = 2.0 * np.pi * harmonic * indices / annual_period
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(columns)


def fit_harmonic_ridge(
    values: np.ndarray,
    horizon: int,
    *,
    annual_period: float,
    harmonic_orders: Sequence[int],
    ridge_penalties: Sequence[float],
) -> FittedForecast:
    """Select harmonic order and ridge penalty by prefix-only GCV."""

    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or y.size < 3 or horizon <= 0:
        raise ValueError("Harmonic ridge needs a nonempty training prefix and horizon")
    indices = np.arange(y.size, dtype=float)
    trend_scale = max(float(y.size - 1), 1.0)
    best: tuple[float, int, float, np.ndarray] | None = None
    for order in harmonic_orders:
        design = _harmonic_design(
            indices, annual_period=annual_period, order=int(order), trend_scale=trend_scale
        )
        xtx = design.T @ design
        xty = design.T @ y
        penalty_mask = np.eye(design.shape[1])
        penalty_mask[0, 0] = 0.0
        for ridge in ridge_penalties:
            system = xtx + float(ridge) * penalty_mask
            try:
                beta = np.linalg.solve(system, xty)
                effective_df = float(np.trace(np.linalg.solve(system, xtx)))
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(system, xty, rcond=None)[0]
                effective_df = float(np.trace(np.linalg.pinv(system) @ xtx))
            residual = y - design @ beta
            denominator = max(1.0 - effective_df / y.size, np.finfo(float).eps)
            gcv = float(np.mean(np.square(residual)) / denominator**2)
            candidate = (gcv, int(order), float(ridge), beta)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:  # pragma: no cover - guarded by nonempty candidate configuration
        raise RuntimeError("No harmonic-ridge specification was evaluated")
    gcv, order, ridge, beta = best
    future_indices = np.arange(y.size, y.size + horizon, dtype=float)
    future_design = _harmonic_design(
        future_indices,
        annual_period=annual_period,
        order=order,
        trend_scale=trend_scale,
    )
    predictions = np.maximum(future_design @ beta, 0.0)
    return FittedForecast(
        predictions=predictions,
        parameters={
            "annual_period_weeks": annual_period,
            "harmonic_order": order,
            "ridge_penalty": ridge,
            "gcv": gcv,
            "nonnegativity_projection": "max(prediction,0)",
        },
    )


def _damped_state_errors(
    values: np.ndarray, alpha: float, beta: float, phi: float
) -> tuple[float, float, np.ndarray]:
    level = float(values[0])
    initial_span = min(values.size - 1, 52)
    trend = float(np.median(np.diff(values[: initial_span + 1]))) if initial_span else 0.0
    errors = np.empty(values.size - 1, dtype=float)
    for index in range(1, values.size):
        forecast = level + phi * trend
        errors[index - 1] = values[index] - forecast
        next_level = alpha * values[index] + (1.0 - alpha) * forecast
        next_trend = beta * (next_level - level) + (1.0 - beta) * phi * trend
        level, trend = float(next_level), float(next_trend)
    return level, trend, errors


def fit_damped_local_trend(
    values: np.ndarray,
    horizon: int,
    *,
    bounds: Mapping[str, Sequence[float]],
    initialisations: Sequence[Sequence[float]],
    max_iterations: int,
) -> FittedForecast:
    """Estimate damping and smoothing parameters on each completed prefix."""

    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or y.size < 3 or horizon <= 0:
        raise ValueError("Damped local trend needs a nonempty training prefix and horizon")
    center = float(np.mean(y))
    scale = float(np.std(y, ddof=0))
    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0
    z = (y - center) / scale
    parameter_bounds = [tuple(bounds[name]) for name in ("alpha", "beta", "phi")]

    def objective(parameters: np.ndarray) -> float:
        _, _, errors = _damped_state_errors(z, *parameters)
        return float(np.mean(np.square(errors)))

    candidates: list[tuple[float, np.ndarray, bool, int]] = []
    for start in initialisations:
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=parameter_bounds,
            options={"maxiter": int(max_iterations), "ftol": 1e-12},
        )
        if np.isfinite(result.fun):
            candidates.append(
                (float(result.fun), np.asarray(result.x), bool(result.success), int(result.nit))
            )
    if not candidates:
        raise RuntimeError("All damped-local-trend deterministic fits failed")
    objective_value, parameters, success, iterations = min(
        candidates, key=lambda item: (item[0], tuple(item[1]))
    )
    alpha, beta, phi = (float(value) for value in parameters)
    level, trend, _ = _damped_state_errors(z, alpha, beta, phi)
    steps = np.arange(1, horizon + 1, dtype=float)
    if abs(1.0 - phi) <= np.finfo(float).eps:
        multiplier = steps
    else:
        multiplier = phi * (1.0 - np.power(phi, steps)) / (1.0 - phi)
    predictions = np.maximum((level + multiplier * trend) * scale + center, 0.0)
    return FittedForecast(
        predictions=predictions,
        parameters={
            "alpha": alpha,
            "beta": beta,
            "phi": phi,
            "scaled_one_step_mse": objective_value,
            "optimizer_success": success,
            "optimizer_iterations": iterations,
            "deterministic_start_count": len(initialisations),
            "nonnegativity_projection": "max(prediction,0)",
        },
    )


def fit_seasonal_naive(
    values: np.ndarray, horizon: int, *, seasonal_lag: int
) -> FittedForecast:
    y = np.asarray(values, dtype=float)
    if y.size < seasonal_lag or horizon <= 0:
        raise ValueError("Seasonal naive requires at least one annual weekly lag")
    predictions = np.empty(horizon, dtype=float)
    history = list(y)
    for step in range(horizon):
        value = float(history[len(history) - seasonal_lag])
        predictions[step] = max(value, 0.0)
        history.append(value)
    return FittedForecast(
        predictions=predictions,
        parameters={
            "seasonal_lag_weeks": int(seasonal_lag),
            "basis": "nearest whole-week Gregorian annual recurrence",
            "nonnegativity_projection": "max(prediction,0)",
        },
    )


def fit_forecast_model(
    model: str, values: np.ndarray, horizon: int, config: Mapping[str, Any]
) -> FittedForecast:
    if model == "harmonic_ridge":
        return fit_harmonic_ridge(
            values,
            horizon,
            annual_period=float(config["annual_period_weeks"]),
            harmonic_orders=config["harmonic_order_candidates"],
            ridge_penalties=config["ridge_penalty_candidates"],
        )
    if model == "damped_local_trend":
        return fit_damped_local_trend(
            values,
            horizon,
            bounds=config["damped_parameter_bounds"],
            initialisations=config["damped_deterministic_initialisations"],
            max_iterations=int(config["damped_optimizer_max_iterations"]),
        )
    if model == "seasonal_naive":
        return fit_seasonal_naive(
            values, horizon, seasonal_lag=int(config["seasonal_naive_lag_weeks"])
        )
    raise ValueError(f"Unknown counterfactual model: {model}")


def run_rolling_origins(
    weekly: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Use every feasible weekly origin without event-window observations."""

    frame = weekly.sort_values("week_start").reset_index(drop=True).copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    if frame["week_start"].duplicated().any():
        raise ValueError("Counterfactual weekly dates must be unique")
    gaps = frame["week_start"].diff().dropna().dt.days
    if not gaps.eq(7).all():
        raise ValueError("Counterfactual training and validation weeks must be contiguous")
    y = frame["observed_activity"].to_numpy(dtype=float)
    dates = frame["week_start"].to_numpy()
    minimum_training = int(config["minimum_training_weeks"])
    maximum_horizon = int(config["maximum_forecast_horizon_weeks"])
    rows: list[dict[str, Any]] = []
    for origin_index in range(minimum_training - 1, len(frame) - 1):
        available = min(maximum_horizon, len(frame) - origin_index - 1)
        training = y[: origin_index + 1]
        for model in config["candidate_models"]:
            fitted = fit_forecast_model(model, training, available, config)
            parameter_json = _json(fitted.parameters)
            for step, prediction in enumerate(fitted.predictions, start=1):
                target_index = origin_index + step
                observed = float(y[target_index])
                predicted = float(prediction)
                rows.append(
                    {
                        "model": model,
                        "origin_date": pd.Timestamp(dates[origin_index]),
                        "training_start": pd.Timestamp(dates[0]),
                        "training_cutoff": pd.Timestamp(dates[origin_index]),
                        "training_weeks": origin_index + 1,
                        "target_date": pd.Timestamp(dates[target_index]),
                        "forecast_horizon": step,
                        "available_horizon_at_origin": available,
                        "observed_value": observed,
                        "predicted_value": predicted,
                        "prediction_error": predicted - observed,
                        "residual": observed - predicted,
                        "fitted_parameters": parameter_json,
                    }
                )
    return pd.DataFrame(rows)


def summarise_rolling_predictions(
    predictions: pd.DataFrame, horizons: Sequence[int]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(predictions["model"].unique()):
        model_rows = predictions.loc[predictions["model"].eq(model)]
        maximum_by_origin = model_rows.groupby("origin_date")[
            "available_horizon_at_origin"
        ].max()
        for horizon in horizons:
            origins = maximum_by_origin.index[maximum_by_origin.ge(horizon)]
            subset = model_rows.loc[
                model_rows["origin_date"].isin(origins)
                & model_rows["forecast_horizon"].le(horizon)
            ]
            denominator = float(subset["observed_value"].abs().sum())
            absolute_error = subset["prediction_error"].abs()
            bias = float(subset["prediction_error"].mean())
            mean_observed = float(subset["observed_value"].abs().mean())
            rows.append(
                {
                    "model": model,
                    "evaluation_horizon_weeks": int(horizon),
                    "origin_count": len(origins),
                    "point_count": len(subset),
                    "wape": float(absolute_error.sum() / denominator),
                    "mae": float(absolute_error.mean()),
                    "bias": bias,
                    "normalised_bias_percent": 100.0 * bias / mean_observed,
                    "validation_start": subset["target_date"].min(),
                    "validation_end": subset["target_date"].max(),
                    "event_observations_used": int(
                        (subset["target_date"] > pd.Timestamp("2026-02-16")).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def one_step_residual_acf(
    predictions: pd.DataFrame, *, maximum_lag: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(predictions["model"].unique()):
        one_step = predictions.loc[
            predictions["model"].eq(model) & predictions["forecast_horizon"].eq(1)
        ].sort_values("target_date")
        if one_step["target_date"].duplicated().any():
            raise ValueError("One-step residual dates must be ordered and nonduplicated")
        residual = one_step["residual"].to_numpy(dtype=float)
        centered = residual - residual.mean()
        denominator = float(centered @ centered)
        confidence = 1.96 / np.sqrt(residual.size)
        for lag in range(1, maximum_lag + 1):
            numerator = float(centered[:-lag] @ centered[lag:])
            acf = 0.0 if denominator <= 0 else numerator / denominator
            rows.append(
                {
                    "model": model,
                    "lag_weeks": lag,
                    "autocorrelation": acf,
                    "confidence_lower": -confidence,
                    "confidence_upper": confidence,
                    "one_step_residual_count": residual.size,
                    "residual_start": one_step["target_date"].min(),
                    "residual_end": one_step["target_date"].max(),
                    "residual_dates_unique": True,
                }
            )
    return pd.DataFrame(rows)


def select_counterfactual_model(
    summary: pd.DataFrame,
    acf: pd.DataFrame,
    *,
    primary_horizon: int,
    relative_wape_tolerance: float,
) -> tuple[str, pd.DataFrame]:
    primary = summary.loc[
        summary["evaluation_horizon_weeks"].eq(primary_horizon)
    ].copy()
    dependence = (
        acf.groupby("model")["autocorrelation"]
        .apply(lambda values: float(values.abs().sum()))
        .rename("one_step_residual_dependence")
    )
    primary = primary.merge(dependence, on="model", how="left", validate="one_to_one")
    best_wape = float(primary["wape"].min())
    primary["within_wape_tolerance"] = primary["wape"].le(
        best_wape * (1.0 + relative_wape_tolerance) + np.finfo(float).eps
    )
    primary["wape_rank"] = primary["wape"].rank(method="min").astype(int)
    eligible = primary.loc[primary["within_wape_tolerance"]].copy()
    eligible["absolute_bias"] = eligible["bias"].abs()
    eligible = eligible.sort_values(
        ["absolute_bias", "one_step_residual_dependence", "model"], ignore_index=True
    )
    selected = str(eligible.loc[0, "model"])
    primary["selected"] = primary["model"].eq(selected)
    primary["primary_metric"] = "cumulative_path_WAPE"
    primary["primary_horizon_weeks"] = primary_horizon
    primary["relative_wape_tie_tolerance"] = relative_wape_tolerance
    primary["first_tie_break"] = "absolute_bias"
    primary["second_tie_break"] = "one_step_residual_dependence"
    primary["selection_rule_locked_before_run"] = True
    primary = primary.sort_values(["selected", "wape"], ascending=[False, True])
    return selected, primary.reset_index(drop=True)


def model_specification_table(config: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "harmonic_ridge",
                "component": "annual seasonality",
                "candidate_or_value": _json(
                    {
                        "annual_period_weeks": config["annual_period_weeks"],
                        "harmonic_orders": config["harmonic_order_candidates"],
                        "ridge_penalties": config["ridge_penalty_candidates"],
                        "design": config["harmonic_design"],
                        "penalty_rule": config["harmonic_penalty_rule"],
                    }
                ),
                "estimation_or_selection": "order and ridge selected by GCV within every completed training prefix",
                "basis": config["harmonic_maximum_order_basis"],
                "event_data_used": False,
            },
            {
                "model": "damped_local_trend",
                "component": "local level, trend, and damping",
                "candidate_or_value": _json(
                    {
                        "bounds": config["damped_parameter_bounds"],
                        "starts": config["damped_deterministic_initialisations"],
                        "numerical_scaling": config["damped_numerical_scaling"],
                        "initial_state": config["damped_initial_state"],
                    }
                ),
                "estimation_or_selection": "alpha, beta, and phi minimise one-step prefix MSE using deterministic multi-start optimisation",
                "basis": "damping is estimated separately in every training prefix and is never fixed at 0.9",
                "event_data_used": False,
            },
            {
                "model": "seasonal_naive",
                "component": "annual weekly recurrence",
                "candidate_or_value": _json(
                    {"seasonal_lag_weeks": config["seasonal_naive_lag_weeks"]}
                ),
                "estimation_or_selection": "fixed calendar identity; no fitted hyperparameter",
                "basis": "52 weeks is the nearest whole-week recurrence to the Gregorian annual period",
                "event_data_used": False,
            },
        ]
    )
