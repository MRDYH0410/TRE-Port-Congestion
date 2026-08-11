"""Two-state released-information HMM estimation and validity diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from tre84.information import FrozenStandardizer, GaussianHMM


@dataclass(frozen=True)
class HMMFitResult:
    model: GaussianHMM
    standardizer: FrozenStandardizer
    feature_names: tuple[str, ...]
    training_rows: int
    heldout_rows: int
    selected_initialisation: int
    selected_initialisation_name: str
    likelihood_history: tuple[float, ...]
    converged: bool
    initialisation_summary: tuple[dict[str, Any], ...]


def _normalise(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Probability normalisation failed")
    return values / total


def _partition_parameters(
    observations: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores, kind="mergesort")
    midpoint = len(order) // 2
    groups = (order[:midpoint], order[midpoint:])
    means = np.vstack([observations[group].mean(axis=0) for group in groups])
    variances = np.vstack([observations[group].var(axis=0) for group in groups])
    return means, np.maximum(variances, 1e-3)


def _initial_models(
    observations: np.ndarray,
    feature_names: Sequence[str],
) -> list[tuple[str, GaussianHMM]]:
    threat_index = feature_names.index("gpr_threat")
    act_index = feature_names.index("gpr_act")
    diff_indices = [
        feature_names.index("gpr_threat_diff"),
        feature_names.index("gpr_act_diff"),
    ]
    centered = observations - observations.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    score_vectors = {
        "risk_level": observations[:, threat_index] + observations[:, act_index],
        "first_principal_component": centered @ vt[0],
        "change_intensity": np.abs(observations[:, diff_indices]).sum(axis=1),
        "time_order": np.arange(observations.shape[0], dtype=float),
    }
    models: list[tuple[str, GaussianHMM]] = []
    for score_name, scores in score_vectors.items():
        means, variances = _partition_parameters(observations, scores)
        for persistence in (0.85, 0.95):
            transition = np.array(
                [[persistence, 1.0 - persistence], [1.0 - persistence, persistence]],
                dtype=float,
            )
            models.append(
                (
                    f"{score_name}_p{persistence:.2f}",
                    GaussianHMM(
                        initial=np.array([0.5, 0.5]),
                        transition=transition,
                        means=means.copy(),
                        variances=variances.copy(),
                    ),
                )
            )
    return models


def _relabel_low_high(
    model: GaussianHMM, feature_names: Sequence[str]
) -> GaussianHMM:
    threat = feature_names.index("gpr_threat")
    act = feature_names.index("gpr_act")
    risk_score = model.means[:, threat] + model.means[:, act]
    order = np.argsort(risk_score, kind="mergesort")
    return GaussianHMM(
        initial=model.initial[order],
        transition=model.transition[np.ix_(order, order)],
        means=model.means[order],
        variances=model.variances[order],
    )


def fit_frozen_hmm(
    features: pd.DataFrame, config: Mapping[str, Any]
) -> HMMFitResult:
    """Fit only through the 2024 calendar boundary using deterministic starts."""

    feature_names = tuple(config["emission_features"])
    missing = sorted(set(("month", *feature_names)) - set(features.columns))
    if missing:
        raise ValueError(f"HMM input is missing columns: {missing}")
    months = pd.to_datetime(features["month"])
    training_end = pd.Timestamp(config["training_end"])
    heldout_start = pd.Timestamp(config["heldout_start_month"])
    heldout_end = pd.Timestamp(config["heldout_end_month"])
    training_mask = months.le(training_end)
    heldout_mask = months.between(heldout_start, heldout_end)
    raw = features.loc[:, list(feature_names)].to_numpy(dtype=float)
    training_raw = raw[training_mask]
    mean = training_raw.mean(axis=0)
    scale = training_raw.std(axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)
    standardizer = FrozenStandardizer(mean=mean, scale=scale)
    transformed = standardizer.transform(raw)
    training = transformed[training_mask]

    initial_models = _initial_models(training, feature_names)
    expected_initialisations = int(config["deterministic_initialisations"])
    if len(initial_models) != expected_initialisations:
        raise ValueError("Configured and constructed HMM initialisation counts disagree")
    summaries: list[dict[str, Any]] = []
    fitted: list[tuple[float, int, str, GaussianHMM, tuple[float, ...], bool]] = []
    tolerance = float(config["em_convergence_tolerance"])
    maximum_iterations = int(config["maximum_em_iterations"])
    variance_floor = float(config["variance_floor"])
    for index, (name, model) in enumerate(initial_models):
        history = model.fit_baum_welch(
            training,
            max_iterations=maximum_iterations,
            tolerance=tolerance,
            variance_floor=variance_floor,
        )
        converged = len(history) >= 2 and abs(history[-1] - history[-2]) <= tolerance
        final_likelihood = float(history[-1])
        summaries.append(
            {
                "initialisation": index,
                "name": name,
                "iterations": len(history),
                "converged": converged,
                "training_log_likelihood": final_likelihood,
            }
        )
        fitted.append(
            (final_likelihood, index, name, model, tuple(history), converged)
        )
    final_likelihood, selected_index, selected_name, selected_model, history, converged = max(
        fitted, key=lambda item: (item[0], -item[1])
    )
    del final_likelihood
    selected_model = _relabel_low_high(selected_model, feature_names)
    return HMMFitResult(
        model=selected_model,
        standardizer=standardizer,
        feature_names=feature_names,
        training_rows=int(training_mask.sum()),
        heldout_rows=int(heldout_mask.sum()),
        selected_initialisation=selected_index,
        selected_initialisation_name=selected_name,
        likelihood_history=history,
        converged=converged,
        initialisation_summary=tuple(summaries),
    )


def filter_feature_history(
    features: pd.DataFrame, fit: HMMFitResult, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, np.ndarray]:
    observations = fit.standardizer.transform(
        features.loc[:, list(fit.feature_names)].to_numpy(dtype=float)
    )
    beliefs = np.empty((len(features), fit.model.n_states), dtype=float)
    belief = fit.model.initial.copy()
    for index, observation in enumerate(observations):
        if index > 0:
            belief = belief @ fit.model.transition
        belief = _normalise(belief * fit.model.emission_density(observation))
        beliefs[index] = belief
    months = pd.to_datetime(features["month"])
    split = np.where(
        months.le(pd.Timestamp(config["training_end"])),
        "training",
        np.where(
            months.between(
                pd.Timestamp(config["heldout_start_month"]),
                pd.Timestamp(config["heldout_end_month"]),
            ),
            "held_out",
            "outside_declared_evaluation",
        ),
    )
    filtered = pd.DataFrame(
        {
            "observation_month": months,
            "assumed_release_date": [conservative_release_date(month) for month in months],
            "sample_split": split,
            "filtered_state_0_probability": beliefs[:, 0],
            "filtered_state_1_probability": beliefs[:, 1],
            "filtered_high_risk_probability": beliefs[:, 1],
            "state_0_label": "normal_risk",
            "state_1_label": "high_risk",
            "interpretation": "current geopolitical risk state belief; not closure probability",
        }
    )
    return filtered, observations


def _stationary_distribution(transition: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(transition.T)
    vector = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    if vector.sum() < 0:
        vector = -vector
    return _normalise(np.maximum(vector, 0.0))


def _mixture_log_density(
    model: GaussianHMM, observation: np.ndarray, state_weights: np.ndarray
) -> float:
    weights = np.maximum(_normalise(state_weights), np.finfo(float).tiny)
    return float(logsumexp(np.log(weights) + model.emission_log_density(observation)))


def heldout_density_scores(
    features: pd.DataFrame,
    observations: np.ndarray,
    filtered_beliefs: np.ndarray,
    fit: HMMFitResult,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = pd.to_datetime(features["month"]).reset_index(drop=True)
    heldout_start = pd.Timestamp(config["heldout_start_month"])
    heldout_end = pd.Timestamp(config["heldout_end_month"])
    stationary = _stationary_distribution(fit.model.transition)
    rows: list[dict[str, Any]] = []
    for horizon in config["heldout_forecast_horizons_months"]:
        horizon = int(horizon)
        for target_index, target_month in enumerate(months):
            if not heldout_start <= target_month <= heldout_end:
                continue
            origin_index = target_index - horizon
            if origin_index < 0:
                continue
            origin_belief = filtered_beliefs[origin_index]
            weights_by_model = {
                "hmm_transition": origin_belief
                @ np.linalg.matrix_power(fit.model.transition, horizon),
                "unconditional": stationary,
                "persistence": origin_belief,
            }
            for model_name, weights in weights_by_model.items():
                rows.append(
                    {
                        "forecast_model": model_name,
                        "horizon_months": horizon,
                        "origin_month": months.iloc[origin_index],
                        "target_month": target_month,
                        "log_predictive_density": _mixture_log_density(
                            fit.model, observations[target_index], weights
                        ),
                        "origin_high_risk_probability": origin_belief[1],
                        "forecast_high_risk_probability": weights[1],
                        "target_in_heldout": True,
                        "hmm_parameters_refit": False,
                    }
                )
    scores = pd.DataFrame(rows)
    summary = (
        scores.groupby(["forecast_model", "horizon_months"], as_index=False)
        .agg(
            mean_log_predictive_density=("log_predictive_density", "mean"),
            standard_deviation=("log_predictive_density", "std"),
            observations=("log_predictive_density", "size"),
            evaluation_start=("target_month", "min"),
            evaluation_end=("target_month", "max"),
        )
    )
    summary["standard_error"] = summary["standard_deviation"] / np.sqrt(
        summary["observations"]
    )
    hmm = summary.loc[
        summary["forecast_model"].eq("hmm_transition"),
        ["horizon_months", "mean_log_predictive_density"],
    ].rename(columns={"mean_log_predictive_density": "hmm_mean_lpd"})
    summary = summary.merge(hmm, on="horizon_months", how="left", validate="many_to_one")
    summary["difference_from_hmm"] = (
        summary["mean_log_predictive_density"] - summary["hmm_mean_lpd"]
    )
    return scores, summary


def conservative_release_date(month: pd.Timestamp) -> pd.Timestamp:
    """End of the following calendar month: a conservative full-month lag."""

    return pd.Timestamp(month).normalize() + pd.offsets.MonthEnd(2)


def calendar_month_transitions(
    source_month: pd.Timestamp, maturity_date: pd.Timestamp
) -> int:
    source = pd.Timestamp(source_month).to_period("M")
    maturity = pd.Timestamp(maturity_date).to_period("M")
    transitions = maturity.ordinal - source.ordinal
    if transitions < 0:
        raise ValueError("Readiness maturity cannot precede the released source month")
    return int(transitions)


def build_release_clock(
    decision_weeks: Sequence[pd.Timestamp],
    filtered: pd.DataFrame,
    fit: HMMFitResult,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Carry monthly beliefs across weeks and power P by calendar transitions only."""

    monthly = filtered.sort_values("observation_month").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    lead_weeks = int(config["readiness_lead_weeks"])
    for decision in pd.to_datetime(pd.Series(decision_weeks)).sort_values():
        cutoff = pd.Timestamp(decision).normalize()
        eligible = monthly.loc[monthly["assumed_release_date"].le(cutoff)]
        if eligible.empty:
            raise ValueError(f"No released GPR observation is available by {cutoff.date()}")
        source = eligible.iloc[-1]
        source_month = pd.Timestamp(source["observation_month"])
        release_date = pd.Timestamp(source["assumed_release_date"])
        belief = np.array(
            [
                source["filtered_state_0_probability"],
                source["filtered_state_1_probability"],
            ],
            dtype=float,
        )
        maturity = cutoff + pd.Timedelta(weeks=lead_weeks)
        transitions = calendar_month_transitions(source_month, maturity)
        forecast = fit.model.forecast(belief, transitions)
        rows.append(
            {
                "decision_week": cutoff,
                "source_observation_month": source_month,
                "release_date": release_date,
                "release_date_status": "assumed_conservative_full_month_lag",
                "decision_cutoff": cutoff,
                "filtered_state_0_probability": belief[0],
                "filtered_state_1_probability": belief[1],
                "filtered_high_risk_probability": belief[1],
                "readiness_maturity_date": maturity,
                "monthly_transitions_to_maturity": transitions,
                "lead_time_state_0_probability": forecast[0],
                "lead_time_state_1_probability": forecast[1],
                "lead_time_high_risk_probability": forecast[1],
                "timing_valid": release_date <= cutoff,
                "transition_basis": "calendar_month_boundaries_from_source_to_maturity",
                "weekly_transition_matrix_applications": 0,
                "risk_information_source": "released_hmm_filter",
                "risk_interpretation": "geopolitical risk state; not closure probability",
            }
        )
    return pd.DataFrame(rows)


def hmm_parameter_manifest(
    fit: HMMFitResult, config: Mapping[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        category: str,
        parameter: str,
        value: Any,
        *,
        state: str = "",
        feature: str = "",
        evidence: str = "frozen_experiment_design",
    ) -> None:
        rows.append(
            {
                "category": category,
                "parameter": parameter,
                "state": state,
                "feature": feature,
                "value": json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list, tuple))
                else value,
                "evidence_or_status": evidence,
            }
        )

    for parameter in (
        "number_of_states",
        "training_end",
        "heldout_start_month",
        "heldout_end_month",
        "emission_features",
        "feature_scaling_rule",
        "em_convergence_tolerance",
        "maximum_em_iterations",
        "variance_floor",
        "deterministic_initialisations",
        "initialisation_construction",
        "state_labelling_rule",
        "release_lag_rule",
        "release_lag_months",
        "heldout_forecast_horizons_months",
        "readiness_lead_weeks",
    ):
        add("design", parameter, config[parameter])
    add("fit", "training_rows", fit.training_rows, evidence="computed")
    add("fit", "heldout_rows", fit.heldout_rows, evidence="computed")
    add(
        "fit",
        "selected_initialisation",
        fit.selected_initialisation,
        evidence=fit.selected_initialisation_name,
    )
    add("fit", "em_iterations", len(fit.likelihood_history), evidence="computed")
    add("fit", "em_converged", fit.converged, evidence="computed")
    add(
        "fit",
        "final_training_log_likelihood",
        fit.likelihood_history[-1],
        evidence="computed",
    )
    for summary in fit.initialisation_summary:
        add(
            "initialisation_audit",
            f"initialisation_{summary['initialisation']}",
            summary,
            evidence="computed_all_deterministic_starts",
        )
    for state in range(fit.model.n_states):
        label = "normal_risk" if state == 0 else "high_risk"
        add("initial_probability", "initial", fit.model.initial[state], state=label)
        for target in range(fit.model.n_states):
            target_label = "normal_risk" if target == 0 else "high_risk"
            add(
                "transition_matrix",
                f"P[{state},{target}]",
                fit.model.transition[state, target],
                state=f"{label}_to_{target_label}",
                evidence="estimated_training_only",
            )
        for feature_index, feature in enumerate(fit.feature_names):
            add(
                "emission_mean",
                "mean",
                fit.model.means[state, feature_index],
                state=label,
                feature=feature,
                evidence="standardised_training_scale",
            )
            add(
                "emission_variance",
                "variance",
                fit.model.variances[state, feature_index],
                state=label,
                feature=feature,
                evidence="standardised_training_scale",
            )
    for feature_index, feature in enumerate(fit.feature_names):
        add(
            "standardizer",
            "training_mean",
            fit.standardizer.mean[feature_index],
            feature=feature,
            evidence="training_only_frozen",
        )
        add(
            "standardizer",
            "training_scale",
            fit.standardizer.scale[feature_index],
            feature=feature,
            evidence="population_standard_deviation_training_only",
        )
    return pd.DataFrame(rows)
