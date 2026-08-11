"""Module 1: released Gaussian HMM filtering and calendar-aligned forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .errors import ContractError, NumericalFailure


def _normalize(vector: np.ndarray) -> np.ndarray:
    total = float(vector.sum())
    if not np.isfinite(total) or total <= 0:
        raise NumericalFailure("Probability normalization failed")
    return vector / total


@dataclass(frozen=True)
class FrozenStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.shape != self.scale.shape or np.any(self.scale <= 0):
            raise ContractError("Frozen normalization arrays are incompatible")

    def transform(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=float)
        return (values - self.mean) / self.scale


@dataclass
class GaussianHMM:
    """Finite-state Gaussian HMM with diagonal emissions and scaled recursions."""

    initial: np.ndarray
    transition: np.ndarray
    means: np.ndarray
    variances: np.ndarray

    def __post_init__(self) -> None:
        self.initial = np.asarray(self.initial, dtype=float)
        self.transition = np.asarray(self.transition, dtype=float)
        self.means = np.asarray(self.means, dtype=float)
        self.variances = np.asarray(self.variances, dtype=float)
        states = self.initial.size
        if self.transition.shape != (states, states):
            raise ContractError("HMM transition matrix has the wrong shape")
        if self.means.shape != self.variances.shape or self.means.shape[0] != states:
            raise ContractError("HMM emission arrays have the wrong shape")
        if np.any(self.initial < 0) or not np.isclose(self.initial.sum(), 1.0):
            raise ContractError("HMM initial probabilities must sum to one")
        if np.any(self.transition < 0) or not np.allclose(self.transition.sum(axis=1), 1.0):
            raise ContractError("Every HMM transition row must sum to one")
        if np.any(self.variances <= 0):
            raise ContractError("HMM emission variances must be positive")

    @property
    def n_states(self) -> int:
        return int(self.initial.size)

    def emission_density(self, observation: np.ndarray) -> np.ndarray:
        log_density = self.emission_log_density(observation)
        offset = float(np.max(log_density))
        return np.exp(log_density - offset)

    def emission_log_density(self, observation: np.ndarray) -> np.ndarray:
        x = np.asarray(observation, dtype=float)
        if x.shape != self.means.shape[1:]:
            raise ContractError("HMM observation dimension does not match the emissions")
        log_density = -0.5 * (
            np.log(2.0 * np.pi * self.variances)
            + np.square(x - self.means) / self.variances
        ).sum(axis=1)
        return log_density

    def released_filter(
        self,
        observations: Sequence[np.ndarray],
        periods: Sequence[int] | None = None,
    ) -> np.ndarray:
        if not observations:
            return self.initial.copy()
        if periods is None:
            periods = list(range(len(observations)))
        if len(periods) != len(observations):
            raise ContractError("Observation periods and observations must have equal length")
        if any(later <= earlier for earlier, later in zip(periods, periods[1:])):
            raise ContractError("Released observation periods must be strictly increasing")
        belief = self.initial.copy()
        previous_period: int | None = None
        for period, observation in zip(periods, observations):
            if previous_period is not None:
                belief = belief @ np.linalg.matrix_power(
                    self.transition, period - previous_period
                )
            belief = _normalize(belief * self.emission_density(observation))
            previous_period = period
        return belief

    def forecast(self, belief: np.ndarray, monthly_transitions: int) -> np.ndarray:
        if monthly_transitions < 0:
            raise ContractError("The lead-time forecast cannot move backward")
        posterior = _normalize(np.asarray(belief, dtype=float))
        return posterior @ np.linalg.matrix_power(self.transition, monthly_transitions)

    def sequence_log_likelihood(self, observations: np.ndarray) -> float:
        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.means.shape[1] or x.shape[0] == 0:
            raise ContractError("HMM likelihood needs a nonempty two-dimensional array")
        belief = self.initial.copy()
        log_likelihood = 0.0
        for index, observation in enumerate(x):
            if index > 0:
                belief = belief @ self.transition
            log_density = self.emission_log_density(observation)
            offset = float(np.max(log_density))
            weighted = belief * np.exp(log_density - offset)
            normalizer = float(weighted.sum())
            if normalizer <= 0:
                raise NumericalFailure("HMM predictive density vanished")
            log_likelihood += np.log(normalizer) + offset
            belief = weighted / normalizer
        return float(log_likelihood)

    def fit_baum_welch(
        self,
        observations: np.ndarray,
        *,
        max_iterations: int,
        tolerance: float,
        variance_floor: float,
    ) -> list[float]:
        """Fit in place on training data only and return the likelihood history."""

        x = np.asarray(observations, dtype=float)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] != self.means.shape[1]:
            raise ContractError("Baum-Welch needs a two-dimensional training array")
        if max_iterations <= 0 or tolerance <= 0 or variance_floor <= 0:
            raise ContractError("Baum-Welch numerical settings must be positive")
        history: list[float] = []
        for _ in range(max_iterations):
            emission_logs = np.vstack([self.emission_log_density(row) for row in x])
            emission_offsets = emission_logs.max(axis=1)
            emissions = np.exp(emission_logs - emission_offsets[:, None])
            alpha = np.zeros((x.shape[0], self.n_states), dtype=float)
            scales = np.zeros(x.shape[0], dtype=float)
            alpha[0] = self.initial * emissions[0]
            scales[0] = alpha[0].sum()
            alpha[0] = _normalize(alpha[0])
            for t in range(1, x.shape[0]):
                alpha[t] = (alpha[t - 1] @ self.transition) * emissions[t]
                scales[t] = alpha[t].sum()
                alpha[t] = _normalize(alpha[t])
            log_likelihood = float(
                (np.log(np.maximum(scales, np.finfo(float).tiny)) + emission_offsets).sum()
            )
            history.append(log_likelihood)

            beta = np.ones_like(alpha)
            for t in range(x.shape[0] - 2, -1, -1):
                beta[t] = self.transition @ (emissions[t + 1] * beta[t + 1])
                beta[t] /= max(scales[t + 1], np.finfo(float).tiny)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)

            xi_sum = np.zeros_like(self.transition)
            for t in range(x.shape[0] - 1):
                xi = (
                    alpha[t, :, None]
                    * self.transition
                    * (emissions[t + 1] * beta[t + 1])[None, :]
                )
                xi_sum += xi / max(float(xi.sum()), np.finfo(float).tiny)

            self.initial = _normalize(gamma[0])
            self.transition = xi_sum / np.maximum(
                xi_sum.sum(axis=1, keepdims=True), np.finfo(float).tiny
            )
            weights = gamma.sum(axis=0)
            self.means = (gamma.T @ x) / weights[:, None]
            diff = x[:, None, :] - self.means[None, :, :]
            self.variances = (
                (gamma[:, :, None] * np.square(diff)).sum(axis=0) / weights[:, None]
            )
            self.variances = np.maximum(self.variances, variance_floor)
            if len(history) >= 2 and abs(history[-1] - history[-2]) <= tolerance:
                break
        return history


@dataclass(frozen=True)
class ReleaseRecord:
    observation_period: int
    release_time: Any
    observation: np.ndarray


@dataclass(frozen=True)
class ReleasedRiskResult:
    belief: np.ndarray
    lead_time_forecast: np.ndarray
    latest_observation_period: int | None
    latest_release_time: Any | None
    information_timestamps: tuple[Any, ...]


class ReleasedRiskInference:
    """Apply the official release clock before filtering or forecasting."""

    def __init__(self, hmm: GaussianHMM) -> None:
        self.hmm = hmm

    def infer(
        self,
        *,
        decision_time: Any,
        readiness_maturity_time: Any,
        records: Iterable[ReleaseRecord],
        monthly_transition_count: Callable[[int | None, Any], int],
    ) -> ReleasedRiskResult:
        eligible = sorted(
            (record for record in records if record.release_time <= decision_time),
            key=lambda record: record.observation_period,
        )
        periods = [record.observation_period for record in eligible]
        observations = [np.asarray(record.observation, dtype=float) for record in eligible]
        belief = self.hmm.released_filter(observations, periods)
        latest_period = periods[-1] if periods else None
        h = monthly_transition_count(latest_period, readiness_maturity_time)
        forecast = self.hmm.forecast(belief, h)
        return ReleasedRiskResult(
            belief=belief,
            lead_time_forecast=forecast,
            latest_observation_period=latest_period,
            latest_release_time=eligible[-1].release_time if eligible else None,
            information_timestamps=tuple(record.release_time for record in eligible),
        )


@dataclass(frozen=True)
class HMMEvaluation:
    log_likelihood: float
    aic: float
    bic: float
    transition_persistence: np.ndarray
    implied_regime_duration: np.ndarray


def evaluate_hmm(model: GaussianHMM, observations: np.ndarray) -> HMMEvaluation:
    values = np.asarray(observations, dtype=float)
    log_likelihood = model.sequence_log_likelihood(values)
    states = model.n_states
    dimensions = model.means.shape[1]
    parameter_count = (
        (states - 1)
        + states * (states - 1)
        + states * dimensions
        + states * dimensions
    )
    persistence = np.diag(model.transition).copy()
    duration = 1.0 / np.maximum(1.0 - persistence, np.finfo(float).eps)
    return HMMEvaluation(
        log_likelihood=log_likelihood,
        aic=2 * parameter_count - 2 * log_likelihood,
        bic=np.log(values.shape[0]) * parameter_count - 2 * log_likelihood,
        transition_persistence=persistence,
        implied_regime_duration=duration,
    )


def rolling_origin_predictive_density(
    observations: np.ndarray,
    *,
    first_test_index: int,
    fit_model: Callable[[np.ndarray], GaussianHMM],
) -> np.ndarray:
    """Refit only on each completed training prefix, then score the next release."""

    values = np.asarray(observations, dtype=float)
    if values.ndim != 2 or not 1 <= first_test_index < values.shape[0]:
        raise ContractError("Rolling-origin split index is invalid")
    scores: list[float] = []
    for test_index in range(first_test_index, values.shape[0]):
        model = fit_model(values[:test_index])
        training_belief = model.released_filter(list(values[:test_index]))
        predictive = training_belief @ model.transition
        log_density = model.emission_log_density(values[test_index])
        offset = float(np.max(log_density))
        density = float((predictive * np.exp(log_density - offset)).sum())
        scores.append(np.log(max(density, np.finfo(float).tiny)) + offset)
    return np.asarray(scores, dtype=float)
