"""Shared Section 5.1 precision and multiplicity rules for path-level inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import t as student_t

from .errors import ContractError


@dataclass(frozen=True)
class PrecisionResult:
    pilot_standard_deviation: float
    target_halfwidth: float
    confidence_level: float
    required_paths: int
    halfwidth_at_required: float
    achieved_within_cap: bool


@dataclass(frozen=True)
class PrecisionRule:
    """Choose physical-path count from pilot paired differences only."""

    target_halfwidth: float
    confidence_level: float
    minimum_paths: int
    maximum_paths: int

    def __post_init__(self) -> None:
        if self.target_halfwidth <= 0 or not np.isfinite(self.target_halfwidth):
            raise ContractError("Precision target halfwidth must be finite and positive")
        if not 0 < self.confidence_level < 1:
            raise ContractError("Precision confidence level must lie in (0, 1)")
        if self.minimum_paths < 2 or self.maximum_paths < self.minimum_paths:
            raise ContractError("Precision path bounds are invalid")

    def required_paths(self, pilot_standard_deviation: float) -> PrecisionResult:
        sd = float(pilot_standard_deviation)
        if sd < 0 or not np.isfinite(sd):
            raise ContractError("Pilot standard deviation must be finite and nonnegative")
        alpha = 1.0 - self.confidence_level
        selected = self.maximum_paths
        halfwidth = 0.0 if sd == 0 else float("inf")
        achieved = sd == 0
        for paths in range(self.minimum_paths, self.maximum_paths + 1):
            critical = float(student_t.ppf(1.0 - alpha / 2.0, df=paths - 1))
            candidate = critical * sd / np.sqrt(paths)
            if candidate <= self.target_halfwidth:
                selected = paths
                halfwidth = float(candidate)
                achieved = True
                break
        if not achieved:
            critical = float(
                student_t.ppf(1.0 - alpha / 2.0, df=self.maximum_paths - 1)
            )
            halfwidth = float(critical * sd / np.sqrt(self.maximum_paths))
        return PrecisionResult(
            pilot_standard_deviation=sd,
            target_halfwidth=self.target_halfwidth,
            confidence_level=self.confidence_level,
            required_paths=selected,
            halfwidth_at_required=halfwidth,
            achieved_within_cap=achieved,
        )


@dataclass(frozen=True)
class StudentInterval:
    count: int
    mean: float
    standard_error: float
    lower: float
    upper: float


def student_interval(
    values: Iterable[float],
    *,
    confidence_level: float,
    family_size: int = 1,
) -> StudentInterval:
    """Two-sided Student interval with optional Bonferroni family coverage."""

    sample = np.asarray(tuple(values), dtype=float)
    if sample.ndim != 1 or sample.size < 2 or np.any(~np.isfinite(sample)):
        raise ContractError("Student intervals require at least two finite path values")
    if not 0 < confidence_level < 1 or family_size < 1:
        raise ContractError("Student interval confidence or family size is invalid")
    mean = float(sample.mean())
    se = float(sample.std(ddof=1) / np.sqrt(sample.size))
    alpha = (1.0 - confidence_level) / family_size
    critical = float(student_t.ppf(1.0 - alpha / 2.0, df=sample.size - 1))
    margin = critical * se
    return StudentInterval(sample.size, mean, se, mean - margin, mean + margin)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm family-wise adjusted p-values in the original hypothesis order."""

    values = np.asarray(tuple(p_values), dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any((values < 0) | (values > 1)):
        raise ContractError("Holm adjustment requires finite p-values in [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.empty(values.size, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (values.size - rank) * values[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(values.size, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted

