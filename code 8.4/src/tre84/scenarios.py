"""Module 2: one event-aligned support with readiness and operational weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .errors import ContractError, NumericalFailure


def _normalized_exp(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ContractError("Scenario log weights must be a finite nonempty vector")
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        raise NumericalFailure("Scenario weight normalization failed")
    return weights / total


@dataclass(frozen=True)
class TimestampedOperationalContext:
    """Operational evidence released no later than one decision time.

    The value is intentionally opaque to the core, but it cannot enter Module
    2 without an explicit timestamp ledger.  This closes the previous ``Any``
    channel through which a caller could accidentally pass future observations.
    """

    value: Any
    information_timestamps: tuple[Any, ...]

    def value_available_at(self, decision_time: Any) -> Any:
        if any(timestamp > decision_time for timestamp in self.information_timestamps):
            raise ContractError("Operational scenario weights used unreleased information")
        return self.value


@dataclass(frozen=True)
class RevealedEventHistory:
    """Read-only prefix available to an MPC continuation decision."""

    path_id: str
    decision_offset: int
    onset: tuple[bool, ...]
    serviceability: Mapping[str, np.ndarray]
    active: tuple[bool, ...]
    demand_residual: Mapping[str, np.ndarray]
    payload: tuple[Any, ...]

    def __post_init__(self) -> None:
        if self.decision_offset < 0:
            raise ContractError("Revealed scenario offset cannot be negative")
        if len(self.payload) != self.decision_offset:
            raise ContractError("Revealed payload must contain past realizations only")


@dataclass(frozen=True)
class EventPath:
    path_id: str
    onset: tuple[bool, ...]
    serviceability: Mapping[str, np.ndarray]
    active: tuple[bool, ...]
    demand_residual: Mapping[str, np.ndarray] = field(default_factory=dict)
    payload: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        horizon = len(self.active)
        if len(self.onset) != horizon:
            raise ContractError("Event onset and active status must share one horizon")
        for cargo_class, values in self.serviceability.items():
            arr = np.asarray(values, dtype=float)
            if arr.shape != (horizon,) or np.any((arr < 0) | (arr > 1)):
                raise ContractError(f"Invalid serviceability path for {cargo_class}")
        for cargo_class, values in self.demand_residual.items():
            arr = np.asarray(values, dtype=float)
            if arr.shape != (horizon,) or np.any(~np.isfinite(arr)):
                raise ContractError(f"Invalid demand-residual path for {cargo_class}")
        if self.payload and len(self.payload) != horizon:
            raise ContractError("Scenario payload must be empty or cover the full horizon")

    @property
    def horizon(self) -> int:
        return len(self.active)

    def has_reclosure(self, start: int = 0) -> bool:
        active = self.active[start:]
        has_opened = False
        for flag in active:
            if not flag:
                has_opened = True
            elif has_opened:
                return True
        return False

    def revealed_before(self, decision_offset: int) -> RevealedEventHistory:
        """Return only outcomes realized before a simulated decision.

        At offset zero the history is empty.  At offset ``ell`` it contains
        exactly payload entries ``0, ..., ell-1``; the current and all later
        scenario realizations remain inaccessible to the continuation policy.
        """

        if decision_offset < 0 or decision_offset > self.horizon:
            raise ContractError("MPC reveal offset lies outside the scenario horizon")
        return RevealedEventHistory(
            path_id=self.path_id,
            decision_offset=decision_offset,
            onset=self.onset[:decision_offset],
            serviceability={
                cargo_class: np.asarray(values, dtype=float)[:decision_offset].copy()
                for cargo_class, values in self.serviceability.items()
            },
            active=self.active[:decision_offset],
            demand_residual={
                cargo_class: np.asarray(values, dtype=float)[:decision_offset].copy()
                for cargo_class, values in self.demand_residual.items()
            },
            payload=self.payload[:decision_offset],
        )


@dataclass(frozen=True)
class ScenarioBundle:
    paths: tuple[EventPath, ...]
    readiness_weights: np.ndarray
    operational_weights: np.ndarray
    active_weights: np.ndarray
    reclosure_probability: float
    weighted_serviceability: Mapping[str, np.ndarray]
    information_timestamps: tuple[Any, ...]
    seed_manifest: Mapping[str, int]
    decision_time: Any | None = None

    def __post_init__(self) -> None:
        size = len(self.paths)
        for name, vector in (
            ("readiness", self.readiness_weights),
            ("operational", self.operational_weights),
            ("active", self.active_weights),
        ):
            arr = np.asarray(vector, dtype=float)
            if arr.shape != (size,) or np.any(arr < 0) or not np.isclose(arr.sum(), 1.0):
                raise ContractError(f"{name} scenario weights are invalid")
        if self.decision_time is not None and any(
            timestamp > self.decision_time for timestamp in self.information_timestamps
        ):
            raise ContractError("Scenario bundle contains post-decision information")


class CommonScenarioConstructor:
    """Reweight a frozen support without changing its paths across policies."""

    def __init__(
        self,
        paths: Sequence[EventPath],
        *,
        readiness_log_weight: Callable[[EventPath, np.ndarray], float],
        operational_log_weight: Callable[[EventPath, Any], float],
        seed_manifest: Mapping[str, int] | None = None,
    ) -> None:
        self.paths = tuple(paths)
        if not self.paths or len({path.path_id for path in self.paths}) != len(self.paths):
            raise ContractError("The common scenario support needs unique path identifiers")
        horizons = {path.horizon for path in self.paths}
        if len(horizons) != 1:
            raise ContractError("All common scenario paths must share one horizon")
        self.readiness_log_weight = readiness_log_weight
        self.operational_log_weight = operational_log_weight
        self.seed_manifest = dict(seed_manifest or {})

    def build(
        self,
        *,
        lead_time_risk_forecast: np.ndarray,
        operational_context: TimestampedOperationalContext,
        phase: int,
        completed_information_timestamps: Sequence[Any],
        decision_time: Any,
    ) -> ScenarioBundle:
        if phase not in (0, 1):
            raise ContractError("The online phase is binary and observable")
        if not isinstance(operational_context, TimestampedOperationalContext):
            raise ContractError(
                "Operational context must carry its complete information-timestamp ledger"
            )
        operational_value = operational_context.value_available_at(decision_time)
        information_timestamps = (
            *tuple(completed_information_timestamps),
            *operational_context.information_timestamps,
        )
        if any(timestamp > decision_time for timestamp in information_timestamps):
            raise ContractError("Scenario construction used post-decision information")
        readiness = _normalized_exp(
            np.asarray(
                [self.readiness_log_weight(path, lead_time_risk_forecast) for path in self.paths],
                dtype=float,
            )
        )
        operational = _normalized_exp(
            np.asarray(
                [self.operational_log_weight(path, operational_value) for path in self.paths],
                dtype=float,
            )
        )
        active = (1 - phase) * readiness + phase * operational
        reclosure = float(
            sum(weight * path.has_reclosure() for weight, path in zip(active, self.paths))
        )
        cargo_classes = sorted(
            {cargo_class for path in self.paths for cargo_class in path.serviceability}
        )
        weighted_serviceability = {
            cargo_class: sum(
                weight * np.asarray(path.serviceability[cargo_class], dtype=float)
                for weight, path in zip(active, self.paths)
            )
            for cargo_class in cargo_classes
        }
        return ScenarioBundle(
            paths=self.paths,
            readiness_weights=readiness,
            operational_weights=operational,
            active_weights=active,
            reclosure_probability=reclosure,
            weighted_serviceability=weighted_serviceability,
            information_timestamps=information_timestamps,
            seed_manifest=self.seed_manifest,
            decision_time=decision_time,
        )


def replace_readiness_weights_only(
    bundle: ScenarioBundle,
    *,
    new_readiness_weights: np.ndarray,
    phase: int,
) -> ScenarioBundle:
    """HMM ablation: preserve support, operational weights, paths, and seeds."""

    readiness = np.asarray(new_readiness_weights, dtype=float)
    if readiness.shape != bundle.readiness_weights.shape or np.any(readiness < 0):
        raise ContractError("Ablated readiness weights do not match the common support")
    if not np.isclose(readiness.sum(), 1.0):
        raise ContractError("Ablated readiness weights must sum to one")
    if phase not in (0, 1):
        raise ContractError("The online phase is binary")
    active = (1 - phase) * readiness + phase * bundle.operational_weights
    reclosure = float(
        sum(weight * path.has_reclosure() for weight, path in zip(active, bundle.paths))
    )
    weighted_serviceability = {
        cargo_class: sum(
            weight * np.asarray(path.serviceability[cargo_class], dtype=float)
            for weight, path in zip(active, bundle.paths)
        )
        for cargo_class in bundle.weighted_serviceability
    }
    return ScenarioBundle(
        paths=bundle.paths,
        readiness_weights=readiness,
        operational_weights=bundle.operational_weights.copy(),
        active_weights=active,
        reclosure_probability=reclosure,
        weighted_serviceability=weighted_serviceability,
        information_timestamps=bundle.information_timestamps,
        seed_manifest=bundle.seed_manifest,
        decision_time=bundle.decision_time,
    )
