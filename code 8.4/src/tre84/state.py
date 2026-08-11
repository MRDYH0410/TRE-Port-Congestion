"""Complete Markov state from Equation (control-state)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .errors import ContractError
from .keys import Provenance, ResourceKey, SourceKey, Tag


@dataclass(frozen=True)
class PipelineLot:
    cargo_class: str
    route: str
    remaining_lag: int
    provenance: Provenance
    mass: float

    def __post_init__(self) -> None:
        if self.remaining_lag < 0:
            raise ContractError("Pipeline remaining lag cannot be negative")
        if not np.isfinite(self.mass) or self.mass < 0:
            raise ContractError("Pipeline mass must be finite and nonnegative")


@dataclass
class CapacityState:
    """Orders are indexed by resource and remaining periods before maturity."""

    orders: dict[ResourceKey, dict[int, float]] = field(default_factory=dict)
    stock: dict[ResourceKey, float] = field(default_factory=dict)

    def total_orders(self) -> float:
        return float(sum(sum(bucket.values()) for bucket in self.orders.values()))


@dataclass
class RiskInformation:
    belief: np.ndarray
    lead_time_forecast: np.ndarray
    scenario_ids: tuple[str, ...] = ()
    readiness_weights: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    operational_weights: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    reclosure_probability: float = 0.0
    latest_release_period: int | None = None
    latest_release_time: Any | None = None


@dataclass
class ModelState:
    period: int
    horizon: int
    risk: RiskInformation
    disruption_seen: bool
    disruption_active: bool
    disruption_duration: int
    waiting: dict[str, np.ndarray]
    berth: dict[Tag, float]
    yard: dict[Tag, float]
    gate: dict[Tag, float]
    corridor: dict[Tag, float]
    maritime_pipeline: list[PipelineLot]
    previous_shares: dict[SourceKey, dict[str, float]]
    corridor_history: dict[tuple[str, str], tuple[float, ...]]
    serviceability_history: tuple[float, ...]
    readiness: CapacityState
    direct_capacity: CapacityState
    budget: float
    observed_covariates: dict[str, Any] = field(default_factory=dict)

    @property
    def phase(self) -> int:
        return int(self.disruption_seen)

    def clone(self) -> "ModelState":
        return deepcopy(self)

    def tagged_mass(self) -> float:
        return float(
            sum(self.berth.values())
            + sum(self.yard.values())
            + sum(self.gate.values())
            + sum(self.corridor.values())
        )

    def pipeline_mass(self) -> float:
        return float(sum(lot.mass for lot in self.maritime_pipeline))

    def waiting_mass(self) -> float:
        return float(sum(np.asarray(v, dtype=float).sum() for v in self.waiting.values()))

    def cargo_mass(self) -> float:
        return self.waiting_mass() + self.pipeline_mass() + self.tagged_mass()

    def validate(self, *, tolerance: float = 1e-10) -> None:
        if self.period < 0 or self.horizon <= 0:
            raise ContractError("State period and horizon are invalid")
        if self.disruption_duration < 0:
            raise ContractError("Disruption duration cannot be negative")
        if self.disruption_active and not self.disruption_seen:
            raise ContractError("An active disruption must already be in the released event state")
        if self.budget < -tolerance or not np.isfinite(self.budget):
            raise ContractError("Remaining budget must be finite and nonnegative")
        for name, vector in (
            ("risk belief", self.risk.belief),
            ("lead-time risk forecast", self.risk.lead_time_forecast),
        ):
            arr = np.asarray(vector, dtype=float)
            if (
                arr.ndim != 1
                or arr.size == 0
                or np.any(~np.isfinite(arr))
                or np.min(arr) < -tolerance
                or not np.isclose(arr.sum(), 1.0, atol=tolerance)
            ):
                raise ContractError(f"{name} must be a finite probability vector")
        if not 0 <= self.risk.reclosure_probability <= 1:
            raise ContractError("Reclosure probability must lie in [0, 1]")
        scenario_count = len(self.risk.scenario_ids)
        for name, weights in (
            ("readiness", self.risk.readiness_weights),
            ("operational", self.risk.operational_weights),
        ):
            arr = np.asarray(weights, dtype=float)
            if scenario_count == 0 and arr.size == 0:
                continue
            if (
                arr.shape != (scenario_count,)
                or np.any(~np.isfinite(arr))
                or np.min(arr) < -tolerance
                or not np.isclose(arr.sum(), 1.0, atol=tolerance)
            ):
                raise ContractError(f"{name} scenario weights are invalid")
        for cargo_class, vintages in self.waiting.items():
            arr = np.asarray(vintages, dtype=float)
            if arr.ndim != 1 or arr.size == 0:
                raise ContractError(f"Waiting vintages for {cargo_class} are malformed")
            if np.any(~np.isfinite(arr)) or np.min(arr) < -tolerance:
                raise ContractError(f"Waiting vintages for {cargo_class} are invalid")
        for name, mapping in self.queue_mappings().items():
            values = np.asarray(tuple(mapping.values()), dtype=float)
            if values.size and (np.any(~np.isfinite(values)) or np.min(values) < -tolerance):
                raise ContractError(f"{name} tagged queue contains invalid mass")
        for lot in self.maritime_pipeline:
            if lot.remaining_lag < 0 or lot.mass < -tolerance or not np.isfinite(lot.mass):
                raise ContractError("Maritime pipeline contains an invalid tagged lot")
        for source, shares in self.previous_shares.items():
            arr = np.asarray(tuple(shares.values()), dtype=float)
            if (
                arr.size == 0
                or np.any(~np.isfinite(arr))
                or np.min(arr) < -tolerance
                or not np.isclose(arr.sum(), 1.0, atol=tolerance)
            ):
                raise ContractError(f"Previous master shares for {source} are invalid")
        for key, history in self.corridor_history.items():
            arr = np.asarray(history, dtype=float)
            if arr.ndim != 1 or np.any(~np.isfinite(arr)) or (
                arr.size and np.min(arr) < -tolerance
            ):
                raise ContractError(f"Corridor history for {key} is invalid")
        serviceability = np.asarray(self.serviceability_history, dtype=float)
        if np.any(~np.isfinite(serviceability)) or np.any(
            (serviceability < -tolerance) | (serviceability > 1.0 + tolerance)
        ):
            raise ContractError("Serviceability history must remain in [0, 1]")
        for capacity_name, capacity in (
            ("readiness", self.readiness),
            ("direct capacity", self.direct_capacity),
        ):
            vals = list(capacity.stock.values()) + [
                value for bucket in capacity.orders.values() for value in bucket.values()
            ]
            arr = np.asarray(vals, dtype=float)
            if arr.size and (np.any(~np.isfinite(arr)) or np.min(arr) < -tolerance):
                raise ContractError(f"{capacity_name} pipeline contains invalid values")
            if any(
                not isinstance(remaining, int) or remaining <= 0
                for bucket in capacity.orders.values()
                for remaining in bucket
            ):
                raise ContractError(
                    f"{capacity_name} pending-order clocks must be positive integers"
                )

    def queue_mappings(self) -> Mapping[str, Mapping[Tag, float]]:
        return {
            "berth": self.berth,
            "yard": self.yard,
            "gate": self.gate,
            "corridor": self.corridor,
        }
