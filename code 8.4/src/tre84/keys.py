"""Stable identifiers for the generic network and tagged physical state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

from .errors import ContractError


class Stage(str, Enum):
    BERTH = "berth"
    YARD = "yard"
    GATE = "gate"
    CORRIDOR = "corridor"


class Provenance(str, Enum):
    COMMITTED = "committed"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True, order=True)
class ResourceKey:
    stage: Stage
    location: str


@dataclass(frozen=True, order=True)
class Tag:
    cargo_class: str
    route: str


@dataclass(frozen=True)
class SourceKey:
    """A new cohort has vintage=None; an old source retains its integer age."""

    cargo_class: str
    vintage: int | None

    @property
    def is_new(self) -> bool:
        return self.vintage is None


@dataclass(frozen=True)
class Route:
    route_id: str
    cargo_class: str
    gateway: str
    corridor: str
    maritime_lag_kernel: tuple[float, ...]

    def __post_init__(self) -> None:
        kernel = np.asarray(self.maritime_lag_kernel, dtype=float)
        if kernel.ndim != 1 or kernel.size == 0:
            raise ContractError("A route must have a nonempty one-dimensional lag kernel")
        if np.any(~np.isfinite(kernel)) or np.any(kernel < 0):
            raise ContractError("Maritime lag weights must be finite and nonnegative")
        if not np.isclose(float(kernel.sum()), 1.0, atol=1e-12):
            raise ContractError("Each dispatch cohort's maritime lag kernel must sum to one")


@dataclass(frozen=True)
class Network:
    """Master route tags remain present even when a route closes to new cargo."""

    routes: Mapping[str, Route]

    def __post_init__(self) -> None:
        if not self.routes:
            raise ContractError("The master route set cannot be empty")
        for route_id, route in self.routes.items():
            if route_id != route.route_id:
                raise ContractError("Route mapping keys must equal Route.route_id")

    def route(self, route_id: str) -> Route:
        try:
            return self.routes[route_id]
        except KeyError as exc:
            raise ContractError(f"Unknown master route tag: {route_id}") from exc

    def routes_for_class(self, cargo_class: str) -> tuple[str, ...]:
        return tuple(
            sorted(route_id for route_id, route in self.routes.items() if route.cargo_class == cargo_class)
        )

    def gateways(self) -> tuple[str, ...]:
        return tuple(sorted({route.gateway for route in self.routes.values()}))

    def corridors(self) -> tuple[str, ...]:
        return tuple(sorted({route.corridor for route in self.routes.values()}))

    def shared_corridors(self) -> dict[str, tuple[str, ...]]:
        result: dict[str, tuple[str, ...]] = {}
        for corridor in self.corridors():
            gateways = tuple(
                sorted({route.gateway for route in self.routes.values() if route.corridor == corridor})
            )
            if len(gateways) >= 2:
                result[corridor] = gateways
        return result
