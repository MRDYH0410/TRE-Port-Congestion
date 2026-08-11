"""Readiness/direct-capacity pipelines and effective service capacities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from .actions import Action, ActionKey
from .errors import ContractError
from .keys import Network, ResourceKey, Stage
from .state import CapacityState, ModelState


@dataclass(frozen=True)
class CapacityActionMap:
    readiness_order: Mapping[ActionKey, ResourceKey]
    direct_order: Mapping[ActionKey, ResourceKey]
    readiness_exercise: Mapping[ActionKey, ResourceKey]


@dataclass(frozen=True)
class CapacityTechnology:
    readiness_lead: Mapping[ResourceKey, int]
    readiness_maturity_yield: Mapping[ResourceKey, float]
    readiness_consumption: Mapping[ResourceKey, float]
    readiness_capacity_yield: Mapping[ResourceKey, float]
    readiness_decay: Mapping[ResourceKey, float]
    direct_lead: Mapping[tuple[int, ResourceKey], int]
    direct_maturity_yield: Mapping[ResourceKey, float]
    direct_decay: Mapping[ResourceKey, float]

    def validate(self, resources: set[ResourceKey]) -> None:
        for resource in resources:
            if resource not in self.readiness_lead or resource not in self.direct_maturity_yield:
                raise ContractError(f"Missing capacity technology for {resource}")
            if self.readiness_lead[resource] < 1:
                raise ContractError("Readiness options must mature after a positive lead time")
            for phase in (0, 1):
                if self.direct_lead[(phase, resource)] < 0:
                    raise ContractError("Direct capacity lead time cannot be negative")
            for value in (
                self.readiness_maturity_yield[resource],
                self.readiness_consumption[resource],
                self.readiness_capacity_yield[resource],
                self.direct_maturity_yield[resource],
            ):
                if value < 0 or not np.isfinite(value):
                    raise ContractError("Capacity conversion values must be finite and nonnegative")
            for decay in (self.readiness_decay[resource], self.direct_decay[resource]):
                if not 0 <= decay <= 1:
                    raise ContractError("Capacity decay rates must lie in [0, 1]")


@dataclass(frozen=True)
class ServiceParameters:
    base_capacity: Mapping[ResourceKey, float]
    thresholds: Mapping[ResourceKey, float]
    yard_feedback: Mapping[str, Callable[[float], float]]
    corridor_feedback: Mapping[str, Callable[[float], float]]
    fallback_corridor_share: Mapping[tuple[str, str], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(value < 0 or not np.isfinite(value) for value in self.base_capacity.values()):
            raise ContractError("Base capacities must be finite and nonnegative")
        if any(value <= 0 or not np.isfinite(value) for value in self.thresholds.values()):
            raise ContractError("Operational thresholds must be finite and positive")


@dataclass(frozen=True)
class CurrentCapacity:
    effective: Mapping[ResourceKey, float]
    direct_spot: Mapping[ResourceKey, float]
    readiness_capacity: Mapping[ResourceKey, float]
    corridor_weights: Mapping[tuple[str, str], float]


@dataclass(frozen=True)
class CapacityTransition:
    current: CurrentCapacity
    next_readiness: CapacityState
    next_direct: CapacityState
    audit: "CapacityTransitionAudit"


@dataclass(frozen=True)
class CapacityTransitionAudit:
    """Equation-level certificate for both registered capacity pipelines."""

    readiness_order_balance_residual: float
    readiness_stock_balance_residual: float
    direct_order_balance_residual: float
    direct_stock_balance_residual: float
    readiness_stock_feasibility_violation: float

    @property
    def maximum_residual(self) -> float:
        return max(
            self.readiness_order_balance_residual,
            self.readiness_stock_balance_residual,
            self.direct_order_balance_residual,
            self.direct_stock_balance_residual,
            self.readiness_stock_feasibility_violation,
        )


def _action_by_resource(
    action: Action, mapping: Mapping[ActionKey, ResourceKey]
) -> dict[ResourceKey, float]:
    result: dict[ResourceKey, float] = {}
    for key, resource in mapping.items():
        result[resource] = result.get(resource, 0.0) + action.value(key)
    return result


def _shift_orders(
    current: CapacityState,
    new_orders: Mapping[ResourceKey, float],
    leads: Mapping[ResourceKey, int],
) -> tuple[dict[ResourceKey, dict[int, float]], dict[ResourceKey, float]]:
    next_orders: dict[ResourceKey, dict[int, float]] = {}
    mature: dict[ResourceKey, float] = {}
    resources = set(current.orders) | set(new_orders) | set(leads)
    for resource in resources:
        bucket: dict[int, float] = {}
        for remaining, amount in current.orders.get(resource, {}).items():
            if remaining <= 0:
                raise ContractError("Pending capacity order keys must be positive")
            if remaining == 1:
                mature[resource] = mature.get(resource, 0.0) + amount
            else:
                bucket[remaining - 1] = bucket.get(remaining - 1, 0.0) + amount
        amount = float(new_orders.get(resource, 0.0))
        lead = int(leads[resource])
        if amount:
            if lead == 1:
                mature[resource] = mature.get(resource, 0.0) + amount
            elif lead > 1:
                bucket[lead - 1] = bucket.get(lead - 1, 0.0) + amount
        if bucket:
            next_orders[resource] = bucket
    return next_orders, mature


def _nested_mapping_residual(
    observed: Mapping[ResourceKey, Mapping[int, float]],
    expected: Mapping[ResourceKey, Mapping[int, float]],
) -> float:
    residual = 0.0
    for resource in set(observed) | set(expected):
        left = observed.get(resource, {})
        right = expected.get(resource, {})
        for remaining in set(left) | set(right):
            residual = max(
                residual,
                abs(float(left.get(remaining, 0.0)) - float(right.get(remaining, 0.0))),
            )
    return residual


def _mapping_residual(
    observed: Mapping[ResourceKey, float], expected: Mapping[ResourceKey, float]
) -> float:
    return max(
        (
            abs(float(observed.get(resource, 0.0)) - float(expected.get(resource, 0.0)))
            for resource in set(observed) | set(expected)
        ),
        default=0.0,
    )


def _reconstruct_pipeline_identity(
    current: CapacityState,
    new_orders: Mapping[ResourceKey, float],
    leads: Mapping[ResourceKey, int],
) -> tuple[dict[ResourceKey, dict[int, float]], dict[ResourceKey, float]]:
    """Independent equation reconstruction used only by the audit."""

    expected_orders: dict[ResourceKey, dict[int, float]] = {}
    expected_mature: dict[ResourceKey, float] = {}
    for resource in set(current.orders) | set(new_orders) | set(leads):
        bucket: dict[int, float] = {}
        for remaining, amount in current.orders.get(resource, {}).items():
            if remaining == 1:
                expected_mature[resource] = expected_mature.get(resource, 0.0) + float(
                    amount
                )
            else:
                bucket[remaining - 1] = bucket.get(remaining - 1, 0.0) + float(amount)
        amount = float(new_orders.get(resource, 0.0))
        lead = int(leads[resource])
        if lead == 1:
            expected_mature[resource] = expected_mature.get(resource, 0.0) + amount
        elif lead > 1:
            bucket[lead - 1] = bucket.get(lead - 1, 0.0) + amount
        if bucket:
            expected_orders[resource] = bucket
    return expected_orders, expected_mature


class CapacityDynamics:
    def __init__(
        self,
        network: Network,
        technology: CapacityTechnology,
        service: ServiceParameters,
        action_map: CapacityActionMap,
    ) -> None:
        self.network = network
        self.technology = technology
        self.service = service
        self.action_map = action_map
        resources = set(service.base_capacity)
        technology.validate(resources)

    def _corridor_weights(self, state: ModelState) -> dict[tuple[str, str], float]:
        weights: dict[tuple[str, str], float] = {}
        for gateway in self.network.gateways():
            corridors = sorted(
                {route.corridor for route in self.network.routes.values() if route.gateway == gateway}
            )
            totals = {
                corridor: float(sum(state.corridor_history.get((gateway, corridor), ())))
                for corridor in corridors
            }
            denominator = sum(totals.values())
            if denominator > 0:
                for corridor in corridors:
                    weights[(gateway, corridor)] = totals[corridor] / denominator
            else:
                fallback = np.asarray(
                    [self.service.fallback_corridor_share.get((gateway, corridor), 0.0) for corridor in corridors],
                    dtype=float,
                )
                if fallback.sum() <= 0:
                    raise ContractError(
                        f"Gateway {gateway} needs lagged flow or a commitment-time corridor share"
                    )
                fallback /= fallback.sum()
                for corridor, share in zip(corridors, fallback):
                    weights[(gateway, corridor)] = float(share)
        return weights

    def transition(self, state: ModelState, action: Action) -> CapacityTransition:
        readiness_orders = _action_by_resource(action, self.action_map.readiness_order)
        direct_orders = _action_by_resource(action, self.action_map.direct_order)
        exercises = _action_by_resource(action, self.action_map.readiness_exercise)
        resources = set(self.service.base_capacity)

        direct_spot: dict[ResourceKey, float] = {}
        positive_lead_direct: dict[ResourceKey, float] = {}
        direct_leads: dict[ResourceKey, int] = {}
        for resource in resources:
            lead = int(self.technology.direct_lead[(state.phase, resource)])
            direct_leads[resource] = max(lead, 1)
            order = direct_orders.get(resource, 0.0)
            if lead == 0:
                direct_spot[resource] = self.technology.direct_maturity_yield[resource] * order
            else:
                positive_lead_direct[resource] = order

        next_readiness_orders, mature_readiness_orders = _shift_orders(
            state.readiness, readiness_orders, self.technology.readiness_lead
        )
        next_direct_orders, mature_direct_orders = _shift_orders(
            state.direct_capacity, positive_lead_direct, direct_leads
        )

        next_readiness_stock: dict[ResourceKey, float] = {}
        next_direct_stock: dict[ResourceKey, float] = {}
        readiness_capacity: dict[ResourceKey, float] = {}
        for resource in resources:
            exercise = exercises.get(resource, 0.0)
            consumed = self.technology.readiness_consumption[resource] * exercise
            stock = state.readiness.stock.get(resource, 0.0)
            if consumed > stock + 1e-10:
                raise ContractError(f"Readiness exercise exceeds mature stock for {resource}")
            readiness_capacity[resource] = (
                self.technology.readiness_capacity_yield[resource] * exercise
            )
            next_readiness_stock[resource] = (
                (1.0 - self.technology.readiness_decay[resource]) * (stock - consumed)
                + self.technology.readiness_maturity_yield[resource]
                * mature_readiness_orders.get(resource, 0.0)
            )
            current_direct = state.direct_capacity.stock.get(resource, 0.0) + direct_spot.get(
                resource, 0.0
            )
            next_direct_stock[resource] = (
                (1.0 - self.technology.direct_decay[resource]) * current_direct
                + self.technology.direct_maturity_yield[resource]
                * mature_direct_orders.get(resource, 0.0)
            )

        yard_mass = {gateway: 0.0 for gateway in self.network.gateways()}
        corridor_mass = {corridor: 0.0 for corridor in self.network.corridors()}
        for tag, mass in state.yard.items():
            yard_mass[self.network.route(tag.route).gateway] += mass
        for tag, mass in state.corridor.items():
            corridor_mass[self.network.route(tag.route).corridor] += mass
        weights = self._corridor_weights(state)

        effective: dict[ResourceKey, float] = {}
        for resource, base in self.service.base_capacity.items():
            active_direct = state.direct_capacity.stock.get(resource, 0.0) + direct_spot.get(
                resource, 0.0
            )
            capacity = base + active_direct + readiness_capacity.get(resource, 0.0)
            if resource.stage == Stage.BERTH:
                yard_key = ResourceKey(Stage.YARD, resource.location)
                multiplier = self.service.yard_feedback[resource.location](
                    yard_mass[resource.location] / self.service.thresholds[yard_key]
                )
                capacity *= multiplier
            elif resource.stage == Stage.GATE:
                linked = [
                    route.corridor
                    for route in self.network.routes.values()
                    if route.gateway == resource.location
                ]
                pressure = sum(
                    weights[(resource.location, corridor)]
                    * corridor_mass[corridor]
                    / self.service.thresholds[ResourceKey(Stage.CORRIDOR, corridor)]
                    for corridor in set(linked)
                )
                capacity *= self.service.corridor_feedback[resource.location](pressure)
            if capacity < 0 or not np.isfinite(capacity):
                raise ContractError(f"Effective capacity is invalid for {resource}")
            effective[resource] = float(capacity)

        next_readiness = CapacityState(next_readiness_orders, next_readiness_stock)
        next_direct = CapacityState(next_direct_orders, next_direct_stock)

        # Reconstruct the four registered identities independently of the
        # returned objects.  These residuals are propagated into the common
        # transition/acceptance certificate rather than being hidden inside
        # the capacity helper.
        expected_readiness_orders, expected_mature_readiness = _reconstruct_pipeline_identity(
            state.readiness, readiness_orders, self.technology.readiness_lead
        )
        expected_direct_orders, expected_mature_direct = _reconstruct_pipeline_identity(
            state.direct_capacity, positive_lead_direct, direct_leads
        )
        expected_readiness_stock: dict[ResourceKey, float] = {}
        expected_direct_stock: dict[ResourceKey, float] = {}
        stock_violation = 0.0
        for resource in resources:
            exercise = exercises.get(resource, 0.0)
            consumed = self.technology.readiness_consumption[resource] * exercise
            stock = state.readiness.stock.get(resource, 0.0)
            stock_violation = max(stock_violation, consumed - stock)
            expected_readiness_stock[resource] = (
                (1.0 - self.technology.readiness_decay[resource]) * (stock - consumed)
                + self.technology.readiness_maturity_yield[resource]
                * expected_mature_readiness.get(resource, 0.0)
            )
            active_direct = state.direct_capacity.stock.get(resource, 0.0) + direct_spot.get(
                resource, 0.0
            )
            expected_direct_stock[resource] = (
                (1.0 - self.technology.direct_decay[resource]) * active_direct
                + self.technology.direct_maturity_yield[resource]
                * expected_mature_direct.get(resource, 0.0)
            )
        audit = CapacityTransitionAudit(
            readiness_order_balance_residual=_nested_mapping_residual(
                next_readiness.orders, expected_readiness_orders
            ),
            readiness_stock_balance_residual=_mapping_residual(
                next_readiness.stock, expected_readiness_stock
            ),
            direct_order_balance_residual=_nested_mapping_residual(
                next_direct.orders, expected_direct_orders
            ),
            direct_stock_balance_residual=_mapping_residual(
                next_direct.stock, expected_direct_stock
            ),
            readiness_stock_feasibility_violation=max(stock_violation, 0.0),
        )

        return CapacityTransition(
            current=CurrentCapacity(effective, direct_spot, readiness_capacity, weights),
            next_readiness=next_readiness,
            next_direct=next_direct,
            audit=audit,
        )
