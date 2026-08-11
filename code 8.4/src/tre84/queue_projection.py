"""Deterministic continuous queue projection used inside, not outside, SUE costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .behavior import PrivateWaitingOracle
from .errors import ContractError
from .keys import Network, SourceKey, Stage


@dataclass(frozen=True)
class ProjectedStage:
    queue: float
    inflow_before_service: float
    effective_capacity: float

    def waiting_time(self, epsilon: float) -> float:
        if self.queue < 0 or self.inflow_before_service < 0 or self.effective_capacity < 0:
            raise ContractError("Projected queue inputs must be nonnegative")
        return (self.queue + self.inflow_before_service) / (self.effective_capacity + epsilon)


ProjectionProvider = Callable[
    [str, Stage, int, Mapping[tuple[str, str], float]], ProjectedStage
]


class ContinuousQueueProjection:
    """Equation (anticipated-wait) with a fixed continuation/exogenous path."""

    def __init__(
        self,
        *,
        network: Network,
        stage_arrival_offsets: Mapping[tuple[str, Stage], int],
        provider: ProjectionProvider,
        epsilon: float,
    ) -> None:
        if epsilon <= 0:
            raise ContractError("Queue projection epsilon must be positive")
        self.network = network
        self.stage_arrival_offsets = dict(stage_arrival_offsets)
        self.provider = provider
        self.epsilon = epsilon

    @staticmethod
    def aggregate_route_dispatch(
        flows: Mapping[SourceKey, Mapping[str, float]], network: Network
    ) -> dict[tuple[str, str], float]:
        dispatch: dict[tuple[str, str], float] = {}
        for source, source_flows in flows.items():
            for route in network.routes_for_class(source.cargo_class):
                key = (source.cargo_class, route)
                dispatch[key] = dispatch.get(key, 0.0) + source_flows.get(route, 0.0)
        return dispatch

    def waiting_by_route(
        self, dispatch: Mapping[tuple[str, str], float]
    ) -> dict[tuple[str, str], float]:
        result: dict[tuple[str, str], float] = {}
        for route_id, route in self.network.routes.items():
            total = 0.0
            for stage in (Stage.BERTH, Stage.YARD, Stage.GATE, Stage.CORRIDOR):
                offset = self.stage_arrival_offsets[(route_id, stage)]
                projected = self.provider(route_id, stage, offset, dispatch)
                total += projected.waiting_time(self.epsilon)
            result[(route.cargo_class, route_id)] = total
        return result

    def oracle(self) -> PrivateWaitingOracle:
        def evaluate(
            flows: Mapping[SourceKey, Mapping[str, float]]
        ) -> Mapping[tuple[str, str], float]:
            dispatch = self.aggregate_route_dispatch(flows, self.network)
            return self.waiting_by_route(dispatch)

        return evaluate

    def frozen_reference_signal(
        self,
        reference_dispatch: Mapping[tuple[str, str], float],
    ) -> dict[tuple[str, str], float]:
        """A predetermined loading rule; never inferred from the induced equilibrium."""

        return self.waiting_by_route(reference_dispatch)

