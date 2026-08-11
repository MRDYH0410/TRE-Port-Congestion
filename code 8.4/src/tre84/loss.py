"""Operational resource loss and explicit terminal-mass correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .errors import ContractError
from .keys import Network, ResourceKey, SourceKey, Stage, Tag
from .state import ModelState


@dataclass(frozen=True)
class LossParameters:
    queue_cost: Mapping[ResourceKey, float]
    waiting_cost: Mapping[tuple[str, int], float]
    exit_failure_cost: Mapping[str, float]
    overflow_cost: Mapping[ResourceKey, float]
    thresholds: Mapping[ResourceKey, float]
    route_resource_increment: Mapping[Tag, float]

    def __post_init__(self) -> None:
        collections = (
            self.queue_cost,
            self.waiting_cost,
            self.exit_failure_cost,
            self.overflow_cost,
            self.route_resource_increment,
        )
        if any(value < 0 or not np.isfinite(value) for mapping in collections for value in mapping.values()):
            raise ContractError("Operational loss coefficients must be finite and nonnegative")


@dataclass(frozen=True)
class LossBreakdown:
    queue: float
    waiting: float
    exit: float
    overflow: float
    route_resource: float
    action: float

    @property
    def total(self) -> float:
        return self.queue + self.waiting + self.exit + self.overflow + self.route_resource + self.action


class OperationalLoss:
    def __init__(self, network: Network, parameters: LossParameters) -> None:
        self.network = network
        self.parameters = parameters

    def _aggregate_queues(self, state: ModelState) -> dict[ResourceKey, float]:
        aggregates: dict[ResourceKey, float] = {}
        for stage, mapping in (
            (Stage.BERTH, state.berth),
            (Stage.YARD, state.yard),
            (Stage.GATE, state.gate),
            (Stage.CORRIDOR, state.corridor),
        ):
            for tag, mass in mapping.items():
                route = self.network.route(tag.route)
                location = route.corridor if stage == Stage.CORRIDOR else route.gateway
                key = ResourceKey(stage, location)
                aggregates[key] = aggregates.get(key, 0.0) + mass
        return aggregates

    def compute(
        self,
        *,
        state: ModelState,
        committed_dispatch: Mapping[Tag, float],
        adaptive_dispatch: Mapping[Tag, float],
        direct_exit: Mapping[SourceKey, float],
        duration_attrition: Mapping[tuple[str, int], float],
        action_cost: float,
    ) -> LossBreakdown:
        aggregates = self._aggregate_queues(state)
        queue = sum(self.parameters.queue_cost[key] * mass for key, mass in aggregates.items())
        waiting = sum(
            self.parameters.waiting_cost[(cargo_class, vintage)] * float(mass)
            for cargo_class, vintages in state.waiting.items()
            for vintage, mass in enumerate(vintages)
        )
        exits_by_class: dict[str, float] = {}
        for source, mass in direct_exit.items():
            exits_by_class[source.cargo_class] = exits_by_class.get(source.cargo_class, 0.0) + mass
        for (cargo_class, _), mass in duration_attrition.items():
            exits_by_class[cargo_class] = exits_by_class.get(cargo_class, 0.0) + mass
        exit_loss = sum(
            self.parameters.exit_failure_cost[cargo_class] * mass
            for cargo_class, mass in exits_by_class.items()
        )
        overflow = sum(
            self.parameters.overflow_cost[key]
            * max(mass - self.parameters.thresholds[key], 0.0)
            for key, mass in aggregates.items()
        )
        route_resource = sum(
            self.parameters.route_resource_increment[tag]
            * (committed_dispatch.get(tag, 0.0) + adaptive_dispatch.get(tag, 0.0))
            for tag in set(committed_dispatch) | set(adaptive_dispatch)
        )
        return LossBreakdown(
            queue=float(queue),
            waiting=float(waiting),
            exit=float(exit_loss),
            overflow=float(overflow),
            route_resource=float(route_resource),
            action=float(action_cost),
        )


@dataclass(frozen=True)
class TerminalCostParameters:
    waiting_unit_cost: Mapping[str, float]
    pipeline_unit_cost: Mapping[str, float]
    tagged_unit_cost: Mapping[ResourceKey, float]


class TerminalMassCorrection:
    """Charge all remaining W, maritime-pipeline, and tagged-queue mass once."""

    def __init__(
        self, network: Network, parameters: TerminalCostParameters
    ) -> None:
        self.network = network
        self.parameters = parameters

    def compute(self, state: ModelState) -> float:
        waiting = sum(
            self.parameters.waiting_unit_cost[cargo_class] * float(vintages.sum())
            for cargo_class, vintages in state.waiting.items()
        )
        pipeline = sum(
            self.parameters.pipeline_unit_cost[lot.cargo_class] * lot.mass
            for lot in state.maritime_pipeline
        )
        tagged = 0.0
        for stage, mapping in (
            (Stage.BERTH, state.berth),
            (Stage.YARD, state.yard),
            (Stage.GATE, state.gate),
            (Stage.CORRIDOR, state.corridor),
        ):
            for tag, mass in mapping.items():
                route = self.network.route(tag.route)
                location = route.corridor if stage == Stage.CORRIDOR else route.gateway
                tagged += self.parameters.tagged_unit_cost[ResourceKey(stage, location)] * mass
        return float(waiting + pipeline + tagged)

