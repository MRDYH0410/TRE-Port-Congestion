"""Model-consistent construction of Module 4 inputs from state, action, and event data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .actions import Action, ActionKey, Block
from .behavior import (
    BehaviorCostParameters,
    BehaviorProblem,
    FrozenDisclosure,
    PrivateWaitingOracle,
    build_decision_masses,
)
from .keys import Network
from .state import ModelState
from .transition import ExogenousRealization, construct_demand_split


@dataclass(frozen=True)
class BehaviorActionMap:
    release: Mapping[ActionKey, str]
    disclosure: Mapping[ActionKey, tuple[str, str]]


@dataclass(frozen=True)
class DisclosureForecast:
    raw_public_signal: Mapping[tuple[str, str], float]
    reference_forecast: Mapping[tuple[str, str], float]
    error_scale: Mapping[tuple[str, str], float]
    gamma: float


CostParameterFactory = Callable[
    [ModelState, Action, ExogenousRealization], BehaviorCostParameters
]
WaitingOracleFactory = Callable[
    [ModelState, Action, ExogenousRealization], PrivateWaitingOracle
]
DisclosureForecastFactory = Callable[
    [ModelState, Action, ExogenousRealization], DisclosureForecast
]


def disclosure_reference_action(action: Action) -> Action:
    """Return the formal ``a_t^{-I}`` input used by the disclosure baseline.

    The reference queue forecast must be independent of the disclosure choice
    whose credibility-bounded effect it anchors.  Enforcing that exclusion at
    the factory boundary prevents a callback from accidentally using the
    current disclosure intensity while constructing its own reference.
    """

    return Action(
        {
            key: float(value)
            for key, value in action.values.items()
            if key.block is not Block.DISCLOSURE
        }
    )


class StandardBehaviorProblemFactory:
    """No fixed exit share: exit remains entirely inside the source simplex/attrition."""

    def __init__(
        self,
        *,
        network: Network,
        action_map: BehaviorActionMap,
        cost_parameters: CostParameterFactory,
        waiting_oracle: WaitingOracleFactory,
        disclosure_forecast: DisclosureForecastFactory,
        tolerance: float,
    ) -> None:
        self.network = network
        self.action_map = action_map
        self.cost_parameter_factory = cost_parameters
        self.waiting_oracle_factory = waiting_oracle
        self.disclosure_forecast_factory = disclosure_forecast
        self.tolerance = tolerance

    def __call__(
        self, state: ModelState, action: Action, realization: ExogenousRealization
    ) -> BehaviorProblem:
        split = construct_demand_split(self.network, realization, tolerance=self.tolerance)
        release_rates = {
            cargo_class: action.value(key) for key, cargo_class in self.action_map.release.items()
        }
        decision = build_decision_masses(
            state.waiting, split.decision_eligible, release_rates
        )
        routes_by_class = {
            cargo_class: tuple(
                route_id
                for route_id in self.network.routes_for_class(cargo_class)
                if route_id in realization.choice_route_available
            )
            for cargo_class in state.waiting
        }
        forecast = self.disclosure_forecast_factory(
            state,
            disclosure_reference_action(action),
            realization,
        )
        intensity = {
            route_key: action.value(action_key)
            for action_key, route_key in self.action_map.disclosure.items()
            if route_key[1] in realization.choice_route_available
        }
        disclosure = FrozenDisclosure.clip_from_reference(
            raw_signal=forecast.raw_public_signal,
            reference_forecast=forecast.reference_forecast,
            error_scale=forecast.error_scale,
            intensity=intensity,
            gamma=forecast.gamma,
        )
        return BehaviorProblem(
            decision=decision,
            waiting_state=state.waiting,
            routes_by_class=routes_by_class,
            disclosure=disclosure,
            parameters=self.cost_parameter_factory(state, action, realization),
            reclosure_probability=state.risk.reclosure_probability,
            private_waiting_oracle=self.waiting_oracle_factory(state, action, realization),
        )
