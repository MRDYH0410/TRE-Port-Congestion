from __future__ import annotations

import numpy as np

from tre84.actions import Action, ActionKey, Block
from tre84.capacity import (
    CapacityActionMap,
    CapacityDynamics,
    CapacityTechnology,
    ServiceParameters,
)
from tre84.keys import Network, ResourceKey, Route, Stage
from tre84.state import CapacityState, ModelState, RiskInformation


def test_readiness_and_direct_capacity_have_distinct_leads_and_stocks() -> None:
    network = Network({"r": Route("r", "c", "g", "e", (1.0,))})
    resources = {
        ResourceKey(Stage.BERTH, "g"),
        ResourceKey(Stage.YARD, "g"),
        ResourceKey(Stage.GATE, "g"),
        ResourceKey(Stage.CORRIDOR, "e"),
    }
    target = ResourceKey(Stage.BERTH, "g")
    readiness_key = ActionKey.one(Block.READINESS_ORDER, "B:g")
    direct_key = ActionKey.one(Block.DIRECT_ORDER, "B:g")
    exercise_key = ActionKey.one(Block.READINESS_EXERCISE, "B:g")
    technology = CapacityTechnology(
        readiness_lead={resource: 2 for resource in resources},
        readiness_maturity_yield={resource: 1.0 for resource in resources},
        readiness_consumption={resource: 1.0 for resource in resources},
        readiness_capacity_yield={resource: 2.0 for resource in resources},
        readiness_decay={resource: 0.0 for resource in resources},
        direct_lead={
            (phase, resource): (0 if phase == 0 and resource == target else 1)
            for phase in (0, 1)
            for resource in resources
        },
        direct_maturity_yield={resource: 1.0 for resource in resources},
        direct_decay={resource: 0.0 for resource in resources},
    )
    dynamics = CapacityDynamics(
        network,
        technology,
        ServiceParameters(
            base_capacity={resource: 1.0 for resource in resources},
            thresholds={resource: 10.0 for resource in resources},
            yard_feedback={"g": lambda ratio: 1.0},
            corridor_feedback={"g": lambda ratio: 1.0},
            fallback_corridor_share={("g", "e"): 1.0},
        ),
        CapacityActionMap(
            readiness_order={readiness_key: target},
            direct_order={direct_key: target},
            readiness_exercise={exercise_key: target},
        ),
    )
    state = ModelState(
        period=0,
        horizon=4,
        risk=RiskInformation(np.array([1.0]), np.array([1.0])),
        disruption_seen=False,
        disruption_active=False,
        disruption_duration=0,
        waiting={"c": np.array([0.0])},
        berth={},
        yard={},
        gate={},
        corridor={},
        maritime_pipeline=[],
        previous_shares={},
        corridor_history={("g", "e"): (0.0,)},
        serviceability_history=(),
        readiness=CapacityState(),
        direct_capacity=CapacityState(),
        budget=10.0,
    )
    first = dynamics.transition(
        state,
        Action({readiness_key: 1.0, direct_key: 1.0, exercise_key: 0.0}),
    )
    assert np.isclose(first.current.effective[target], 2.0)  # direct lead zero is current spot
    assert np.isclose(first.next_direct.stock[target], 1.0)
    assert first.next_readiness.stock[target] == 0.0
    assert first.next_readiness.orders[target] == {1: 1.0}

    state.readiness = first.next_readiness
    state.direct_capacity = first.next_direct
    state.period = 1
    second = dynamics.transition(state, Action())
    assert np.isclose(second.current.effective[target], 2.0)  # active direct stock persists
    assert np.isclose(second.next_readiness.stock[target], 1.0)  # readiness matures next period
    assert second.current.readiness_capacity[target] == 0.0  # maturity is not consuming exercise

