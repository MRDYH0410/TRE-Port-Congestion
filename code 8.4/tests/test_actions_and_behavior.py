from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tre84.actions import (
    Action,
    ActionDomain,
    ActionKey,
    ActionProjector,
    Block,
    ConvexPiecewiseLinearCurve,
    StockConstraint,
)
from tre84.acceptance import _behavior_certificate
from tre84.behavior import (
    BehaviorCostParameters,
    BehaviorProblem,
    FrozenDisclosure,
    RCMSASettings,
    RCMSASolver,
    SourceKey,
    build_decision_masses,
    oldest_first,
)
from tre84.factory import (
    BehaviorActionMap,
    DisclosureForecast,
    StandardBehaviorProblemFactory,
)
from tre84.keys import Network, ResourceKey, Route, Stage
from tre84.state import CapacityState, ModelState, RiskInformation
from tre84.transition import ExogenousRealization


def empty_state() -> ModelState:
    risk = RiskInformation(np.array([1.0]), np.array([1.0]))
    return ModelState(
        period=0,
        horizon=4,
        risk=risk,
        disruption_seen=False,
        disruption_active=False,
        disruption_duration=0,
        waiting={"c": np.array([0.0, 2.0])},
        berth={},
        yard={},
        gate={},
        corridor={},
        maritime_pipeline=[],
        previous_shares={},
        corridor_history={("g", "e"): (0.0,)},
        serviceability_history=(),
        readiness=CapacityState(stock={ResourceKey(Stage.BERTH, "g"): 0.0}),
        direct_capacity=CapacityState(),
        budget=2.0,
    )


def test_oldest_first_and_projection_enforce_stock_phase_and_budget() -> None:
    assert np.allclose(oldest_first(3.0, np.array([2.0, 2.0, 2.0])), [0.0, 1.0, 2.0])
    resource = ResourceKey(Stage.BERTH, "g")
    keys = (
        ActionKey.one(Block.READINESS_ORDER, "g"),
        ActionKey.one(Block.DIRECT_ORDER, "g"),
        ActionKey.one(Block.READINESS_EXERCISE, "g"),
        ActionKey.one(Block.RELEASE, "c"),
        ActionKey(Block.DISCLOSURE, ("c", "r")),
    )
    paid = ConvexPiecewiseLinearCurve((0.0,), (1.0,))
    free = ConvexPiecewiseLinearCurve((0.0,), (0.0,))
    domain = ActionDomain(
        keys=keys,
        phase_upper={0: {key: 1.0 for key in keys}, 1: {key: 1.0 for key in keys}},
        cost_curves={keys[0]: paid, keys[1]: paid, keys[2]: paid, keys[3]: free, keys[4]: free},
        period_budget_cap=lambda state: 2.0,
        stock_constraints={keys[2]: StockConstraint(resource, 1.0)},
    )
    projector = ActionProjector(
        domain, scaling={key: 1.0 for key in keys}, tolerance=1e-9, max_iterations=200
    )
    result = projector.project(Action({key: 1.0 for key in keys}), empty_state())
    assert result.action.value(keys[2]) == 0.0
    assert result.action.value(keys[1]) > 0.0  # direct procurement is not conditional on readiness stock
    assert domain.action_cost(result.action) <= 2.0 + 1e-8
    assert result.action.value(keys[3]) == 1.0
    assert result.action.value(keys[4]) == 1.0


def test_rcmsa_uses_one_route_wait_exit_simplex_for_every_source() -> None:
    waiting = {"c": np.array([0.0, 2.0])}
    decision = build_decision_masses(waiting, {"c": 3.0}, {"c": 0.5})
    sources = tuple(decision.masses)
    direct_exit = {source: 8.0 for source in sources}
    parameters = BehaviorCostParameters(
        theta={"c": 1.0},
        route_private_resource={("c", "r"): 0.0},
        route_market_cost={("c", "r"): 0.0},
        value_of_time={"c": 1.0},
        waiting_base={"c": 5.0},
        waiting_age_cost={("c", 0): 0.0, ("c", 1): 1.0, ("c", 2): 2.0},
        waiting_inventory_cost={"c": 0.0},
        waiting_scale={"c": 1.0},
        waiting_reclosure_cost={"c": 0.0},
        continuation_value={("c", 0): 0.0, ("c", 1): 0.0},
        direct_exit_cost=direct_exit,
        late_exit_cost={("c", 0): 0.0, ("c", 1): 0.0},
        hazard={"c": np.array([0.2, 1.0])},
    )
    disclosure = FrozenDisclosure(
        public_signal={("c", "r"): 1.0},
        reference_forecast={("c", "r"): 1.0},
        error_scale={("c", "r"): 1.0},
        intensity={("c", "r"): 0.0},
        gamma=1.0,
    )
    problem = BehaviorProblem(
        decision=decision,
        waiting_state=waiting,
        routes_by_class={"c": ("r",)},
        disclosure=disclosure,
        parameters=parameters,
        reclosure_probability=0.0,
        private_waiting_oracle=lambda flows: {
            ("c", "r"): 0.1
            * sum(source_flow.get("r", 0.0) for source_flow in flows.values())
        },
    )
    result = RCMSASolver(RCMSASettings(1e-8, 500, 1e-7)).solve(
        problem, previous_shares={}
    )
    assert result.status == "converged"
    assert result.residual <= 1e-8
    assert np.allclose(result.releases["c"], [0.0, 1.0])
    for source, mass in decision.masses.items():
        assert np.isclose(sum(result.flows[source].values()), mass)
        assert set(result.flows[source]) == {"r", "__WAIT__", "__EXIT__"}
    assert result.direct_exit[SourceKey("c", None)] > 0.0  # endogenous Logit outcome, not a fixed share


def test_rcmsa_certifies_the_final_trial_and_preserves_start_provenance() -> None:
    source = SourceKey("c", None)

    class _Problem:
        decision = SimpleNamespace(masses={source: 1.0})
        sources = (source,)

        @staticmethod
        def choices(_source):
            return ("left", "right")

    class _ScriptedSolver(RCMSASolver):
        def __init__(self):
            super().__init__(RCMSASettings(0.08, 1, 1e-8))
            self.loading_inputs: list[np.ndarray] = []

        def _loading(self, problem, slices, vector):
            self.loading_inputs.append(np.asarray(vector, dtype=float).copy())
            return np.array([0.55, 0.45])

        def _residual(self, problem, slices, vector):
            return abs(float(vector[0]) - 0.60)

        @staticmethod
        def _kl(problem, slices, vector, loading):
            return 0.0

        def _build_result(
            self,
            problem,
            slices,
            vector,
            residual,
            kl,
            iterations,
            status,
            records,
            dispersion,
            selected_start="",
            selected_step_multipliers=(),
        ):
            return SimpleNamespace(
                residual=residual,
                iterations=iterations,
                status=status,
                starts=records,
                selected_start=selected_start,
                selected_step_multipliers=selected_step_multipliers,
            )

    solver = _ScriptedSolver()
    result = solver.solve(_Problem(), previous_shares={})
    previous = next(record for record in result.starts if record.name == "previous")
    assert np.allclose(solver.loading_inputs[0], 0.0)  # strict zero-load/free-flow start
    assert previous.converged and np.isclose(previous.residual, 0.05)
    assert previous.selected_step_multipliers == (1.0,)
    assert result.status == "converged"
    assert result.selected_start == "previous"
    assert np.isclose(result.residual, previous.residual)
    assert result.selected_step_multipliers == previous.selected_step_multipliers


def test_rcmsa_historical_distance_uses_complete_master_choice_support() -> None:
    source = SourceKey("c", None)

    class _Problem:
        decision = SimpleNamespace(masses={source: 1.0})
        sources = (source,)

        @staticmethod
        def choices(_source):
            return ("available", "__WAIT__", "__EXIT__")

    solver = RCMSASolver(RCMSASettings(1e-8, 10, 1e-8))
    problem = _Problem()
    slices, _ = solver._layout(problem)
    candidate = np.array([1.0, 0.0, 0.0])
    previous = {
        source: {
            "available": 0.0,
            "now_unavailable": 1.0,
            "__WAIT__": 0.0,
            "__EXIT__": 0.0,
        }
    }
    # One unit moves off the unavailable historical route and one unit moves
    # onto the current route.  Omitting the unavailable coordinate gives 1.0;
    # the complete-master L1 distance required by Sel_t is 2.0.
    assert np.isclose(
        solver._distance_to_previous(problem, slices, candidate, previous),
        2.0,
    )


def test_disclosure_reference_factory_receives_a_minus_information() -> None:
    release_key = ActionKey.one(Block.RELEASE, "c")
    disclosure_key = ActionKey(Block.DISCLOSURE, ("c", "r"))
    observed_actions: list[Action] = []

    def disclosure_forecast(state, reference_action, realization):
        observed_actions.append(reference_action)
        return DisclosureForecast(
            raw_public_signal={("c", "r"): 1.0},
            reference_forecast={("c", "r"): 1.0},
            error_scale={("c", "r"): 1.0},
            gamma=1.0,
        )

    parameters = BehaviorCostParameters(
        theta={"c": 1.0},
        route_private_resource={("c", "r"): 0.0},
        route_market_cost={("c", "r"): 0.0},
        value_of_time={"c": 1.0},
        waiting_base={"c": 0.0},
        waiting_age_cost={("c", 0): 0.0, ("c", 1): 0.0, ("c", 2): 0.0},
        waiting_inventory_cost={"c": 0.0},
        waiting_scale={"c": 1.0},
        waiting_reclosure_cost={"c": 0.0},
        continuation_value={("c", 0): 0.0, ("c", 1): 0.0},
        direct_exit_cost={SourceKey("c", None): 1.0},
        late_exit_cost={("c", 0): 0.0, ("c", 1): 0.0},
        hazard={"c": np.array([0.2, 1.0])},
    )
    factory = StandardBehaviorProblemFactory(
        network=Network({"r": Route("r", "c", "g", "e", (1.0,))}),
        action_map=BehaviorActionMap(
            release={release_key: "c"},
            disclosure={disclosure_key: ("c", "r")},
        ),
        cost_parameters=lambda state, action, realization: parameters,
        waiting_oracle=lambda state, action, realization: (
            lambda flows: {("c", "r"): 1.0}
        ),
        disclosure_forecast=disclosure_forecast,
        tolerance=1e-9,
    )
    realization = ExogenousRealization(
        gulf_demand={"c": 1.0},
        serviceable_share={"c": 0.0},
        committed_fraction={"c": 0.0},
        committed_route_share={},
        base_arrivals={},
        choice_route_available=frozenset({"r"}),
        physical_route_available=frozenset({"r"}),
        serviceability_observation=0.0,
        next_disruption_seen=False,
        next_disruption_active=False,
        next_disruption_duration=0,
    )
    full_action = Action({release_key: 0.0, disclosure_key: 0.75})
    problem = factory(empty_state(), full_action, realization)
    assert len(observed_actions) == 1
    assert observed_actions[0].value(disclosure_key) == 0.0
    assert disclosure_key not in observed_actions[0].values
    assert problem.disclosure.intensity[("c", "r")] == 0.75


def test_rcmsa_historical_shares_remain_a_simplex_for_subnormal_vintage_flow() -> None:
    source = SourceKey("c", 40)
    choices = ("r1", "r2", "__WAIT__", "__EXIT__")
    problem = SimpleNamespace(
        sources=(source,),
        choices=lambda current: choices,
        decision=SimpleNamespace(masses={source: np.nextafter(0.0, 1.0) * 1000}),
    )
    slices = {source: slice(0, len(choices))}
    # Independent underflow of alternatives can leave only part of the source
    # mass represented, but the historical RC-MSA start is still a probability
    # vector on the master choice set.
    vector = np.asarray([np.nextafter(0.0, 1.0) * 300, 0.0, 0.0, 0.0])
    shares = RCMSASolver._shares(problem, slices, vector)[source]
    assert set(shares) == set(choices)
    assert np.isclose(sum(shares.values()), 1.0, atol=0.0)
    assert shares["r1"] == 1.0
    equilibrium = SimpleNamespace(
        flows={source: {choice: float(vector[index]) for index, choice in enumerate(choices)}},
        releases={}, normalized_shares={source: shares}, status="converged",
        residual=0.0, starts=(), selected_start="", iterations=0,
        selected_step_multipliers=(),
    )
    assert _behavior_certificate(
        equilibrium, {source: problem.decision.masses[source]}, empty_state(), 1e-8
    )
