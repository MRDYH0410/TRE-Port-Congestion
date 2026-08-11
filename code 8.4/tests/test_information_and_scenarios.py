from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tre84.actions import Action
from tre84.control import (
    MPCCandidate,
    ProjectedStochasticMPC,
    TwoProposalSelector,
)
from tre84.errors import ContractError
from tre84.information import GaussianHMM, ReleaseRecord, ReleasedRiskInference
from tre84.orchestration import InformationContext, InformationScenarioModules
from tre84.scenarios import (
    CommonScenarioConstructor,
    EventPath,
    ScenarioBundle,
    TimestampedOperationalContext,
)
from tre84.transition import ExogenousRealization
from tre84.state import CapacityState, ModelState, RiskInformation


def test_release_clock_excludes_unreleased_observations_and_uses_monthly_power() -> None:
    hmm = GaussianHMM(
        initial=np.array([0.5, 0.5]),
        transition=np.array([[0.9, 0.1], [0.2, 0.8]]),
        means=np.array([[0.0], [3.0]]),
        variances=np.ones((2, 1)),
    )
    records = (
        ReleaseRecord(0, 2, np.array([0.0])),
        ReleaseRecord(1, 6, np.array([3.0])),
    )
    result = ReleasedRiskInference(hmm).infer(
        decision_time=5,
        readiness_maturity_time=8,
        records=records,
        monthly_transition_count=lambda latest, maturity: 2,
    )
    expected_belief = hmm.released_filter([np.array([0.0])], [0])
    assert np.allclose(result.belief, expected_belief)
    assert np.allclose(result.lead_time_forecast, expected_belief @ np.linalg.matrix_power(hmm.transition, 2))
    assert result.latest_observation_period == 0
    assert result.information_timestamps == (2,)


def test_common_support_keeps_dual_weight_systems_separate() -> None:
    paths = (
        EventPath("a", (True, False, True), {"c": np.array([0.2, 1.0, 0.1])}, (True, False, True)),
        EventPath("b", (False, False, False), {"c": np.ones(3)}, (False, False, False)),
    )
    constructor = CommonScenarioConstructor(
        paths,
        readiness_log_weight=lambda path, risk: float(risk[1]) if path.path_id == "a" else 0.0,
        operational_log_weight=lambda path, context: 0.0 if path.path_id == context else -3.0,
        seed_manifest={"support": 17},
    )
    pre = constructor.build(
        lead_time_risk_forecast=np.array([0.1, 0.9]),
        operational_context=TimestampedOperationalContext("b", (4,)),
        phase=0,
        completed_information_timestamps=(4,),
        decision_time=5,
    )
    post = constructor.build(
        lead_time_risk_forecast=np.array([0.1, 0.9]),
        operational_context=TimestampedOperationalContext("b", (4,)),
        phase=1,
        completed_information_timestamps=(4,),
        decision_time=5,
    )
    assert tuple(path.path_id for path in pre.paths) == tuple(path.path_id for path in post.paths)
    assert np.allclose(pre.active_weights, pre.readiness_weights)
    assert np.allclose(post.active_weights, post.operational_weights)
    assert pre.reclosure_probability > post.reclosure_probability


def test_operational_context_rejects_post_decision_information() -> None:
    path = EventPath("a", (False,), {"c": np.ones(1)}, (False,))
    constructor = CommonScenarioConstructor(
        (path,),
        readiness_log_weight=lambda path, risk: 0.0,
        operational_log_weight=lambda path, context: 0.0,
    )
    with pytest.raises(ContractError, match="unreleased"):
        constructor.build(
            lead_time_risk_forecast=np.array([1.0]),
            operational_context=TimestampedOperationalContext({}, (6,)),
            phase=1,
            completed_information_timestamps=(4,),
            decision_time=5,
        )


def test_preparation_crops_serviceability_values_with_their_release_timestamps() -> None:
    state = ModelState(
        period=0,
        horizon=2,
        risk=RiskInformation(np.array([1.0]), np.array([1.0])),
        disruption_seen=False,
        disruption_active=False,
        disruption_duration=0,
        waiting={},
        berth={},
        yard={},
        gate={},
        corridor={},
        maritime_pipeline=[],
        previous_shares={},
        corridor_history={},
        serviceability_history=(0.1, 0.9),
        readiness=CapacityState(),
        direct_capacity=CapacityState(),
        budget=1.0,
        observed_covariates={"serviceability_timestamps": (4, 6)},
    )
    hmm = GaussianHMM(
        initial=np.array([1.0]),
        transition=np.array([[1.0]]),
        means=np.array([[0.0]]),
        variances=np.array([[1.0]]),
    )
    path = EventPath("a", (False,), {}, (False,))
    modules = InformationScenarioModules(
        risk_inference=ReleasedRiskInference(hmm),
        scenario_constructor=CommonScenarioConstructor(
            (path,),
            readiness_log_weight=lambda path, risk: 0.0,
            operational_log_weight=lambda path, context: 0.0,
        ),
    )
    prepared = modules.prepare(
        state,
        InformationContext(
            decision_time=5,
            readiness_maturity_time=5,
            release_records=(),
            operational_context=TimestampedOperationalContext({}, (4,)),
            monthly_transition_count=lambda latest, maturity: 0,
        ),
    )
    assert prepared.state.serviceability_history == (0.1,)
    assert prepared.state.observed_covariates["serviceability_timestamps"] == (4,)
    assert state.serviceability_history == (0.1, 0.9)  # input state remains immutable here


def test_event_path_reveals_only_strictly_past_payload() -> None:
    path = EventPath(
        "a",
        (True, False, True),
        {"c": np.array([0.2, 0.8, 1.0])},
        (True, False, True),
        demand_residual={"c": np.array([0.1, -0.2, 0.0])},
        payload=("t0", "t1", "t2"),
    )
    history = path.revealed_before(2)
    assert history.decision_offset == 2
    assert history.payload == ("t0", "t1")
    assert history.active == (True, False)
    assert np.allclose(history.serviceability["c"], [0.2, 0.8])
    with pytest.raises(IndexError):
        _ = history.payload[2]


def test_mpc_continuation_is_called_with_revealed_prefix_not_full_path() -> None:
    def realization() -> ExogenousRealization:
        return ExogenousRealization(
            gulf_demand={},
            serviceable_share={},
            committed_fraction={},
            committed_route_share={},
            base_arrivals={},
            choice_route_available=frozenset(),
            physical_route_available=frozenset(),
            serviceability_observation=1.0,
            next_disruption_seen=False,
            next_disruption_active=False,
            next_disruption_duration=0,
        )

    payload = (realization(), realization())
    path = EventPath(
        "a",
        (False, False),
        {},
        (False, False),
        payload=payload,
    )
    bundle = ScenarioBundle(
        paths=(path,),
        readiness_weights=np.array([1.0]),
        operational_weights=np.array([1.0]),
        active_weights=np.array([1.0]),
        reclosure_probability=0.0,
        weighted_serviceability={},
        information_timestamps=(),
        seed_manifest={},
        decision_time=0,
    )

    class _State:
        def __init__(self, period=0):
            self.period = period

        def clone(self):
            return _State(self.period)

    class _Projector:
        @staticmethod
        def project(raw, state):
            return SimpleNamespace(
                action=raw,
                raw_action=raw,
                objective=0.0,
                active_budget=False,
                feasibility_violation=0.0,
                iterations=1,
            )

    class _Kernel:
        @staticmethod
        def execute(**kwargs):
            return SimpleNamespace(
                equilibrium=SimpleNamespace(
                    status="converged",
                    residual=0.0,
                    iterations=1,
                    selected_start="free_flow",
                ),
                transition=SimpleNamespace(
                    audit=SimpleNamespace(passed=True),
                    loss=SimpleNamespace(total=0.0),
                    next_state=_State(kwargs["state"].period + 1),
                )
            )

    seen = []

    def continuation(state, offset, history):
        seen.append(history)
        assert offset == 1
        assert history.payload == (payload[0],)
        with pytest.raises(IndexError):
            _ = history.payload[1]
        return Action()

    mpc = ProjectedStochasticMPC(
        kernel=_Kernel(),
        projector=_Projector(),
        lookahead=2,
        terminal_value=lambda state: 0.0,
    )
    evaluation = mpc.evaluate(
        _State(), bundle, MPCCandidate("candidate", Action(), continuation)
    )
    assert evaluation.valid
    assert len(seen) == 1
    assert len(evaluation.module_certificates) == 2
    assert evaluation.scenario_ids == ("a",)
    assert evaluation.scenario_weights == (1.0,)
    assert evaluation.terminal_losses == (0.0,)

    result = mpc.solve(
        state=_State(),
        bundle=bundle,
        candidates=(
            MPCCandidate("first", Action(), continuation),
            MPCCandidate("second", Action(), continuation),
        ),
    )
    assert result.selection_log.considered_candidate_ids == ("first", "second")
    assert result.selection_log.selected_candidate_id == "first"

    selector = TwoProposalSelector(
        mpc_evaluator=mpc,
        fallback_raw_action=lambda state: Action(),
        continuation=continuation,
    )
    selection = selector.select(
        state=_State(),
        bundle=bundle,
        bc_raw_action=Action(),
        sac_raw_action=Action(),
    )
    assert selection.source == "BC"
    assert selection.selection_log.considered_candidate_ids == ("BC", "SAC")
    assert selection.selection_log.selected_candidate_id == "BC"
