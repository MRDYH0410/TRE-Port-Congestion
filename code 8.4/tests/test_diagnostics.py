from __future__ import annotations

import numpy as np
import pytest

from tre84.diagnostics import (
    AbsorptionBoundaryLP,
    AbsorptionInput,
    ThresholdSnapshot,
    check_adaptive_shaping_bound,
    ex_post_cascade_statistics,
    local_selected_branch_jacobian,
)
from tre84.errors import ContractError


def test_structural_and_local_diagnostics_keep_their_declared_boundaries() -> None:
    shaping = check_adaptive_shaping_bound(
        {"s": {"r": 2.0, "__WAIT__": 0.0}},
        {"s": {"r": 0.0, "__WAIT__": 2.0}},
        {"s": 2.0},
        tolerance=1e-12,
    )
    assert shaping.satisfied and np.isclose(shaping.half_l1_distance, 2.0)

    boundary = AbsorptionBoundaryLP.solve(
        AbsorptionInput(
            initial_berth=0.0,
            initial_yard=0.0,
            initial_gate=0.0,
            initial_corridor=0.0,
            berth_threshold=1.0,
            yard_threshold=1.0,
            gate_threshold=1.0,
            corridor_threshold=1.0,
            berth_capacity=np.array([1.0]),
            yard_capacity=np.array([1.0]),
            gate_capacity=np.array([1.0]),
            optimistic_corridor_capacity=np.array([1.0]),
        )
    )
    assert np.isclose(boundary.maximum_absorbable, 2.0)

    local = local_selected_branch_jacobian(
        d_phi_d_state=np.array([[0.5]]),
        d_phi_d_choice=np.array([[1.0]]),
        d_loading_d_choice=np.array([[0.2]]),
        d_loading_d_state=np.array([[0.1]]),
    )
    assert np.allclose(local.jacobian, [[0.625]])
    assert np.isclose(local.spectral_radius, 0.625)


def test_cascade_times_are_computed_only_against_completed_matched_paths() -> None:
    realized = (
        ThresholdSnapshot(0, frozenset(), frozenset()),
        ThresholdSnapshot(1, frozenset({"g1"}), frozenset()),
        ThresholdSnapshot(2, frozenset({"g1", "g2"}), frozenset({"e"})),
    )
    counterfactual = tuple(
        ThresholdSnapshot(period, frozenset(), frozenset()) for period in range(3)
    )
    stats = ex_post_cascade_statistics(
        disruption_start=0, realized=realized, matched_no_disruption=counterfactual
    )
    assert stats.first_gateway_time == 1
    assert stats.propagation_time == 2
    assert not stats.propagation_right_censored

    with pytest.raises(ContractError, match="completed, period-matched"):
        ex_post_cascade_statistics(
            disruption_start=0,
            realized=realized,
            matched_no_disruption=counterfactual[:-1],
        )
