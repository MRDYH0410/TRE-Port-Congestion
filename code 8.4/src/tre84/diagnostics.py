"""Structural boundaries, local diagnostics, and ex-post cascade statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from .errors import ContractError, NumericalFailure


@dataclass(frozen=True)
class ShapingBoundResult:
    half_l1_distance: float
    decision_mass: float
    satisfied: bool


def check_adaptive_shaping_bound(
    left: Mapping[object, Mapping[str, float]],
    right: Mapping[object, Mapping[str, float]],
    decision_masses: Mapping[object, float],
    *,
    tolerance: float,
) -> ShapingBoundResult:
    half_l1 = 0.0
    for source in decision_masses:
        choices = set(left.get(source, {})) | set(right.get(source, {}))
        half_l1 += 0.5 * sum(
            abs(left.get(source, {}).get(choice, 0.0) - right.get(source, {}).get(choice, 0.0))
            for choice in choices
        )
    total = float(sum(decision_masses.values()))
    return ShapingBoundResult(half_l1, total, half_l1 <= total + tolerance)


@dataclass(frozen=True)
class AbsorptionInput:
    initial_berth: float
    initial_yard: float
    initial_gate: float
    initial_corridor: float
    berth_threshold: float
    yard_threshold: float
    gate_threshold: float
    corridor_threshold: float
    berth_capacity: np.ndarray
    yard_capacity: np.ndarray
    gate_capacity: np.ndarray
    optimistic_corridor_capacity: np.ndarray

    @property
    def horizon(self) -> int:
        return int(np.asarray(self.berth_capacity).size)


@dataclass(frozen=True)
class AbsorptionResult:
    horizon: int
    maximum_absorbable: float
    arrivals: np.ndarray
    state_path: np.ndarray
    service_path: np.ndarray


class AbsorptionBoundaryLP:
    """Optimistic, non-work-conserving four-stage time-expanded LP."""

    _A, _SB, _SY, _SG, _SM, _B, _Y, _G, _M = range(9)

    @classmethod
    def solve(cls, data: AbsorptionInput) -> AbsorptionResult:
        horizon = data.horizon
        if horizon <= 0:
            raise ContractError("Absorption horizon must be positive")
        capacities = [
            np.asarray(data.berth_capacity, dtype=float),
            np.asarray(data.yard_capacity, dtype=float),
            np.asarray(data.gate_capacity, dtype=float),
            np.asarray(data.optimistic_corridor_capacity, dtype=float),
        ]
        if any(values.shape != (horizon,) for values in capacities):
            raise ContractError("Every capacity envelope must cover the full horizon")
        if any(np.any(values < 0) or np.any(~np.isfinite(values)) for values in capacities):
            raise ContractError("Capacity envelopes must be finite and nonnegative")
        thresholds = np.asarray(
            [data.berth_threshold, data.yard_threshold, data.gate_threshold, data.corridor_threshold],
            dtype=float,
        )
        initial = np.asarray(
            [data.initial_berth, data.initial_yard, data.initial_gate, data.initial_corridor],
            dtype=float,
        )
        if np.any(thresholds <= 0) or np.any(initial < 0) or np.any(initial > thresholds):
            raise ContractError("Boundary LP needs a feasible observed initial state and positive thresholds")

        size = 9 * horizon
        index = lambda t, component: 9 * t + component
        objective = np.zeros(size, dtype=float)
        for t in range(horizon):
            objective[index(t, cls._A)] = -1.0
        bounds: list[tuple[float, float | None]] = [(0.0, None)] * size
        for t in range(horizon):
            for component, capacity in zip(
                (cls._SB, cls._SY, cls._SG, cls._SM), capacities
            ):
                bounds[index(t, component)] = (0.0, float(capacity[t]))
            for component, threshold in zip((cls._B, cls._Y, cls._G, cls._M), thresholds):
                bounds[index(t, component)] = (0.0, float(threshold))

        equalities: list[np.ndarray] = []
        equality_rhs: list[float] = []
        inequalities: list[np.ndarray] = []
        inequality_rhs: list[float] = []
        for t in range(horizon):
            current_components = (cls._B, cls._Y, cls._G, cls._M)
            current_constants = initial

            # B+ = B + A - SB
            row = np.zeros(size)
            row[index(t, cls._B)] = 1
            row[index(t, cls._A)] = -1
            row[index(t, cls._SB)] = 1
            if t == 0:
                rhs = initial[0]
            else:
                row[index(t - 1, cls._B)] = -1
                rhs = 0.0
            equalities.append(row)
            equality_rhs.append(rhs)

            # Y+ = Y + SB - SY, then G+, then M+
            for next_component, inflow, outflow, initial_value in (
                (cls._Y, cls._SB, cls._SY, initial[1]),
                (cls._G, cls._SY, cls._SG, initial[2]),
                (cls._M, cls._SG, cls._SM, initial[3]),
            ):
                row = np.zeros(size)
                row[index(t, next_component)] = 1
                row[index(t, inflow)] = -1
                row[index(t, outflow)] = 1
                if t == 0:
                    rhs = initial_value
                else:
                    row[index(t - 1, next_component)] = -1
                    rhs = 0.0
                equalities.append(row)
                equality_rhs.append(rhs)

            # Service may be less than capacity but cannot exceed pre-service workload.
            row = np.zeros(size)
            row[index(t, cls._SB)] = 1
            row[index(t, cls._A)] = -1
            if t == 0:
                rhs = initial[0]
            else:
                row[index(t - 1, cls._B)] = -1
                rhs = 0.0
            inequalities.append(row)
            inequality_rhs.append(rhs)
            for service, state_component, initial_value in (
                (cls._SY, cls._Y, initial[1]),
                (cls._SG, cls._G, initial[2]),
                (cls._SM, cls._M, initial[3]),
            ):
                row = np.zeros(size)
                row[index(t, service)] = 1
                if t == 0:
                    rhs = initial_value
                else:
                    row[index(t - 1, state_component)] = -1
                    rhs = 0.0
                inequalities.append(row)
                inequality_rhs.append(rhs)

        result = linprog(
            objective,
            A_ub=np.vstack(inequalities),
            b_ub=np.asarray(inequality_rhs),
            A_eq=np.vstack(equalities),
            b_eq=np.asarray(equality_rhs),
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            raise NumericalFailure(f"Absorption boundary LP failed: {result.message}")
        vector = np.asarray(result.x, dtype=float).reshape(horizon, 9)
        return AbsorptionResult(
            horizon=horizon,
            maximum_absorbable=float(vector[:, cls._A].sum()),
            arrivals=vector[:, cls._A],
            state_path=vector[:, [cls._B, cls._Y, cls._G, cls._M]],
            service_path=vector[:, [cls._SB, cls._SY, cls._SG, cls._SM]],
        )

    @classmethod
    def first_violating_horizon(
        cls,
        inputs: Sequence[AbsorptionInput],
        committed_arrivals: Sequence[float],
        *,
        tolerance: float,
    ) -> int | None:
        if len(inputs) != len(committed_arrivals):
            raise ContractError("Each candidate horizon needs its committed-arrival quantity")
        for horizon, (data, committed) in enumerate(zip(inputs, committed_arrivals), start=1):
            capacity = cls.solve(data).maximum_absorbable
            if committed > capacity + tolerance:
                return horizon
        return None


@dataclass(frozen=True)
class LocalJacobianResult:
    jacobian: np.ndarray
    spectral_radius: float
    loading_inverse_condition: float


def local_selected_branch_jacobian(
    *,
    d_phi_d_state: np.ndarray,
    d_phi_d_choice: np.ndarray,
    d_loading_d_choice: np.ndarray,
    d_loading_d_state: np.ndarray,
) -> LocalJacobianResult:
    identity = np.eye(d_loading_d_choice.shape[0])
    matrix = identity - d_loading_d_choice
    condition = float(np.linalg.cond(matrix))
    try:
        sensitivity = np.linalg.solve(matrix, d_loading_d_state)
    except np.linalg.LinAlgError as exc:
        raise NumericalFailure("The fixed-branch loading Jacobian is singular") from exc
    jacobian = d_phi_d_state + d_phi_d_choice @ sensitivity
    radius = float(np.max(np.abs(np.linalg.eigvals(jacobian))))
    return LocalJacobianResult(jacobian, radius, condition)


@dataclass(frozen=True)
class ThresholdSnapshot:
    period: int
    gateway_crossed: frozenset[str]
    corridor_crossed: frozenset[str]


@dataclass(frozen=True)
class CascadeStatistics:
    first_gateway_time: int | None
    propagation_time: int | None
    first_gateway: str | None
    first_gateway_right_censored: bool
    propagation_right_censored: bool


def ex_post_cascade_statistics(
    *,
    disruption_start: int,
    realized: Sequence[ThresholdSnapshot],
    matched_no_disruption: Sequence[ThresholdSnapshot],
) -> CascadeStatistics:
    realized_periods = [snapshot.period for snapshot in realized]
    matched_periods = [snapshot.period for snapshot in matched_no_disruption]
    if len(set(realized_periods)) != len(realized_periods) or len(
        set(matched_periods)
    ) != len(matched_periods):
        raise ContractError("Cascade diagnostics require one snapshot per path and period")
    realized_post = sorted(
        (snapshot for snapshot in realized if snapshot.period >= disruption_start),
        key=lambda snapshot: snapshot.period,
    )
    matched_post = {
        snapshot.period: snapshot
        for snapshot in matched_no_disruption
        if snapshot.period >= disruption_start
    }
    if {snapshot.period for snapshot in realized_post} != set(matched_post):
        raise ContractError(
            "Attributable cascade timing requires completed, period-matched paths"
        )
    attributable: list[ThresholdSnapshot] = []
    for snapshot in realized_post:
        base = matched_post[snapshot.period]
        attributable.append(
            ThresholdSnapshot(
                snapshot.period,
                snapshot.gateway_crossed - base.gateway_crossed,
                snapshot.corridor_crossed - base.corridor_crossed,
            )
        )
    first_snapshot = next((item for item in attributable if item.gateway_crossed), None)
    if first_snapshot is None:
        return CascadeStatistics(None, None, None, True, True)
    first_gateway = sorted(first_snapshot.gateway_crossed)[0]
    propagation = next(
        (
            item
            for item in attributable
            if item.period >= first_snapshot.period
            and ((item.gateway_crossed - {first_gateway}) or item.corridor_crossed)
        ),
        None,
    )
    return CascadeStatistics(
        first_gateway_time=first_snapshot.period,
        propagation_time=propagation.period if propagation else None,
        first_gateway=first_gateway,
        first_gateway_right_censored=False,
        propagation_right_censored=propagation is None,
    )
