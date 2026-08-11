"""Formal five-block action and exact hard-feasibility projection (Module 3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping

import numpy as np
from scipy.optimize import minimize

from .errors import ContractError, InfeasibleAction, NumericalFailure
from .keys import ResourceKey
from .state import ModelState


class Block(str, Enum):
    READINESS_ORDER = "readiness_order"
    DIRECT_ORDER = "direct_order"
    READINESS_EXERCISE = "readiness_exercise"
    RELEASE = "release"
    DISCLOSURE = "disclosure"


@dataclass(frozen=True, order=True)
class ActionKey:
    block: Block
    coordinate: tuple[str, ...]

    @classmethod
    def one(cls, block: Block, coordinate: str) -> "ActionKey":
        return cls(block, (coordinate,))


@dataclass
class Action:
    values: dict[ActionKey, float] = field(default_factory=dict)

    def value(self, key: ActionKey) -> float:
        return float(self.values.get(key, 0.0))

    def block(self, block: Block) -> dict[tuple[str, ...], float]:
        return {key.coordinate: float(value) for key, value in self.values.items() if key.block == block}

    def vector(self, keys: tuple[ActionKey, ...]) -> np.ndarray:
        return np.asarray([self.value(key) for key in keys], dtype=float)

    @classmethod
    def from_vector(cls, keys: tuple[ActionKey, ...], vector: np.ndarray) -> "Action":
        values = np.asarray(vector, dtype=float)
        if values.shape != (len(keys),):
            raise ContractError("Action vector and layout are incompatible")
        return cls({key: float(value) for key, value in zip(keys, values)})


@dataclass(frozen=True)
class ConvexPiecewiseLinearCurve:
    """Breakpoints start at zero; the final slope extends to infinity."""

    breakpoints: tuple[float, ...]
    slopes: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.breakpoints or self.breakpoints[0] != 0:
            raise ContractError("An action cost curve must start at zero")
        if len(self.breakpoints) != len(self.slopes):
            raise ContractError("Each action cost segment needs one slope")
        if any(b < 0 for b in self.breakpoints) or any(
            later <= earlier for earlier, later in zip(self.breakpoints, self.breakpoints[1:])
        ):
            raise ContractError("Action cost breakpoints must be strictly increasing")
        if any(s < 0 for s in self.slopes) or any(
            later < earlier for earlier, later in zip(self.slopes, self.slopes[1:])
        ):
            raise ContractError("Action cost slopes must be nonnegative and nondecreasing")

    def cost(self, quantity: float) -> float:
        x = max(float(quantity), 0.0)
        total = 0.0
        for index, start in enumerate(self.breakpoints):
            end = self.breakpoints[index + 1] if index + 1 < len(self.breakpoints) else x
            width = max(min(x, end) - start, 0.0)
            total += self.slopes[index] * width
            if x <= end:
                break
        return float(total)

    def marginal(self, quantity: float) -> float:
        x = max(float(quantity), 0.0)
        index = int(np.searchsorted(np.asarray(self.breakpoints), x, side="right") - 1)
        return float(self.slopes[max(index, 0)])


@dataclass(frozen=True)
class StockConstraint:
    resource: ResourceKey
    units_per_action: float


@dataclass
class ActionDomain:
    """The phase changes declared bounds but never invents action rights."""

    keys: tuple[ActionKey, ...]
    phase_upper: Mapping[int, Mapping[ActionKey, float]]
    cost_curves: Mapping[ActionKey, ConvexPiecewiseLinearCurve]
    period_budget_cap: Callable[[ModelState], float]
    stock_constraints: Mapping[ActionKey, StockConstraint] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.keys)) != len(self.keys):
            raise ContractError("Action layout keys must be unique")
        for phase in (0, 1):
            if phase not in self.phase_upper:
                raise ContractError("Both observable phases require explicit action bounds")
            missing = set(self.keys) - set(self.phase_upper[phase])
            if missing:
                raise ContractError(f"Missing phase {phase} bounds for {sorted(missing)}")
        for key in self.keys:
            if key not in self.cost_curves:
                raise ContractError(f"Missing declared action cost for {key}")

    def bounds(self, state: ModelState) -> tuple[np.ndarray, np.ndarray]:
        lower = np.zeros(len(self.keys), dtype=float)
        upper = np.asarray([self.phase_upper[state.phase][key] for key in self.keys], dtype=float)
        if np.any(~np.isfinite(upper)) or np.any(upper < 0):
            raise ContractError("Phase-dependent action bounds must be finite and nonnegative")
        for index, key in enumerate(self.keys):
            constraint = self.stock_constraints.get(key)
            if constraint is not None:
                if constraint.units_per_action <= 0:
                    raise ContractError("Readiness consumption units must be positive")
                stock = state.readiness.stock.get(constraint.resource, 0.0)
                upper[index] = min(upper[index], stock / constraint.units_per_action)
        return lower, upper

    def action_cost(self, action: Action) -> float:
        return float(sum(self.cost_curves[key].cost(action.value(key)) for key in self.keys))

    def budget_cap(self, state: ModelState) -> float:
        cap = min(float(state.budget), float(self.period_budget_cap(state)))
        if not np.isfinite(cap) or cap < 0:
            raise ContractError("The period or remaining budget cap is invalid")
        return cap

    def violation(self, action: Action, state: ModelState) -> float:
        if set(action.values) - set(self.keys):
            return float("inf")
        vector = action.vector(self.keys)
        if np.any(~np.isfinite(vector)):
            return float("inf")
        lower, upper = self.bounds(state)
        bound_violation = max(
            float(np.max(np.maximum(lower - vector, 0.0), initial=0.0)),
            float(np.max(np.maximum(vector - upper, 0.0), initial=0.0)),
        )
        budget_violation = max(self.action_cost(action) - self.budget_cap(state), 0.0)
        return max(bound_violation, budget_violation)

    def assert_feasible(self, action: Action, state: ModelState, *, tolerance: float) -> None:
        violation = self.violation(action, state)
        if violation > tolerance:
            raise InfeasibleAction(f"Hard action-feasibility violation is {violation:.6g}")


@dataclass(frozen=True)
class ProjectionResult:
    action: Action
    raw_action: Action
    objective: float
    active_budget: bool
    feasibility_violation: float
    iterations: int


class ActionProjector:
    """Weighted Euclidean projection onto phase, stock, component, and budget bounds."""

    def __init__(
        self,
        domain: ActionDomain,
        *,
        scaling: Mapping[ActionKey, float],
        tolerance: float,
        max_iterations: int,
    ) -> None:
        self.domain = domain
        self.scale = np.asarray([scaling[key] for key in domain.keys], dtype=float)
        if np.any(self.scale <= 0) or np.any(~np.isfinite(self.scale)):
            raise ContractError("Every action block needs a positive finite projection scale")
        if tolerance <= 0 or max_iterations <= 0:
            raise ContractError("Projection settings must be positive")
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def project(self, raw_action: Action, state: ModelState) -> ProjectionResult:
        raw = raw_action.vector(self.domain.keys)
        if np.any(~np.isfinite(raw)):
            raise ContractError("Raw action proposals must be finite")
        lower, upper = self.domain.bounds(state)
        initial = np.clip(raw, lower, upper)
        cap = self.domain.budget_cap(state)

        def objective(vector: np.ndarray) -> float:
            delta = self.scale * (vector - raw)
            return 0.5 * float(delta @ delta)

        def objective_jac(vector: np.ndarray) -> np.ndarray:
            return np.square(self.scale) * (vector - raw)

        def budget_slack(vector: np.ndarray) -> float:
            return cap - self.domain.action_cost(Action.from_vector(self.domain.keys, vector))

        def budget_jac(vector: np.ndarray) -> np.ndarray:
            return -np.asarray(
                [
                    self.domain.cost_curves[key].marginal(value)
                    for key, value in zip(self.domain.keys, vector)
                ],
                dtype=float,
            )

        if budget_slack(initial) >= -self.tolerance:
            vector = initial
            success = True
            message = "box projection"
            iterations = 0
        else:
            result = minimize(
                objective,
                initial,
                jac=objective_jac,
                method="SLSQP",
                bounds=list(zip(lower, upper)),
                constraints=[{"type": "ineq", "fun": budget_slack, "jac": budget_jac}],
                options={"ftol": self.tolerance, "maxiter": self.max_iterations, "disp": False},
            )
            vector = np.asarray(result.x, dtype=float)
            success = bool(result.success)
            message = str(result.message)
            iterations = int(result.nit)
        action = Action.from_vector(self.domain.keys, vector)
        violation = self.domain.violation(action, state)
        if not success or violation > 10 * self.tolerance:
            raise NumericalFailure(
                f"Action projection failed ({message}); feasibility violation={violation:.6g}"
            )
        return ProjectionResult(
            action=action,
            raw_action=raw_action,
            objective=objective(vector),
            active_budget=abs(self.domain.action_cost(action) - cap) <= 10 * self.tolerance,
            feasibility_violation=violation,
            iterations=iterations,
        )

    def local_jacobian(
        self,
        raw_action: Action,
        state: ModelState,
        *,
        projection: ProjectionResult | None = None,
    ) -> np.ndarray:
        """Piecewise active-set derivative of the existing convex projection.

        This is a read-only differentiation interface: it does not alter the
        SLSQP/box projection or any feasibility rule.  It is valid away from
        bound and piecewise-cost active-set ties, which is the subgradient
        convention declared for projected SAC in Chapter 4.
        """

        result = projection if projection is not None else self.project(raw_action, state)
        raw = raw_action.vector(self.domain.keys)
        vector = result.action.vector(self.domain.keys)
        lower, upper = self.domain.bounds(state)
        active_tolerance = max(100.0 * self.tolerance, 1e-9)
        free = (vector > lower + active_tolerance) & (vector < upper - active_tolerance)
        jacobian = np.zeros((len(raw), len(raw)), dtype=float)
        free_indices = np.flatnonzero(free)
        if free_indices.size == 0:
            return jacobian
        jacobian[free_indices, free_indices] = 1.0
        if not result.active_budget:
            return jacobian

        marginal = np.asarray(
            [
                self.domain.cost_curves[key].marginal(value)
                for key, value in zip(self.domain.keys, vector)
            ],
            dtype=float,
        )
        inverse_metric = np.divide(
            1.0,
            np.square(self.scale),
            out=np.zeros_like(self.scale),
            where=self.scale > 0,
        )
        c = marginal[free_indices]
        denominator = float(np.sum(inverse_metric[free_indices] * np.square(c)))
        if denominator <= np.finfo(float).eps:
            return jacobian
        correction = np.outer(inverse_metric[free_indices] * c, c) / denominator
        jacobian[np.ix_(free_indices, free_indices)] -= correction
        return jacobian
