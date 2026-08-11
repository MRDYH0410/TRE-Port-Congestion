"""Module 4: oldest-first release and source/vintage route-wait-exit SUE."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Mapping, Sequence

import numpy as np

from .errors import ContractError
from .keys import SourceKey

WAIT = "__WAIT__"
EXIT = "__EXIT__"


def oldest_first(total_release: float, vintages: np.ndarray) -> np.ndarray:
    """Exhaust older vintages before younger ones; index increases with age."""

    stock = np.asarray(vintages, dtype=float)
    if stock.ndim != 1 or np.any(stock < 0) or np.any(~np.isfinite(stock)):
        raise ContractError("Oldest-first needs a finite nonnegative vintage vector")
    if total_release < 0 or total_release > float(stock.sum()) + 1e-12:
        raise ContractError("Requested waiting release is outside the available stock")
    release = np.zeros_like(stock)
    remaining = float(total_release)
    for age in range(stock.size - 1, -1, -1):
        amount = min(float(stock[age]), remaining)
        release[age] = amount
        remaining -= amount
        if remaining <= 1e-14:
            break
    return release


@dataclass(frozen=True)
class DecisionMasses:
    masses: Mapping[SourceKey, float]
    releases: Mapping[str, np.ndarray]


def build_decision_masses(
    waiting: Mapping[str, np.ndarray],
    new_decision_mass: Mapping[str, float],
    release_rates: Mapping[str, float],
) -> DecisionMasses:
    masses: dict[SourceKey, float] = {}
    releases: dict[str, np.ndarray] = {}
    cargo_classes = sorted(set(waiting) | set(new_decision_mass))
    for cargo_class in cargo_classes:
        stock = np.asarray(waiting.get(cargo_class, np.zeros(1)), dtype=float)
        rho = float(release_rates.get(cargo_class, 0.0))
        if not 0 <= rho <= 1:
            raise ContractError("Release rates must lie in [0, 1]")
        release = oldest_first(rho * float(stock.sum()), stock)
        releases[cargo_class] = release
        # Keep the new source even when its mass is zero so clearance periods
        # retain a well-defined (possibly zero-loaded) RC-MSA layout.  Do not
        # add zero-release vintages: they stay in the physical waiting vector
        # and age normally, but have no route/wait/exit decision this period.
        masses[SourceKey(cargo_class, None)] = float(new_decision_mass.get(cargo_class, 0.0))
        for vintage, amount in enumerate(release):
            if amount > 0.0:
                masses[SourceKey(cargo_class, vintage)] = float(amount)
    if any(value < 0 or not np.isfinite(value) for value in masses.values()):
        raise ContractError("Decision source masses must be finite and nonnegative")
    return DecisionMasses(masses=masses, releases=releases)


@dataclass(frozen=True)
class FrozenDisclosure:
    """The signal, reference, error scale, and intensity are frozen before RC-MSA."""

    public_signal: Mapping[tuple[str, str], float]
    reference_forecast: Mapping[tuple[str, str], float]
    error_scale: Mapping[tuple[str, str], float]
    intensity: Mapping[tuple[str, str], float]
    gamma: float

    def __post_init__(self) -> None:
        if not 0 <= self.gamma <= 1:
            raise ContractError("Disclosure credibility gamma must lie in [0, 1]")
        for key, signal in self.public_signal.items():
            if key not in self.reference_forecast or key not in self.error_scale:
                raise ContractError("Every public signal needs a frozen reference and error scale")
            bound = self.gamma * self.error_scale[key]
            if bound < 0 or abs(signal - self.reference_forecast[key]) > bound + 1e-12:
                raise ContractError(f"Public signal {key} lies outside its credibility box")
            intensity = self.intensity.get(key, 0.0)
            if not 0 <= intensity <= 1:
                raise ContractError("Disclosure intensities must lie in [0, 1]")

    @classmethod
    def clip_from_reference(
        cls,
        *,
        raw_signal: Mapping[tuple[str, str], float],
        reference_forecast: Mapping[tuple[str, str], float],
        error_scale: Mapping[tuple[str, str], float],
        intensity: Mapping[tuple[str, str], float],
        gamma: float,
    ) -> "FrozenDisclosure":
        public: dict[tuple[str, str], float] = {}
        for key, raw in raw_signal.items():
            reference = float(reference_forecast[key])
            delta = gamma * float(error_scale[key])
            public[key] = float(np.clip(raw, reference - delta, reference + delta))
        return cls(public, reference_forecast, error_scale, intensity, gamma)


@dataclass(frozen=True)
class BehaviorCostParameters:
    theta: Mapping[str, float]
    route_private_resource: Mapping[tuple[str, str], float]
    route_market_cost: Mapping[tuple[str, str], float]
    value_of_time: Mapping[str, float]
    waiting_base: Mapping[str, float]
    waiting_age_cost: Mapping[tuple[str, int], float]
    waiting_inventory_cost: Mapping[str, float]
    waiting_scale: Mapping[str, float]
    waiting_reclosure_cost: Mapping[str, float]
    continuation_value: Mapping[tuple[str, int], float]
    direct_exit_cost: Mapping[SourceKey, float]
    late_exit_cost: Mapping[tuple[str, int], float]
    hazard: Mapping[str, np.ndarray]

    def validate(self, waiting: Mapping[str, np.ndarray]) -> None:
        for cargo_class, stock in waiting.items():
            hazard = np.asarray(self.hazard[cargo_class], dtype=float)
            if hazard.shape != np.asarray(stock).shape:
                raise ContractError("Waiting stock and attrition hazard need the same vintages")
            if np.any(np.diff(hazard) < 0) or np.any((hazard < 0) | (hazard > 1)):
                raise ContractError("Attrition hazards must be nondecreasing probabilities")
            if not np.isclose(hazard[-1], 1.0):
                raise ContractError("The maximum waiting vintage must have terminal hazard one")
            if self.theta[cargo_class] <= 0 or self.waiting_scale[cargo_class] <= 0:
                raise ContractError("Logit sensitivity and waiting scale must be positive")


PrivateWaitingOracle = Callable[
    [Mapping[SourceKey, Mapping[str, float]]], Mapping[tuple[str, str], float]
]


@dataclass
class BehaviorProblem:
    decision: DecisionMasses
    waiting_state: Mapping[str, np.ndarray]
    routes_by_class: Mapping[str, tuple[str, ...]]
    disclosure: FrozenDisclosure
    parameters: BehaviorCostParameters
    reclosure_probability: float
    private_waiting_oracle: PrivateWaitingOracle

    def __post_init__(self) -> None:
        self.parameters.validate(self.waiting_state)
        if not 0 <= self.reclosure_probability <= 1:
            raise ContractError("Reclosure probability must lie in [0, 1]")
        for source in self.decision.masses:
            if source.cargo_class not in self.routes_by_class:
                raise ContractError(f"Missing current route set for {source.cargo_class}")

    @property
    def sources(self) -> tuple[SourceKey, ...]:
        return tuple(
            sorted(
                self.decision.masses,
                key=lambda source: (
                    source.cargo_class,
                    source.vintage is not None,
                    -1 if source.vintage is None else source.vintage,
                ),
            )
        )

    def choices(self, source: SourceKey) -> tuple[str, ...]:
        return tuple(self.routes_by_class[source.cargo_class]) + (WAIT, EXIT)

    def projected_waiting_out(
        self, flows: Mapping[SourceKey, Mapping[str, float]], cargo_class: str
    ) -> float:
        stock = np.asarray(self.waiting_state[cargo_class], dtype=float)
        release = np.asarray(self.decision.releases[cargo_class], dtype=float)
        hazard = np.asarray(self.parameters.hazard[cargo_class], dtype=float)
        new_wait = flows.get(SourceKey(cargo_class, None), {}).get(WAIT, 0.0)
        renewed = np.zeros_like(stock)
        for source, source_flows in flows.items():
            if source.cargo_class == cargo_class and source.vintage is not None:
                renewed[source.vintage] = source_flows.get(WAIT, 0.0)
        surviving_old = float(np.dot(1.0 - hazard, stock - release + renewed))
        return float(new_wait + surviving_old)

    def costs(
        self, flows: Mapping[SourceKey, Mapping[str, float]]
    ) -> dict[SourceKey, dict[str, float]]:
        private_waiting = self.private_waiting_oracle(flows)
        projected = {
            cargo_class: self.projected_waiting_out(flows, cargo_class)
            for cargo_class in self.waiting_state
        }
        result: dict[SourceKey, dict[str, float]] = {}
        for source in self.sources:
            cargo_class = source.cargo_class
            if self.decision.masses[source] <= 0:
                result[source] = {}
                continue
            source_costs: dict[str, float] = {}
            for route in self.routes_by_class[cargo_class]:
                key = (cargo_class, route)
                intensity = float(self.disclosure.intensity.get(key, 0.0))
                perceived = (1.0 - intensity) * float(private_waiting[key]) + intensity * float(
                    self.disclosure.public_signal[key]
                )
                source_costs[route] = (
                    float(self.parameters.route_private_resource[key])
                    + float(self.parameters.route_market_cost[key])
                    + float(self.parameters.value_of_time[cargo_class]) * perceived
                )

            if source.is_new:
                age = 0
                next_age = 0
                attrition_probability = 0.0
            else:
                assert source.vintage is not None
                age = source.vintage + 1
                next_age = source.vintage + 1
                attrition_probability = float(
                    self.parameters.hazard[cargo_class][source.vintage]
                )
            resource_wait = (
                float(self.parameters.waiting_base[cargo_class])
                + float(self.parameters.waiting_age_cost[(cargo_class, age)])
                + float(self.parameters.waiting_inventory_cost[cargo_class])
                * projected[cargo_class]
                / self.parameters.waiting_scale[cargo_class]
                + float(self.parameters.waiting_reclosure_cost[cargo_class])
                * self.reclosure_probability
            )
            exit_cost = float(self.parameters.direct_exit_cost[source])
            late_cost = 0.0 if source.is_new else float(
                self.parameters.late_exit_cost[(cargo_class, source.vintage)]
            )
            max_vintage = len(self.parameters.hazard[cargo_class]) - 1
            continuation = (
                0.0
                if next_age > max_vintage
                else float(self.parameters.continuation_value[(cargo_class, next_age)])
            )
            source_costs[WAIT] = (
                resource_wait
                + attrition_probability * (exit_cost + late_cost)
                + (1.0 - attrition_probability) * continuation
            )
            source_costs[EXIT] = exit_cost
            result[source] = source_costs
        return result


@dataclass(frozen=True)
class RCMSASettings:
    tolerance: float
    max_iterations: int
    deduplication_tolerance: float
    step_multipliers: tuple[float, ...] = (1.0, 2.0, 4.0)

    def __post_init__(self) -> None:
        if self.tolerance <= 0 or self.max_iterations <= 0 or self.deduplication_tolerance <= 0:
            raise ContractError("RC-MSA settings must be positive")
        if not self.step_multipliers or any(value <= 0 for value in self.step_multipliers):
            raise ContractError("RC-MSA step multipliers must be positive")


@dataclass(frozen=True)
class StartRecord:
    name: str
    residual: float
    iterations: int
    converged: bool
    selected_step_multipliers: tuple[float, ...] = ()


@dataclass
class EquilibriumResult:
    flows: dict[SourceKey, dict[str, float]]
    releases: dict[str, np.ndarray]
    route_dispatch: dict[tuple[str, str], float]
    renewed_waiting: dict[SourceKey, float]
    direct_exit: dict[SourceKey, float]
    normalized_shares: dict[SourceKey, dict[str, float]]
    residual: float
    kl_discrepancy: float
    multi_start_dispersion: float
    iterations: int
    status: str
    starts: tuple[StartRecord, ...]
    selected_start: str = ""
    selected_step_multipliers: tuple[float, ...] = ()


class RCMSASolver:
    """Multi-start residual-controlled MSA with deterministic selection."""

    def __init__(self, settings: RCMSASettings) -> None:
        self.settings = settings

    @staticmethod
    def _layout(problem: BehaviorProblem) -> tuple[dict[SourceKey, slice], int]:
        slices: dict[SourceKey, slice] = {}
        cursor = 0
        for source in problem.sources:
            width = len(problem.choices(source))
            slices[source] = slice(cursor, cursor + width)
            cursor += width
        return slices, cursor

    @staticmethod
    def _to_mapping(
        problem: BehaviorProblem, slices: Mapping[SourceKey, slice], vector: np.ndarray
    ) -> dict[SourceKey, dict[str, float]]:
        return {
            source: {
                choice: float(value)
                for choice, value in zip(problem.choices(source), vector[slices[source]])
            }
            for source in problem.sources
        }

    def _loading(
        self,
        problem: BehaviorProblem,
        slices: Mapping[SourceKey, slice],
        vector: np.ndarray,
    ) -> np.ndarray:
        flows = self._to_mapping(problem, slices, vector)
        costs = problem.costs(flows)
        loaded = np.zeros_like(vector)
        for source in problem.sources:
            mass = float(problem.decision.masses[source])
            if mass == 0:
                continue
            choice_cost = np.asarray(
                [costs[source][choice] for choice in problem.choices(source)], dtype=float
            )
            logits = -float(problem.parameters.theta[source.cargo_class]) * choice_cost
            logits -= float(np.max(logits))
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            loaded[slices[source]] = mass * probabilities
        return loaded

    def _residual(
        self,
        problem: BehaviorProblem,
        slices: Mapping[SourceKey, slice],
        vector: np.ndarray,
    ) -> float:
        demand = float(sum(problem.decision.masses.values()))
        if demand == 0:
            return 0.0
        return float(np.abs(self._loading(problem, slices, vector) - vector).sum() / (2 * demand))

    @staticmethod
    def _shares(
        problem: BehaviorProblem,
        slices: Mapping[SourceKey, slice],
        vector: np.ndarray,
    ) -> dict[SourceKey, dict[str, float]]:
        shares: dict[SourceKey, dict[str, float]] = {}
        for source in problem.sources:
            mass = float(problem.decision.masses[source])
            if mass <= 0:
                continue
            # ``normalized_shares`` is a probability simplex used only as the
            # next-period historical start.  At ordinary scales every RC-MSA
            # iterate conserves the source mass and division by ``mass`` is
            # equivalent.  For an extremely old waiting vintage, however,
            # source flows can be IEEE-754 subnormal values: convex updates may
            # underflow selected alternatives independently, so dividing by
            # the pre-update source mass need not sum to one.  Normalise the
            # represented source-flow slice itself.  This neither drops nor
            # reallocates physical flow; it only preserves the Chapter 4
            # master-choice simplex used by the deterministic historical tie
            # break.  No empirical or solver threshold is introduced.
            source_flow = np.asarray(vector[slices[source]], dtype=float)
            represented = float(source_flow.sum())
            if represented > 0.0:
                probabilities = source_flow / represented
                probabilities /= float(probabilities.sum())
            else:
                probabilities = np.full(len(problem.choices(source)), 1.0 / len(problem.choices(source)))
            shares[source] = {
                choice: float(value)
                for choice, value in zip(problem.choices(source), probabilities)
            }
        return shares

    def _previous_start(
        self,
        problem: BehaviorProblem,
        slices: Mapping[SourceKey, slice],
        previous_shares: Mapping[SourceKey, Mapping[str, float]],
    ) -> np.ndarray:
        vector = np.zeros(max(section.stop for section in slices.values()), dtype=float)
        for source in problem.sources:
            mass = float(problem.decision.masses[source])
            choices = problem.choices(source)
            prior = np.asarray([previous_shares.get(source, {}).get(choice, 0.0) for choice in choices])
            if prior.sum() <= 0:
                prior = np.ones(len(choices), dtype=float)
            prior /= prior.sum()
            vector[slices[source]] = mass * prior
        return vector

    @staticmethod
    def _lexicographic_key(vector: np.ndarray) -> tuple[float, ...]:
        return tuple(np.round(vector, decimals=14).tolist())

    def _distance_to_previous(
        self,
        problem: BehaviorProblem,
        slices: Mapping[SourceKey, slice],
        vector: np.ndarray,
        previous_shares: Mapping[SourceKey, Mapping[str, float]],
    ) -> float:
        distance = 0.0
        compared = 0
        shares = self._shares(problem, slices, vector)
        for source, candidate in shares.items():
            if source not in previous_shares:
                continue
            # Sel_t is defined on the complete master choice support.  A route
            # that was used in the preceding solution can be absent from the
            # current feasible simplex; its current share is then exactly zero,
            # but the resulting historical distance must still be counted.
            # The union is the smallest lossless representation of that master
            # support available at this interface (all other choices are
            # zero in both vectors and contribute nothing).
            master_choices = set(candidate) | set(previous_shares[source])
            for choice in master_choices:
                distance += abs(candidate.get(choice, 0.0) - previous_shares[source].get(choice, 0.0))
            compared += 1
        return distance if compared else 0.0

    def _normalized_distance(
        self,
        problem: BehaviorProblem,
        slices: Mapping[SourceKey, slice],
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        """Maximum total-variation distance across positive-mass source simplexes.

        RC-MSA fixed points are deduplicated on normalized master choice shares,
        not on demand-weighted flow vectors.  Using a per-source maximum also
        prevents a large cargo class from hiding a distinct fixed point for a
        smaller class.
        """

        distance = 0.0
        for source in problem.sources:
            mass = float(problem.decision.masses[source])
            if mass <= 0.0:
                continue
            section = slices[source]
            distance = max(
                distance,
                float(np.abs(left[section] / mass - right[section] / mass).sum() / 2.0),
            )
        return distance

    def solve(
        self,
        problem: BehaviorProblem,
        *,
        previous_shares: Mapping[SourceKey, Mapping[str, float]],
    ) -> EquilibriumResult:
        slices, size = self._layout(problem)
        demand = float(sum(problem.decision.masses.values()))
        if demand == 0:
            zero = np.zeros(size, dtype=float)
            return self._build_result(
                problem, slices, zero, 0.0, 0.0, 0, "converged", (), 0.0
            )

        uniform = np.zeros(size, dtype=float)
        for source in problem.sources:
            mass = float(problem.decision.masses[source])
            uniform[slices[source]] = mass / len(problem.choices(source))
        previous = self._previous_start(problem, slices, previous_shares)
        # The registered free-flow start is a Logit loading evaluated at zero
        # adaptive flow, not a loading of the dispersed/uniform assignment.
        free_flow = self._loading(problem, slices, np.zeros(size, dtype=float))
        starts = (("previous", previous), ("free_flow", free_flow), ("dispersed", uniform))

        best_vectors: list[tuple[np.ndarray, int]] = []
        records: list[StartRecord] = []
        for name, initial in starts:
            current = initial.copy()
            best = current.copy()
            best_residual = self._residual(problem, slices, current)
            converged = best_residual <= self.settings.tolerance
            iterations = 0
            selected_steps: list[float] = []
            for iteration in range(self.settings.max_iterations):
                iterations = iteration
                residual = self._residual(problem, slices, current)
                if residual < best_residual:
                    best, best_residual = current.copy(), residual
                if residual <= self.settings.tolerance:
                    converged = True
                    best, best_residual = current.copy(), residual
                    break
                loading = self._loading(problem, slices, current)
                candidates: list[tuple[float, float, np.ndarray]] = []
                for multiplier in self.settings.step_multipliers:
                    step = min(1.0, multiplier / (iteration + 1.0))
                    trial = (1.0 - step) * current + step * loading
                    candidates.append((self._residual(problem, slices, trial), multiplier, trial))
                selected_residual, selected_multiplier, current = min(
                    candidates,
                    key=lambda item: (item[0], self._lexicographic_key(item[2])),
                )
                selected_steps.append(float(selected_multiplier))
                # The selected trial is an actual RC-MSA iterate and must be
                # certified even when it was generated by the final allowed
                # update.  The previous implementation silently discarded it.
                if selected_residual < best_residual:
                    best, best_residual = current.copy(), float(selected_residual)
                if selected_residual <= self.settings.tolerance:
                    converged = True
                    break
            final_residual = self._residual(problem, slices, best)
            records.append(
                StartRecord(
                    name,
                    final_residual,
                    iterations + 1,
                    converged,
                    tuple(selected_steps),
                )
            )
            best_vectors.append((best, len(records) - 1))

        qualified: list[tuple[np.ndarray, int]] = []
        for vector, record_index in best_vectors:
            record = records[record_index]
            if record.residual > self.settings.tolerance:
                continue
            if not any(
                self._normalized_distance(problem, slices, vector, existing)
                <= self.settings.deduplication_tolerance
                for existing, _ in qualified
            ):
                qualified.append((vector, record_index))
        pool = qualified or [
            min(
                best_vectors,
                key=lambda item: (
                    records[item[1]].residual,
                    self._lexicographic_key(item[0]),
                    records[item[1]].name,
                ),
            )
        ]
        selected, selected_record_index = min(
            pool,
            key=lambda item: (
                self._distance_to_previous(problem, slices, item[0], previous_shares),
                self._lexicographic_key(item[0]),
                records[item[1]].name,
            ),
        )
        residual = self._residual(problem, slices, selected)
        status = "converged" if qualified else "nonconverged"
        dispersion = 0.0
        for (left, _), (right, _) in combinations(best_vectors, 2):
            dispersion = max(
                dispersion,
                self._normalized_distance(problem, slices, left, right),
            )
        loading = self._loading(problem, slices, selected)
        kl = self._kl(problem, slices, selected, loading)
        selected_record = records[selected_record_index]
        return self._build_result(
            problem,
            slices,
            selected,
            residual,
            kl,
            selected_record.iterations,
            status,
            tuple(records),
            dispersion,
            selected_record.name,
            selected_record.selected_step_multipliers,
        )

    @staticmethod
    def _kl(
        problem: BehaviorProblem,
        slices: Mapping[SourceKey, slice],
        vector: np.ndarray,
        loading: np.ndarray,
    ) -> float:
        total_mass = float(sum(problem.decision.masses.values()))
        if total_mass == 0:
            return 0.0
        value = 0.0
        for source in problem.sources:
            mass = float(problem.decision.masses[source])
            if mass == 0:
                continue
            p = vector[slices[source]] / mass
            q = loading[slices[source]] / mass
            positive = p > 0
            value += mass * float((p[positive] * np.log(p[positive] / q[positive])).sum())
        return value / total_mass

    def _build_result(
        self,
        problem: BehaviorProblem,
        slices: Mapping[SourceKey, slice],
        vector: np.ndarray,
        residual: float,
        kl: float,
        iterations: int,
        status: str,
        records: tuple[StartRecord, ...],
        dispersion: float,
        selected_start: str = "",
        selected_step_multipliers: tuple[float, ...] = (),
    ) -> EquilibriumResult:
        flows = self._to_mapping(problem, slices, vector)
        route_dispatch: dict[tuple[str, str], float] = {}
        renewed_waiting: dict[SourceKey, float] = {}
        direct_exit: dict[SourceKey, float] = {}
        for source, source_flows in flows.items():
            renewed_waiting[source] = source_flows[WAIT]
            direct_exit[source] = source_flows[EXIT]
            for route in problem.routes_by_class[source.cargo_class]:
                key = (source.cargo_class, route)
                route_dispatch[key] = route_dispatch.get(key, 0.0) + source_flows[route]
        return EquilibriumResult(
            flows=flows,
            releases={key: np.asarray(value, dtype=float) for key, value in problem.decision.releases.items()},
            route_dispatch=route_dispatch,
            renewed_waiting=renewed_waiting,
            direct_exit=direct_exit,
            normalized_shares=self._shares(problem, slices, vector),
            residual=residual,
            kl_discrepancy=kl,
            multi_start_dispersion=dispersion,
            iterations=iterations,
            status=status,
            starts=records,
            selected_start=selected_start,
            selected_step_multipliers=selected_step_multipliers,
        )
