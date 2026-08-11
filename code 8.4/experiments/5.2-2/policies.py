"""Common-authority policy definitions and formal MPC/BC-SAC selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from tre84.actions import Action
from tre84.control import MPCCandidate, ProjectedStochasticMPC, TwoProposalSelector
from tre84.scenarios import ScenarioBundle
from tre84.state import ModelState

from features import LinearActor
from model import BenchmarkModel
from simulator import PolicyDecision


def _zero_continuation(model: BenchmarkModel):
    return lambda state, offset, path: model.zero_action()


def _reactive_normalised(state: ModelState, model: BenchmarkModel) -> np.ndarray:
    threshold_ratio = float(model.config["reactive_policy"]["pressure_threshold_ratio"])
    pressure_by_resource: dict[Any, float] = {}
    for resource, threshold in model.thresholds.items():
        if resource.stage.value == "berth":
            mapping = state.berth
        elif resource.stage.value == "yard":
            mapping = state.yard
        elif resource.stage.value == "gate":
            mapping = state.gate
        else:
            mapping = state.corridor
        mass = 0.0
        for tag, amount in mapping.items():
            route = model.network.route(tag.route)
            location = route.corridor if resource.stage.value == "corridor" else route.gateway
            if location == resource.location:
                mass += amount
        pressure_by_resource[resource] = mass / threshold
    response_by_resource = {
        resource: max(pressure - threshold_ratio, 0.0)
        / (1.0 + max(pressure - threshold_ratio, 0.0))
        for resource, pressure in pressure_by_resource.items()
    }
    waiting_scale = max(sum(model.gateway_scales.values()), 1.0)
    waiting_ratio = state.waiting_mass() / waiting_scale
    release = waiting_ratio / (1.0 + waiting_ratio)
    lead_risk = float(state.risk.lead_time_forecast[-1])
    current_risk = float(state.risk.belief[-1])
    resource_count = len(model.controlled_resources)
    readiness = np.asarray(
        [lead_risk * (1.0 - response_by_resource[resource]) for resource in model.controlled_resources]
    )
    direct = np.asarray([response_by_resource[resource] for resource in model.controlled_resources])
    exercise = np.asarray(
        [
            min(
                state.readiness.stock.get(resource, 0.0)
                / max(model.action_upper[2 * resource_count + index], 1e-12),
                response_by_resource[resource],
            )
            for index, resource in enumerate(model.controlled_resources)
        ]
    )
    disclosure = []
    for route_id in sorted(model.network.routes):
        route = model.network.route(route_id)
        route_pressure = max(
            pressure
            for resource, pressure in pressure_by_resource.items()
            if resource.location in {route.gateway, route.corridor}
        )
        route_overload = max(route_pressure - threshold_ratio, 0.0) / (
            1.0 + max(route_pressure - threshold_ratio, 0.0)
        )
        disclosure.append(max(current_risk, route_overload))
    return np.concatenate(
        [readiness, direct, exercise, np.asarray([release]), np.asarray(disclosure)]
    )


class PassivePolicy:
    name = "Passive"
    training_seed = None

    def __init__(self, model: BenchmarkModel) -> None:
        self.model = model

    def decide(self, **kwargs) -> PolicyDecision:
        return PolicyDecision(self.model.zero_action(), "zero_coordination_action")


class ReactivePolicy:
    name = "Reactive"
    training_seed = None

    def __init__(self, model: BenchmarkModel) -> None:
        self.model = model

    def decide(self, *, state: ModelState, **kwargs) -> PolicyDecision:
        return PolicyDecision(
            self.model.action_from_normalised(_reactive_normalised(state, self.model)),
            "current_state_formula",
        )


def _candidate_profiles(model: BenchmarkModel) -> tuple[MPCCandidate, ...]:
    block_sizes = [
        len(model.layout.readiness_order),
        len(model.layout.direct_order),
        len(model.layout.readiness_exercise),
        len(model.layout.release),
        len(model.layout.disclosure),
    ]

    def direction(block_index: int, level: float = 1.0) -> np.ndarray:
        vector = np.zeros(len(model.layout.keys), dtype=float)
        start = sum(block_sizes[:block_index])
        vector[start : start + block_sizes[block_index]] = level
        return vector

    levels = tuple(float(value) for value in model.config["mpc"]["normalised_levels"])
    if levels != (0.0, 0.5, 1.0):
        raise ValueError("The frozen MPC grid must retain endpoint-midpoint levels")
    high = levels[-1]
    middle = levels[1]
    profiles = {
        "zero": np.zeros(len(model.layout.keys), dtype=float),
        "readiness_full": direction(0, high),
        "direct_full": direction(1, high),
        "exercise_full": direction(2, high),
        "release_full": direction(3, high),
        "disclosure_full": direction(4, high),
        "balanced_mid": np.full(len(model.layout.keys), middle, dtype=float),
    }
    expected = tuple(model.config["mpc"]["candidate_profiles"])
    if tuple(profiles) != expected:
        raise ValueError("MPC candidate profiles differ from the frozen manifest")
    continuation = lambda state, offset, path: model.action_from_normalised(
        _reactive_normalised(state, model)
    )
    return tuple(
        MPCCandidate(name, model.action_from_normalised(np.asarray(values)), continuation)
        for name, values in profiles.items()
    )


def build_mpc(model: BenchmarkModel) -> ProjectedStochasticMPC:
    return ProjectedStochasticMPC(
        kernel=model.kernel,
        projector=model.projector,
        lookahead=int(model.config["mpc"]["control_horizon_weeks"]),
        terminal_value=model.terminal_cost.compute,
    )


class MPCPolicy:
    name = "Projected stochastic MPC"
    training_seed = None

    def __init__(self, model: BenchmarkModel) -> None:
        self.model = model
        self.mpc = build_mpc(model)

    def decide(
        self, *, state: ModelState, bundle: ScenarioBundle, **kwargs
    ) -> PolicyDecision:
        candidates = _candidate_profiles(self.model)
        result = self.mpc.solve(state=state, bundle=bundle, candidates=candidates)
        selected = next(item for item in candidates if item.candidate_id == result.candidate_id)
        return PolicyDecision(
            selected.first_raw_action,
            f"formal_mpc:{result.candidate_id}",
            {
                "mpc_objective": result.objective,
                "valid_candidates": sum(item.valid for item in result.evaluations),
                "failed_candidates": sum(not item.valid for item in result.evaluations),
            },
        )


@dataclass
class ActorPolicy:
    name: str
    model: BenchmarkModel
    actor: LinearActor
    training_seed: int

    def decide(self, *, state: ModelState, **kwargs) -> PolicyDecision:
        return PolicyDecision(
            self.actor.raw_action(state, self.model),
            f"frozen_checkpoint:{self.name}",
        )


class ModelGuidedPolicy:
    name = "Model-guided constrained SAC"

    def __init__(
        self,
        *,
        model: BenchmarkModel,
        bc_actor: LinearActor,
        sac_actor: LinearActor,
        training_seed: int,
    ) -> None:
        self.model = model
        self.bc_actor = bc_actor
        self.sac_actor = sac_actor
        self.training_seed = training_seed
        self.mpc = build_mpc(model)
        continuation = lambda state, offset, path: model.action_from_normalised(
            _reactive_normalised(state, model)
        )
        self.selector = TwoProposalSelector(
            mpc_evaluator=self.mpc,
            fallback_raw_action=lambda state: model.zero_action(),
            continuation=continuation,
        )

    def decide(
        self, *, state: ModelState, bundle: ScenarioBundle, **kwargs
    ) -> PolicyDecision:
        bc = self.bc_actor.raw_action(state, self.model)
        sac = self.sac_actor.raw_action(state, self.model)
        selection = self.selector.select(
            state=state,
            bundle=bundle,
            bc_raw_action=bc,
            sac_raw_action=sac,
        )
        if selection.source == "BC":
            raw = bc
        elif selection.source == "SAC":
            raw = sac
        else:
            raw = self.model.zero_action()
        records = []
        for source, evaluation in selection.evaluations.items():
            records.append(
                {
                    "proposal_source": source,
                    "selected": source == selection.source,
                    "rejected": source != selection.source,
                    "nested_objective": evaluation.objective,
                    "solver_valid": evaluation.valid,
                    "solver_failure": evaluation.failure or "",
                    "used_fallback": selection.used_fallback,
                    "common_scenario_path_ids": "|".join(path.path_id for path in bundle.paths),
                    "common_kernel": "RC-MSA + complete tagged transition + formal objective",
                }
            )
        return PolicyDecision(raw, f"bc_sac_selector:{selection.source}", proposal_records=records)


def mpc_teacher_action(
    *, model: BenchmarkModel, state: ModelState, bundle: ScenarioBundle
) -> tuple[Action, str, float]:
    mpc = build_mpc(model)
    result = mpc.solve(state=state, bundle=bundle, candidates=_candidate_profiles(model))
    return result.action, result.candidate_id, result.objective
