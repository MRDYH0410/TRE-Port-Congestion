"""Declared reference network and the shared Chapter 3/4 execution kernel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tre84.actions import (
    Action,
    ActionDomain,
    ActionKey,
    ActionProjector,
    Block,
    ConvexPiecewiseLinearCurve,
    StockConstraint,
)
from tre84.behavior import (
    BehaviorCostParameters,
    BehaviorProblem,
    FrozenDisclosure,
    RCMSASettings,
    RCMSASolver,
    SourceKey,
    build_decision_masses,
)
from tre84.capacity import (
    CapacityActionMap,
    CapacityDynamics,
    CapacityTechnology,
    ServiceParameters,
)
from tre84.engine import ModelKernel
from tre84.keys import Network, ResourceKey, Route, Stage, Tag
from tre84.loss import (
    LossParameters,
    OperationalLoss,
    TerminalCostParameters,
    TerminalMassCorrection,
)
from tre84.state import CapacityState, ModelState, RiskInformation
from tre84.transition import ExogenousRealization, TaggedTransition, construct_demand_split


CODE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ActionLayout:
    readiness_order: tuple[ActionKey, ...]
    direct_order: tuple[ActionKey, ...]
    readiness_exercise: tuple[ActionKey, ...]
    release: tuple[ActionKey, ...]
    disclosure: tuple[ActionKey, ...]

    @property
    def keys(self) -> tuple[ActionKey, ...]:
        return (
            *self.readiness_order,
            *self.direct_order,
            *self.readiness_exercise,
            *self.release,
            *self.disclosure,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(
            f"{key.block.value}__{'__'.join(key.coordinate)}".replace(" ", "_")
            for key in self.keys
        )


@dataclass
class BenchmarkModel:
    config: Mapping[str, Any]
    network: Network
    resources: tuple[ResourceKey, ...]
    thresholds: Mapping[ResourceKey, float]
    base_capacity: Mapping[ResourceKey, float]
    gateway_scales: Mapping[str, float]
    committed_shares: Mapping[Tag, float]
    route_costs: Mapping[Tag, float]
    route_cost_register: pd.DataFrame
    layout: ActionLayout
    domain: ActionDomain
    projector: ActionProjector
    kernel: ModelKernel
    terminal_cost: TerminalMassCorrection
    controlled_resources: tuple[ResourceKey, ...]
    action_upper: np.ndarray
    initial_budget: float
    waiting_hazard: np.ndarray

    def zero_action(self) -> Action:
        return Action({key: 0.0 for key in self.layout.keys})

    def action_from_normalised(self, values: np.ndarray) -> Action:
        vector = np.clip(np.asarray(values, dtype=float), 0.0, 1.0) * self.action_upper
        return Action.from_vector(self.layout.keys, vector)

    def normalise_action(self, action: Action) -> np.ndarray:
        return np.divide(
            action.vector(self.layout.keys),
            self.action_upper,
            out=np.zeros_like(self.action_upper),
            where=self.action_upper > 0,
        )

    def initial_state(self, first_row: Mapping[str, Any]) -> ModelState:
        high = float(first_row["filtered_high_risk_probability"])
        lead = float(first_row["lead_time_high_risk_probability"])
        cargo = str(self.config["cargo_class"])
        corridor = str(self.config["routes"][0]["corridor"])
        scenario_ids = tuple(str(value) for value in self.config["mpc"]["scenario_ids"])
        scenario_weights = np.full(len(scenario_ids), 1.0 / len(scenario_ids))
        market_covariates = {
            f"market_cost_{route_id}": float(
                self.config["behavior"]["route_market_cost_default"]
            )
            for route_id in self.network.routes
        }
        return ModelState(
            period=0,
            horizon=int(self.config["event_weeks"]),
            risk=RiskInformation(
                belief=np.asarray([1.0 - high, high]),
                lead_time_forecast=np.asarray([1.0 - lead, lead]),
                scenario_ids=scenario_ids,
                readiness_weights=scenario_weights.copy(),
                operational_weights=scenario_weights.copy(),
                latest_release_period=0,
                latest_release_time=first_row["release_date"],
            ),
            disruption_seen=False,
            disruption_active=False,
            disruption_duration=0,
            waiting={cargo: np.zeros(len(self.waiting_hazard), dtype=float)},
            berth={},
            yard={},
            gate={},
            corridor={},
            maritime_pipeline=[],
            previous_shares={},
            corridor_history={
                (gateway, corridor): tuple(
                    0.0
                    for _ in range(
                        int(self.config["physical_feedback"]["corridor_history_window_weeks"])
                    )
                )
                for gateway in self.network.gateways()
            },
            serviceability_history=(),
            readiness=CapacityState(stock={resource: 0.0 for resource in self.controlled_resources}),
            direct_capacity=CapacityState(stock={resource: 0.0 for resource in self.controlled_resources}),
            budget=self.initial_budget,
            observed_covariates={
                "estimated_normal_demand": float(sum(self.gateway_scales.values())),
                "observed_serviceability": 1.0,
                "released_high_risk_belief": high,
                "lead_high_risk_forecast": lead,
                "release_date": first_row["release_date"],
                "decision_week": first_row["week"],
                "serviceability_timestamps": (),
                **market_covariates,
            },
        )


class CommonDisclosureBehaviorFactory:
    """Build the formal route-wait-exit problem with one common disclosure block."""

    def __init__(self, model: BenchmarkModel) -> None:
        self.model = model

    def __call__(
        self, state: ModelState, action: Action, realization: ExogenousRealization
    ) -> BehaviorProblem:
        model = self.model
        cargo = str(model.config["cargo_class"])
        split = construct_demand_split(model.network, realization, tolerance=1e-9)
        decision = build_decision_masses(
            state.waiting,
            split.decision_eligible,
            {cargo: action.value(model.layout.release[0])},
        )
        routes = tuple(
            route_id
            for route_id in model.network.routes_for_class(cargo)
            if route_id in realization.choice_route_available
        )
        route_keys = tuple((cargo, route_id) for route_id in routes)
        reference = {
            key: _current_route_wait(state, model, key[1]) for key in route_keys
        }
        reference_shares = {
            tag.route: float(share) for tag, share in model.committed_shares.items()
        }
        adaptive_decision_mass = float(sum(decision.masses.values()))
        raw_signal = {
            key: _public_route_wait(
                state,
                model,
                action,
                key[1],
                reference_loading=(
                    adaptive_decision_mass * reference_shares[key[1]]
                    + float(split.committed_by_tag.get(Tag(cargo, key[1]), 0.0))
                ),
            )
            for key in route_keys
        }
        registered_scales = model.config["information"]["waiting_error_scale_weeks_by_route"]
        error_scale = {key: float(registered_scales[key[1]]) for key in route_keys}
        disclosure_keys = dict(zip(sorted(model.network.routes), model.layout.disclosure))
        disclosure = FrozenDisclosure.clip_from_reference(
            raw_signal=raw_signal,
            reference_forecast=reference,
            error_scale=error_scale,
            intensity={key: action.value(disclosure_keys[key[1]]) for key in route_keys},
            gamma=float(model.config["information"]["gamma_I"]),
        )
        vintages = len(model.waiting_hazard)
        direct_exit: dict[SourceKey, float] = {}
        for source in decision.masses:
            direct_exit[source] = float(model.config["behavior"]["exit_failure_cost_per_unit"])
        parameters = BehaviorCostParameters(
            theta={cargo: float(model.config["behavior"]["logit_theta"])},
            route_private_resource={
                (cargo, tag.route): value for tag, value in model.route_costs.items()
            },
            route_market_cost={
                key: float(
                    state.observed_covariates.get(
                        f"market_cost_{key[1]}",
                        model.config["behavior"]["route_market_cost_default"],
                    )
                )
                for key in route_keys
            },
            value_of_time={cargo: float(model.config["behavior"]["value_of_time"])},
            waiting_base={cargo: float(model.config["behavior"]["waiting_base"])},
            waiting_age_cost={
                (cargo, age): float(model.config["behavior"]["waiting_age_cost_per_week"])
                * age
                for age in range(vintages + 1)
            },
            waiting_inventory_cost={
                cargo: float(model.config["behavior"]["waiting_inventory_coefficient"])
            },
            waiting_scale={cargo: max(sum(model.gateway_scales.values()), 1.0)},
            waiting_reclosure_cost={
                cargo: float(model.config["behavior"]["waiting_reclosure_coefficient"])
            },
            continuation_value={
                (cargo, age): float(model.config["behavior"]["waiting_continuation_value"])
                for age in range(vintages)
            },
            direct_exit_cost=direct_exit,
            late_exit_cost={
                (cargo, age): float(model.config["behavior"]["late_exit_cost_per_vintage"])
                for age in range(vintages)
            },
            hazard={cargo: model.waiting_hazard},
        )

        def waiting_oracle(
            flows: Mapping[SourceKey, Mapping[str, float]],
        ) -> Mapping[tuple[str, str], float]:
            result: dict[tuple[str, str], float] = {}
            for route_id in routes:
                incoming = sum(source.get(route_id, 0.0) for source in flows.values())
                result[(cargo, route_id)] = _current_route_wait(
                    state, model, route_id, marginal_dispatch=incoming
                )
            return result

        return BehaviorProblem(
            decision=decision,
            waiting_state=state.waiting,
            routes_by_class={cargo: routes},
            disclosure=disclosure,
            parameters=parameters,
            reclosure_probability=state.risk.reclosure_probability,
            private_waiting_oracle=waiting_oracle,
        )


def _current_route_wait(
    state: ModelState,
    model: BenchmarkModel,
    route_id: str,
    *,
    marginal_dispatch: float = 0.0,
) -> float:
    route = model.network.route(route_id)
    tag = Tag(str(model.config["cargo_class"]), route_id)
    stages = (
        (Stage.BERTH, state.berth.get(tag, 0.0), route.gateway),
        (Stage.YARD, state.yard.get(tag, 0.0), route.gateway),
        (Stage.GATE, state.gate.get(tag, 0.0), route.gateway),
        (Stage.CORRIDOR, state.corridor.get(tag, 0.0), route.corridor),
    )
    wait = float(route.maritime_lag_kernel.index(1.0))
    for stage, mass, location in stages:
        capacity = max(model.base_capacity[ResourceKey(stage, location)], 1e-12)
        wait += mass / capacity
    berth_capacity = max(model.base_capacity[ResourceKey(Stage.BERTH, route.gateway)], 1e-12)
    wait += marginal_dispatch / berth_capacity
    return float(wait)


def _public_route_wait(
    state: ModelState,
    model: BenchmarkModel,
    action: Action,
    route_id: str,
    *,
    reference_loading: float,
) -> float:
    """Predetermined action-aware queue signal; never uses realised SUE flows."""

    route = model.network.route(route_id)
    tag = Tag(str(model.config["cargo_class"]), route_id)
    current_capacity = model.kernel.transition.capacity_model.transition(
        state, action
    ).current.effective
    stages = (
        (Stage.BERTH, state.berth.get(tag, 0.0), route.gateway),
        (Stage.YARD, state.yard.get(tag, 0.0), route.gateway),
        (Stage.GATE, state.gate.get(tag, 0.0), route.gateway),
        (Stage.CORRIDOR, state.corridor.get(tag, 0.0), route.corridor),
    )
    wait = float(route.maritime_lag_kernel.index(1.0))
    for stage, mass, location in stages:
        capacity = max(current_capacity[ResourceKey(stage, location)], 1e-12)
        wait += mass / capacity
    berth_capacity = max(
        current_capacity[ResourceKey(Stage.BERTH, route.gateway)], 1e-12
    )
    wait += max(float(reference_loading), 0.0) / berth_capacity
    return float(wait)


def route_resource_cost_register(config: Mapping[str, Any]) -> pd.DataFrame:
    routes = list(config["routes"])
    minimum_lag = min(int(route["maritime_lag_weeks"]) for route in routes)
    baseline = str(config["route_resource_cost"]["common_baseline"])
    rows = []
    for route in routes:
        ocean = float(int(route["maritime_lag_weeks"]) - minimum_lag)
        rows.append(
            {
                "route": route["route_id"],
                "gateway": route["gateway"],
                "common_baseline": baseline,
                "ocean_component": ocean,
                "handling_component": 0.0,
                "trucking_component": 0.0,
                "border_component": 0.0,
                "total_incremental_resource_cost": ocean,
                "unit": "designed resource-index units per model cargo unit",
                "construction_formula": (
                    "excess maritime lag weeks relative to the minimum-lag route; "
                    "other components are zero only where topology is identical or evidence absent"
                ),
                "source": "declared 5.2.2 reference-network lag and stage topology",
                "evidence_status": config["route_resource_cost"]["evidence_status"],
                "sensitivity_destination": "5.3.4",
            }
        )
    frame = pd.DataFrame(rows)
    if frame["total_incremental_resource_cost"].isna().any():
        raise ValueError("Route resource costs cannot be missing")
    if not (frame["total_incremental_resource_cost"] > 0).any():
        raise ValueError("The declared reference network cannot have all-zero route costs")
    return frame


def build_model(config: Mapping[str, Any]) -> BenchmarkModel:
    cargo = str(config["cargo_class"])
    if float(config["behavior"]["late_exit_cost_per_vintage"]) != 0.0:
        raise ValueError(
            "Chapter 3 requires unidentified c^{E,late}_{kj} to be zero"
        )
    gamma = float(config["information"]["gamma_I"])
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("Registered disclosure credibility gamma_I must lie in [0,1]")
    calibration_path = CODE_ROOT / str(
        config["information"]["waiting_error_scale_calibration_file"]
    )
    calibration = pd.read_csv(calibration_path)
    calibrated_scales = dict(
        zip(calibration["route"], calibration["sigma_W_rmse_weeks"])
    )
    registered_scales = config["information"]["waiting_error_scale_weeks_by_route"]
    if set(calibrated_scales) != set(registered_scales) or any(
        not np.isclose(float(calibrated_scales[route]), float(registered_scales[route]))
        for route in calibrated_scales
    ):
        raise ValueError("Registered sigma^W values differ from frozen validation calibration")
    if any(float(value) <= 0.0 for value in registered_scales.values()):
        raise ValueError("Frozen waiting forecast error scales must be positive")
    scale_path = CODE_ROOT / "experiments" / "data" / "processed" / "anchors" / "gateway_reference_scales.csv"
    share_path = CODE_ROOT / "experiments" / "data" / "processed" / "anchors" / "committed_itinerary_reference.csv"
    scale_frame = pd.read_csv(scale_path)
    share_frame = pd.read_csv(share_path)
    gateway_scales = {
        str(row.gateway): float(row.activity_scale_model_units)
        for row in scale_frame.itertuples(index=False)
    }
    route_register = route_resource_cost_register(config)
    route_cost_by_id = dict(
        zip(route_register["route"], route_register["total_incremental_resource_cost"])
    )
    route_definitions: dict[str, Route] = {}
    for item in config["routes"]:
        lag = int(item["maritime_lag_weeks"])
        kernel = tuple([0.0] * lag + [1.0])
        route_definitions[str(item["route_id"])] = Route(
            route_id=str(item["route_id"]),
            cargo_class=cargo,
            gateway=str(item["gateway"]),
            corridor=str(item["corridor"]),
            maritime_lag_kernel=kernel,
        )
    network = Network(route_definitions)
    if network.shared_corridors() != {"landbridge_shared": tuple(sorted(network.gateways()))}:
        raise ValueError("The reference network must retain one corridor shared by all gateways")

    committed_lookup = {
        str(row.gateway): float(row.committed_itinerary_share)
        for row in share_frame.itertuples(index=False)
    }
    committed_shares = {
        Tag(cargo, route_id): committed_lookup[route.gateway]
        for route_id, route in network.routes.items()
    }
    if not np.isclose(sum(committed_shares.values()), 1.0):
        raise ValueError("Committed reference shares must sum to one")
    route_costs = {
        Tag(cargo, route_id): float(route_cost_by_id[route_id]) for route_id in network.routes
    }

    base_capacity: dict[ResourceKey, float] = {}
    for gateway in network.gateways():
        for stage in (Stage.BERTH, Stage.YARD, Stage.GATE):
            base_capacity[ResourceKey(stage, gateway)] = gateway_scales[gateway]
    shared_capacity = float(sum(gateway_scales.values()))
    base_capacity[ResourceKey(Stage.CORRIDOR, "landbridge_shared")] = shared_capacity
    threshold_weeks = float(config["loss"]["threshold_service_weeks"])
    thresholds = {
        resource: threshold_weeks * value for resource, value in base_capacity.items()
    }
    resources = tuple(sorted(base_capacity))
    controlled = resources
    resource_coordinates = tuple(
        f"{resource.stage.value}:{resource.location}" for resource in controlled
    )
    route_ids = tuple(sorted(network.routes))
    layout = ActionLayout(
        tuple(ActionKey.one(Block.READINESS_ORDER, coordinate) for coordinate in resource_coordinates),
        tuple(ActionKey.one(Block.DIRECT_ORDER, coordinate) for coordinate in resource_coordinates),
        tuple(ActionKey.one(Block.READINESS_EXERCISE, coordinate) for coordinate in resource_coordinates),
        (ActionKey.one(Block.RELEASE, cargo),),
        tuple(ActionKey.one(Block.DISCLOSURE, route_id) for route_id in route_ids),
    )
    resource_upper = np.asarray([base_capacity[resource] for resource in controlled], dtype=float)
    action_upper = np.concatenate(
        [resource_upper, resource_upper, resource_upper, np.ones(1), np.ones(len(route_ids))]
    )
    readiness_cost = float(config["action"]["readiness_order_cost_per_unit"])
    direct_cost = float(config["action"]["direct_order_cost_per_unit"])
    exercise_cost = float(config["action"]["readiness_exercise_cost_per_unit"])
    publication_cost = float(config["action"]["publication_cost_per_unit"])
    slopes = np.concatenate(
        [
            np.full(len(controlled), readiness_cost),
            np.full(len(controlled), direct_cost),
            np.full(len(controlled), exercise_cost),
            np.zeros(1),
            np.full(len(route_ids), publication_cost),
        ]
    )
    curves = {
        key: ConvexPiecewiseLinearCurve((0.0,), (slope,))
        for key, slope in zip(layout.keys, slopes)
    }
    weekly_full_cost = float(np.dot(action_upper, slopes))
    period_cap = float(config["action"]["period_budget_fraction"]) * weekly_full_cost
    total_budget = (
        float(config["action"]["cumulative_budget_fraction"])
        * weekly_full_cost
        * int(config["event_weeks"])
    )
    upper_map = {key: float(value) for key, value in zip(layout.keys, action_upper)}
    domain = ActionDomain(
        keys=layout.keys,
        phase_upper={0: dict(upper_map), 1: dict(upper_map)},
        cost_curves=curves,
        period_budget_cap=lambda state: min(period_cap, state.budget),
        stock_constraints={
            key: StockConstraint(resource, 1.0)
            for key, resource in zip(layout.readiness_exercise, controlled)
        },
    )
    projector = ActionProjector(
        domain,
        scaling={key: 1.0 / upper_map[key] for key in layout.keys},
        tolerance=float(config["action"]["projection_tolerance"]),
        max_iterations=int(config["action"]["projection_max_iterations"]),
    )

    lead = int(config["action"]["readiness_lead_weeks"])
    direct_lead = int(config["action"]["direct_lead_weeks"])
    technology = CapacityTechnology(
        readiness_lead={resource: lead for resource in resources},
        readiness_maturity_yield={
            resource: float(config["capacity_technology"]["readiness_maturity_yield"])
            for resource in resources
        },
        readiness_consumption={
            resource: float(config["capacity_technology"]["readiness_consumption"])
            for resource in resources
        },
        readiness_capacity_yield={
            resource: float(config["capacity_technology"]["readiness_capacity_yield"])
            for resource in resources
        },
        readiness_decay={
            resource: float(config["capacity_technology"]["readiness_decay"])
            for resource in resources
        },
        direct_lead={(phase, resource): direct_lead for phase in (0, 1) for resource in resources},
        direct_maturity_yield={
            resource: float(config["capacity_technology"]["direct_maturity_yield"])
            for resource in resources
        },
        direct_decay={
            resource: float(config["capacity_technology"]["direct_decay"])
            for resource in resources
        },
    )

    def feedback(ratio: float) -> float:
        threshold = float(config["physical_feedback"]["threshold_ratio"])
        strength = float(config["physical_feedback"]["strength"])
        return float(1.0 / (1.0 + strength * max(float(ratio) - threshold, 0.0)))

    service = ServiceParameters(
        base_capacity=base_capacity,
        thresholds=thresholds,
        yard_feedback={gateway: feedback for gateway in network.gateways()},
        corridor_feedback={gateway: feedback for gateway in network.gateways()},
        fallback_corridor_share={
            (gateway, "landbridge_shared"): float(
                config["physical_feedback"]["fallback_shared_corridor_weight"]
            )
            for gateway in network.gateways()
        },
    )
    capacity = CapacityDynamics(
        network,
        technology,
        service,
        CapacityActionMap(
            readiness_order=dict(zip(layout.readiness_order, controlled)),
            direct_order=dict(zip(layout.direct_order, controlled)),
            readiness_exercise=dict(zip(layout.readiness_exercise, controlled)),
        ),
    )
    vintages = int(config["behavior"]["waiting_vintages"])
    denominator = max(vintages - 1, 1)
    hazard_power = float(config["behavior"]["hazard_power"])
    waiting_hazard = np.asarray(
        [(age / denominator) ** hazard_power for age in range(vintages)]
    )
    waiting_hazard[-1] = 1.0
    exit_cost = float(config["behavior"]["exit_failure_cost_per_unit"])
    loss_parameters = LossParameters(
        queue_cost={resource: float(config["loss"]["queue_cost_per_model_unit_week"]) for resource in resources},
        waiting_cost={
            (cargo, age): float(config["loss"]["waiting_cost_per_vintage_week"])
            * (age + 1)
            for age in range(vintages)
        },
        exit_failure_cost={cargo: exit_cost},
        overflow_cost={resource: float(config["loss"]["overflow_cost_per_model_unit_week"]) for resource in resources},
        thresholds=thresholds,
        route_resource_increment=route_costs,
    )
    transition = TaggedTransition(
        network=network,
        action_domain=domain,
        capacity=capacity,
        loss=OperationalLoss(network, loss_parameters),
        waiting_hazard={cargo: waiting_hazard},
        release_action_map={cargo: layout.release[0]},
        corridor_history_window=int(
            config["physical_feedback"]["corridor_history_window_weeks"]
        ),
        audit_tolerance=float(config["numerics"]["mass_tolerance"]),
    )
    terminal_cost = TerminalMassCorrection(
        network,
        TerminalCostParameters(
            waiting_unit_cost={cargo: exit_cost},
            pipeline_unit_cost={cargo: exit_cost},
            tagged_unit_cost={resource: exit_cost for resource in resources},
        ),
    )
    placeholder = BenchmarkModel(
        config=config,
        network=network,
        resources=resources,
        thresholds=thresholds,
        base_capacity=base_capacity,
        gateway_scales=gateway_scales,
        committed_shares=committed_shares,
        route_costs=route_costs,
        route_cost_register=route_register,
        layout=layout,
        domain=domain,
        projector=projector,
        kernel=None,  # type: ignore[arg-type]
        terminal_cost=terminal_cost,
        controlled_resources=controlled,
        action_upper=action_upper,
        initial_budget=total_budget,
        waiting_hazard=waiting_hazard,
    )
    kernel = ModelKernel(
        behavior_factory=CommonDisclosureBehaviorFactory(placeholder),
        equilibrium_solver=RCMSASolver(
            RCMSASettings(
                tolerance=float(config["behavior"]["rcmsa_tolerance"]),
                max_iterations=int(config["behavior"]["rcmsa_max_iterations"]),
                deduplication_tolerance=float(config["behavior"]["rcmsa_deduplication_tolerance"]),
            )
        ),
        transition=transition,
        projector=projector,
    )
    placeholder.kernel = kernel
    return placeholder
