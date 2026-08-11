"""Fixed full-state vectorisation and lightweight from-scratch linear actors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tre84.behavior import EXIT, WAIT
from tre84.keys import Provenance, SourceKey, Stage, Tag
from tre84.state import ModelState

from model import BenchmarkModel, _current_route_wait


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def state_feature_names(model: BenchmarkModel) -> tuple[str, ...]:
    names = [
        "intercept",
        "period_fraction",
        "phase",
        "disruption_active",
        "disruption_duration_fraction",
        "current_high_risk_belief",
        "lead_high_risk_forecast",
        "reclosure_scenario_weight",
        "remaining_budget_fraction",
        "waiting_total_scaled",
    ]
    vintages = len(model.waiting_hazard)
    names.extend(f"waiting_vintage_{age}_scaled" for age in range(vintages))
    for stage in (Stage.BERTH, Stage.YARD, Stage.GATE, Stage.CORRIDOR):
        for route_id in sorted(model.network.routes):
            names.append(f"queue_{stage.value}_{route_id}_scaled")
    max_lag = max(len(route.maritime_lag_kernel) - 1 for route in model.network.routes.values())
    for provenance in (Provenance.COMMITTED, Provenance.ADAPTIVE):
        for route_id in sorted(model.network.routes):
            for lag in range(max_lag + 1):
                names.append(f"pipeline_{provenance.value}_{route_id}_lag_{lag}_scaled")
    for system in ("readiness", "direct"):
        for resource in model.controlled_resources:
            coordinate = f"{resource.stage.value}_{resource.location}".replace(" ", "_")
            names.append(f"{system}_stock_{coordinate}_scaled")
            for remaining in range(1, int(model.config["action"]["readiness_lead_weeks"]) + 1):
                names.append(f"{system}_order_{coordinate}_remaining_{remaining}_scaled")
    corridor_lookback = int(
        model.config["state_and_actor"]["corridor_history_lookback_weeks"]
    )
    for gateway in model.network.gateways():
        for lag in range(corridor_lookback):
            names.append(f"corridor_history_{gateway}_lag_{lag}_scaled")
    service_lookback = int(
        model.config["state_and_actor"]["serviceability_history_lookback_weeks"]
    )
    for lookback in range(service_lookback):
        names.append(f"serviceability_history_lag_{lookback}")
    choices = tuple(sorted(model.network.routes)) + (WAIT, EXIT)
    sources = (SourceKey(str(model.config["cargo_class"]), None),) + tuple(
        SourceKey(str(model.config["cargo_class"]), age) for age in range(vintages)
    )
    for source in sources:
        source_name = "new" if source.vintage is None else f"v{source.vintage}"
        for choice in choices:
            names.append(f"previous_share_{source_name}_{choice}")
    for route_id in sorted(model.network.routes):
        names.append(f"market_cost_{route_id}")
        names.append(f"reference_wait_{route_id}_scaled")
    return tuple(name.replace(" ", "_") for name in names)


def state_features(state: ModelState, model: BenchmarkModel) -> np.ndarray:
    cargo = str(model.config["cargo_class"])
    scale = max(sum(model.gateway_scales.values()), 1.0)
    values: list[float] = [
        1.0,
        state.period / max(state.horizon, 1),
        float(state.phase),
        float(state.disruption_active),
        state.disruption_duration / max(state.horizon, 1),
        float(state.risk.belief[-1]),
        float(state.risk.lead_time_forecast[-1]),
        float(state.risk.reclosure_probability),
        state.budget / max(model.initial_budget, 1e-12),
        state.waiting_mass() / scale,
    ]
    waiting = np.asarray(state.waiting[cargo], dtype=float)
    values.extend((waiting / scale).tolist())
    queue_maps = {
        Stage.BERTH: state.berth,
        Stage.YARD: state.yard,
        Stage.GATE: state.gate,
        Stage.CORRIDOR: state.corridor,
    }
    for stage in (Stage.BERTH, Stage.YARD, Stage.GATE, Stage.CORRIDOR):
        for route_id in sorted(model.network.routes):
            values.append(queue_maps[stage].get(Tag(cargo, route_id), 0.0) / scale)
    max_lag = max(len(route.maritime_lag_kernel) - 1 for route in model.network.routes.values())
    for provenance in (Provenance.COMMITTED, Provenance.ADAPTIVE):
        for route_id in sorted(model.network.routes):
            for lag in range(max_lag + 1):
                values.append(
                    sum(
                        lot.mass
                        for lot in state.maritime_pipeline
                        if lot.provenance == provenance
                        and lot.route == route_id
                        and lot.remaining_lag == lag
                    )
                    / scale
                )
    for capacity in (state.readiness, state.direct_capacity):
        for resource in model.controlled_resources:
            values.append(capacity.stock.get(resource, 0.0) / scale)
            for remaining in range(1, int(model.config["action"]["readiness_lead_weeks"]) + 1):
                values.append(
                    capacity.orders.get(resource, {}).get(remaining, 0.0) / scale
                )
    corridor = str(model.config["routes"][0]["corridor"])
    corridor_lookback = int(
        model.config["state_and_actor"]["corridor_history_lookback_weeks"]
    )
    for gateway in model.network.gateways():
        history = tuple(state.corridor_history.get((gateway, corridor), ()))
        for lag in range(corridor_lookback):
            values.append((history[lag] if lag < len(history) else 0.0) / scale)
    service_lookback = int(
        model.config["state_and_actor"]["serviceability_history_lookback_weeks"]
    )
    history = tuple(reversed(state.serviceability_history[-service_lookback:]))
    for lookback in range(service_lookback):
        values.append(history[lookback] if lookback < len(history) else 1.0)
    choices = tuple(sorted(model.network.routes)) + (WAIT, EXIT)
    sources = (SourceKey(cargo, None),) + tuple(SourceKey(cargo, age) for age in range(len(model.waiting_hazard)))
    for source in sources:
        shares = state.previous_shares.get(source, {})
        values.extend(float(shares.get(choice, 0.0)) for choice in choices)
    for route_id in sorted(model.network.routes):
        values.append(
            float(
                state.observed_covariates.get(
                    f"market_cost_{route_id}",
                    model.config["behavior"]["route_market_cost_default"],
                )
            )
        )
        longest_lag = max(
            len(route.maritime_lag_kernel) - 1
            for route in model.network.routes.values()
        )
        values.append(
            _current_route_wait(state, model, route_id) / max(longest_lag, 1)
        )
    vector = np.asarray(values, dtype=float)
    if vector.shape != (len(state_feature_names(model)),) or np.any(~np.isfinite(vector)):
        raise ValueError("Full-state feature vector is incomplete or non-finite")
    return vector


@dataclass
class LinearActor:
    weights: np.ndarray
    log_standard_deviation: np.ndarray

    @classmethod
    def random(cls, model: BenchmarkModel, seed: int) -> "LinearActor":
        rng = np.random.default_rng(seed)
        feature_count = len(state_feature_names(model))
        scale = float(
            model.config["state_and_actor"]["initial_weight_standard_deviation"]
        )
        weights = rng.normal(0.0, scale, size=(len(model.layout.keys), feature_count))
        weights[:, 0] = float(
            model.config["state_and_actor"]["initial_intercept_weight"]
        )
        return cls(
            weights=weights,
            log_standard_deviation=np.full(
                len(model.layout.keys),
                float(
                    model.config["state_and_actor"][
                        "initial_log_action_standard_deviation"
                    ]
                ),
            ),
        )

    def normalised_mean(self, state: ModelState, model: BenchmarkModel) -> np.ndarray:
        return _sigmoid(self.weights @ state_features(state, model))

    def raw_action(self, state: ModelState, model: BenchmarkModel) -> object:
        return model.action_from_normalised(self.normalised_mean(state, model))

    def sample_normalised(
        self, state: ModelState, model: BenchmarkModel, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Legacy bounded action-space Gaussian used only by PPO."""

        mean = self.normalised_mean(state, model)
        noise = rng.normal(size=mean.shape)
        sample = np.clip(mean + np.exp(self.log_standard_deviation) * noise, 0.0, 1.0)
        return sample, noise

    def latent_mean(self, state: ModelState, model: BenchmarkModel) -> np.ndarray:
        """Preprojection Gaussian mean used by the formal SAC policy."""

        return self.weights @ state_features(state, model)

    def sample_latent_normalised(
        self, state: ModelState, model: BenchmarkModel, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Reparameterised latent Gaussian followed by the bounded logistic map.

        The returned log density is the density of the preprojection latent
        policy, as required by Eq. (constrained-sac-loss); it is deliberately
        not described as the density of the many-to-one projected action.
        """

        mean = self.latent_mean(state, model)
        noise = rng.normal(size=mean.shape)
        standard_deviation = np.exp(np.clip(self.log_standard_deviation, -30.0, 20.0))
        latent = mean + standard_deviation * noise
        normalised = _sigmoid(latent)
        log_probability = float(
            -0.5
            * np.sum(
                np.square(noise)
                + 2.0 * self.log_standard_deviation
                + np.log(2.0 * np.pi)
            )
        )
        return normalised, latent, noise, log_probability

    def clone(self) -> "LinearActor":
        return LinearActor(self.weights.copy(), self.log_standard_deviation.copy())
