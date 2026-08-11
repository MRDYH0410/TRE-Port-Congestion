"""From-scratch BC, PPO, SAC, and constrained-SAC training on formal model loss."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from tre84.clearance import ClearanceRunner
from tre84.learning import SACActorLossInput, constrained_sac_actor_loss
from tre84.metrics import stage_pressures

from features import LinearActor, state_feature_names, state_features
from model import BenchmarkModel
from paths import PhysicalPath, deterministic_seed
from policies import ActorPolicy, ModelGuidedPolicy, mpc_teacher_action
from preparation import build_realization, prepare_period
from simulator import RecoveryRule, run_replication


@dataclass(frozen=True)
class TeacherRecord:
    path_id: str
    period_offset: int
    state_vector: np.ndarray
    target_normalised_action: np.ndarray
    candidate_id: str
    nested_objective: float


@dataclass
class TrainingResult:
    policy: str
    seed_index: int
    seed: int
    actor: LinearActor
    training_curve: list[dict[str, Any]]
    validation_curve: list[dict[str, Any]]
    best_validation_loss: float
    selected_episode: int
    stopped_reason: str
    final_dual: float
    teacher_hash: str
    entropy_temperature: float = float("nan")


@dataclass
class EpisodeStep:
    state_snapshot: Any
    state_vector: np.ndarray
    action_normalised: np.ndarray
    raw_action_normalised: np.ndarray
    projected_action_normalised: np.ndarray
    mean_normalised: np.ndarray
    reward: float
    constraint_cost: float
    old_log_probability: float
    latent_action: np.ndarray
    standard_normal_noise: np.ndarray


def _teacher_hash(records: Sequence[TeacherRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.path_id.encode("utf-8"))
        digest.update(str(record.period_offset).encode("ascii"))
        digest.update(np.asarray(record.state_vector, dtype="<f8").tobytes())
        digest.update(np.asarray(record.target_normalised_action, dtype="<f8").tobytes())
        digest.update(record.candidate_id.encode("utf-8"))
    return digest.hexdigest()


def generate_teacher_data(
    *,
    model: BenchmarkModel,
    paths: Sequence[PhysicalPath],
) -> tuple[list[TeacherRecord], str]:
    records: list[TeacherRecord] = []
    for path in paths:
        rows = [row._asdict() for row in path.frame.itertuples(index=False)]
        state = model.initial_state(rows[0])
        for offset, row in enumerate(rows):
            prepared = prepare_period(model=model, state=state, row=row)
            state = prepared.state
            action, candidate_id, objective = mpc_teacher_action(
                model=model, state=state, bundle=prepared.scenarios
            )
            records.append(
                TeacherRecord(
                    path.path_id,
                    offset,
                    state_features(state, model),
                    model.normalise_action(action),
                    candidate_id,
                    objective,
                )
            )
            projection = model.projector.project(action, state)
            realization = build_realization(model=model, state=state, row=row)
            result = model.kernel.execute(
                state=state,
                action=projection.action,
                realization=realization,
                projection=projection,
            )
            state = result.transition.next_state
    return records, _teacher_hash(records)


def _validation_loss(
    *,
    model: BenchmarkModel,
    actor: LinearActor,
    policy_name: str,
    seed: int,
    paths: Sequence[PhysicalPath],
) -> float:
    policy = ActorPolicy(policy_name, model, actor, seed)
    return float(
        np.mean(
            [
                run_replication(model=model, policy=policy, path=path).replication[
                    "total_operational_objective"
                ]
                for path in paths
            ]
        )
    )


def _stop_update(
    *,
    loss: float,
    best_loss: float,
    tolerance_fraction: float,
    stale: int,
) -> tuple[bool, int]:
    threshold = tolerance_fraction * max(abs(best_loss), 1.0)
    improved = not np.isfinite(best_loss) or loss < best_loss - threshold
    return improved, 0 if improved else stale + 1


def train_bc(
    *,
    model: BenchmarkModel,
    teacher: Sequence[TeacherRecord],
    teacher_hash: str,
    validation_paths: Sequence[PhysicalPath],
    seed_index: int,
) -> TrainingResult:
    policy_name = "Behaviour cloning"
    seed = deterministic_seed(f"{model.config['experiment_id']}|{policy_name}", seed_index)
    actor = LinearActor.random(model, seed)
    rng = np.random.default_rng(seed)
    x = np.vstack([record.state_vector for record in teacher])
    y = np.vstack([record.target_normalised_action for record in teacher])
    learning_rate = float(model.config["training"]["learning_rate"])
    ridge = float(model.config["training"]["bc_ridge"])
    interval = int(model.config["training"]["validation_interval_episodes"])
    minimum = int(model.config["training"]["minimum_episodes"])
    maximum = int(model.config["training"]["maximum_episodes"])
    patience = int(model.config["training"]["patience_evaluations"])
    tolerance = float(model.config["training"]["validation_improvement_tolerance_fraction"])
    best_actor = actor.clone()
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    training_curve: list[dict[str, Any]] = []
    validation_curve: list[dict[str, Any]] = []
    stopped = "maximum_computational_cap"
    for epoch in range(1, maximum + 1):
        order = rng.permutation(len(x))
        prediction = 1.0 / (1.0 + np.exp(-np.clip(x[order] @ actor.weights.T, -30.0, 30.0)))
        error = prediction - y[order]
        gradient = ((error * prediction * (1.0 - prediction)).T @ x[order]) / len(x)
        gradient += ridge * actor.weights
        actor.weights -= learning_rate * gradient
        actor.weights = np.clip(actor.weights, -10.0, 10.0)
        mse = float(np.mean(np.square(error)))
        training_curve.append(
            {
                "policy": policy_name,
                "seed_index": seed_index,
                "training_seed": seed,
                "episode": epoch,
                "training_loss": mse,
                "mean_formal_reward": np.nan,
                "mean_constraint_violation": 0.0,
                "constraint_dual": 0.0,
                "update_method": "projected_action_mse_on_formal_mpc_teacher",
            }
        )
        if epoch % interval == 0:
            validation = _validation_loss(
                model=model,
                actor=actor,
                policy_name=policy_name,
                seed=seed,
                paths=validation_paths,
            )
            improved, stale = _stop_update(
                loss=validation,
                best_loss=best_loss,
                tolerance_fraction=tolerance,
                stale=stale,
            )
            validation_curve.append(
                {
                    "policy": policy_name,
                    "seed_index": seed_index,
                    "training_seed": seed,
                    "episode": epoch,
                    "validation_operational_loss": validation,
                    "improved": improved,
                    "stale_evaluations": stale,
                    "checkpoint_selected": False,
                }
            )
            if improved:
                best_loss = validation
                best_actor = actor.clone()
                best_epoch = epoch
            if epoch >= minimum and stale >= patience:
                stopped = "validation_patience"
                break
    for row in validation_curve:
        row["checkpoint_selected"] = row["episode"] == best_epoch
    return TrainingResult(
        policy_name,
        seed_index,
        seed,
        best_actor,
        training_curve,
        validation_curve,
        best_loss,
        best_epoch,
        stopped,
        0.0,
        teacher_hash,
    )


def _gaussian_log_probability(
    sample: np.ndarray, mean: np.ndarray, log_std: np.ndarray
) -> float:
    variance = np.exp(2.0 * log_std)
    return float(
        -0.5
        * np.sum(
            np.square(sample - mean) / variance
            + 2.0 * log_std
            + np.log(2.0 * np.pi)
        )
    )


def _training_episode(
    *,
    model: BenchmarkModel,
    actor: LinearActor,
    path: PhysicalPath,
    rng: np.random.Generator,
    sampling_mode: str = "ppo_action_gaussian",
) -> tuple[list[EpisodeStep], float, float]:
    rows = [row._asdict() for row in path.frame.itertuples(index=False)]
    state = model.initial_state(rows[0])
    steps: list[EpisodeStep] = []
    decision_loss = 0.0
    for offset, row in enumerate(rows):
        prepared = prepare_period(model=model, state=state, row=row)
        state = prepared.state
        features = state_features(state, model)
        mean = actor.normalised_mean(state, model)
        if sampling_mode == "sac_latent_gaussian":
            sample, latent, noise, log_probability = actor.sample_latent_normalised(
                state, model, rng
            )
        elif sampling_mode == "ppo_action_gaussian":
            sample, noise = actor.sample_normalised(state, model, rng)
            latent = np.full_like(sample, np.nan)
            log_probability = _gaussian_log_probability(
                sample, mean, actor.log_standard_deviation
            )
        else:
            raise ValueError(f"Unknown training sampling mode: {sampling_mode}")
        raw = model.action_from_normalised(sample)
        projection = model.projector.project(raw, state)
        projected = model.normalise_action(projection.action)
        realization = build_realization(model=model, state=state, row=row)
        result = model.kernel.execute(
            state=state,
            action=projection.action,
            realization=realization,
            projection=projection,
        )
        pressures = stage_pressures(result.transition.next_state, model.network, model.thresholds)
        constraint = float(max(max(pressures.values(), default=0.0) - 1.0, 0.0))
        reward = -float(result.transition.loss.total)
        steps.append(
            EpisodeStep(
                state.clone(),
                features,
                projected if sampling_mode == "sac_latent_gaussian" else sample,
                sample,
                projected,
                mean,
                reward,
                constraint,
                log_probability,
                latent,
                noise,
            )
        )
        decision_loss -= reward
        state = result.transition.next_state
    clearance = ClearanceRunner(
        kernel=model.kernel,
        recovery_rule=RecoveryRule(model),
        terminal_cost=model.terminal_cost,
        maximum_weeks=int(model.config["clearance"]["maximum_weeks"]),
        empty_tolerance=float(model.config["clearance"]["empty_tolerance"]),
    ).run(state)
    tail_loss = clearance.total_loss
    if steps:
        steps[-1].reward -= tail_loss
    total_loss = decision_loss + tail_loss
    mean_constraint = float(np.mean([step.constraint_cost for step in steps]))
    return steps, float(total_loss), mean_constraint


def _discounted_returns(steps: Sequence[EpisodeStep], gamma: float) -> np.ndarray:
    result = np.zeros(len(steps), dtype=float)
    following = 0.0
    for index in range(len(steps) - 1, -1, -1):
        following = steps[index].reward + gamma * following
        result[index] = following
    return result


def train_ppo(
    *,
    model: BenchmarkModel,
    training_paths: Sequence[PhysicalPath],
    validation_paths: Sequence[PhysicalPath],
    seed_index: int,
) -> TrainingResult:
    policy_name = "PPO"
    seed = deterministic_seed(f"{model.config['experiment_id']}|{policy_name}", seed_index)
    actor = LinearActor.random(model, seed)
    rng = np.random.default_rng(seed)
    cfg = model.config["training"]
    interval = int(cfg["validation_interval_episodes"])
    minimum, maximum = int(cfg["minimum_episodes"]), int(cfg["maximum_episodes"])
    best_actor, best_loss, best_episode = actor.clone(), float("inf"), 0
    stale = 0
    training_curve: list[dict[str, Any]] = []
    validation_curve: list[dict[str, Any]] = []
    stopped = "maximum_computational_cap"
    for episode in range(1, maximum + 1):
        path = training_paths[(episode - 1) % len(training_paths)]
        steps, total_loss, constraint = _training_episode(
            model=model, actor=actor, path=path, rng=rng
        )
        returns = _discounted_returns(steps, float(cfg["discount_factor"]))
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        clip = float(cfg["ppo_clip_ratio"])
        learning_rate = float(cfg["learning_rate"])
        surrogate_terms = []
        for _ in range(2):
            for step, advantage in zip(steps, advantages):
                logits = actor.weights @ step.state_vector
                mean = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
                logp = _gaussian_log_probability(
                    step.action_normalised, mean, actor.log_standard_deviation
                )
                ratio = float(np.exp(np.clip(logp - step.old_log_probability, -20.0, 20.0)))
                clipped_ratio = float(np.clip(ratio, 1.0 - clip, 1.0 + clip))
                surrogate = min(ratio * advantage, clipped_ratio * advantage)
                surrogate_terms.append(surrogate)
                clipped_binding = (advantage >= 0 and ratio > 1.0 + clip) or (
                    advantage < 0 and ratio < 1.0 - clip
                )
                if clipped_binding:
                    continue
                variance = np.exp(2.0 * actor.log_standard_deviation)
                score = (step.action_normalised - mean) / variance
                gradient = np.outer(score * mean * (1.0 - mean), step.state_vector)
                actor.weights += learning_rate * float(np.clip(advantage, -3.0, 3.0)) * gradient / len(steps)
        actor.weights = np.clip(actor.weights, -10.0, 10.0)
        training_curve.append(
            {
                "policy": policy_name,
                "seed_index": seed_index,
                "training_seed": seed,
                "episode": episode,
                "training_loss": -float(np.mean(surrogate_terms)),
                "mean_formal_reward": -total_loss,
                "mean_constraint_violation": constraint,
                "constraint_dual": 0.0,
                "update_method": "clipped_on_policy_gaussian_linear_actor",
            }
        )
        if episode % interval == 0:
            validation = _validation_loss(
                model=model, actor=actor, policy_name=policy_name, seed=seed, paths=validation_paths
            )
            improved, stale = _stop_update(
                loss=validation,
                best_loss=best_loss,
                tolerance_fraction=float(cfg["validation_improvement_tolerance_fraction"]),
                stale=stale,
            )
            validation_curve.append(
                {
                    "policy": policy_name,
                    "seed_index": seed_index,
                    "training_seed": seed,
                    "episode": episode,
                    "validation_operational_loss": validation,
                    "improved": improved,
                    "stale_evaluations": stale,
                    "checkpoint_selected": False,
                }
            )
            if improved:
                best_actor, best_loss, best_episode = actor.clone(), validation, episode
            if episode >= minimum and stale >= int(cfg["patience_evaluations"]):
                stopped = "validation_patience"
                break
    for row in validation_curve:
        row["checkpoint_selected"] = row["episode"] == best_episode
    return TrainingResult(
        policy_name, seed_index, seed, best_actor, training_curve, validation_curve,
        best_loss, best_episode, stopped, 0.0, "not_applicable"
    )


def _critic_features(
    state: np.ndarray,
    action: np.ndarray,
    projection: np.ndarray,
    interaction_head: int,
) -> np.ndarray:
    compact = state @ projection
    head = compact[:interaction_head]
    return np.concatenate(
        [compact, action, np.square(action), np.outer(head, action).ravel(), np.ones(1)]
    )


def _critic_action_gradient(
    coefficients: np.ndarray,
    compact_state: np.ndarray,
    action: np.ndarray,
    interaction_head: int,
) -> np.ndarray:
    state_size = compact_state.size
    action_size = action.size
    direct = coefficients[state_size : state_size + action_size]
    quadratic = coefficients[state_size + action_size : state_size + 2 * action_size]
    interaction = coefficients[
        state_size + 2 * action_size : state_size + 2 * action_size + interaction_head * action_size
    ].reshape(interaction_head, action_size)
    return direct + 2.0 * quadratic * action + compact_state[:interaction_head] @ interaction


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-4) -> np.ndarray:
    identity = np.eye(x.shape[1])
    identity[-1, -1] = 0.0
    return np.linalg.solve(x.T @ x + ridge * identity, x.T @ y)


def _normalised_projection_jacobian(
    *,
    model: BenchmarkModel,
    state: Any,
    raw_action: Any,
    projection: Any,
) -> np.ndarray:
    """Map the formal projector's physical-unit subgradient to [0,1] units."""

    physical = model.projector.local_jacobian(
        raw_action, state, projection=projection
    )
    upper = np.asarray(model.action_upper, dtype=float)
    scaled = physical * upper[np.newaxis, :]
    return np.divide(
        scaled,
        upper[:, np.newaxis],
        out=np.zeros_like(scaled),
        where=upper[:, np.newaxis] > 0,
    )


def _sac_sample_objective_and_gradient(
    *,
    model: BenchmarkModel,
    state_snapshot: Any,
    state_vector: np.ndarray,
    weights: np.ndarray,
    log_standard_deviation: np.ndarray,
    noise: np.ndarray,
    critic_projection: np.ndarray,
    interaction_head: int,
    q1: np.ndarray,
    q2: np.ndarray,
    constraint_q: np.ndarray,
    entropy_temperature: float,
    constraint_dual: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """One reparameterised projected SAC sample and its analytic gradient."""

    latent_mean = weights @ state_vector
    standard_deviation = np.exp(np.clip(log_standard_deviation, -30.0, 20.0))
    latent = latent_mean + standard_deviation * noise
    raw_normalised = 1.0 / (1.0 + np.exp(-np.clip(latent, -30.0, 30.0)))
    raw_action = model.action_from_normalised(raw_normalised)
    projection_result = model.projector.project(raw_action, state_snapshot)
    projected_normalised = model.normalise_action(projection_result.action)
    phi = _critic_features(
        state_vector, projected_normalised, critic_projection, interaction_head
    )
    q1_value, q2_value = float(phi @ q1), float(phi @ q2)
    constraint_value = float(phi @ constraint_q)
    log_probability = float(
        -0.5
        * np.sum(
            np.square(noise)
            + 2.0 * log_standard_deviation
            + np.log(2.0 * np.pi)
        )
    )
    objective = (
        entropy_temperature * log_probability
        - min(q1_value, q2_value)
        + constraint_dual * constraint_value
    )
    compact_state = state_vector @ critic_projection
    selected_q = q1 if q1_value <= q2_value else q2
    gradient_projected = -_critic_action_gradient(
        selected_q, compact_state, projected_normalised, interaction_head
    )
    gradient_projected += constraint_dual * _critic_action_gradient(
        constraint_q, compact_state, projected_normalised, interaction_head
    )
    projector_jacobian = _normalised_projection_jacobian(
        model=model,
        state=state_snapshot,
        raw_action=raw_action,
        projection=projection_result,
    )
    gradient_raw = projector_jacobian.T @ gradient_projected
    gradient_latent = gradient_raw * raw_normalised * (1.0 - raw_normalised)
    weight_gradient = np.outer(gradient_latent, state_vector)
    log_std_gradient = (
        gradient_latent * standard_deviation * noise - entropy_temperature
    )
    return float(objective), weight_gradient, log_std_gradient


def sac_actor_gradient_check(model: BenchmarkModel) -> list[dict[str, Any]]:
    """Independent central-difference check of mean and log-std gradients."""

    cfg = model.config["numerics"]
    step = float(cfg["sac_gradient_check_step"])
    tolerance = float(cfg["sac_gradient_check_relative_tolerance"])
    seed = deterministic_seed(f"{model.config['experiment_id']}|sac-gradient-check", 0)
    rng = np.random.default_rng(seed)
    first_row = {
        "filtered_high_risk_probability": 0.25,
        "lead_time_high_risk_probability": 0.5,
        "release_date": pd.Timestamp("2000-01-01"),
        "week": pd.Timestamp("2000-01-03"),
    }
    state = model.initial_state(first_row)
    actor = LinearActor.random(model, seed)
    state_vector = state_features(state, model)
    projection_dimension = int(model.config["training"]["sac_critic_projection_dimension"])
    interaction_head = int(model.config["training"]["sac_critic_interaction_head"])
    critic_projection = rng.normal(
        0.0,
        1.0 / np.sqrt(projection_dimension),
        size=(state_vector.size, projection_dimension),
    )
    critic_size = _critic_features(
        state_vector,
        np.full(len(model.layout.keys), 0.05),
        critic_projection,
        interaction_head,
    ).size
    q1 = rng.normal(0.0, 0.1, size=critic_size)
    q2 = q1.copy()
    q2[-1] += 10.0
    constraint_q = rng.normal(0.0, 0.05, size=critic_size)
    noise = rng.normal(size=len(model.layout.keys))
    entropy_temperature = float(model.config["training"]["sac_entropy_temperature"])
    constraint_dual = 0.3
    objective, weight_gradient, log_std_gradient = _sac_sample_objective_and_gradient(
        model=model,
        state_snapshot=state,
        state_vector=state_vector,
        weights=actor.weights,
        log_standard_deviation=actor.log_standard_deviation,
        noise=noise,
        critic_projection=critic_projection,
        interaction_head=interaction_head,
        q1=q1,
        q2=q2,
        constraint_q=constraint_q,
        entropy_temperature=entropy_temperature,
        constraint_dual=constraint_dual,
    )

    def value(weights: np.ndarray, log_std: np.ndarray) -> float:
        return _sac_sample_objective_and_gradient(
            model=model,
            state_snapshot=state,
            state_vector=state_vector,
            weights=weights,
            log_standard_deviation=log_std,
            noise=noise,
            critic_projection=critic_projection,
            interaction_head=interaction_head,
            q1=q1,
            q2=q2,
            constraint_q=constraint_q,
            entropy_temperature=entropy_temperature,
            constraint_dual=constraint_dual,
        )[0]

    rows: list[dict[str, Any]] = []
    weight_coordinates = [
        (action_index, feature_index)
        for action_index in range(min(5, actor.weights.shape[0]))
        for feature_index in (0, min(5, actor.weights.shape[1] - 1))
    ]
    for action_index, feature_index in weight_coordinates:
        plus, minus = actor.weights.copy(), actor.weights.copy()
        plus[action_index, feature_index] += step
        minus[action_index, feature_index] -= step
        numerical = (
            value(plus, actor.log_standard_deviation)
            - value(minus, actor.log_standard_deviation)
        ) / (2.0 * step)
        analytic = float(weight_gradient[action_index, feature_index])
        relative = abs(numerical - analytic) / max(1.0, abs(numerical), abs(analytic))
        rows.append(
            {
                "parameter": f"mean_weight[{action_index},{feature_index}]",
                "objective": objective,
                "analytic_gradient": analytic,
                "finite_difference_gradient": numerical,
                "relative_error": relative,
                "tolerance": tolerance,
                "passed": relative <= tolerance,
            }
        )
    for action_index in range(actor.log_standard_deviation.size):
        plus, minus = actor.log_standard_deviation.copy(), actor.log_standard_deviation.copy()
        plus[action_index] += step
        minus[action_index] -= step
        numerical = (
            value(actor.weights, plus) - value(actor.weights, minus)
        ) / (2.0 * step)
        analytic = float(log_std_gradient[action_index])
        relative = abs(numerical - analytic) / max(1.0, abs(numerical), abs(analytic))
        rows.append(
            {
                "parameter": f"log_standard_deviation[{action_index}]",
                "objective": objective,
                "analytic_gradient": analytic,
                "finite_difference_gradient": numerical,
                "relative_error": relative,
                "tolerance": tolerance,
                "passed": relative <= tolerance,
            }
        )
    return rows


def train_sac(
    *,
    model: BenchmarkModel,
    training_paths: Sequence[PhysicalPath],
    validation_paths: Sequence[PhysicalPath],
    seed_index: int,
    constrained: bool,
) -> TrainingResult:
    policy_name = "Constrained SAC" if constrained else "Vanilla SAC"
    seed = deterministic_seed(f"{model.config['experiment_id']}|{policy_name}", seed_index)
    actor = LinearActor.random(model, seed)
    rng = np.random.default_rng(seed)
    cfg = model.config["training"]
    projection_dimension = int(cfg["sac_critic_projection_dimension"])
    interaction_head = int(cfg["sac_critic_interaction_head"])
    feature_count = len(state_feature_names(model))
    critic_projection = rng.normal(
        0.0,
        1.0 / np.sqrt(projection_dimension),
        size=(feature_count, projection_dimension),
    )
    interval = int(cfg["validation_interval_episodes"])
    minimum, maximum = int(cfg["minimum_episodes"]), int(cfg["maximum_episodes"])
    replay: list[tuple[EpisodeStep, float, float]] = []
    dual = 0.0
    initial_entropy_temperature = float(cfg["sac_entropy_temperature"])
    if initial_entropy_temperature <= 0:
        raise ValueError("The initial SAC entropy temperature must be positive")
    log_entropy_temperature = float(np.log(initial_entropy_temperature))
    target_entropy = -float(len(model.layout.keys))
    if str(cfg["sac_target_entropy_rule"]) != "negative_action_dimension":
        raise ValueError("SAC target entropy must use the registered negative action dimension rule")
    best_actor, best_loss, best_episode = actor.clone(), float("inf"), 0
    best_entropy_temperature = initial_entropy_temperature
    q1: np.ndarray | None = None
    q2: np.ndarray | None = None
    constraint_q: np.ndarray | None = None
    stale = 0
    training_curve: list[dict[str, Any]] = []
    validation_curve: list[dict[str, Any]] = []
    stopped = "maximum_computational_cap"
    for episode in range(1, maximum + 1):
        path = training_paths[(episode - 1) % len(training_paths)]
        steps, total_loss, constraint = _training_episode(
            model=model,
            actor=actor,
            path=path,
            rng=rng,
            sampling_mode="sac_latent_gaussian",
        )
        returns = _discounted_returns(steps, float(cfg["discount_factor"]))
        constraint_returns = np.asarray([step.constraint_cost for step in steps], dtype=float)
        replay.extend(zip(steps, returns, constraint_returns))
        actor_losses: list[float] = []
        temperature_losses: list[float] = []
        entropies: list[float] = []
        projection_distances_all: list[float] = []
        critic_losses_q1: list[float] = []
        critic_losses_q2: list[float] = []
        constraint_critic_losses: list[float] = []
        learning_rate = float(cfg["learning_rate"])
        # One complete twin-critic, actor, entropy-temperature, and (when
        # applicable) dual update is executed for every completed decision
        # transition.  Validation still occurs only at the frozen episode
        # interval and test paths never enter this replay buffer.
        for current_step, return_target, constraint_target in zip(
            steps, returns, constraint_returns
        ):
            critic_phi = _critic_features(
                current_step.state_vector,
                current_step.action_normalised,
                critic_projection,
                interaction_head,
            )
            if q1 is None:
                q1 = rng.normal(0.0, 1e-4, size=critic_phi.size)
                q2 = rng.normal(0.0, 1e-4, size=critic_phi.size)
                constraint_q = np.zeros(critic_phi.size, dtype=float)
            assert q2 is not None and constraint_q is not None
            critic_scale = max(float(critic_phi @ critic_phi), 1.0)
            q1_error = float(q1 @ critic_phi - return_target)
            q2_error = float(q2 @ critic_phi - return_target)
            q1 -= learning_rate * q1_error * critic_phi / critic_scale
            q2 -= learning_rate * q2_error * critic_phi / critic_scale
            constraint_error = float(constraint_q @ critic_phi - constraint_target)
            if constrained:
                constraint_q -= (
                    learning_rate
                    * constraint_error
                    * critic_phi
                    / critic_scale
                )
            entropy_temperature = float(np.exp(log_entropy_temperature))
            actor_dual = dual
            raw_normalised, _, noise, log_probability = (
                actor.sample_latent_normalised(
                    current_step.state_snapshot, model, rng
                )
            )
            actor_objective, weight_gradient, log_std_gradient = (
                _sac_sample_objective_and_gradient(
                    model=model,
                    state_snapshot=current_step.state_snapshot,
                    state_vector=current_step.state_vector,
                    weights=actor.weights,
                    log_standard_deviation=actor.log_standard_deviation,
                    noise=noise,
                    critic_projection=critic_projection,
                    interaction_head=interaction_head,
                    q1=q1,
                    q2=q2,
                    constraint_q=constraint_q,
                    entropy_temperature=entropy_temperature,
                    constraint_dual=actor_dual if constrained else 0.0,
                )
            )
            raw_action = model.action_from_normalised(raw_normalised)
            projected = model.projector.project(
                raw_action, current_step.state_snapshot
            )
            projected_normalised = model.normalise_action(projected.action)
            projection_distance = float(
                np.linalg.norm(projected_normalised - raw_normalised)
            )
            actor.weights -= learning_rate * weight_gradient
            actor.log_standard_deviation -= learning_rate * log_std_gradient
            actor.weights = np.clip(actor.weights, -10.0, 10.0)
            if np.any(~np.isfinite(actor.log_standard_deviation)):
                raise FloatingPointError(
                    "SAC log standard deviation update became non-finite"
                )
            actor_losses.append(actor_objective)
            temperature_gradient = -entropy_temperature * float(
                log_probability + target_entropy
            )
            temperature_losses.append(temperature_gradient)
            log_entropy_temperature -= float(
                cfg["sac_entropy_temperature_learning_rate"]
            ) * temperature_gradient
            if not np.isfinite(log_entropy_temperature):
                raise FloatingPointError(
                    "SAC entropy-temperature update became non-finite"
                )
            if constrained:
                dual = max(
                    0.0,
                    dual
                    + float(cfg["sac_constraint_dual_step"])
                    * (
                        current_step.constraint_cost
                        - float(cfg["constraint_violation_tolerance"])
                    ),
                )
            entropies.append(-float(log_probability))
            projection_distances_all.append(projection_distance)
            critic_losses_q1.append(q1_error * q1_error)
            critic_losses_q2.append(q2_error * q2_error)
            constraint_critic_losses.append(
                constraint_error * constraint_error if constrained else 0.0
            )
        actor_loss = float(np.mean(actor_losses))
        temperature_loss = float(np.mean(temperature_losses))
        entropy = float(np.mean(entropies))
        critic_loss_q1 = float(np.mean(critic_losses_q1))
        critic_loss_q2 = float(np.mean(critic_losses_q2))
        constraint_critic_loss = float(np.mean(constraint_critic_losses))
        training_curve.append(
            {
                "policy": policy_name,
                "seed_index": seed_index,
                "training_seed": seed,
                "episode": episode,
                "training_loss": actor_loss,
                "mean_formal_reward": -total_loss,
                "mean_constraint_violation": constraint,
                "constraint_dual": dual,
                "critic_loss_q1": critic_loss_q1,
                "critic_loss_q2": critic_loss_q2,
                "constraint_critic_loss": constraint_critic_loss,
                "latent_policy_entropy": entropy,
                "mean_log_standard_deviation": float(
                    np.mean(actor.log_standard_deviation)
                ),
                "entropy_temperature": float(np.exp(log_entropy_temperature)),
                "entropy_temperature_loss": temperature_loss,
                "target_entropy": target_entropy,
                "period_update_count": len(steps),
                "reward_critic_q1_update_count": len(steps),
                "reward_critic_q2_update_count": len(steps),
                "constraint_critic_update_count": len(steps) if constrained else 0,
                "actor_update_count": len(steps),
                "entropy_temperature_update_count": len(steps),
                "constraint_dual_update_count": len(steps) if constrained else 0,
                "mean_projection_distance": float(
                    np.mean(projection_distances_all)
                ),
                "projected_action_fraction": float(
                    np.mean(np.asarray(projection_distances_all) > 1e-12)
                ),
                "update_method": (
                    "reparameterised_latent_gaussian_projected_twin_reward_critics_constraint_critic_auto_entropy_dual"
                    if constrained
                    else "reparameterised_latent_gaussian_projected_twin_reward_critics_auto_entropy"
                ),
            }
        )
        if episode % interval == 0:
            validation = _validation_loss(
                model=model, actor=actor, policy_name=policy_name, seed=seed, paths=validation_paths
            )
            improved, stale = _stop_update(
                loss=validation,
                best_loss=best_loss,
                tolerance_fraction=float(cfg["validation_improvement_tolerance_fraction"]),
                stale=stale,
            )
            validation_curve.append(
                {
                    "policy": policy_name,
                    "seed_index": seed_index,
                    "training_seed": seed,
                    "episode": episode,
                    "validation_operational_loss": validation,
                    "improved": improved,
                    "stale_evaluations": stale,
                    "checkpoint_selected": False,
                }
            )
            if improved:
                best_actor, best_loss, best_episode = actor.clone(), validation, episode
                best_entropy_temperature = float(np.exp(log_entropy_temperature))
            if episode >= minimum and stale >= int(cfg["patience_evaluations"]):
                stopped = "validation_patience"
                break
    for row in validation_curve:
        row["checkpoint_selected"] = row["episode"] == best_episode
    return TrainingResult(
        policy_name, seed_index, seed, best_actor, training_curve, validation_curve,
        best_loss, best_episode, stopped, dual, "not_applicable",
        best_entropy_temperature,
    )


def save_checkpoint(
    *,
    result: TrainingResult,
    directory: Path,
    feature_names: Sequence[str],
    config_hash: str,
) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = result.policy.lower().replace(" ", "_").replace("-", "_")
    path = directory / f"{safe_name}_seed_{result.seed_index}.npz"
    np.savez_compressed(
        path,
        weights=result.actor.weights,
        log_standard_deviation=result.actor.log_standard_deviation,
        policy=result.policy,
        seed_index=result.seed_index,
        training_seed=result.seed,
        feature_names=np.asarray(feature_names),
        config_hash=config_hash,
        selected_episode=result.selected_episode,
        validation_loss=result.best_validation_loss,
        entropy_temperature=result.entropy_temperature,
        sac_training_contract=(
            "latent_gaussian_projected_twin_reward_critics_auto_entropy"
            if "SAC" in result.policy
            else "not_applicable"
        ),
        generated_from_scratch=True,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def validate_model_guided(
    *,
    model: BenchmarkModel,
    bc: TrainingResult,
    constrained_sac: TrainingResult,
    validation_paths: Sequence[PhysicalPath],
) -> tuple[float, list[dict[str, Any]]]:
    policy = ModelGuidedPolicy(
        model=model,
        bc_actor=bc.actor,
        sac_actor=constrained_sac.actor,
        training_seed=constrained_sac.seed,
    )
    artifacts = [run_replication(model=model, policy=policy, path=path) for path in validation_paths]
    loss = float(np.mean([item.replication["total_operational_objective"] for item in artifacts]))
    proposals = [row for item in artifacts for row in item.proposal_records]
    return loss, proposals
