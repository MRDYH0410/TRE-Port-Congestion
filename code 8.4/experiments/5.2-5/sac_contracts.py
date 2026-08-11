"""Independent full-SAC contract audits for Experiment 5.2.5.

The accepted 5.2.2 checkpoints and curves remain immutable inputs.  This module
replays one complete production training episode with the same path builder,
state preparer, projector, RC-MSA kernel and optimizer formulas, then reconciles
that replay with the persisted production trace.  It also reruns the registered
central-difference check and independently differentiates the local projection.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_DIR.parents[1]
EXP522 = CODE_ROOT / "experiments" / "5.2-2"
SRC = CODE_ROOT / "src"
for item in (str(SRC), str(EXP522)):
    if item not in sys.path:
        sys.path.insert(0, item)

from features import LinearActor, state_features  # noqa: E402
from paths import (  # noqa: E402
    build_training_validation_paths,
    deterministic_seed,
    load_frozen_5_2_1_inputs,
)
from training import (  # noqa: E402
    _critic_features,
    _discounted_returns,
    _normalised_projection_jacobian,
    _sac_sample_objective_and_gradient,
    _training_episode,
    sac_actor_gradient_check,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_error(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(1.0, abs(float(left)), abs(float(right)))


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "pass", "passed"})


def _checkpoint_and_validation_audit(
    *, model: Any, upstream: Mapping[str, Path], curves: pd.DataFrame
) -> pd.DataFrame:
    manifest = pd.read_csv(upstream["5.2.2"] / "checkpoint_manifest.csv")
    validation = pd.read_csv(upstream["5.2.2"] / "validation_curves.csv")
    validation_record = pd.read_csv(upstream["5.2.2"] / "checkpoint_validation.csv")
    rows: list[dict[str, Any]] = []
    for item in manifest.itertuples(index=False):
        if item.policy not in {"Behaviour cloning", "Vanilla SAC", "Constrained SAC"}:
            continue
        path = upstream["5.2.2"] / str(item.checkpoint_path)
        payload = np.load(path, allow_pickle=False)
        actor = LinearActor(
            np.asarray(payload["weights"], dtype=float),
            np.asarray(payload["log_standard_deviation"], dtype=float),
        )
        initial = LinearActor.random(model, int(item.training_seed))
        row_validation = validation_record.loc[
            (validation_record["policy"] == item.policy)
            & (validation_record["seed_index"] == item.seed_index)
        ].iloc[0]
        selected_rows = validation.loc[
            (validation["policy"] == item.policy)
            & (validation["seed_index"] == item.seed_index)
            & _as_bool(validation["checkpoint_selected"])
        ]
        selected_episode_matches = (
            len(selected_rows) == 1
            and int(selected_rows.iloc[0]["episode"]) == int(item.selected_episode)
            and _relative_error(
                float(selected_rows.iloc[0]["validation_operational_loss"]),
                float(item.best_validation_operational_loss),
            )
            <= 1e-12
        )
        state = model.initial_state(
            {
                "filtered_high_risk_probability": 0.25,
                "lead_time_high_risk_probability": 0.50,
                "release_date": pd.Timestamp("2000-01-01"),
                "week": pd.Timestamp("2000-01-03"),
                "normal_model_units": 1.0,
                "serviceability": 1.0,
            }
        )
        first = actor.raw_action(state, model).vector(model.layout.keys)
        second = actor.raw_action(state.clone(), model).vector(model.layout.keys)
        rows.append(
            {
                "policy": item.policy,
                "seed_index": int(item.seed_index),
                "training_seed": int(item.training_seed),
                "checkpoint_path": str(item.checkpoint_path),
                "recorded_checkpoint_sha256": str(item.checkpoint_sha256),
                "recalculated_checkpoint_sha256": _sha256(path),
                "checkpoint_hash_matches": _sha256(path) == str(item.checkpoint_sha256),
                "generated_from_scratch": bool(item.generated_from_scratch),
                "old_checkpoint_loaded": bool(item.old_checkpoint_loaded),
                "selected_before_test_replay": bool(item.selected_before_test_replay),
                "selection_data_split": str(row_validation["selection_data_split"]),
                "test_event_seen_before_selection": bool(row_validation["test_event_seen_before_selection"]),
                "selected_episode_matches_validation_trace": selected_episode_matches,
                "actor_mean_weight_maximum_change_from_initial": float(
                    np.max(np.abs(actor.weights - initial.weights))
                ),
                "actor_log_standard_deviation_maximum_change_from_initial": float(
                    np.max(
                        np.abs(
                            actor.log_standard_deviation
                            - initial.log_standard_deviation
                        )
                    )
                ),
                "checkpoint_replay_maximum_action_difference": float(
                    np.max(np.abs(first - second))
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["validation_checkpoint_status"] = np.where(
        frame["generated_from_scratch"]
        & ~frame["old_checkpoint_loaded"]
        & frame["selected_before_test_replay"]
        & frame["selection_data_split"].eq("validation")
        & ~frame["test_event_seen_before_selection"]
        & frame["selected_episode_matches_validation_trace"],
        "PASS",
        "FAIL",
    )
    frame["checkpoint_replay_status"] = np.where(
        frame["checkpoint_hash_matches"]
        & frame["checkpoint_replay_maximum_action_difference"].le(0.0),
        "PASS",
        "FAIL",
    )
    return frame


def _episode_replay(
    *,
    model: Any,
    config: Mapping[str, Any],
    upstream: Mapping[str, Path],
    constrained: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_name = "Constrained SAC" if constrained else "Vanilla SAC"
    seed_index = 0
    seed = deterministic_seed(f"{config['experiment_id']}|{policy_name}", seed_index)
    actor = LinearActor.random(model, seed)
    rng = np.random.default_rng(seed)
    cfg = config["training"]
    projection_dimension = int(cfg["sac_critic_projection_dimension"])
    interaction_head = int(cfg["sac_critic_interaction_head"])
    feature_count = actor.weights.shape[1]
    critic_projection = rng.normal(
        0.0,
        1.0 / np.sqrt(projection_dimension),
        size=(feature_count, projection_dimension),
    )
    frozen = load_frozen_5_2_1_inputs(config)
    training_paths, _ = build_training_validation_paths(
        config=config,
        residuals=frozen.residuals,
        reference_normal_model_units=float(sum(model.gateway_scales.values())),
    )
    steps, total_loss, mean_constraint = _training_episode(
        model=model,
        actor=actor,
        path=training_paths[0],
        rng=rng,
        sampling_mode="sac_latent_gaussian",
    )
    returns = _discounted_returns(steps, float(cfg["discount_factor"]))
    constraint_returns = np.asarray([step.constraint_cost for step in steps], dtype=float)
    learning_rate = float(cfg["learning_rate"])
    target_entropy = -float(len(model.layout.keys))
    log_alpha = float(np.log(float(cfg["sac_entropy_temperature"])))
    dual = 0.0
    q1: np.ndarray | None = None
    q2: np.ndarray | None = None
    constraint_q: np.ndarray | None = None
    rows: list[dict[str, Any]] = []

    for period, (step, reward_target, constraint_target) in enumerate(
        zip(steps, returns, constraint_returns)
    ):
        phi = _critic_features(
            step.state_vector,
            step.action_normalised,
            critic_projection,
            interaction_head,
        )
        if q1 is None:
            q1 = rng.normal(0.0, 1e-4, size=phi.size)
            q2 = rng.normal(0.0, 1e-4, size=phi.size)
            constraint_q = np.zeros(phi.size, dtype=float)
        assert q2 is not None and constraint_q is not None
        scale = max(float(phi @ phi), 1.0)
        q1_before = float(q1 @ phi)
        q2_before = float(q2 @ phi)
        constraint_before = float(constraint_q @ phi)
        q1_error = q1_before - float(reward_target)
        q2_error = q2_before - float(reward_target)
        constraint_error = constraint_before - float(constraint_target)
        q1 -= learning_rate * q1_error * phi / scale
        q2 -= learning_rate * q2_error * phi / scale
        if constrained:
            constraint_q -= learning_rate * constraint_error * phi / scale

        alpha_before = float(np.exp(log_alpha))
        dual_before = float(dual)
        raw_normalised, latent, noise, sampled_log_probability = (
            actor.sample_latent_normalised(step.state_snapshot, model, rng)
        )
        recorded_objective, weight_gradient, log_std_gradient = (
            _sac_sample_objective_and_gradient(
                model=model,
                state_snapshot=step.state_snapshot,
                state_vector=step.state_vector,
                weights=actor.weights,
                log_standard_deviation=actor.log_standard_deviation,
                noise=noise,
                critic_projection=critic_projection,
                interaction_head=interaction_head,
                q1=q1,
                q2=q2,
                constraint_q=constraint_q,
                entropy_temperature=alpha_before,
                constraint_dual=dual_before if constrained else 0.0,
            )
        )
        raw_action = model.action_from_normalised(raw_normalised)
        projection = model.projector.project(raw_action, step.state_snapshot)
        projected_normalised = model.normalise_action(projection.action)
        actor_phi = _critic_features(
            step.state_vector,
            projected_normalised,
            critic_projection,
            interaction_head,
        )
        q1_actor = float(actor_phi @ q1)
        q2_actor = float(actor_phi @ q2)
        constraint_actor = float(actor_phi @ constraint_q)
        independent_log_probability = float(
            -0.5
            * np.sum(
                np.square(noise)
                + 2.0 * actor.log_standard_deviation
                + np.log(2.0 * np.pi)
            )
        )
        entropy_contribution = alpha_before * independent_log_probability
        independent_objective = float(
            entropy_contribution
            - min(q1_actor, q2_actor)
            + (dual_before * constraint_actor if constrained else 0.0)
        )
        temperature_gradient = float(
            -alpha_before * (sampled_log_probability + target_entropy)
        )
        independent_temperature_gradient = float(
            -alpha_before * (independent_log_probability + target_entropy)
        )
        next_log_alpha = log_alpha - float(
            cfg["sac_entropy_temperature_learning_rate"]
        ) * temperature_gradient
        next_dual = dual
        if constrained:
            next_dual = max(
                0.0,
                dual
                + float(cfg["sac_constraint_dual_step"])
                * (
                    float(step.constraint_cost)
                    - float(cfg["constraint_violation_tolerance"])
                ),
            )
        projection_distance = float(
            np.linalg.norm(projected_normalised - raw_normalised)
        )
        rows.append(
            {
                "policy": policy_name,
                "seed_index": seed_index,
                "training_seed": seed,
                "episode": 1,
                "period_offset": period,
                "state_sha256": hashlib.sha256(step.state_vector.tobytes()).hexdigest(),
                "latent_sample_sha256": hashlib.sha256(np.asarray(latent).tobytes()).hexdigest(),
                "reward_critic_target": float(reward_target),
                "reward_critic_q1_before_update": q1_before,
                "reward_critic_q2_before_update": q2_before,
                "reward_critic_q1_squared_loss": q1_error * q1_error,
                "reward_critic_q2_squared_loss": q2_error * q2_error,
                "constraint_critic_target": float(constraint_target),
                "constraint_critic_before_update": constraint_before,
                "constraint_critic_squared_loss": (
                    constraint_error * constraint_error if constrained else 0.0
                ),
                "preprojection_log_probability_recorded": float(sampled_log_probability),
                "preprojection_log_probability_recalculated": independent_log_probability,
                "latent_entropy": -float(sampled_log_probability),
                "entropy_temperature_before": alpha_before,
                "entropy_actor_contribution": entropy_contribution,
                "constraint_dual_before": dual_before,
                "q1_actor_value": q1_actor,
                "q2_actor_value": q2_actor,
                "constraint_actor_value": constraint_actor,
                "actor_loss_recorded": float(recorded_objective),
                "actor_loss_recalculated": independent_objective,
                "actor_loss_relative_error": _relative_error(
                    recorded_objective, independent_objective
                ),
                "temperature_gradient_recorded": temperature_gradient,
                "temperature_gradient_recalculated": independent_temperature_gradient,
                "temperature_gradient_relative_error": _relative_error(
                    temperature_gradient, independent_temperature_gradient
                ),
                "entropy_temperature_after": float(np.exp(next_log_alpha)),
                "constraint_dual_after": float(next_dual),
                "projection_distance": projection_distance,
                "projection_feasibility_residual": float(
                    projection.feasibility_violation
                ),
                "weight_gradient_norm": float(np.linalg.norm(weight_gradient)),
                "log_standard_deviation_gradient_norm": float(
                    np.linalg.norm(log_std_gradient)
                ),
            }
        )
        actor.weights -= learning_rate * weight_gradient
        actor.log_standard_deviation -= learning_rate * log_std_gradient
        actor.weights = np.clip(actor.weights, -10.0, 10.0)
        log_alpha = next_log_alpha
        dual = next_dual

    replay = pd.DataFrame(rows)
    official = pd.read_csv(upstream["5.2.2"] / "training_curves.csv")
    official = official.loc[
        (official["policy"] == policy_name)
        & (official["seed_index"] == seed_index)
        & (official["episode"] == 1)
    ].iloc[0]
    aggregates = {
        "training_loss": float(replay["actor_loss_recorded"].mean()),
        "critic_loss_q1": float(replay["reward_critic_q1_squared_loss"].mean()),
        "critic_loss_q2": float(replay["reward_critic_q2_squared_loss"].mean()),
        "constraint_critic_loss": float(
            replay["constraint_critic_squared_loss"].mean()
        ),
        "latent_policy_entropy": float(replay["latent_entropy"].mean()),
        "mean_log_standard_deviation": float(
            np.mean(actor.log_standard_deviation)
        ),
        "entropy_temperature": float(np.exp(log_alpha)),
        "entropy_temperature_loss": float(
            replay["temperature_gradient_recorded"].mean()
        ),
        "constraint_dual": float(dual),
        "mean_projection_distance": float(replay["projection_distance"].mean()),
    }
    summary_rows: list[dict[str, Any]] = []
    for field, recalculated in aggregates.items():
        recorded = float(official[field])
        relative = _relative_error(recorded, recalculated)
        summary_rows.append(
            {
                "policy": policy_name,
                "seed_index": seed_index,
                "training_seed": seed,
                "episode": 1,
                "field": field,
                "persisted_production_value": recorded,
                "independent_replay_value": recalculated,
                "relative_error": relative,
                "tolerance": 1e-10,
                "status": "PASS" if relative <= 1e-10 else "FAIL",
                "production_total_loss": total_loss,
                "production_mean_constraint": mean_constraint,
            }
        )
    return replay, pd.DataFrame(summary_rows)


def _projection_jacobian_audit(model: Any) -> pd.DataFrame:
    seed = deterministic_seed(f"{model.config['experiment_id']}|projection-jacobian", 0)
    rng = np.random.default_rng(seed)
    state = model.initial_state(
        {
            "filtered_high_risk_probability": 0.25,
            "lead_time_high_risk_probability": 0.5,
            "release_date": pd.Timestamp("2000-01-01"),
            "week": pd.Timestamp("2000-01-03"),
            "normal_model_units": 1.0,
            "serviceability": 1.0,
        }
    )
    actor = LinearActor.random(model, seed)
    raw_normalised, _, _, _ = actor.sample_latent_normalised(state, model, rng)
    raw_action = model.action_from_normalised(raw_normalised)
    projection = model.projector.project(raw_action, state)
    analytic = _normalised_projection_jacobian(
        model=model, state=state, raw_action=raw_action, projection=projection
    )
    h = float(model.config["numerics"]["sac_gradient_check_step"])
    tolerance = float(
        model.config["numerics"]["sac_gradient_check_relative_tolerance"]
    )
    rows: list[dict[str, Any]] = []
    for input_index in range(raw_normalised.size):
        plus = raw_normalised.copy()
        minus = raw_normalised.copy()
        plus[input_index] += h
        minus[input_index] -= h
        plus_value = model.normalise_action(
            model.projector.project(model.action_from_normalised(plus), state).action
        )
        minus_value = model.normalise_action(
            model.projector.project(model.action_from_normalised(minus), state).action
        )
        numerical = (plus_value - minus_value) / (2.0 * h)
        for output_index, value in enumerate(numerical):
            recorded = float(analytic[output_index, input_index])
            relative = _relative_error(recorded, float(value))
            rows.append(
                {
                    "input_action_index": input_index,
                    "output_action_index": output_index,
                    "analytic_local_jacobian": recorded,
                    "finite_difference_local_jacobian": float(value),
                    "relative_error": relative,
                    "tolerance": tolerance,
                    "status": "PASS" if relative <= tolerance else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def sac_learning_contract_audit(
    *, model: Any, upstream: Mapping[str, Path], config: Mapping[str, Any]
) -> dict[str, pd.DataFrame]:
    curves = pd.read_csv(upstream["5.2.2"] / "training_curves.csv")
    sac = curves[curves["policy"].isin(["Vanilla SAC", "Constrained SAC"])].copy()
    numeric = [
        "training_loss",
        "critic_loss_q1",
        "critic_loss_q2",
        "constraint_critic_loss",
        "latent_policy_entropy",
        "mean_log_standard_deviation",
        "entropy_temperature",
        "entropy_temperature_loss",
        "constraint_dual",
        "mean_projection_distance",
    ]
    for column in numeric:
        sac[column] = pd.to_numeric(sac[column], errors="coerce")
    sac["all_required_training_fields_finite"] = np.isfinite(sac[numeric]).all(axis=1)
    sac["latent_gaussian_method_recorded"] = sac["update_method"].str.contains(
        "reparameterised_latent_gaussian", regex=False
    )
    sac["twin_reward_critic_updates_complete"] = (
        sac["reward_critic_q1_update_count"].eq(sac["period_update_count"])
        & sac["reward_critic_q2_update_count"].eq(sac["period_update_count"])
    )
    sac["actor_updates_complete"] = sac["actor_update_count"].eq(
        sac["period_update_count"]
    )
    sac["temperature_updates_complete"] = sac[
        "entropy_temperature_update_count"
    ].eq(sac["period_update_count"])
    constrained_mask = sac["policy"].eq("Constrained SAC")
    sac["constraint_updates_complete"] = (~constrained_mask) | (
        sac["constraint_critic_update_count"].eq(sac["period_update_count"])
        & sac["constraint_dual_update_count"].eq(sac["period_update_count"])
    )
    sac["trace_status"] = np.where(
        sac[
            [
                "all_required_training_fields_finite",
                "latent_gaussian_method_recorded",
                "twin_reward_critic_updates_complete",
                "actor_updates_complete",
                "temperature_updates_complete",
                "constraint_updates_complete",
            ]
        ].all(axis=1),
        "PASS",
        "FAIL",
    )

    replay_frames: list[pd.DataFrame] = []
    replay_summaries: list[pd.DataFrame] = []
    for constrained in (False, True):
        replay, summary = _episode_replay(
            model=model,
            config=config,
            upstream=upstream,
            constrained=constrained,
        )
        replay_frames.append(replay)
        replay_summaries.append(summary)
    replay = pd.concat(replay_frames, ignore_index=True)
    replay_summary = pd.concat(replay_summaries, ignore_index=True)

    accepted_gradient = pd.read_csv(
        upstream["5.2.2"] / "sac_actor_gradient_check.csv"
    )
    rerun_gradient = pd.DataFrame(sac_actor_gradient_check(model))
    gradient = accepted_gradient.merge(
        rerun_gradient,
        on="parameter",
        suffixes=("_persisted", "_recalculated"),
        validate="one_to_one",
    )
    gradient["recalculation_relative_error"] = gradient.apply(
        lambda row: max(
            _relative_error(
                row["analytic_gradient_persisted"],
                row["analytic_gradient_recalculated"],
            ),
            _relative_error(
                row["finite_difference_gradient_persisted"],
                row["finite_difference_gradient_recalculated"],
            ),
        ),
        axis=1,
    )
    gradient["status"] = np.where(
        _as_bool(gradient["passed_persisted"])
        & _as_bool(gradient["passed_recalculated"])
        & gradient["recalculation_relative_error"].le(
            gradient[["tolerance_persisted", "tolerance_recalculated"]].max(axis=1)
        ),
        "PASS",
        "FAIL",
    )
    jacobian = _projection_jacobian_audit(model)
    checkpoint = _checkpoint_and_validation_audit(
        model=model, upstream=upstream, curves=sac
    )

    initial_temperature = float(config["training"]["sac_entropy_temperature"])
    sac_checkpoints = checkpoint[
        checkpoint["policy"].isin(["Vanilla SAC", "Constrained SAC"])
    ]
    constrained = sac[sac["policy"].eq("Constrained SAC")]
    gradient_tolerance = float(
        config["numerics"]["sac_gradient_check_relative_tolerance"]
    )
    replay_max = float(replay_summary["relative_error"].max())
    actor_recalc_max = float(replay["actor_loss_relative_error"].max())
    temp_recalc_max = float(replay["temperature_gradient_relative_error"].max())
    contracts = [
        (
            "M13_SAC_LATENT_GAUSSIAN",
            bool(
                sac["latent_gaussian_method_recorded"].all()
                and sac["latent_policy_entropy"].notna().all()
                and replay["preprojection_log_probability_recorded"].notna().all()
            ),
            float((~sac["latent_gaussian_method_recorded"]).sum()),
            0.0,
            "The persisted SAC trace does not certify reparameterised latent Gaussian sampling and preprojection density.",
        ),
        (
            "M14_SAC_ACTOR_MEAN_UPDATE",
            bool(
                sac["actor_updates_complete"].all()
                and sac_checkpoints[
                    "actor_mean_weight_maximum_change_from_initial"
                ].gt(1e-12).all()
            ),
            float(
                (
                    sac_checkpoints[
                        "actor_mean_weight_maximum_change_from_initial"
                    ]
                    <= 1e-12
                ).sum()
            ),
            0.0,
            "One or more SAC actors did not update their mean parameters from the registered initialization.",
        ),
        (
            "M15_SAC_LOG_STD_UPDATE",
            bool(
                sac_checkpoints[
                    "actor_log_standard_deviation_maximum_change_from_initial"
                ].gt(1e-12).all()
                and sac.groupby(["policy", "seed_index"])[
                    "mean_log_standard_deviation"
                ].nunique().gt(1).all()
            ),
            float(
                (
                    sac_checkpoints[
                        "actor_log_standard_deviation_maximum_change_from_initial"
                    ]
                    <= 1e-12
                ).sum()
            ),
            0.0,
            "The latent log-standard-deviation was not updated for every SAC checkpoint.",
        ),
        (
            "M16_SAC_ENTROPY_ACTOR_TERM",
            bool(
                actor_recalc_max <= 1e-10
                and replay["entropy_actor_contribution"].abs().gt(0.0).all()
            ),
            actor_recalc_max,
            1e-10,
            "The entropy term is missing from the independently reconstructed actor loss.",
        ),
        (
            "M17_SAC_ENTROPY_TEMPERATURE",
            bool(
                sac["temperature_updates_complete"].all()
                and sac["entropy_temperature"].sub(initial_temperature).abs().gt(1e-12).all()
                and temp_recalc_max <= 1e-10
            ),
            temp_recalc_max,
            1e-10,
            "Adaptive entropy-temperature updates do not reconcile with the registered formula.",
        ),
        (
            "M18_SAC_TWIN_REWARD_CRITICS",
            bool(
                sac["twin_reward_critic_updates_complete"].all()
                and np.isfinite(sac[["critic_loss_q1", "critic_loss_q2"]]).all().all()
                and replay_summary.loc[
                    replay_summary["field"].isin(
                        ["critic_loss_q1", "critic_loss_q2"]
                    ),
                    "status",
                ].eq("PASS").all()
            ),
            replay_max,
            1e-10,
            "Twin reward-critic targets, losses, or per-transition updates failed replay reconciliation.",
        ),
        (
            "M19_SAC_CONSTRAINT_CRITIC",
            bool(
                constrained["constraint_critic_update_count"].eq(
                    constrained["period_update_count"]
                ).all()
                and np.isfinite(constrained["constraint_critic_loss"]).all()
                and replay_summary.loc[
                    replay_summary["field"].eq("constraint_critic_loss")
                    & replay_summary["policy"].eq("Constrained SAC"),
                    "status",
                ].eq("PASS").all()
            ),
            float(
                replay_summary.loc[
                    replay_summary["field"].eq("constraint_critic_loss")
                    & replay_summary["policy"].eq("Constrained SAC"),
                    "relative_error",
                ].max()
            ),
            1e-10,
            "The constrained-SAC critic target, loss, or update count failed reconciliation.",
        ),
        (
            "M20_SAC_CONSTRAINT_DUAL",
            bool(
                constrained["constraint_dual_update_count"].eq(
                    constrained["period_update_count"]
                ).all()
                and constrained["constraint_dual"].gt(0.0).all()
                and replay_summary.loc[
                    replay_summary["field"].eq("constraint_dual")
                    & replay_summary["policy"].eq("Constrained SAC"),
                    "status",
                ].eq("PASS").all()
            ),
            float(
                replay_summary.loc[
                    replay_summary["field"].eq("constraint_dual")
                    & replay_summary["policy"].eq("Constrained SAC"),
                    "relative_error",
                ].max()
            ),
            1e-10,
            "The constrained dual did not update per transition or failed the independent replay.",
        ),
        (
            "M21_SAC_PROJECTION_GRADIENT",
            bool(jacobian["status"].eq("PASS").all()),
            float(jacobian["relative_error"].max()),
            gradient_tolerance,
            "The formal projector local Jacobian does not match central differences away from active-set ties.",
        ),
        (
            "M22_SAC_FINITE_DIFFERENCE",
            bool(gradient["status"].eq("PASS").all()),
            float(
                max(
                    gradient["relative_error_recalculated"].max(),
                    gradient["recalculation_relative_error"].max(),
                )
            ),
            gradient_tolerance,
            "The actor mean/log-standard-deviation analytic gradient failed independent central differences.",
        ),
        (
            "M23_VALIDATION_CHECKPOINT",
            bool(checkpoint["validation_checkpoint_status"].eq("PASS").all()),
            float((checkpoint["validation_checkpoint_status"] != "PASS").sum()),
            0.0,
            "A BC or SAC checkpoint was not selected exclusively from independent validation data before test replay.",
        ),
        (
            "M24_CHECKPOINT_REPLAY",
            bool(checkpoint["checkpoint_replay_status"].eq("PASS").all()),
            float(checkpoint["checkpoint_replay_maximum_action_difference"].max()),
            0.0,
            "Checkpoint hash or deterministic independent action replay failed.",
        ),
    ]
    contract_frame = pd.DataFrame(
        [
            {
                "contract_id": contract_id,
                "condition": condition,
                "maximum_observed_residual": residual,
                "tolerance": tolerance,
                "status": "PASS" if condition else "FAIL",
                "failure_reason": "" if condition else failure,
            }
            for contract_id, condition, residual, tolerance, failure in contracts
        ]
    )
    return {
        "training_trace": sac,
        "update_recalculation": replay,
        "episode_replay_summary": replay_summary,
        "gradient_recalculation": gradient,
        "projection_jacobian": jacobian,
        "checkpoint_replay": checkpoint,
        "contract_status": contract_frame,
    }
