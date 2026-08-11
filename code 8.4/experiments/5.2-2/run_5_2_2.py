"""Single reproducible command for experiment 5.2.2.

Run from the code root:
    python experiments/5.2-2/run_5_2_2.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_DIR.parents[1]
SRC_ROOT = CODE_ROOT / "src"
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(SRC_ROOT))

from features import state_feature_names  # noqa: E402
from experiment_validation import policy_nonanticipativity_checks  # noqa: E402
from evaluation_worker import evaluate_task, initialise_worker  # noqa: E402
from model import build_model, route_resource_cost_register  # noqa: E402
from paths import (  # noqa: E402
    PhysicalPath,
    build_test_paths,
    build_training_validation_paths,
    load_frozen_5_2_1_inputs,
    manifest_frame,
    sha256_file,
)
from policies import (  # noqa: E402
    ActorPolicy,
    MPCPolicy,
    ModelGuidedPolicy,
    PassivePolicy,
    ReactivePolicy,
)
from reporting import (  # noqa: E402
    acceptance_payload,
    create_figures,
    parameter_registry,
    policy_authority_register,
    scientific_parameter_traceability,
    write_acceptance_report,
    write_run_manifest,
)
from simulator import ReplicationArtifacts, run_replication  # noqa: E402
from statistics import (  # noqa: E402
    aggregate_learning_seeds,
    clearance_summary,
    decision_time_summary,
    loss_component_summary,
    paired_policy_effects,
    policy_activation_summary,
    policy_confidence_set,
    select_path_count,
    update_precision_achievement,
)
from training import (  # noqa: E402
    TrainingResult,
    generate_teacher_data,
    save_checkpoint,
    train_bc,
    train_ppo,
    train_sac,
    sac_actor_gradient_check,
    validate_model_guided,
)


TABLE_FILES = [
    "training_path_manifest.csv",
    "validation_path_manifest.csv",
    "test_path_manifest.csv",
    "training_curves.csv",
    "validation_curves.csv",
    "checkpoint_manifest.csv",
    "checkpoint_validation.csv",
    "checkpoint_hashes.csv",
    "proposal_selection_log.csv",
    "benchmark_replications.csv",
    "benchmark_period_paths.csv",
    "requested_and_implemented_actions.csv",
    "solver_diagnostics.csv",
    "trajectory_contract_checks.csv",
    "path_level_seed_aggregated.csv",
    "pilot_precision.csv",
    "selected_path_count.csv",
    "paired_policy_effects.csv",
    "policy_confidence_set.csv",
    "loss_component_summary.csv",
    "clearance_summary.csv",
    "policy_activation_summary.csv",
    "decision_time_summary.csv",
    "parameter_registry_5_2_2.csv",
    "policy_authority_register.csv",
    "route_resource_cost_register.csv",
    "waiting_forecast_error_calibration.csv",
    "waiting_forecast_error_residuals.csv",
    "scientific_parameter_traceability.csv",
    "sac_actor_gradient_check.csv",
    "policy_nonanticipativity_checks.csv",
]


def _config() -> tuple[dict[str, Any], Path, str]:
    path = EXPERIMENT_DIR / "config_5_2_2.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    return config, path, digest


def _teacher_frame(records: Sequence[Any]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "path_id": record.path_id,
            "period_offset": record.period_offset,
            "candidate_id": record.candidate_id,
            "nested_formal_objective": record.nested_objective,
            "state_sha256": hashlib.sha256(np.asarray(record.state_vector, dtype="<f8").tobytes()).hexdigest(),
            "teacher_action_sha256": hashlib.sha256(np.asarray(record.target_normalised_action, dtype="<f8").tobytes()).hexdigest(),
        }
        for index, value in enumerate(record.target_normalised_action):
            row[f"teacher_action_{index}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _train(
    *,
    model: Any,
    training_paths: Sequence[PhysicalPath],
    validation_paths: Sequence[PhysicalPath],
    checkpoint_directory: Path,
    config_hash: str,
) -> tuple[
    dict[str, list[TrainingResult]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    print("[5.2.2] Generating formal stochastic-MPC demonstrations...", flush=True)
    teacher, teacher_hash = generate_teacher_data(model=model, paths=training_paths)
    teacher_frame = _teacher_frame(teacher)
    results: dict[str, list[TrainingResult]] = {
        "Behaviour cloning": [],
        "PPO": [],
        "Vanilla SAC": [],
        "Constrained SAC": [],
    }
    training_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_validation_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    features = state_feature_names(model)
    seed_count = int(model.config["training"]["seeds"])
    for seed_index in range(seed_count):
        print(f"[5.2.2] Training seed {seed_index + 1}/{seed_count}: BC, PPO, SAC, constrained SAC...", flush=True)
        batch = [
            train_bc(
                model=model,
                teacher=teacher,
                teacher_hash=teacher_hash,
                validation_paths=validation_paths,
                seed_index=seed_index,
            ),
            train_ppo(
                model=model,
                training_paths=training_paths,
                validation_paths=validation_paths,
                seed_index=seed_index,
            ),
            train_sac(
                model=model,
                training_paths=training_paths,
                validation_paths=validation_paths,
                seed_index=seed_index,
                constrained=False,
            ),
            train_sac(
                model=model,
                training_paths=training_paths,
                validation_paths=validation_paths,
                seed_index=seed_index,
                constrained=True,
            ),
        ]
        for result in batch:
            results[result.policy].append(result)
            training_rows.extend(result.training_curve)
            validation_rows.extend(result.validation_curve)
            checkpoint_path, checkpoint_hash = save_checkpoint(
                result=result,
                directory=checkpoint_directory,
                feature_names=features,
                config_hash=config_hash,
            )
            checkpoint_rows.append(
                {
                    "policy": result.policy,
                    "seed_index": result.seed_index,
                    "training_seed": result.seed,
                    "checkpoint_path": checkpoint_path.relative_to(checkpoint_directory.parent).as_posix(),
                    "checkpoint_sha256": checkpoint_hash,
                    "selected_episode": result.selected_episode,
                    "best_validation_operational_loss": result.best_validation_loss,
                    "stopped_reason": result.stopped_reason,
                    "final_constraint_dual": result.final_dual,
                    "selected_entropy_temperature": result.entropy_temperature,
                    "selected_mean_log_standard_deviation": float(
                        np.mean(result.actor.log_standard_deviation)
                    ),
                    "sac_training_contract": (
                        "latent Gaussian entropy + twin reward critics + constraint critic/dual + learned temperature"
                        if "SAC" in result.policy
                        else "not_applicable"
                    ),
                    "teacher_action_hash": result.teacher_hash,
                    "generated_from_scratch": True,
                    "selected_before_test_replay": True,
                    "old_checkpoint_loaded": False,
                }
            )
            checkpoint_validation_rows.append(
                {
                    "policy": result.policy,
                    "seed_index": result.seed_index,
                    "training_seed": result.seed,
                    "selected_episode": result.selected_episode,
                    "validation_operational_loss": result.best_validation_loss,
                    "selection_data_split": "validation",
                    "test_event_seen_before_selection": False,
                }
            )
        mg_loss, mg_proposals = validate_model_guided(
            model=model,
            bc=results["Behaviour cloning"][-1],
            constrained_sac=results["Constrained SAC"][-1],
            validation_paths=validation_paths,
        )
        for row in mg_proposals:
            proposal_rows.append({"evaluation_split": "validation", **row})
        mg_metadata = {
            "policy": "Model-guided constrained SAC",
            "seed_index": seed_index,
            "training_seed": results["Constrained SAC"][-1].seed,
            "bc_seed": results["Behaviour cloning"][-1].seed,
            "constrained_sac_seed": results["Constrained SAC"][-1].seed,
            "selector": "formal common nested objective over BC and constrained-SAC proposals",
            "validation_operational_loss": mg_loss,
            "generated_from_scratch": True,
        }
        mg_path = checkpoint_directory / f"model_guided_constrained_sac_seed_{seed_index}.json"
        mg_path.write_text(json.dumps(mg_metadata, indent=2) + "\n", encoding="utf-8")
        mg_hash = sha256_file(mg_path)
        checkpoint_rows.append(
            {
                "policy": "Model-guided constrained SAC",
                "seed_index": seed_index,
                "training_seed": results["Constrained SAC"][-1].seed,
                "checkpoint_path": mg_path.relative_to(checkpoint_directory.parent).as_posix(),
                "checkpoint_sha256": mg_hash,
                "selected_episode": np.nan,
                "best_validation_operational_loss": mg_loss,
                "stopped_reason": "selector_has_no_additional_training",
                "final_constraint_dual": results["Constrained SAC"][-1].final_dual,
                "selected_entropy_temperature": results["Constrained SAC"][-1].entropy_temperature,
                "selected_mean_log_standard_deviation": float(
                    np.mean(results["Constrained SAC"][-1].actor.log_standard_deviation)
                ),
                "sac_training_contract": "inherits constrained-SAC checkpoint; selector logic unchanged",
                "teacher_action_hash": teacher_hash,
                "generated_from_scratch": True,
                "selected_before_test_replay": True,
                "old_checkpoint_loaded": False,
            }
        )
        checkpoint_validation_rows.append(
            {
                "policy": "Model-guided constrained SAC",
                "seed_index": seed_index,
                "training_seed": results["Constrained SAC"][-1].seed,
                "selected_episode": np.nan,
                "validation_operational_loss": mg_loss,
                "selection_data_split": "validation",
                "test_event_seen_before_selection": False,
            }
        )
        validation_rows.append(
            {
                "policy": "Model-guided constrained SAC",
                "seed_index": seed_index,
                "training_seed": results["Constrained SAC"][-1].seed,
                "episode": np.nan,
                "validation_operational_loss": mg_loss,
                "improved": True,
                "stale_evaluations": 0,
                "checkpoint_selected": True,
            }
        )
    return (
        results,
        teacher_frame,
        pd.DataFrame(training_rows),
        pd.DataFrame(validation_rows),
        pd.DataFrame(checkpoint_rows),
        pd.DataFrame(checkpoint_validation_rows),
        pd.DataFrame(proposal_rows),
    )


def _policy_instances(model: Any, training: Mapping[str, Sequence[TrainingResult]]) -> list[Any]:
    policies: list[Any] = [PassivePolicy(model), ReactivePolicy(model), MPCPolicy(model)]
    for name in ("Behaviour cloning", "PPO", "Vanilla SAC", "Constrained SAC"):
        policies.extend(ActorPolicy(name, model, result.actor, result.seed) for result in training[name])
    policies.extend(
        ModelGuidedPolicy(
            model=model,
            bc_actor=bc.actor,
            sac_actor=sac.actor,
            training_seed=sac.seed,
        )
        for bc, sac in zip(training["Behaviour cloning"], training["Constrained SAC"])
    )
    return policies


def _evaluate(
    *,
    model: Any,
    policies: Sequence[Any],
    paths: Sequence[PhysicalPath],
    label: str,
) -> list[ReplicationArtifacts]:
    def policy_spec(policy: Any) -> dict[str, Any]:
        if isinstance(policy, PassivePolicy):
            return {"kind": "passive"}
        if isinstance(policy, ReactivePolicy):
            return {"kind": "reactive"}
        if isinstance(policy, MPCPolicy):
            return {"kind": "mpc"}
        if isinstance(policy, ActorPolicy):
            return {
                "kind": "actor",
                "name": policy.name,
                "training_seed": policy.training_seed,
                "weights": policy.actor.weights,
                "log_standard_deviation": policy.actor.log_standard_deviation,
            }
        if isinstance(policy, ModelGuidedPolicy):
            return {
                "kind": "model_guided",
                "training_seed": policy.training_seed,
                "bc_weights": policy.bc_actor.weights,
                "bc_log_standard_deviation": policy.bc_actor.log_standard_deviation,
                "sac_weights": policy.sac_actor.weights,
                "sac_log_standard_deviation": policy.sac_actor.log_standard_deviation,
            }
        raise TypeError(f"Unsupported evaluation policy: {policy.__class__.__name__}")

    artifacts: list[ReplicationArtifacts] = []
    total = len(policies) * len(paths)
    workers = int(model.config["computation"]["parallel_evaluation_workers"])
    specs = [policy_spec(policy) for policy in policies]
    tasks = [(path, policy_index) for path in paths for policy_index in range(len(policies))]
    if workers <= 1:
        for path, policy_index in tasks:
            artifacts.append(
                run_replication(model=model, policy=policies[policy_index], path=path)
            )
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialise_worker,
            initargs=(dict(model.config), specs),
        ) as executor:
            for completed, artifact in enumerate(
                executor.map(evaluate_task, tasks, chunksize=1), start=1
            ):
                artifacts.append(artifact)
                if completed % len(policies) == 0 or completed == total:
                    path_index = completed // len(policies) - 1
                    print(
                        f"[5.2.2] {label}: completed path {paths[path_index].path_id} "
                        f"({completed}/{total} policy-path-seed runs).",
                        flush=True,
                    )
    return artifacts


def _artifact_frames(artifacts: Sequence[ReplicationArtifacts]) -> dict[str, pd.DataFrame]:
    return {
        "replications": pd.DataFrame([item.replication for item in artifacts]),
        "periods": pd.DataFrame([row for item in artifacts for row in item.periods]),
        "actions": pd.DataFrame([row for item in artifacts for row in item.actions]),
        "diagnostics": pd.DataFrame([row for item in artifacts for row in item.solver_diagnostics]),
        "contracts": pd.DataFrame([item.contract_checks for item in artifacts]),
        "proposals": pd.DataFrame([row for item in artifacts for row in item.proposal_records]),
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def run() -> int:
    started = time.perf_counter()
    config, config_path, config_hash = _config()
    output_final = CODE_ROOT / str(config["output_directory"])
    staging = output_final.parent / f".{output_final.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    checkpoint_directory = staging / "checkpoints"
    figure_output = staging / "figures"

    print("[5.2.2] Verifying the unique frozen 5.2.1 interface and constructing the common model...", flush=True)
    frozen = load_frozen_5_2_1_inputs(config)
    model = build_model(config)
    sac_gradient_checks = pd.DataFrame(sac_actor_gradient_check(model))
    if not sac_gradient_checks["passed"].astype(bool).all():
        raise RuntimeError("Independent SAC actor-gradient acceptance failed")
    route_costs = route_resource_cost_register(config)
    reference_normal = float(sum(model.gateway_scales.values()))
    training_paths, validation_paths = build_training_validation_paths(
        config=config,
        residuals=frozen.residuals,
        reference_normal_model_units=reference_normal,
    )
    train_manifest = manifest_frame(training_paths)
    validation_manifest = manifest_frame(validation_paths)
    if set(train_manifest["path_content_sha256"]) & set(validation_manifest["path_content_sha256"]):
        raise RuntimeError("Training and validation path hashes overlap")

    (
        training_results,
        teacher_actions,
        training_curves,
        validation_curves,
        checkpoint_manifest,
        checkpoint_validation,
        validation_proposals,
    ) = _train(
        model=model,
        training_paths=training_paths,
        validation_paths=validation_paths,
        checkpoint_directory=checkpoint_directory,
        config_hash=config_hash,
    )
    # Persist the complete training/validation evidence before any historical
    # test replay. This is an auditable stage boundary, not an old-checkpoint
    # shortcut: every file above was generated from scratch in this command.
    _write_csv(train_manifest, staging / "training_path_manifest.csv")
    _write_csv(validation_manifest, staging / "validation_path_manifest.csv")
    _write_csv(teacher_actions, staging / "teacher_actions.csv")
    _write_csv(training_curves, staging / "training_curves.csv")
    _write_csv(validation_curves, staging / "validation_curves.csv")
    _write_csv(checkpoint_manifest, staging / "checkpoint_manifest.csv")
    _write_csv(checkpoint_validation, staging / "checkpoint_validation.csv")
    _write_csv(
        checkpoint_manifest[
            ["policy", "seed_index", "training_seed", "checkpoint_path", "checkpoint_sha256"]
        ],
        staging / "checkpoint_hashes.csv",
    )
    _write_csv(validation_proposals, staging / "proposal_selection_validation.csv")
    (staging / "training_stage_complete.json").write_text(
        json.dumps(
            {
                "config_sha256": config_hash,
                "historical_interface_sha256": frozen.interface_hash,
                "residual_library_sha256": frozen.residual_hash,
                "generated_from_scratch": True,
                "test_event_replayed": False,
                "checkpoint_count": len(checkpoint_manifest),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    policies = _policy_instances(model, training_results)
    nonanticipativity_checks = pd.DataFrame(
        policy_nonanticipativity_checks(
            model=model,
            policies=policies,
            path=validation_paths[0],
        )
    )
    if not nonanticipativity_checks["passed"].astype(bool).all():
        raise RuntimeError("Production-policy nonanticipativity probe failed")
    expected_policy_runs = 3 + 5 * int(config["training"]["seeds"])
    if len(policies) != expected_policy_runs:
        raise RuntimeError("Policy/seed execution count is incomplete")

    pilot_count = int(config["paths"]["pilot_count"])
    pilot_paths = build_test_paths(config=config, frozen=frozen, count=pilot_count)
    print("[5.2.2] Checkpoints frozen. Running preregistered pilot physical paths...", flush=True)
    pilot_artifacts = _evaluate(model=model, policies=policies, paths=pilot_paths, label="pilot")
    pilot_frames = _artifact_frames(pilot_artifacts)
    pilot_path_level = aggregate_learning_seeds(
        pilot_frames["replications"], learning_policies=config["learning_policies"]
    )
    reference_failure_loss = (
        reference_normal
        * int(config["event_weeks"])
        * float(config["behavior"]["exit_failure_cost_per_unit"])
    )
    pilot_precision, selected_path_count = select_path_count(
        pilot_path_level=pilot_path_level,
        config=config,
        reference_failure_loss=reference_failure_loss,
    )
    executed_count = int(selected_path_count.loc[0, "executed_paths"])
    final_paths = build_test_paths(config=config, frozen=frozen, count=executed_count)
    if [path.path_hash for path in final_paths[:pilot_count]] != [path.path_hash for path in pilot_paths]:
        raise RuntimeError("Pilot paths are not the deterministic prefix of final test paths")
    extra_paths = final_paths[pilot_count:]
    extra_artifacts: list[ReplicationArtifacts] = []
    if extra_paths:
        print(f"[5.2.2] Precision rule selected {executed_count} paths; running {len(extra_paths)} additional matched paths...", flush=True)
        extra_artifacts = _evaluate(model=model, policies=policies, paths=extra_paths, label="final")
    artifacts = pilot_artifacts + extra_artifacts
    frames = _artifact_frames(artifacts)
    if not validation_proposals.empty:
        if frames["proposals"].empty:
            frames["proposals"] = validation_proposals
        else:
            test_proposals = frames["proposals"].copy()
            test_proposals.insert(0, "evaluation_split", "test")
            frames["proposals"] = pd.concat([validation_proposals, test_proposals], ignore_index=True, sort=False)

    path_level = aggregate_learning_seeds(
        frames["replications"], learning_policies=config["learning_policies"]
    )
    paired = paired_policy_effects(
        path_level,
        policies=config["main_policies"],
        confidence_level=float(config["paths"]["confidence_level"]),
    )
    pilot_precision, selected_path_count = update_precision_achievement(
        pilot_precision=pilot_precision,
        selection=selected_path_count,
        paired_effects=paired,
    )
    confidence = policy_confidence_set(
        path_level,
        policies=config["main_policies"],
        confidence_level=float(config["paths"]["confidence_level"]),
    )
    loss_summary = loss_component_summary(path_level)
    clearance = clearance_summary(
        path_level, cap=int(config["clearance"]["maximum_weeks"])
    )
    activation = policy_activation_summary(
        frames["actions"],
        action_names=model.layout.names,
        learning_policies=config["learning_policies"],
    )
    decision_times = decision_time_summary(frames["actions"])
    parameters = parameter_registry(config)
    scientific_trace = scientific_parameter_traceability(config)
    waiting_calibration = pd.read_csv(
        EXPERIMENT_DIR / "waiting_forecast_error_calibration.csv"
    )
    waiting_residuals = pd.read_csv(
        EXPERIMENT_DIR / "waiting_forecast_error_residuals.csv"
    )
    authority = policy_authority_register(config)
    test_manifest = manifest_frame(final_paths)

    checkpoint_hashes = checkpoint_manifest[
        ["policy", "seed_index", "training_seed", "checkpoint_path", "checkpoint_sha256"]
    ].copy()
    output_frames = {
        "training_path_manifest.csv": train_manifest,
        "validation_path_manifest.csv": validation_manifest,
        "test_path_manifest.csv": test_manifest,
        "training_curves.csv": training_curves,
        "validation_curves.csv": validation_curves,
        "checkpoint_manifest.csv": checkpoint_manifest,
        "checkpoint_validation.csv": checkpoint_validation,
        "checkpoint_hashes.csv": checkpoint_hashes,
        "proposal_selection_log.csv": frames["proposals"],
        "benchmark_replications.csv": frames["replications"],
        "benchmark_period_paths.csv": frames["periods"],
        "requested_and_implemented_actions.csv": frames["actions"],
        "solver_diagnostics.csv": frames["diagnostics"],
        "trajectory_contract_checks.csv": frames["contracts"],
        "path_level_seed_aggregated.csv": path_level,
        "pilot_precision.csv": pilot_precision,
        "selected_path_count.csv": selected_path_count,
        "paired_policy_effects.csv": paired,
        "policy_confidence_set.csv": confidence,
        "loss_component_summary.csv": loss_summary,
        "clearance_summary.csv": clearance,
        "policy_activation_summary.csv": activation,
        "decision_time_summary.csv": decision_times,
        "parameter_registry_5_2_2.csv": parameters,
        "policy_authority_register.csv": authority,
        "route_resource_cost_register.csv": route_costs,
        "waiting_forecast_error_calibration.csv": waiting_calibration,
        "waiting_forecast_error_residuals.csv": waiting_residuals,
        "scientific_parameter_traceability.csv": scientific_trace,
        "sac_actor_gradient_check.csv": sac_gradient_checks,
        "policy_nonanticipativity_checks.csv": nonanticipativity_checks,
    }
    for name, frame in output_frames.items():
        _write_csv(frame, staging / name)
    _write_csv(teacher_actions, staging / "teacher_actions.csv")

    figures = create_figures(
        path_level=path_level,
        replications=frames["replications"],
        paired_effects=paired,
        confidence_set=confidence,
        loss_summary=loss_summary,
        clearance=clearance,
        policies=config["main_policies"],
        output_directory=figure_output,
        dpi=int(config["numerics"]["figure_dpi"]),
    )
    acceptance = acceptance_payload(
        config=config,
        frozen=frozen,
        model=model,
        training_manifest=train_manifest,
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        replications=frames["replications"],
        path_level=path_level,
        actions=frames["actions"],
        diagnostics=frames["diagnostics"],
        contracts=frames["contracts"],
        paired_effects=paired,
        pilot_precision=pilot_precision,
        selected_path_count=selected_path_count,
        loss_summary=loss_summary,
        clearance=clearance,
        authority=authority,
        route_costs=route_costs,
        checkpoint_manifest=checkpoint_manifest,
        parameter_registry_frame=parameters,
        scientific_traceability=scientific_trace,
        waiting_calibration=waiting_calibration,
        training_curves=training_curves,
        sac_gradient_checks=sac_gradient_checks,
        nonanticipativity_checks=nonanticipativity_checks,
        figures=figures,
    )
    (staging / "acceptance_5_2_2.json").write_text(
        json.dumps(acceptance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_acceptance_report(
        acceptance=acceptance,
        confidence_set=confidence,
        paired_effects=paired,
        clearance=clearance,
        output_path=staging / "ACCEPTANCE_REPORT.md",
    )
    provenance = {
        "5.2.1 historical information event path": frozen.interface_hash,
        "5.2.1 pre-event residual library": frozen.residual_hash,
        "5.2.2 frozen config": config_hash,
        "waiting forecast error calibration": sha256_file(
            EXPERIMENT_DIR / "waiting_forecast_error_calibration.csv"
        ),
        "waiting forecast error residuals": sha256_file(
            EXPERIMENT_DIR / "waiting_forecast_error_residuals.csv"
        ),
        "chapter_3_4_common_model": "src/tre84 imported by model.py; hashes are in run_manifest code files and package source register",
    }
    (staging / "parameter_source_mapping.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    publish_directory = CODE_ROOT / str(config["figure_directory"])
    publish_directory.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    for figure in figures:
        target = publish_directory / figure.name
        shutil.copy2(figure, target)
        published.append(target)
    command = "python experiments/5.2-2/run_5_2_2.py"
    write_run_manifest(
        output_directory=staging,
        experiment_directory=EXPERIMENT_DIR,
        config_path=config_path,
        frozen=frozen,
        figures_published=published,
        command=command,
    )
    if output_final.exists():
        shutil.rmtree(output_final)
    staging.replace(output_final)
    elapsed = time.perf_counter() - started
    print(f"[5.2.2] Finished in {elapsed:.1f}s. Acceptance: {acceptance['status']}.", flush=True)
    print(f"[5.2.2] Output: {output_final}", flush=True)
    print(f"[5.2.2] Figures: {publish_directory}", flush=True)
    if acceptance["honest_result_warnings"]:
        print("[5.2.2] Honest-result warnings:", flush=True)
        for warning in acceptance["honest_result_warnings"]:
            print(f"  - {warning}", flush=True)
    return 0 if acceptance["status"] == "complete" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
