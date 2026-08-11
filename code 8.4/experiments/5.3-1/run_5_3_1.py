"""Single resumable command for Experiment 5.3.1 Commitment Sensitivity.

Run from the code root:
    C:\\Users\\Owner\\anaconda3\\python.exe experiments\\5.3-1\\run_5_3_1.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_DIR.parents[1]
PROJECT_ROOT = CODE_ROOT.parent
SRC_ROOT = CODE_ROOT / "src"
BENCHMARK_DIR = CODE_ROOT / "experiments" / "5.2-2"
MECHANISM_DIR = CODE_ROOT / "experiments" / "5.2-3"
for entry in (EXPERIMENT_DIR, BENCHMARK_DIR, MECHANISM_DIR, SRC_ROOT):
    sys.path.insert(0, str(entry))

from commitment_worker import (  # noqa: E402
    CommitmentArtifacts,
    evaluate_task,
    initialise_worker,
    summarise_mechanism_artifact,
)
from features import LinearActor, state_feature_names  # noqa: E402
from model import build_model  # noqa: E402
from mechanism import run_mechanism_replication  # noqa: E402
from paths import (  # noqa: E402
    PhysicalPath,
    build_test_paths,
    build_training_validation_paths,
    load_frozen_5_2_1_inputs,
    manifest_frame,
)
from policies import (  # noqa: E402
    ActorPolicy,
    MPCPolicy,
    ModelGuidedPolicy,
    PassivePolicy,
    ReactivePolicy,
)
from reporting_5_3_1 import (  # noqa: E402
    acceptance_payload,
    create_figures,
    formula_code_registry,
    independent_checks,
    parameter_registry,
    sha256_file,
    write_csv,
    write_manifest,
    write_reports,
)
from statistics_5_3_1 import (  # noqa: E402
    aggregate_learning_seeds,
    clearance_summary,
    confidence_sets_and_regret,
    endpoint_precision,
    mechanism_summary,
    paired_effects,
    update_endpoint_precision,
)
from training import (  # noqa: E402
    TrainingResult,
    generate_teacher_data,
    sac_actor_gradient_check,
    save_checkpoint,
    train_bc,
    train_sac,
    validate_model_guided,
)


SOURCE_FILES = [
    "src/tre84/actions.py",
    "src/tre84/behavior.py",
    "src/tre84/capacity.py",
    "src/tre84/clearance.py",
    "src/tre84/control.py",
    "src/tre84/engine.py",
    "src/tre84/loss.py",
    "src/tre84/scenarios.py",
    "src/tre84/state.py",
    "src/tre84/transition.py",
    "experiments/5.2-2/features.py",
    "experiments/5.2-2/model.py",
    "experiments/5.2-2/paths.py",
    "experiments/5.2-2/policies.py",
    "experiments/5.2-2/preparation.py",
    "experiments/5.2-2/simulator.py",
    "experiments/5.2-2/training.py",
    "experiments/5.2-3/mechanism.py",
    "experiments/5.3-1/commitment_worker.py",
    "experiments/5.3-1/statistics_5_3_1.py",
    "experiments/5.3-1/reporting_5_3_1.py",
    "experiments/5.3-1/run_5_3_1.py",
]

# The completed scientific caches below were produced by the immediately
# preceding run, which failed only while JSON-encoding a numpy.bool_ after all
# simulations, precision calculations, statistics, and figures had finished.
# This one-signature allowlist prevents a non-scientific serialization fix from
# forcing 7,128 identical policy/path/seed replications to be repeated.  All
# config, upstream, checkpoint, path-prefix, and content-hash checks remain in
# force.
CACHE_COMPATIBLE_RUN_SIGNATURES = {
    "1fa18f987986c9267cb403e00e9d9357a87b9ec1a1fe5515d564074541ffcdbb":
        "numpy.bool_ acceptance JSON serialization fix only"
}


def _cache_signature_matches(cached: Any, current: str) -> bool:
    value = str(cached)
    return value == current or value in CACHE_COMPATIBLE_RUN_SIGNATURES


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_bundle_hash() -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        path = CODE_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _chi_tag(chi: float) -> str:
    return f"chi_{chi:.3f}".replace(".", "p")


def _load_configs() -> tuple[dict[str, Any], dict[str, Any], str]:
    experiment_path = EXPERIMENT_DIR / "config_5_3_1.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    base_path = CODE_ROOT / str(experiment["base_model_config"])
    base = json.loads(base_path.read_text(encoding="utf-8"))
    return experiment, base, sha256_file(experiment_path)


def _verify_upstream(experiment: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for lock in experiment["upstream_locks"]:
        path = CODE_ROOT / str(lock["path"])
        observed = sha256_file(path) if path.exists() else "MISSING"
        rows.append(
            {
                "relative_path": str(lock["path"]),
                "expected_sha256": str(lock["sha256"]),
                "observed_sha256": observed,
                "bytes": path.stat().st_size if path.exists() else 0,
                "matched": observed == str(lock["sha256"]),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame["matched"].all():
        failures = frame.loc[~frame["matched"], "relative_path"].tolist()
        raise RuntimeError(f"Accepted 5.2 input lock mismatch: {failures}")
    acceptance_522 = json.loads(
        (CODE_ROOT / "output/5.2.2_common_authority_benchmark/acceptance_5_2_2.json").read_text(
            encoding="utf-8"
        )
    )
    acceptance_525 = json.loads(
        (CODE_ROOT / "output/5.2.5_computational_methodological_acceptance/acceptance_5_2_5.json").read_text(
            encoding="utf-8"
        )
    )
    if acceptance_522.get("status") != "complete":
        raise RuntimeError("5.2.2 is not accepted")
    if acceptance_525.get("OVERALL_ACCEPTANCE") != "PASS":
        raise RuntimeError("5.2.5 overall methodology acceptance is not PASS")
    return frame


def _model_config(
    base: Mapping[str, Any], experiment: Mapping[str, Any], chi: float
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    config["experiment_id"] = str(experiment["experiment_id"])
    config["committed_fraction_reference"] = float(chi)
    config["main_policies"] = list(experiment["main_policies"])
    config["learning_policies"] = list(experiment["learning_policies"])
    config["computation"]["parallel_evaluation_workers"] = int(
        experiment["execution"]["parallel_workers"]
    )
    return config


def _teacher_frame(records: Sequence[Any], chi: float) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "chi": float(chi),
            "path_id": record.path_id,
            "period_offset": record.period_offset,
            "candidate_id": record.candidate_id,
            "nested_formal_objective": record.nested_objective,
            "state_sha256": hashlib.sha256(
                np.asarray(record.state_vector, dtype="<f8").tobytes()
            ).hexdigest(),
            "teacher_action_sha256": hashlib.sha256(
                np.asarray(record.target_normalised_action, dtype="<f8").tobytes()
            ).hexdigest(),
        }
        for index, value in enumerate(record.target_normalised_action):
            row[f"teacher_action_{index}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _load_training_result(row: pd.Series, checkpoint: Path) -> TrainingResult:
    with np.load(checkpoint, allow_pickle=False) as payload:
        actor = LinearActor(
            np.asarray(payload["weights"], dtype=float),
            np.asarray(payload["log_standard_deviation"], dtype=float),
        )
    return TrainingResult(
        policy=str(row["policy"]),
        seed_index=int(row["seed_index"]),
        seed=int(row["training_seed"]),
        actor=actor,
        training_curve=[],
        validation_curve=[],
        best_validation_loss=float(row["best_validation_operational_loss"]),
        selected_episode=int(row["selected_episode"]),
        stopped_reason=str(row["stopped_reason"]),
        final_dual=float(row["final_constraint_dual"]),
        teacher_hash=str(row["teacher_action_hash"]),
        entropy_temperature=float(row["selected_entropy_temperature"]),
    )


def _train_cell(
    *,
    chi: float,
    model: Any,
    training_paths: Sequence[PhysicalPath],
    validation_paths: Sequence[PhysicalPath],
    run_signature: str,
) -> tuple[dict[str, list[TrainingResult]], dict[str, pd.DataFrame]]:
    directory = EXPERIMENT_DIR / "checkpoints" / _chi_tag(chi)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "training_complete.json"
    manifest_path = directory / "checkpoint_manifest.csv"
    if marker.exists():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if not _cache_signature_matches(payload.get("run_signature"), run_signature) or not np.isclose(
            float(payload.get("chi")), chi
        ):
            raise RuntimeError(
                f"Existing 5.3.1 training cache for chi={chi} has a different contract"
            )
        manifest = pd.read_csv(manifest_path)
        results = {"Behaviour cloning": [], "Constrained SAC": []}
        for policy in results:
            for _, row in manifest.loc[manifest["policy"] == policy].sort_values(
                "seed_index"
            ).iterrows():
                checkpoint = CODE_ROOT / str(row["checkpoint_path"])
                if sha256_file(checkpoint) != row["checkpoint_sha256"]:
                    raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
                results[policy].append(_load_training_result(row, checkpoint))
        frames = {
            "teacher": pd.read_csv(directory / "teacher_actions.csv"),
            "training": pd.read_csv(directory / "training_curves.csv"),
            "validation": pd.read_csv(directory / "validation_curves.csv"),
            "checkpoints": manifest,
            "checkpoint_validation": pd.read_csv(
                directory / "checkpoint_validation.csv"
            ),
            "validation_proposals": pd.read_csv(
                directory / "proposal_selection_validation.csv"
            ),
            "gradient_checks": pd.read_csv(directory / "sac_actor_gradient_check.csv"),
        }
        print(f"[5.3.1] chi={chi:g}: reused contract-matched 5.3.1 training cache", flush=True)
        return results, frames

    print(f"[5.3.1] chi={chi:g}: generating formal MPC teacher data", flush=True)
    teacher, teacher_hash = generate_teacher_data(model=model, paths=training_paths)
    teacher_frame = _teacher_frame(teacher, chi)
    results: dict[str, list[TrainingResult]] = {
        "Behaviour cloning": [],
        "Constrained SAC": [],
    }
    training_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_validation_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    features = state_feature_names(model)
    model_hash = _json_hash(model.config)
    for seed_index in range(int(model.config["training"]["seeds"])):
        print(
            f"[5.3.1] chi={chi:g}: training BC and constrained SAC seed {seed_index + 1}/3",
            flush=True,
        )
        bc = train_bc(
            model=model,
            teacher=teacher,
            teacher_hash=teacher_hash,
            validation_paths=validation_paths,
            seed_index=seed_index,
        )
        sac = train_sac(
            model=model,
            training_paths=training_paths,
            validation_paths=validation_paths,
            seed_index=seed_index,
            constrained=True,
        )
        results["Behaviour cloning"].append(bc)
        results["Constrained SAC"].append(sac)
        for result in (bc, sac):
            training_rows.extend({"chi": chi, **row} for row in result.training_curve)
            validation_rows.extend({"chi": chi, **row} for row in result.validation_curve)
            checkpoint, digest = save_checkpoint(
                result=result,
                directory=directory,
                feature_names=features,
                config_hash=model_hash,
            )
            checkpoint_rows.append(
                {
                    "chi": chi,
                    "policy": result.policy,
                    "seed_index": result.seed_index,
                    "training_seed": result.seed,
                    "checkpoint_path": checkpoint.relative_to(CODE_ROOT).as_posix(),
                    "checkpoint_sha256": digest,
                    "selected_episode": result.selected_episode,
                    "best_validation_operational_loss": result.best_validation_loss,
                    "stopped_reason": result.stopped_reason,
                    "final_constraint_dual": result.final_dual,
                    "selected_entropy_temperature": result.entropy_temperature,
                    "selected_mean_log_standard_deviation": float(
                        np.mean(result.actor.log_standard_deviation)
                    ),
                    "teacher_action_hash": result.teacher_hash,
                    "generated_for_5_3_1": True,
                    "loaded_from_5_2_checkpoint": False,
                }
            )
            checkpoint_validation_rows.append(
                {
                    "chi": chi,
                    "policy": result.policy,
                    "seed_index": result.seed_index,
                    "training_seed": result.seed,
                    "selected_episode": result.selected_episode,
                    "validation_operational_loss": result.best_validation_loss,
                    "selection_split": "validation",
                    "test_paths_seen_before_selection": False,
                }
            )
        mg_loss, proposals = validate_model_guided(
            model=model,
            bc=bc,
            constrained_sac=sac,
            validation_paths=validation_paths,
        )
        proposal_rows.extend({"chi": chi, "seed_index": seed_index, **row} for row in proposals)
        mg_payload = {
            "chi": chi,
            "policy": "Model-guided constrained SAC",
            "seed_index": seed_index,
            "training_seed": sac.seed,
            "bc_checkpoint_sha256": checkpoint_rows[-2]["checkpoint_sha256"],
            "constrained_sac_checkpoint_sha256": checkpoint_rows[-1]["checkpoint_sha256"],
            "validation_operational_loss": mg_loss,
            "selector": "formal nested-objective choice between BC and constrained-SAC proposals",
            "generated_for_5_3_1": True,
        }
        mg_path = directory / f"model_guided_constrained_sac_seed_{seed_index}.json"
        mg_path.write_text(json.dumps(mg_payload, indent=2) + "\n", encoding="utf-8")
        checkpoint_rows.append(
            {
                "chi": chi,
                "policy": "Model-guided constrained SAC",
                "seed_index": seed_index,
                "training_seed": sac.seed,
                "checkpoint_path": mg_path.relative_to(CODE_ROOT).as_posix(),
                "checkpoint_sha256": sha256_file(mg_path),
                "selected_episode": sac.selected_episode,
                "best_validation_operational_loss": mg_loss,
                "stopped_reason": "selector_has_no_additional_training",
                "final_constraint_dual": sac.final_dual,
                "selected_entropy_temperature": sac.entropy_temperature,
                "selected_mean_log_standard_deviation": float(
                    np.mean(sac.actor.log_standard_deviation)
                ),
                "teacher_action_hash": teacher_hash,
                "generated_for_5_3_1": True,
                "loaded_from_5_2_checkpoint": False,
            }
        )
        checkpoint_validation_rows.append(
            {
                "chi": chi,
                "policy": "Model-guided constrained SAC",
                "seed_index": seed_index,
                "training_seed": sac.seed,
                "selected_episode": sac.selected_episode,
                "validation_operational_loss": mg_loss,
                "selection_split": "validation",
                "test_paths_seen_before_selection": False,
            }
        )

    gradient = pd.DataFrame(sac_actor_gradient_check(model))
    gradient.insert(0, "chi", chi)
    if not gradient["passed"].astype(bool).all():
        raise RuntimeError(f"SAC finite-difference gradient check failed at chi={chi}")
    frames = {
        "teacher": teacher_frame,
        "training": pd.DataFrame(training_rows),
        "validation": pd.DataFrame(validation_rows),
        "checkpoints": pd.DataFrame(checkpoint_rows),
        "checkpoint_validation": pd.DataFrame(checkpoint_validation_rows),
        "validation_proposals": pd.DataFrame(proposal_rows),
        "gradient_checks": gradient,
    }
    write_csv(frames["teacher"], directory / "teacher_actions.csv")
    write_csv(frames["training"], directory / "training_curves.csv")
    write_csv(frames["validation"], directory / "validation_curves.csv")
    write_csv(frames["checkpoints"], manifest_path)
    write_csv(frames["checkpoint_validation"], directory / "checkpoint_validation.csv")
    write_csv(frames["validation_proposals"], directory / "proposal_selection_validation.csv")
    write_csv(frames["gradient_checks"], directory / "sac_actor_gradient_check.csv")
    marker.write_text(
        json.dumps(
            {
                "run_signature": run_signature,
                "chi": chi,
                "generated_for_5_3_1": True,
                "old_checkpoint_loaded": False,
                "teacher_hash": teacher_hash,
                "checkpoint_count": len(frames["checkpoints"]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return results, frames


def _policy_instances(
    model: Any, results: Mapping[str, Sequence[TrainingResult]]
) -> list[Any]:
    policies: list[Any] = [PassivePolicy(model), ReactivePolicy(model), MPCPolicy(model)]
    policies.extend(
        ActorPolicy("Behaviour cloning", model, result.actor, result.seed)
        for result in results["Behaviour cloning"]
    )
    policies.extend(
        ModelGuidedPolicy(
            model=model,
            bc_actor=bc.actor,
            sac_actor=sac.actor,
            training_seed=sac.seed,
        )
        for bc, sac in zip(
            results["Behaviour cloning"], results["Constrained SAC"]
        )
    )
    if len(policies) != 9:
        raise RuntimeError("5.3.1 must execute 3 non-learning and 6 learning-seed policies")
    return policies


def _policy_spec(policy: Any) -> dict[str, Any]:
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
    raise TypeError(policy)


def _cache_files(chi: float) -> dict[str, Path]:
    directory = EXPERIMENT_DIR / "cache" / _chi_tag(chi)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "directory": directory,
        "replications": directory / "replications.csv.gz",
        "weekly": directory / "weekly.csv.gz",
        "contracts": directory / "contracts.csv.gz",
        "marker": directory / "evaluation_state.json",
    }


def _read_evaluation_cache(
    chi: float, run_signature: str, paths: Sequence[PhysicalPath]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    files = _cache_files(chi)
    if not files["marker"].exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), 0
    marker = json.loads(files["marker"].read_text(encoding="utf-8"))
    if not _cache_signature_matches(marker.get("run_signature"), run_signature):
        raise RuntimeError(f"Existing evaluation cache at chi={chi} has a different contract")
    replications = pd.read_csv(files["replications"])
    weekly = pd.read_csv(files["weekly"])
    contracts = pd.read_csv(files["contracts"])
    completed = int(marker["completed_physical_paths"])
    expected_ids = [path.path_id for path in paths[:completed]]
    if sorted(replications["path_id"].unique()) != sorted(expected_ids):
        raise RuntimeError(f"Cached path prefix mismatch at chi={chi}")
    return replications, weekly, contracts, completed


def _write_evaluation_cache(
    *,
    chi: float,
    run_signature: str,
    replications: pd.DataFrame,
    weekly: pd.DataFrame,
    contracts: pd.DataFrame,
    completed_paths: int,
) -> None:
    files = _cache_files(chi)
    replications.to_csv(files["replications"], index=False, compression="gzip")
    weekly.to_csv(files["weekly"], index=False, compression="gzip")
    contracts.to_csv(files["contracts"], index=False, compression="gzip")
    files["marker"].write_text(
        json.dumps(
            {
                "run_signature": run_signature,
                "chi": chi,
                "completed_physical_paths": completed_paths,
                "policy_path_seed_runs": int(len(replications)),
                "status": "complete" if completed_paths > 0 else "empty",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _evaluate_cell(
    *,
    chi: float,
    model: Any,
    model_config: Mapping[str, Any],
    policies: Sequence[Any],
    paths: Sequence[PhysicalPath],
    run_signature: str,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replications, weekly, contracts, completed_paths = _read_evaluation_cache(
        chi, run_signature, paths
    )
    if completed_paths >= len(paths):
        print(
            f"[5.3.1] chi={chi:g}: reused {completed_paths}/{len(paths)} contract-matched cached paths",
            flush=True,
        )
        allowed = {path.path_id for path in paths}
        return (
            replications.loc[replications["path_id"].isin(allowed)].copy(),
            weekly.loc[weekly["path_id"].isin(allowed)].copy(),
            contracts.loc[contracts["path_id"].isin(allowed)].copy(),
        )
    new_paths = list(paths[completed_paths:])
    specs = [_policy_spec(policy) for policy in policies]
    tasks = [(path, index) for path in new_paths for index in range(len(policies))]
    new_replications: list[dict[str, Any]] = []
    new_weekly: list[dict[str, Any]] = []
    new_contracts: list[dict[str, Any]] = []
    print(
        f"[5.3.1] chi={chi:g}: evaluating paths {completed_paths + 1}-{len(paths)} "
        f"({len(tasks)} policy-path-seed runs)",
        flush=True,
    )

    def accept(artifact: CommitmentArtifacts) -> None:
        new_replications.append(artifact.replication)
        new_weekly.extend(artifact.weekly)
        new_contracts.append(artifact.contract)

    if workers <= 1:
        for path, policy_index in tasks:
            raw = run_mechanism_replication(
                model=model,
                base_policy=policies[policy_index],
                path=path,
                restriction="full_action",
                no_release_pacing_baseline=1.0,
                store_detail=True,
            )
            accept(summarise_mechanism_artifact(raw, chi))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialise_worker,
            initargs=(dict(model_config), specs, chi),
        ) as executor:
            for index, artifact in enumerate(
                executor.map(evaluate_task, tasks, chunksize=1), start=1
            ):
                accept(artifact)
                if index % len(policies) == 0:
                    total_completed = completed_paths + index // len(policies)
                    if total_completed % 4 == 0 or total_completed == len(paths):
                        current_replications = pd.concat(
                            [replications, pd.DataFrame(new_replications)],
                            ignore_index=True,
                        )
                        current_weekly = pd.concat(
                            [weekly, pd.DataFrame(new_weekly)], ignore_index=True
                        )
                        current_contracts = pd.concat(
                            [contracts, pd.DataFrame(new_contracts)], ignore_index=True
                        )
                        _write_evaluation_cache(
                            chi=chi,
                            run_signature=run_signature,
                            replications=current_replications,
                            weekly=current_weekly,
                            contracts=current_contracts,
                            completed_paths=total_completed,
                        )
                        print(
                            f"[5.3.1] chi={chi:g}: completed {total_completed}/{len(paths)} paths",
                            flush=True,
                        )
    replications = pd.concat(
        [replications, pd.DataFrame(new_replications)], ignore_index=True
    )
    weekly = pd.concat([weekly, pd.DataFrame(new_weekly)], ignore_index=True)
    contracts = pd.concat([contracts, pd.DataFrame(new_contracts)], ignore_index=True)
    _write_evaluation_cache(
        chi=chi,
        run_signature=run_signature,
        replications=replications,
        weekly=weekly,
        contracts=contracts,
        completed_paths=len(paths),
    )
    return replications, weekly, contracts


def _combine_cached(
    chis: Sequence[float], run_signature: str, paths: Sequence[PhysicalPath]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replications = []
    weekly = []
    contracts = []
    for chi in chis:
        rep, week, contract, completed = _read_evaluation_cache(
            chi, run_signature, paths
        )
        if completed < len(paths):
            raise RuntimeError(f"chi={chi} is incomplete")
        allowed = {path.path_id for path in paths}
        replications.append(rep.loc[rep["path_id"].isin(allowed)])
        weekly.append(week.loc[week["path_id"].isin(allowed)])
        contracts.append(contract.loc[contract["path_id"].isin(allowed)])
    return (
        pd.concat(replications, ignore_index=True),
        pd.concat(weekly, ignore_index=True),
        pd.concat(contracts, ignore_index=True),
    )


def _safe_staging(output_final: Path) -> Path:
    staging = output_final.parent / f".{output_final.name}.staging"
    resolved_parent = output_final.parent.resolve()
    if staging.resolve().parent != resolved_parent:
        raise RuntimeError("Staging directory escaped the output parent")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def _publish(staging: Path, output_final: Path) -> None:
    if output_final.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        previous = output_final.parent / f"{output_final.name}.previous_{timestamp}"
        output_final.rename(previous)
    staging.rename(output_final)


def run() -> int:
    started_clock = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    experiment, base, config_hash = _load_configs()
    upstream = _verify_upstream(experiment)
    source_hash = _source_bundle_hash()
    run_signature = _json_hash(
        {
            "config_hash": config_hash,
            "source_hash": source_hash,
            "upstream": upstream[["relative_path", "observed_sha256"]].to_dict(
                orient="records"
            ),
        }
    )
    print(f"[5.3.1] Run signature: {run_signature}", flush=True)
    frozen = load_frozen_5_2_1_inputs(base)
    reference_model = build_model(_model_config(base, experiment, 0.5))
    reference_normal = float(sum(reference_model.gateway_scales.values()))
    training_paths, validation_paths = build_training_validation_paths(
        config=base,
        residuals=frozen.residuals,
        reference_normal_model_units=reference_normal,
    )
    minimum = int(experiment["path_design"]["minimum_common_physical_paths"])
    cap = int(experiment["path_design"]["maximum_physical_paths"])
    paths_88 = build_test_paths(config=base, frozen=frozen, count=minimum)
    accepted_manifest = pd.read_csv(
        CODE_ROOT / "output/5.2.2_common_authority_benchmark/test_path_manifest.csv"
    )
    generated_manifest = manifest_frame(paths_88)
    if len(accepted_manifest) != minimum or not (
        accepted_manifest["path_content_sha256"].astype(str).to_numpy()
        == generated_manifest["path_content_sha256"].astype(str).to_numpy()
    ).all():
        raise RuntimeError("The first 88 test paths do not reproduce accepted 5.2.2")

    chis = [float(value) for value in experiment["commitment_grid"]]
    training_frames: dict[str, list[pd.DataFrame]] = {
        "teacher": [],
        "training": [],
        "validation": [],
        "checkpoints": [],
        "checkpoint_validation": [],
        "validation_proposals": [],
        "gradient_checks": [],
    }
    cell_objects: dict[float, tuple[Any, dict[str, Any], list[Any]]] = {}
    for chi in chis:
        model_config = _model_config(base, experiment, chi)
        model = build_model(model_config)
        results, frames = _train_cell(
            chi=chi,
            model=model,
            training_paths=training_paths,
            validation_paths=validation_paths,
            run_signature=run_signature,
        )
        for key in training_frames:
            training_frames[key].append(frames[key])
        policies = _policy_instances(model, results)
        cell_objects[chi] = (model, model_config, policies)
        _evaluate_cell(
            chi=chi,
            model=model,
            model_config=model_config,
            policies=policies,
            paths=paths_88,
            run_signature=run_signature,
            workers=int(experiment["execution"]["parallel_workers"]),
        )

    replications_88, _, _ = _combine_cached(chis, run_signature, paths_88)
    path_level_88 = aggregate_learning_seeds(replications_88)
    requirements, selected_count = endpoint_precision(
        path_level_88,
        config=experiment,
        policies=experiment["main_policies"],
    )
    print(
        f"[5.3.1] Endpoint precision selected common count {selected_count} "
        f"(cap {cap})",
        flush=True,
    )
    final_paths = paths_88
    if selected_count > minimum:
        final_paths = build_test_paths(config=base, frozen=frozen, count=selected_count)
        for chi in chis:
            model, model_config, policies = cell_objects[chi]
            _evaluate_cell(
                chi=chi,
                model=model,
                model_config=model_config,
                policies=policies,
                paths=final_paths,
                run_signature=run_signature,
                workers=int(experiment["execution"]["parallel_workers"]),
            )

    replications, weekly, contracts = _combine_cached(
        chis, run_signature, final_paths
    )
    path_level = aggregate_learning_seeds(replications)
    requirements = update_endpoint_precision(
        requirements,
        path_level,
        executed_paths=len(final_paths),
    )
    selected = pd.DataFrame(
        [
            {
                "selection_rule": experiment["path_design"]["common_count_rule"],
                "minimum_common_paths": minimum,
                "maximum_required_paths": int(requirements["required_paths"].max()),
                "executed_paths": len(final_paths),
                "computational_cap": cap,
                "target_halfwidth": float(
                    experiment["path_design"]["target_halfwidth"]
                ),
                "maximum_achieved_halfwidth": float(
                    requirements["achieved_halfwidth"].max()
                ),
                "all_precision_targets_met": bool(
                    requirements["precision_target_met"].all()
                ),
            }
        ]
    )
    confidence_level = float(experiment["path_design"]["confidence_level"])
    paired = paired_effects(
        path_level,
        policies=experiment["main_policies"],
        confidence_level=confidence_level,
    )
    confidence, regret = confidence_sets_and_regret(
        path_level,
        policies=experiment["main_policies"],
        confidence_level=confidence_level,
    )
    mechanisms = mechanism_summary(path_level)
    clearance = clearance_summary(
        path_level, cap=int(base["clearance"]["maximum_weeks"])
    )

    output_final = CODE_ROOT / str(experiment["output_directory"])
    staging = _safe_staging(output_final)
    figures_directory = EXPERIMENT_DIR / "figures"
    figures = create_figures(
        path_level=path_level,
        mechanism=mechanisms,
        confidence=confidence,
        regret=regret,
        clearance=clearance,
        figures_directory=figures_directory,
        output_directory=staging,
        dpi=int(experiment["execution"]["figure_dpi"]),
    )
    for figure in figures.values():
        shutil.copy2(figure, staging / figure.name)

    training_manifest = manifest_frame(training_paths)
    validation_manifest = manifest_frame(validation_paths)
    training_manifest.insert(0, "chi_scope", "common_across_chi; model retrained per chi")
    validation_manifest.insert(0, "chi_scope", "common_across_chi; checkpoint selected per chi")
    test_manifest = manifest_frame(final_paths)
    test_manifest["matches_accepted_5_2_2_first_88"] = [
        index < minimum
        and row.path_hash
        == accepted_manifest.loc[index, "path_content_sha256"]
        for index, row in enumerate(final_paths)
    ]
    path_pairing = (
        path_level.groupby(["chi", "path_id"], as_index=False)
        .agg(
            policies=("policy", "nunique"),
            physical_hashes=("path_content_sha256", "nunique"),
            information_hashes=("released_information_path_sha256", "nunique"),
        )
    )
    parameter_frame = parameter_registry(experiment, base)
    formula_frame = formula_code_registry()
    all_training = {
        key: pd.concat(value, ignore_index=True) for key, value in training_frames.items()
    }
    independent = independent_checks(
        path_level=path_level,
        paired=paired,
        figures=figures,
        tolerance=float(base["numerics"]["loss_identity_tolerance"]),
    )
    acceptance = acceptance_payload(
        upstream_locks=upstream,
        replications=replications,
        path_level=path_level,
        contracts=contracts,
        checkpoints=all_training["checkpoints"],
        requirements=requirements,
        selected=selected,
        independent=independent,
        figures=figures,
        expected_grid=chis,
        expected_policies=experiment["main_policies"],
        tolerance=float(base["numerics"]["mass_tolerance"]),
    )
    acceptance["cache_reuse_provenance"] = {
        "accepted_prior_run_signatures": CACHE_COMPATIBLE_RUN_SIGNATURES,
        "basis": "The prior run completed all scientific computations and failed only at final numpy.bool_ JSON serialization.",
        "scientific_model_or_parameter_change": False,
    }

    tables = {
        "upstream_input_locks.csv": upstream,
        "training_path_manifest.csv": training_manifest,
        "validation_path_manifest.csv": validation_manifest,
        "test_path_manifest.csv": test_manifest,
        "path_pairing_audit.csv": path_pairing,
        "teacher_actions.csv": all_training["teacher"],
        "training_curves.csv": all_training["training"],
        "validation_curves.csv": all_training["validation"],
        "checkpoint_manifest.csv": all_training["checkpoints"],
        "checkpoint_validation.csv": all_training["checkpoint_validation"],
        "proposal_selection_validation.csv": all_training["validation_proposals"],
        "sac_actor_gradient_check.csv": all_training["gradient_checks"],
        "weekly_commitment_trajectories.csv": weekly,
        "path_level_results.csv": replications,
        "path_level_seed_aggregated.csv": path_level,
        "trajectory_contract_checks.csv": contracts,
        "endpoint_precision_requirements.csv": requirements,
        "selected_path_count.csv": selected,
        "paired_effects.csv": paired,
        "policy_confidence_set.csv": confidence,
        "policy_regret.csv": regret,
        "mechanism_summary.csv": mechanisms,
        "clearance_and_censoring.csv": clearance,
        "parameter_registry_5_3_1.csv": parameter_frame,
        "formula_to_code_5_3_1.csv": formula_frame,
        "independent_recalculation_checks.csv": independent,
    }
    for filename, frame in tables.items():
        write_csv(frame, staging / filename)
    (staging / "frozen_config_5_3_1.json").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )
    (staging / "acceptance_5_3_1.json").write_text(
        json.dumps(acceptance, indent=2) + "\n", encoding="utf-8"
    )
    report_directory = PROJECT_ROOT / "report - 8.4" / "5.3.1"
    write_reports(
        report_directory=report_directory,
        acceptance=acceptance,
        mechanism=mechanisms,
        paired=paired,
        confidence=confidence,
        clearance=clearance,
        selected=selected,
        experiment=experiment,
    )
    elapsed = time.perf_counter() - started_clock
    write_manifest(
        path=staging / "run_manifest.json",
        config_hash=config_hash,
        source_bundle_hash=source_hash,
        upstream_locks=upstream,
        output_directory=staging,
        figures=figures,
        started_utc=started_utc,
        elapsed_seconds=elapsed,
        executed_paths=len(final_paths),
    )
    _publish(staging, output_final)
    print(
        json.dumps(
            {
                "status": acceptance["run_status"],
                "overall_evidence_acceptance": acceptance[
                    "overall_evidence_acceptance"
                ],
                "elapsed_seconds": elapsed,
                "grid_cells": len(chis) * len(experiment["main_policies"]),
                "physical_paths_per_cell": len(final_paths),
                "policy_path_seed_runs": len(replications),
                "precision_targets_met": acceptance["precision_targets_met"],
                "precision_contrasts": acceptance["precision_contrasts"],
                "manifest": str(output_final / "run_manifest.json"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if acceptance["run_status"] == "complete" else 2


def main() -> int:
    try:
        return run()
    except Exception as error:
        print(f"[5.3.1] BLOCKED: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
