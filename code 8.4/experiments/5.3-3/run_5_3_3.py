"""Resumable formal runner for Experiment 5.3.3 Gateway Network Sensitivity."""

from __future__ import annotations

import hashlib
import json
import math
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

from features import LinearActor, state_feature_names  # noqa: E402
from gateway_worker import GatewayArtifacts, evaluate_task, initialise_worker  # noqa: E402
from model import build_model  # noqa: E402
from network_5_3_3 import (  # noqa: E402
    NetworkCell, build_cell_config, declared_cells, is_full_policy_anchor,
    network_register, training_cell,
)
from paths import (  # noqa: E402
    PhysicalPath, build_test_paths, build_training_validation_paths,
    load_frozen_5_2_1_inputs, manifest_frame,
)
from policies import ActorPolicy, MPCPolicy, ModelGuidedPolicy, PassivePolicy, ReactivePolicy  # noqa: E402
from reporting_5_3_3 import (  # noqa: E402
    acceptance_payload, create_figures, sha256_file, write_csv, write_manifest,
    write_reports,
)
from statistics_5_3_3 import (  # noqa: E402
    aggregate_learning_seeds, component_values, policy_regret, policy_summary,
    precision_audit,
)
from training import (  # noqa: E402
    TrainingResult, _normalised_projection_jacobian, generate_teacher_data,
    sac_actor_gradient_check, save_checkpoint, train_bc, train_sac,
    validate_model_guided,
)


SOURCE_FILES = [
    "src/tre84/actions.py", "src/tre84/behavior.py", "src/tre84/capacity.py",
    "src/tre84/clearance.py", "src/tre84/control.py", "src/tre84/engine.py",
    "src/tre84/loss.py", "src/tre84/scenarios.py", "src/tre84/state.py",
    "src/tre84/transition.py", "experiments/5.2-2/features.py",
    "experiments/5.2-2/model.py", "experiments/5.2-2/paths.py",
    "experiments/5.2-2/policies.py", "experiments/5.2-2/preparation.py",
    "experiments/5.2-2/simulator.py", "experiments/5.2-2/training.py",
    "experiments/5.2-3/mechanism.py", "experiments/5.3-3/network_5_3_3.py",
    "experiments/5.3-3/gateway_worker.py", "experiments/5.3-3/statistics_5_3_3.py",
    "experiments/5.3-3/reporting_5_3_3.py", "experiments/5.3-3/run_5_3_3.py",
]


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _source_hash() -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        path = CODE_ROOT / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load() -> tuple[dict[str, Any], dict[str, Any], str]:
    path = EXPERIMENT_DIR / "config_5_3_3.json"
    experiment = json.loads(path.read_text(encoding="utf-8"))
    base = json.loads((CODE_ROOT / experiment["base_model_config"]).read_text(encoding="utf-8"))
    return experiment, base, sha256_file(path)


def _verify_upstream(experiment: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for lock in experiment["upstream_locks"]:
        path = CODE_ROOT / str(lock["path"])
        observed = sha256_file(path) if path.exists() else "MISSING"
        rows.append({"relative_path": lock["path"], "expected_sha256": lock["sha256"], "observed_sha256": observed, "bytes": path.stat().st_size if path.exists() else 0, "matched": observed == lock["sha256"]})
    frame = pd.DataFrame(rows)
    if not frame["matched"].all():
        raise RuntimeError(f"Upstream lock mismatch: {frame.loc[~frame['matched'], 'relative_path'].tolist()}")
    a525 = json.loads((CODE_ROOT / "output/5.2.5_computational_methodological_acceptance/acceptance_5_2_5.json").read_text())
    if a525.get("OVERALL_ACCEPTANCE") != "PASS":
        raise RuntimeError("Accepted Chapter 4 methodology gate 5.2.5 is not PASS")
    a532 = json.loads((CODE_ROOT / "output/5.3.2_reclosure_sensitivity/acceptance_5_3_2.json").read_text())
    if a532.get("run_status") != "complete" or a532.get("engineering_acceptance") != "PASS" or a532.get("numerical_acceptance") != "PASS":
        raise RuntimeError("5.3.2 is not frozen as complete engineering/numerical evidence")
    return frame


def _load_result(row: pd.Series, checkpoint: Path) -> TrainingResult:
    with np.load(checkpoint, allow_pickle=False) as payload:
        actor = LinearActor(np.asarray(payload["weights"]), np.asarray(payload["log_standard_deviation"]))
    return TrainingResult(
        policy=str(row.policy), seed_index=int(row.seed_index), seed=int(row.training_seed), actor=actor,
        training_curve=[], validation_curve=[], best_validation_loss=float(row.best_validation_operational_loss),
        selected_episode=int(row.selected_episode), stopped_reason=str(row.stopped_reason),
        final_dual=float(row.final_constraint_dual), teacher_hash=str(row.teacher_action_hash),
        entropy_temperature=float(row.selected_entropy_temperature),
    )


def _teacher_frame(records: Sequence[Any], n: int) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append({
            "gateway_count": n, "path_id": record.path_id, "period_offset": record.period_offset,
            "candidate_id": record.candidate_id, "nested_formal_objective": record.nested_objective,
            "state_sha256": hashlib.sha256(np.asarray(record.state_vector, dtype="<f8").tobytes()).hexdigest(),
            "teacher_action_sha256": hashlib.sha256(np.asarray(record.target_normalised_action, dtype="<f8").tobytes()).hexdigest(),
            "action_dimension": len(record.target_normalised_action),
        })
    return pd.DataFrame(rows)


def _projection_jacobian_audit(model: Any, n: int) -> pd.DataFrame:
    state = model.initial_state({"filtered_high_risk_probability": 0.25, "lead_time_high_risk_probability": 0.5, "release_date": pd.Timestamp("2000-01-01"), "week": pd.Timestamp("2000-01-03")})
    rng = np.random.default_rng(int(hashlib.sha256(f"5.3.3|jacobian|{n}".encode()).hexdigest()[:8], 16))
    actor = LinearActor.random(model, int(rng.integers(0, 2**31 - 1)))
    raw, _, _, _ = actor.sample_latent_normalised(state, model, rng)
    raw_action = model.action_from_normalised(raw)
    projection = model.projector.project(raw_action, state)
    analytic = _normalised_projection_jacobian(model=model, state=state, raw_action=raw_action, projection=projection)
    h = float(model.config["numerics"]["sac_gradient_check_step"])
    tolerance = float(model.config["numerics"]["sac_gradient_check_relative_tolerance"])
    rows = []
    for index in range(raw.size):
        plus, minus = raw.copy(), raw.copy()
        plus[index] += h
        minus[index] -= h
        p = model.normalise_action(model.projector.project(model.action_from_normalised(plus), state).action)
        m = model.normalise_action(model.projector.project(model.action_from_normalised(minus), state).action)
        numerical = (p - m) / (2.0 * h)
        for output, value in enumerate(numerical):
            recorded = float(analytic[output, index])
            relative = abs(recorded - float(value)) / max(1.0, abs(recorded), abs(float(value)))
            rows.append({"gateway_count": n, "input_action_index": index, "output_action_index": output, "analytic": recorded, "finite_difference": float(value), "relative_error": relative, "tolerance": tolerance, "status": "PASS" if relative <= tolerance else "FAIL"})
    return pd.DataFrame(rows)


def _train_size(
    *, n: int, model: Any, training_paths: Sequence[PhysicalPath],
    validation_paths: Sequence[PhysicalPath], run_signature: str,
) -> tuple[dict[str, list[TrainingResult]], dict[str, pd.DataFrame]]:
    directory = EXPERIMENT_DIR / "checkpoints" / f"n{n:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "training_complete.json"
    manifest_path = directory / "checkpoint_manifest.csv"
    if marker.exists():
        marker_data = json.loads(marker.read_text())
        if marker_data.get("run_signature") != run_signature:
            raise RuntimeError(f"Training cache contract mismatch for n={n}")
        manifest = pd.read_csv(manifest_path)
        results = {"Behaviour cloning": [], "Constrained SAC": []}
        for policy in results:
            for _, row in manifest.loc[manifest["policy"] == policy].sort_values("seed_index").iterrows():
                path = CODE_ROOT / row.checkpoint_path
                if sha256_file(path) != row.checkpoint_sha256:
                    raise RuntimeError(f"Checkpoint hash mismatch: {path}")
                results[policy].append(_load_result(row, path))
        frames = {name: pd.read_csv(directory / filename) for name, filename in {
            "teacher": "teacher_actions.csv", "training": "training_curves.csv",
            "validation": "validation_curves.csv", "checkpoints": "checkpoint_manifest.csv",
            "checkpoint_validation": "checkpoint_validation.csv", "validation_proposals": "proposal_selection_validation.csv",
            "gradient": "sac_actor_gradient_check.csv", "jacobian": "projection_jacobian_check.csv",
        }.items()}
        print(f"[5.3.3] n={n}: reused contract-matched training", flush=True)
        return results, frames

    print(f"[5.3.3] n={n}: generating teacher and training BC/full constrained SAC", flush=True)
    teacher, teacher_hash = generate_teacher_data(model=model, paths=training_paths)
    results = {"Behaviour cloning": [], "Constrained SAC": []}
    curves, validation, checkpoint_rows, validation_rows, proposal_rows = [], [], [], [], []
    model_hash = _json_hash(model.config)
    for seed_index in range(int(model.config["training"]["seeds"])):
        bc = train_bc(model=model, teacher=teacher, teacher_hash=teacher_hash, validation_paths=validation_paths, seed_index=seed_index)
        sac = train_sac(model=model, training_paths=training_paths, validation_paths=validation_paths, seed_index=seed_index, constrained=True)
        results["Behaviour cloning"].append(bc)
        results["Constrained SAC"].append(sac)
        for result in (bc, sac):
            curves.extend({"gateway_count": n, **row} for row in result.training_curve)
            validation.extend({"gateway_count": n, **row} for row in result.validation_curve)
            checkpoint, digest = save_checkpoint(result=result, directory=directory, feature_names=state_feature_names(model), config_hash=model_hash)
            checkpoint_rows.append({
                "gateway_count": n, "policy": result.policy, "seed_index": result.seed_index,
                "training_seed": result.seed, "checkpoint_path": checkpoint.relative_to(CODE_ROOT).as_posix(),
                "checkpoint_sha256": digest, "selected_episode": result.selected_episode,
                "best_validation_operational_loss": result.best_validation_loss, "stopped_reason": result.stopped_reason,
                "final_constraint_dual": result.final_dual, "selected_entropy_temperature": result.entropy_temperature,
                "teacher_action_hash": result.teacher_hash, "checkpoint_action_dimension": result.actor.weights.shape[0],
                "expected_action_dimension": len(model.layout.keys), "generated_for_5_3_3": True,
                "loaded_from_5_2_checkpoint": False,
            })
            validation_rows.append({"gateway_count": n, "policy": result.policy, "seed_index": seed_index, "selected_episode": result.selected_episode, "validation_operational_loss": result.best_validation_loss, "test_paths_seen_before_selection": False})
        mg_loss, proposals = validate_model_guided(model=model, bc=bc, constrained_sac=sac, validation_paths=validation_paths)
        proposal_rows.extend({"gateway_count": n, "seed_index": seed_index, **row} for row in proposals)
        mg_path = directory / f"model_guided_constrained_sac_seed_{seed_index}.json"
        mg_path.write_text(json.dumps({"gateway_count": n, "seed_index": seed_index, "training_seed": sac.seed, "bc_checkpoint_sha256": checkpoint_rows[-2]["checkpoint_sha256"], "sac_checkpoint_sha256": checkpoint_rows[-1]["checkpoint_sha256"], "validation_operational_loss": mg_loss}, indent=2) + "\n")
        checkpoint_rows.append({
            "gateway_count": n, "policy": "Model-guided constrained SAC", "seed_index": seed_index,
            "training_seed": sac.seed, "checkpoint_path": mg_path.relative_to(CODE_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(mg_path), "selected_episode": sac.selected_episode,
            "best_validation_operational_loss": mg_loss, "stopped_reason": "selector_has_no_additional_training",
            "final_constraint_dual": sac.final_dual, "selected_entropy_temperature": sac.entropy_temperature,
            "teacher_action_hash": teacher_hash, "checkpoint_action_dimension": sac.actor.weights.shape[0],
            "expected_action_dimension": len(model.layout.keys), "generated_for_5_3_3": True,
            "loaded_from_5_2_checkpoint": False,
        })
        validation_rows.append({"gateway_count": n, "policy": "Model-guided constrained SAC", "seed_index": seed_index, "selected_episode": sac.selected_episode, "validation_operational_loss": mg_loss, "test_paths_seen_before_selection": False})

    gradient = pd.DataFrame(sac_actor_gradient_check(model)); gradient.insert(0, "gateway_count", n)
    jacobian = _projection_jacobian_audit(model, n)
    if not gradient["passed"].astype(bool).all() or not jacobian["status"].eq("PASS").all():
        raise RuntimeError(f"Learning derivative contract failed for n={n}")
    frames = {
        "teacher": _teacher_frame(teacher, n), "training": pd.DataFrame(curves),
        "validation": pd.DataFrame(validation), "checkpoints": pd.DataFrame(checkpoint_rows),
        "checkpoint_validation": pd.DataFrame(validation_rows), "validation_proposals": pd.DataFrame(proposal_rows),
        "gradient": gradient, "jacobian": jacobian,
    }
    filenames = {
        "teacher": "teacher_actions.csv", "training": "training_curves.csv", "validation": "validation_curves.csv",
        "checkpoints": "checkpoint_manifest.csv", "checkpoint_validation": "checkpoint_validation.csv",
        "validation_proposals": "proposal_selection_validation.csv", "gradient": "sac_actor_gradient_check.csv",
        "jacobian": "projection_jacobian_check.csv",
    }
    for key, filename in filenames.items():
        frames[key].to_csv(directory / filename, index=False)
    marker.write_text(json.dumps({"run_signature": run_signature, "gateway_count": n, "status": "complete"}, indent=2) + "\n")
    return results, frames


def _policies(model: Any, results: Mapping[str, Sequence[TrainingResult]], full: bool) -> list[Any]:
    policies: list[Any] = [PassivePolicy(model), ReactivePolicy(model)]
    policies += [ActorPolicy("Behaviour cloning", model, result.actor, result.seed) for result in results["Behaviour cloning"]]
    if full:
        policies.insert(2, MPCPolicy(model))
        policies += [ModelGuidedPolicy(model=model, bc_actor=bc.actor, sac_actor=sac.actor, training_seed=sac.seed) for bc, sac in zip(results["Behaviour cloning"], results["Constrained SAC"])]
    return policies


def _spec(policy: Any) -> dict[str, Any]:
    if isinstance(policy, PassivePolicy): return {"kind": "passive"}
    if isinstance(policy, ReactivePolicy): return {"kind": "reactive"}
    if isinstance(policy, MPCPolicy): return {"kind": "mpc"}
    if isinstance(policy, ActorPolicy): return {"kind": "actor", "name": policy.name, "training_seed": policy.training_seed, "weights": policy.actor.weights, "log_standard_deviation": policy.actor.log_standard_deviation}
    if isinstance(policy, ModelGuidedPolicy): return {"kind": "model_guided", "training_seed": policy.training_seed, "bc_weights": policy.bc_actor.weights, "bc_log_standard_deviation": policy.bc_actor.log_standard_deviation, "sac_weights": policy.sac_actor.weights, "sac_log_standard_deviation": policy.sac_actor.log_standard_deviation}
    raise TypeError(policy)


def _cell_dict(cell: NetworkCell) -> dict[str, Any]:
    return {"cell_id": cell.cell_id, "gateway_count": cell.gateway_count, "architecture": cell.architecture, "eligibility": cell.eligibility, "full_policy_anchor": is_full_policy_anchor(cell)}


def _cache_paths(cell: NetworkCell, scope: str) -> dict[str, Path]:
    directory = EXPERIMENT_DIR / "cache" / scope / cell.cell_id
    directory.mkdir(parents=True, exist_ok=True)
    return {"rep": directory / "replications.csv.gz", "res": directory / "resources.csv.gz", "contract": directory / "contracts.csv.gz", "marker": directory / "complete.json"}


def _evaluate_cell(
    *, cell: NetworkCell, model_config: Mapping[str, Any], policies: Sequence[Any],
    paths: Sequence[PhysicalPath], run_signature: str, workers: int, scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = _cache_paths(cell, scope)
    if files["marker"].exists():
        marker = json.loads(files["marker"].read_text())
        if marker.get("run_signature") != run_signature or marker.get("path_ids") != [path.path_id for path in paths]:
            raise RuntimeError(f"Evaluation cache contract mismatch: {cell.cell_id}/{scope}")
        return pd.read_csv(files["rep"]), pd.read_csv(files["res"]), pd.read_csv(files["contract"])
    specs = [_spec(policy) for policy in policies]
    tasks = [(path, index) for path in paths for index in range(len(policies))]
    reps, resources, contracts = [], [], []
    print(f"[5.3.3] {scope} {cell.cell_id}: {len(tasks)} policy-path-seed runs", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialise_worker, initargs=(dict(model_config), specs, _cell_dict(cell))) as executor:
        for artifact in executor.map(evaluate_task, tasks, chunksize=1):
            reps.append(artifact.replication); resources.extend(artifact.resources); contracts.append(artifact.contract)
    rep, res, contract = pd.DataFrame(reps), pd.DataFrame(resources), pd.DataFrame(contracts)
    rep.to_csv(files["rep"], index=False, compression="gzip")
    res.to_csv(files["res"], index=False, compression="gzip")
    contract.to_csv(files["contract"], index=False, compression="gzip")
    files["marker"].write_text(json.dumps({"run_signature": run_signature, "path_ids": [path.path_id for path in paths], "policy_path_seed_runs": len(rep), "status": "complete"}, indent=2) + "\n")
    return rep, res, contract


def _safe_staging(final: Path) -> Path:
    staging = final.with_name(final.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def run() -> int:
    started = time.perf_counter(); started_utc = datetime.now(timezone.utc).isoformat()
    experiment, base, config_hash = _load()
    upstream = _verify_upstream(experiment)
    source_hash = _source_hash()
    run_signature = _json_hash({"config": config_hash, "source": source_hash, "upstream": upstream[["relative_path", "observed_sha256"]].to_dict(orient="records")})
    print(f"[5.3.3] run signature {run_signature}", flush=True)

    frozen = load_frozen_5_2_1_inputs(base)
    observed_model = build_model(base)
    reference_normal = float(sum(observed_model.gateway_scales.values()))
    training_paths, validation_paths = build_training_validation_paths(config=base, residuals=frozen.residuals, reference_normal_model_units=reference_normal)
    cells = declared_cells()
    size_results: dict[int, dict[str, list[TrainingResult]]] = {}
    training_frames: dict[str, list[pd.DataFrame]] = {key: [] for key in ("teacher", "training", "validation", "checkpoints", "checkpoint_validation", "validation_proposals", "gradient", "jacobian")}
    network_rows = []
    for n in experiment["gateway_counts"]:
        model_config = build_cell_config(base, experiment, training_cell(int(n)))
        model = build_model(model_config)
        if len(model.layout.keys) != 10 * int(n) + 4:
            raise RuntimeError("Dynamic action dimension formula failed")
        results, frames = _train_size(n=int(n), model=model, training_paths=training_paths, validation_paths=validation_paths, run_signature=run_signature)
        size_results[int(n)] = results
        for key in training_frames: training_frames[key].append(frames[key])
    training_elapsed = time.perf_counter() - started

    gate_started = time.perf_counter()
    for cell in cells:
        model_config = build_cell_config(base, experiment, cell)
        model = build_model(model_config)
        policies = _policies(model, size_results[cell.gateway_count], is_full_policy_anchor(cell))
        _evaluate_cell(cell=cell, model_config=model_config, policies=policies, paths=validation_paths[:int(experiment["path_design"]["gate_paths"])], run_signature=run_signature, workers=int(experiment["execution"]["parallel_workers"]), scope="computational_gate")
    gate_elapsed = time.perf_counter() - gate_started
    gate_paths = int(experiment["path_design"]["gate_paths"])
    per_path_seconds = gate_elapsed / gate_paths
    budget = float(experiment["path_design"]["runtime_budget_seconds"])
    reserve = float(experiment["path_design"]["reporting_reserve_seconds"])
    safety = float(experiment["path_design"]["runtime_safety_fraction"])
    supported = math.floor(max(budget - training_elapsed - gate_elapsed - reserve, 0.0) * safety / max(per_path_seconds, 1e-9))
    minimum = int(experiment["path_design"]["minimum_formal_paths"])
    maximum = int(experiment["path_design"]["maximum_formal_paths"])
    if supported < minimum:
        raise RuntimeError(f"Computational gate supports only {supported} paths, below the preregistered minimum {minimum}")
    formal_count = min(maximum, supported)
    print(f"[5.3.3] computational gate {gate_elapsed:.1f}s for {gate_paths} validation paths; selected {formal_count} formal paths", flush=True)
    gate_frame = pd.DataFrame([{"training_elapsed_seconds": training_elapsed, "gate_elapsed_seconds": gate_elapsed, "gate_paths": gate_paths, "estimated_seconds_per_complete_network_path": per_path_seconds, "runtime_supported_paths": supported, "selected_formal_paths": formal_count, "selection_uses_test_outcomes": False}])

    formal_paths = build_test_paths(config=base, frozen=frozen, count=formal_count)
    accepted = pd.read_csv(CODE_ROOT / "output/5.2.2_common_authority_benchmark/test_path_manifest.csv")
    generated = manifest_frame(formal_paths)
    if not (generated["path_content_sha256"].astype(str).to_numpy() == accepted.loc[:formal_count-1, "path_content_sha256"].astype(str).to_numpy()).all():
        raise RuntimeError("Formal paths do not reproduce the accepted 5.2.2 prefix")
    replication_frames, resource_frames, contract_frames = [], [], []
    cell_rows = []
    formal_started = time.perf_counter()
    for cell in cells:
        model_config = build_cell_config(base, experiment, cell)
        network_rows.extend(network_register(model_config))
        cell_rows.append(_cell_dict(cell))
        model = build_model(model_config)
        policies = _policies(model, size_results[cell.gateway_count], is_full_policy_anchor(cell))
        rep, res, contract = _evaluate_cell(cell=cell, model_config=model_config, policies=policies, paths=formal_paths, run_signature=run_signature, workers=int(experiment["execution"]["parallel_workers"]), scope="formal")
        replication_frames.append(rep); resource_frames.append(res); contract_frames.append(contract)
    formal_elapsed = time.perf_counter() - formal_started
    replications = pd.concat(replication_frames, ignore_index=True)
    resources = pd.concat(resource_frames, ignore_index=True)
    contracts = pd.concat(contract_frames, ignore_index=True)
    path_level = aggregate_learning_seeds(replications)
    confidence_level = float(experiment["path_design"]["confidence_level"])
    summary = policy_summary(path_level, confidence_level)
    regret, confidence = policy_regret(path_level, confidence_level)
    components = component_values(path_level, confidence_level)
    precision = precision_audit(components, float(experiment["path_design"]["target_halfwidth"]))
    not_evaluated = []
    for cell in cells:
        if not is_full_policy_anchor(cell):
            for policy in ("Projected stochastic MPC", "Model-guided constrained SAC"):
                not_evaluated.append({**_cell_dict(cell), "policy": policy, "status": "NOT_EVALUATED_BY_DESIGN", "value": np.nan})
    total_elapsed = time.perf_counter() - started
    runtime = pd.DataFrame([{
        "training_elapsed_seconds": training_elapsed, "computational_gate_elapsed_seconds": gate_elapsed,
        "formal_evaluation_elapsed_seconds": formal_elapsed, "total_elapsed_seconds": total_elapsed,
        "runtime_budget_seconds": budget, "formal_paths": formal_count,
        "mean_decision_time_seconds": path_level["mean_decision_time_seconds"].mean(),
        "maximum_decision_time_seconds": path_level["maximum_decision_time_seconds"].max(),
        "maximum_action_dimension": path_level["action_dimension"].max(),
    }])

    output_final = CODE_ROOT / experiment["output_directory"]
    staging = _safe_staging(output_final)
    figures = create_figures(summary=summary, regret=regret, figures_directory=EXPERIMENT_DIR / "figures", output_directory=staging, dpi=int(experiment["execution"]["figure_dpi"]))
    all_training = {key: pd.concat(value, ignore_index=True) for key, value in training_frames.items()}
    cells_frame = pd.DataFrame(cell_rows)
    tables = {
        "upstream_input_locks.csv": upstream, "declared_network_cells.csv": cells_frame,
        "network_resource_registry.csv": pd.DataFrame(network_rows),
        "training_path_manifest.csv": manifest_frame(training_paths), "validation_path_manifest.csv": manifest_frame(validation_paths),
        "test_path_manifest.csv": generated, "computational_gate.csv": gate_frame,
        "teacher_actions.csv": all_training["teacher"], "training_curves.csv": all_training["training"],
        "validation_curves.csv": all_training["validation"], "checkpoint_manifest.csv": all_training["checkpoints"],
        "checkpoint_validation.csv": all_training["checkpoint_validation"], "proposal_selection_validation.csv": all_training["validation_proposals"],
        "sac_actor_gradient_check.csv": all_training["gradient"], "projection_jacobian_check.csv": all_training["jacobian"],
        "path_level_policy_seed_results.csv": replications, "path_level_seed_aggregated.csv": path_level,
        "resource_week_results.csv": resources, "trajectory_contract_checks.csv": contracts,
        "policy_summary.csv": summary, "policy_regret.csv": regret, "policy_confidence_set.csv": confidence,
        "network_component_values.csv": components, "precision_audit.csv": precision,
        "not_evaluated_policy_cells.csv": pd.DataFrame(not_evaluated), "runtime_and_scalability.csv": runtime,
        "figure_5_3_3a_data.csv": summary.merge(regret[["cell_id", "policy", "regret_mean", "regret_lower", "regret_upper"]], on=["cell_id", "policy"], how="left"),
        "figure_5_3_3b_data.csv": summary[[column for column in summary if "overload" in column or column in ["cell_id", "gateway_count", "architecture", "eligibility", "policy"]]],
        "figure_5_3_3c_data.csv": summary[["cell_id", "gateway_count", "architecture", "eligibility", "policy", "mean_waiting_exposure", "mean_delivery", "clearance_probability", "mean_terminal_outstanding"]],
    }
    for filename, frame in tables.items(): write_csv(frame, staging / filename)
    (staging / "frozen_config_5_3_3.json").write_text(json.dumps(experiment, indent=2) + "\n")
    report_directory = PROJECT_ROOT / "report - 8.4" / "5.3.3"
    total_elapsed = time.perf_counter() - started
    runtime.loc[0, "total_elapsed_seconds"] = total_elapsed
    write_csv(runtime, staging / "runtime_and_scalability.csv")
    acceptance = acceptance_payload(
        upstream=upstream, cells=cells_frame, path_level=path_level, contracts=contracts,
        checkpoints=all_training["checkpoints"], gradient=all_training["gradient"], jacobian=all_training["jacobian"],
        precision=precision, figures=figures, runtime=runtime, expected_paths=formal_count,
    )
    (staging / "acceptance_5_3_3.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    write_reports(report_directory, acceptance, summary, components, confidence, runtime, cells_frame)
    total_elapsed = time.perf_counter() - started
    runtime.loc[0, "total_elapsed_seconds"] = total_elapsed
    write_csv(runtime, staging / "runtime_and_scalability.csv")
    acceptance = acceptance_payload(
        upstream=upstream, cells=cells_frame, path_level=path_level, contracts=contracts,
        checkpoints=all_training["checkpoints"], gradient=all_training["gradient"], jacobian=all_training["jacobian"],
        precision=precision, figures=figures, runtime=runtime, expected_paths=formal_count,
    )
    (staging / "acceptance_5_3_3.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    write_reports(report_directory, acceptance, summary, components, confidence, runtime, cells_frame)
    write_manifest(staging / "run_manifest.json", config_hash=config_hash, source_hash=source_hash, upstream=upstream, output_directory=staging, figures=figures, started_utc=started_utc, elapsed_seconds=total_elapsed, formal_paths=formal_count)
    if output_final.exists(): shutil.rmtree(output_final)
    staging.rename(output_final)
    print(json.dumps({"status": acceptance["run_status"], "overall_evidence_acceptance": acceptance["OVERALL_EVIDENCE_ACCEPTANCE"], "formal_paths": formal_count, "elapsed_seconds": total_elapsed, "output": str(output_final)}, indent=2), flush=True)
    return 0 if acceptance["run_status"] == "complete" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as error:
        print(f"[5.3.3] FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
