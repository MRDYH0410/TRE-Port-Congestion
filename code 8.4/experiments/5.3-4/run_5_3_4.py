"""Resumable runner for Experiment 5.3.4 Parameter Robustness."""

from __future__ import annotations

import argparse
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
from model import build_model  # noqa: E402
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
from reporting_5_3_4 import (  # noqa: E402
    acceptance_payload,
    create_figures,
    formula_registry,
    independent_checks,
    parameter_registry,
    sha256_file,
    write_csv,
    write_manifest,
    write_reports,
)
from robustness_5_3_4 import (  # noqa: E402
    RobustnessCell,
    build_cells,
    cell_registry,
    dimension_changed_cell,
    full_policy_anchor,
    model_config,
    policy_family,
    transform_paths,
)
from robustness_worker import (  # noqa: E402
    RobustnessArtifacts,
    evaluate_task,
    initialise_worker,
)
from statistics_5_3_4 import (  # noqa: E402
    aggregate_learning_seeds,
    clearance_endpoint_diagnostic,
    paired_cell_effects,
    policy_regret,
    policy_summary,
)
from training import (  # noqa: E402
    TrainingResult,
    _normalised_projection_jacobian,
    generate_teacher_data,
    sac_actor_gradient_check,
    save_checkpoint,
    train_bc,
    train_sac,
    validate_model_guided,
)


SOURCE_FILES = [
    "src/tre84/actions.py",
    "src/tre84/acceptance.py",
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
    "experiments/5.3-3/network_5_3_3.py",
    "experiments/5.3-4/robustness_5_3_4.py",
    "experiments/5.3-4/robustness_worker.py",
    "experiments/5.3-4/statistics_5_3_4.py",
    "experiments/5.3-4/reporting_5_3_4.py",
    "experiments/5.3-4/run_5_3_4.py",
]


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_hash() -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        path = CODE_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load() -> tuple[dict[str, Any], dict[str, Any], str]:
    path = EXPERIMENT_DIR / "config_5_3_4.json"
    experiment = json.loads(path.read_text(encoding="utf-8"))
    base = json.loads((CODE_ROOT / experiment["base_model_config"]).read_text(encoding="utf-8"))
    return experiment, base, sha256_file(path)


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
        raise RuntimeError(f"Upstream lock mismatch: {frame.loc[~frame['matched'], 'relative_path'].tolist()}")
    acceptance = json.loads(
        (CODE_ROOT / "output/5.2.5_computational_methodological_acceptance/acceptance_5_2_5.json").read_text(encoding="utf-8")
    )
    if acceptance.get("OVERALL_ACCEPTANCE") != "PASS":
        raise RuntimeError("The accepted Chapter 4 methodology gate 5.2.5 is not PASS")
    reclosure = json.loads(
        (CODE_ROOT / "output/5.3.2_reclosure_sensitivity/acceptance_5_3_2.json").read_text(encoding="utf-8")
    )
    if reclosure.get("run_status") != "complete" or reclosure.get("engineering_acceptance") != "PASS" or reclosure.get("numerical_acceptance") != "PASS":
        raise RuntimeError("The reclosure constructor is not frozen as complete engineering/numerical evidence")
    gateway = json.loads(
        (CODE_ROOT / "output/5.3.3_gateway_network_sensitivity/acceptance_5_3_3.json").read_text(encoding="utf-8")
    )
    if gateway.get("run_status") != "complete" or gateway.get("ENGINEERING_ACCEPTANCE") != "PASS" or gateway.get("METHODOLOGY_CONTRACT_ACCEPTANCE") != "PASS":
        raise RuntimeError("The nine-gateway constructor/checkpoints are not accepted for structural use")
    return frame


def _load_actor(path: Path) -> LinearActor:
    with np.load(path, allow_pickle=False) as payload:
        return LinearActor(
            np.asarray(payload["weights"], dtype=float),
            np.asarray(payload["log_standard_deviation"], dtype=float),
        )


def _checkpoint_bundle(
    *,
    manifest_path: Path,
    checkpoint_base: Path,
    source_experiment: str,
    gateway_count: int | None = None,
) -> tuple[dict[str, list[Any]], pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    if gateway_count is not None:
        manifest = manifest.loc[manifest["gateway_count"].astype(int) == gateway_count].copy()
    bc, sac, audit = [], [], []
    for seed_index in range(3):
        bc_row = manifest.loc[
            (manifest["policy"] == "Behaviour cloning") & (manifest["seed_index"] == seed_index)
        ].iloc[0]
        sac_row = manifest.loc[
            (manifest["policy"] == "Constrained SAC") & (manifest["seed_index"] == seed_index)
        ].iloc[0]
        def resolve(value: str) -> Path:
            candidate = Path(str(value))
            if candidate.parts and candidate.parts[0] == "experiments":
                return CODE_ROOT / candidate
            return checkpoint_base / candidate

        bc_path = resolve(str(bc_row["checkpoint_path"]))
        sac_path = resolve(str(sac_row["checkpoint_path"]))
        for row, path in ((bc_row, bc_path), (sac_row, sac_path)):
            digest = sha256_file(path)
            if digest != str(row["checkpoint_sha256"]):
                raise RuntimeError(f"Frozen checkpoint hash mismatch: {path}")
            audit.append(
                {
                    "policy": row["policy"],
                    "seed_index": seed_index,
                    "training_seed": int(row["training_seed"]),
                    "checkpoint_path": path.relative_to(CODE_ROOT).as_posix(),
                    "checkpoint_sha256": digest,
                    "source_experiment": source_experiment,
                    "retrained_for_5_3_4": False,
                }
            )
        bc_actor = _load_actor(bc_path)
        sac_actor = _load_actor(sac_path)
        bc.append((bc_actor, int(bc_row["training_seed"])))
        sac.append(
            (
                bc_actor,
                sac_actor,
                int(sac_row["training_seed"]),
            )
        )
    return {"bc": bc, "sac": sac}, pd.DataFrame(audit)


def _projection_jacobian_audit(model: Any, cell_id: str) -> pd.DataFrame:
    state = model.initial_state(
        {
            "filtered_high_risk_probability": 0.25,
            "lead_time_high_risk_probability": 0.5,
            "release_date": pd.Timestamp("2000-01-01"),
            "week": pd.Timestamp("2000-01-03"),
        }
    )
    seed = int(hashlib.sha256(f"5.3.4|jacobian|{cell_id}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    actor = LinearActor.random(model, seed)
    raw, _, _, _ = actor.sample_latent_normalised(state, model, rng)
    raw_action = model.action_from_normalised(raw)
    projection = model.projector.project(raw_action, state)
    analytic = _normalised_projection_jacobian(
        model=model, state=state, raw_action=raw_action, projection=projection
    )
    h = float(model.config["numerics"]["sac_gradient_check_step"])
    tolerance = float(model.config["numerics"]["sac_gradient_check_relative_tolerance"])
    rows = []
    for index in range(raw.size):
        plus, minus = raw.copy(), raw.copy()
        plus[index] += h
        minus[index] -= h
        p = model.normalise_action(
            model.projector.project(model.action_from_normalised(plus), state).action
        )
        m = model.normalise_action(
            model.projector.project(model.action_from_normalised(minus), state).action
        )
        numerical = (p - m) / (2.0 * h)
        for output, value in enumerate(numerical):
            recorded = float(analytic[output, index])
            relative = abs(recorded - float(value)) / max(1.0, abs(recorded), abs(float(value)))
            rows.append(
                {
                    "cell_id": cell_id,
                    "input_action_index": index,
                    "output_action_index": output,
                    "analytic": recorded,
                    "finite_difference": float(value),
                    "relative_error": relative,
                    "tolerance": tolerance,
                    "status": "PASS" if relative <= tolerance else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def _train_matched_bundle(
    *,
    cell: RobustnessCell,
    model: Any,
    training_paths: Sequence[PhysicalPath],
    validation_paths: Sequence[PhysicalPath],
    training_signature: str,
) -> tuple[dict[str, list[Any]], pd.DataFrame, dict[str, pd.DataFrame]]:
    directory = EXPERIMENT_DIR / "checkpoints" / cell.cell_id
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "training_complete.json"
    manifest_path = directory / "checkpoint_manifest.csv"
    if marker.exists():
        marker_data = json.loads(marker.read_text(encoding="utf-8"))
        if marker_data.get("training_signature") != training_signature:
            raise RuntimeError(f"Matched-training cache contract mismatch: {cell.cell_id}")
        bundle, audit = _checkpoint_bundle(
            manifest_path=manifest_path,
            checkpoint_base=CODE_ROOT,
            source_experiment=f"5.3.4 matched {cell.cell_id}",
        )
        frames = {
            name: pd.read_csv(directory / filename)
            for name, filename in {
                "training": "training_curves.csv",
                "validation": "validation_curves.csv",
                "gradient": "sac_actor_gradient_check.csv",
                "jacobian": "projection_jacobian_check.csv",
                "proposal": "proposal_selection_validation.csv",
            }.items()
        }
        print(f"[5.3.4] reused matched training for {cell.cell_id}", flush=True)
        return bundle, audit, frames

    print(f"[5.3.4] matched teacher/BC/full-SAC training: {cell.cell_id}", flush=True)
    teacher, teacher_hash = generate_teacher_data(model=model, paths=training_paths)
    model_hash = _json_hash(model.config)
    results: dict[str, list[TrainingResult]] = {"Behaviour cloning": [], "Constrained SAC": []}
    curves, validations, checkpoints, proposals = [], [], [], []
    for seed_index in range(int(model.config["training"]["seeds"])):
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
            curves.extend({"cell_id": cell.cell_id, **row} for row in result.training_curve)
            validations.extend({"cell_id": cell.cell_id, **row} for row in result.validation_curve)
            checkpoint, digest = save_checkpoint(
                result=result,
                directory=directory,
                feature_names=state_feature_names(model),
                config_hash=model_hash,
            )
            checkpoints.append(
                {
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
                    "teacher_action_hash": result.teacher_hash,
                    "retrained_for_5_3_4": True,
                    "source_experiment": f"5.3.4 matched {cell.cell_id}",
                }
            )
        _, proposal_rows = validate_model_guided(
            model=model, bc=bc, constrained_sac=sac, validation_paths=validation_paths
        )
        proposals.extend({"cell_id": cell.cell_id, "seed_index": seed_index, **row} for row in proposal_rows)
    manifest = pd.DataFrame(checkpoints)
    gradient = pd.DataFrame(sac_actor_gradient_check(model))
    gradient.insert(0, "cell_id", cell.cell_id)
    jacobian = _projection_jacobian_audit(model, cell.cell_id)
    if not gradient["passed"].astype(bool).all() or not jacobian["status"].eq("PASS").all():
        raise RuntimeError(f"Matched learning derivative contract failed: {cell.cell_id}")
    frames = {
        "training": pd.DataFrame(curves),
        "validation": pd.DataFrame(validations),
        "gradient": gradient,
        "jacobian": jacobian,
        "proposal": pd.DataFrame(proposals),
    }
    manifest.to_csv(manifest_path, index=False)
    for name, filename in {
        "training": "training_curves.csv",
        "validation": "validation_curves.csv",
        "gradient": "sac_actor_gradient_check.csv",
        "jacobian": "projection_jacobian_check.csv",
        "proposal": "proposal_selection_validation.csv",
    }.items():
        frames[name].to_csv(directory / filename, index=False)
    marker.write_text(
        json.dumps({"training_signature": training_signature, "cell_id": cell.cell_id, "status": "complete"}, indent=2) + "\n",
        encoding="utf-8",
    )
    bundle = {
        "bc": [(result.actor, result.seed) for result in results["Behaviour cloning"]],
        "sac": [
            (bc.actor, sac.actor, sac.seed)
            for bc, sac in zip(results["Behaviour cloning"], results["Constrained SAC"])
        ],
    }
    return bundle, manifest, frames


def _policies(
    model: Any,
    bundle: Mapping[str, Sequence[Any]],
    names: Sequence[str],
) -> list[Any]:
    result: list[Any] = []
    for name in names:
        if name == "Passive":
            result.append(PassivePolicy(model))
        elif name == "Reactive":
            result.append(ReactivePolicy(model))
        elif name == "Projected stochastic MPC":
            result.append(MPCPolicy(model))
        elif name == "Behaviour cloning":
            result.extend(
                ActorPolicy("Behaviour cloning", model, actor, seed)
                for actor, seed in bundle["bc"]
            )
        elif name == "Model-guided constrained SAC":
            result.extend(
                ModelGuidedPolicy(
                    model=model,
                    bc_actor=bc_actor,
                    sac_actor=sac_actor,
                    training_seed=seed,
                )
                for bc_actor, sac_actor, seed in bundle["sac"]
            )
        else:
            raise ValueError(f"Unknown 5.3.4 policy: {name}")
    return result


def _spec(policy: Any) -> dict[str, Any]:
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


def _cell_dict(cell: RobustnessCell) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "cell_type": cell.cell_type,
        "family": cell.family,
        "display_factor": cell.display_factor,
        "display_level": cell.display_level,
        "full_policy_anchor": full_policy_anchor(cell),
        "path_stress": cell.path_stress,
        "network_stress": cell.network_stress,
    }


def _cache_paths(
    cell: RobustnessCell, scope: str, run_signature: str | None = None
) -> dict[str, Path]:
    signature = run_signature[:16] if run_signature else "unscoped"
    directory = EXPERIMENT_DIR / "cache" / scope / signature / cell.cell_id
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "rep": directory / "replications.csv.gz",
        "resources": directory / "resources.csv.gz",
        "contracts": directory / "contracts.csv.gz",
        "marker": directory / "complete.json",
    }


def _evaluate_cell(
    *,
    cell: RobustnessCell,
    cell_model_config: Mapping[str, Any],
    policies: Sequence[Any],
    paths: Sequence[PhysicalPath],
    run_signature: str,
    workers: int,
    scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = _cache_paths(cell, scope, run_signature)
    path_contract = [{"path_id": path.path_id, "path_hash": path.path_hash} for path in paths]
    if files["marker"].exists():
        marker = json.loads(files["marker"].read_text(encoding="utf-8"))
        if marker.get("run_signature") != run_signature or marker.get("paths") != path_contract:
            raise RuntimeError(f"Evaluation cache contract mismatch: {cell.cell_id}/{scope}")
        return (
            pd.read_csv(files["rep"]),
            pd.read_csv(files["resources"]),
            pd.read_csv(files["contracts"]),
        )
    specs = [_spec(policy) for policy in policies]
    tasks = [(path, index) for path in paths for index in range(len(policies))]
    replications, resources, contracts = [], [], []
    print(f"[5.3.4] {scope} {cell.cell_id}: {len(tasks)} policy-path-seed runs", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialise_worker,
        initargs=(dict(cell_model_config), specs, _cell_dict(cell)),
    ) as executor:
        for artifact in executor.map(evaluate_task, tasks, chunksize=1):
            replications.append(artifact.replication)
            resources.extend(artifact.resources)
            contracts.append(artifact.contract)
    rep = pd.DataFrame(replications)
    res = pd.DataFrame(resources)
    contract = pd.DataFrame(contracts)
    rep.to_csv(files["rep"], index=False, compression="gzip")
    res.to_csv(files["resources"], index=False, compression="gzip")
    contract.to_csv(files["contracts"], index=False, compression="gzip")
    files["marker"].write_text(
        json.dumps(
            {
                "run_signature": run_signature,
                "paths": path_contract,
                "policy_path_seed_runs": len(rep),
                "status": "complete",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rep, res, contract


def _run_cells(
    *,
    cells: Sequence[RobustnessCell],
    experiment: Mapping[str, Any],
    base: Mapping[str, Any],
    policy_bundles: Mapping[str, Mapping[str, Sequence[Any]]],
    paths: Sequence[PhysicalPath],
    run_signature: str,
    workers: int,
    scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replications, resources, contracts = [], [], []
    for cell in cells:
        config = model_config(base, experiment, cell)
        model = build_model(config)
        if cell.cell_id == "interaction__long_lag__severe_reclosure":
            bundle = policy_bundles["matched_severe"]
        elif cell.network_stress != "reference":
            bundle = policy_bundles["n09"]
        else:
            bundle = policy_bundles["reference"]
        policies = _policies(model, bundle, policy_family(cell, experiment))
        cell_paths = transform_paths(paths, cell, experiment)
        rep, res, contract = _evaluate_cell(
            cell=cell,
            cell_model_config=config,
            policies=policies,
            paths=cell_paths,
            run_signature=run_signature,
            workers=workers,
            scope=scope,
        )
        replications.append(rep)
        resources.append(res)
        contracts.append(contract)
    return (
        pd.concat(replications, ignore_index=True),
        pd.concat(resources, ignore_index=True),
        pd.concat(contracts, ignore_index=True),
    )


def _safe_staging(final: Path) -> Path:
    staging = final.with_name(final.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def _publish(staging: Path, final: Path) -> None:
    if final.exists():
        previous = final.with_name(final.name + f".previous_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        final.rename(previous)
    staging.rename(final)


def _status(cells: Sequence[RobustnessCell], run_signature: str) -> int:
    rows = []
    for scope in ("computational_gate", "formal"):
        complete = 0
        runs = 0
        for cell in cells:
            marker = _cache_paths(cell, scope, run_signature)["marker"]
            if marker.exists():
                payload = json.loads(marker.read_text(encoding="utf-8"))
                if payload.get("run_signature") == run_signature:
                    complete += 1
                    runs += int(payload.get("policy_path_seed_runs", 0))
        rows.append({"scope": scope, "completed_cells": complete, "expected_cells": len(cells), "policy_path_seed_runs": runs})
    print(json.dumps(rows, indent=2), flush=True)
    return 0


def run(phase: str) -> int:
    started_clock = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    experiment, base, config_hash = _load()
    cells, diagnostic_design = build_cells(experiment)
    upstream = _verify_upstream(experiment)
    source_hash = _source_hash()
    frozen = load_frozen_5_2_1_inputs(base)
    reference_config = model_config(base, experiment, cells[0])
    reference_model = build_model(reference_config)
    reference_bundle, reference_audit = _checkpoint_bundle(
        manifest_path=CODE_ROOT / "output/5.2.2_common_authority_benchmark/checkpoint_manifest.csv",
        checkpoint_base=CODE_ROOT / "output/5.2.2_common_authority_benchmark",
        source_experiment="accepted 5.2.2 historical anchor",
    )
    n09_bundle, n09_audit = _checkpoint_bundle(
        manifest_path=CODE_ROOT / "output/5.3.3_gateway_network_sensitivity/checkpoint_manifest.csv",
        checkpoint_base=CODE_ROOT,
        source_experiment="accepted 5.3.3 n=9 size-matched frozen deployment",
        gateway_count=9,
    )
    training_signature = _json_hash(
        {
            "config_hash": config_hash,
            "source_hash": source_hash,
            "upstream": upstream[["relative_path", "observed_sha256"]].to_dict(orient="records"),
            "reference_checkpoints": reference_audit[["checkpoint_path", "checkpoint_sha256"]].to_dict(orient="records"),
            "n09_checkpoints": n09_audit[["checkpoint_path", "checkpoint_sha256"]].to_dict(orient="records"),
        }
    )
    reference_normal = float(sum(reference_model.gateway_scales.values()))
    training_paths, validation_paths = build_training_validation_paths(
        config=base,
        residuals=frozen.residuals,
        reference_normal_model_units=reference_normal,
    )
    matched_cell = next(
        cell for cell in cells if cell.cell_id == "interaction__long_lag__severe_reclosure"
    )
    matched_config = model_config(base, experiment, matched_cell)
    matched_model = build_model(matched_config)
    matched_training_paths = transform_paths(
        training_paths[: int(experiment["matched_training"]["training_paths"])],
        matched_cell,
        experiment,
    )
    matched_validation_paths = transform_paths(
        validation_paths[: int(experiment["matched_training"]["validation_paths"])],
        matched_cell,
        experiment,
    )
    training_started = time.perf_counter()
    matched_bundle, matched_audit, training_frames = _train_matched_bundle(
        cell=matched_cell,
        model=matched_model,
        training_paths=matched_training_paths,
        validation_paths=matched_validation_paths,
        training_signature=training_signature,
    )
    training_elapsed = time.perf_counter() - training_started
    checkpoint_audit = pd.concat(
        [reference_audit, n09_audit, matched_audit], ignore_index=True, sort=False
    )
    policy_bundles = {
        "reference": reference_bundle,
        "n09": n09_bundle,
        "matched_severe": matched_bundle,
    }
    run_signature = _json_hash(
        {
            "training_signature": training_signature,
            "checkpoints": checkpoint_audit[["checkpoint_path", "checkpoint_sha256"]].to_dict(orient="records"),
        }
    )
    print(f"[5.3.4] run signature {run_signature}", flush=True)
    if phase == "status":
        return _status(cells, run_signature)

    gate_started = time.perf_counter()
    gate_count = int(experiment["path_design"]["gate_paths"])
    gate_rep, _, gate_contracts = _run_cells(
        cells=cells,
        experiment=experiment,
        base=base,
        policy_bundles=policy_bundles,
        paths=validation_paths[:gate_count],
        run_signature=run_signature,
        workers=min(int(experiment["execution"]["parallel_workers"]), 4),
        scope="computational_gate",
    )
    gate_elapsed = time.perf_counter() - gate_started
    if not gate_contracts["all_step_acceptance_passed"].astype(bool).all():
        raise RuntimeError("The computational gate failed a trajectory contract")
    per_path_seconds = gate_elapsed / gate_count
    budget = float(experiment["path_design"]["runtime_budget_seconds"])
    reserve = float(experiment["path_design"]["reporting_reserve_seconds"])
    safety = float(experiment["path_design"]["runtime_safety_fraction"])
    supported = math.floor(
        max(budget - training_elapsed - gate_elapsed - reserve, 0.0)
        * safety
        / max(per_path_seconds, 1e-9)
    )
    minimum = int(experiment["path_design"]["minimum_formal_paths"])
    maximum = int(experiment["path_design"]["maximum_formal_paths"])
    if supported < minimum:
        raise RuntimeError(f"Computational gate supports only {supported} paths, below minimum {minimum}")
    formal_count = min(maximum, supported)
    gate_frame = pd.DataFrame(
        [
            {
                "gate_elapsed_seconds": gate_elapsed,
                "matched_training_elapsed_seconds": training_elapsed,
                "gate_paths": gate_count,
                "estimated_seconds_per_complete_parameter_path": per_path_seconds,
                "runtime_supported_paths": supported,
                "selected_formal_paths": formal_count,
                "selection_uses_test_outcomes": False,
                "gate_policy_path_seed_runs": len(gate_rep),
            }
        ]
    )
    EXPERIMENT_DIR.joinpath("gate_result.json").write_text(
        json.dumps({"run_signature": run_signature, **gate_frame.iloc[0].to_dict()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[5.3.4] gate {gate_elapsed:.1f}s; selected {formal_count} formal paths", flush=True)
    if phase == "gate":
        return 0

    formal_paths = build_test_paths(config=base, frozen=frozen, count=formal_count)
    accepted = pd.read_csv(CODE_ROOT / "output/5.2.2_common_authority_benchmark/test_path_manifest.csv")
    generated = manifest_frame(formal_paths)
    if not (
        generated["path_content_sha256"].astype(str).to_numpy()
        == accepted.loc[: formal_count - 1, "path_content_sha256"].astype(str).to_numpy()
    ).all():
        raise RuntimeError("Formal paths do not reproduce the accepted benchmark prefix")
    formal_started = time.perf_counter()
    replications, resources, contracts = _run_cells(
        cells=cells,
        experiment=experiment,
        base=base,
        policy_bundles=policy_bundles,
        paths=formal_paths,
        run_signature=run_signature,
        workers=int(experiment["execution"]["parallel_workers"]),
        scope="formal",
    )
    formal_elapsed = time.perf_counter() - formal_started
    path_level = aggregate_learning_seeds(replications)
    confidence_level = float(experiment["path_design"]["confidence_level"])
    summary = policy_summary(path_level, confidence_level)
    regret, confidence = policy_regret(path_level, confidence_level)
    effects = paired_cell_effects(path_level, cells, confidence_level)
    clearance_diagnostic = clearance_endpoint_diagnostic(
        path_level.loc[path_level["cell_id"] == "reference"], diagnostic_design
    )
    registry = cell_registry(cells, experiment)
    runtime = pd.DataFrame(
        [
            {
                "matched_training_elapsed_seconds": training_elapsed,
                "computational_gate_elapsed_seconds": gate_elapsed,
                "formal_evaluation_elapsed_seconds": formal_elapsed,
                "total_elapsed_seconds": time.perf_counter() - started_clock,
                "runtime_budget_seconds": budget,
                "formal_paths": formal_count,
                "mean_recorded_decision_time_seconds": replications["mean_decision_time_seconds"].mean(),
                "maximum_recorded_decision_time_seconds": replications["maximum_decision_time_seconds"].max(),
                "eight_hour_wall_clock_respected": (time.perf_counter() - started_clock) <= budget,
            }
        ]
    )
    cell_path_rows = []
    for cell in cells:
        for base_path, transformed in zip(
            formal_paths, transform_paths(formal_paths, cell, experiment)
        ):
            cell_path_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "path_id": base_path.path_id,
                    "base_path_sha256": base_path.path_hash,
                    "cell_path_sha256": transformed.path_hash,
                    "same_released_information_clock": True,
                    "construction": transformed.construction,
                }
            )
    cell_paths = pd.DataFrame(cell_path_rows)
    pairing = (
        path_level.groupby(["cell_id", "path_id"], as_index=False)
        .agg(policies=("policy", "nunique"), path_hashes=("path_content_sha256", "nunique"))
    )
    output_final = CODE_ROOT / experiment["output_directory"]
    staging = _safe_staging(output_final)
    figures = create_figures(
        effects=effects,
        confidence=confidence,
        summary=summary,
        figure_directory=EXPERIMENT_DIR / "figures",
        output_directory=staging,
        dpi=int(experiment["execution"]["figure_dpi"]),
    )
    independent = independent_checks(
        path_level=path_level,
        effects=effects,
        figures=figures,
        tolerance=float(base["numerics"]["loss_identity_tolerance"]),
    )
    acceptance = acceptance_payload(
        upstream=upstream,
        replications=replications,
        path_level=path_level,
        contracts=contracts,
        registry=registry,
        effects=effects,
        diagnostics=clearance_diagnostic,
        independent=independent,
        figures=figures,
        expected_paths=formal_count,
        target_halfwidth=float(experiment["path_design"]["target_halfwidth"]),
        tolerance=float(base["numerics"]["mass_tolerance"]),
    )
    gradient_pass = bool(training_frames["gradient"]["passed"].astype(bool).all())
    jacobian_pass = bool(training_frames["jacobian"]["status"].eq("PASS").all())
    runtime_pass = bool((time.perf_counter() - started_clock) <= budget)
    acceptance["checks"]["matched_sac_gradient_check_pass"] = gradient_pass
    acceptance["checks"]["matched_projection_jacobian_check_pass"] = jacobian_pass
    acceptance["checks"]["eight_hour_wall_clock_respected_before_publication"] = runtime_pass
    if not (gradient_pass and jacobian_pass and runtime_pass):
        acceptance["ENGINEERING_ACCEPTANCE"] = "FAIL"
        acceptance["OVERALL_EVIDENCE_ACCEPTANCE"] = "FAIL"
        acceptance["run_status"] = "failed"
    tables = {
        "upstream_input_locks.csv": upstream,
        "checkpoint_manifest_5_3_4.csv": checkpoint_audit,
        "matched_training_path_manifest.csv": manifest_frame(matched_training_paths),
        "matched_validation_path_manifest.csv": manifest_frame(matched_validation_paths),
        "matched_training_curves.csv": training_frames["training"],
        "matched_validation_curves.csv": training_frames["validation"],
        "matched_sac_actor_gradient_check.csv": training_frames["gradient"],
        "matched_projection_jacobian_check.csv": training_frames["jacobian"],
        "matched_proposal_selection_validation.csv": training_frames["proposal"],
        "cell_policy_coverage_registry.csv": registry,
        "test_path_manifest.csv": generated,
        "test_path_cell_manifest.csv": cell_paths,
        "path_pairing_audit.csv": pairing,
        "computational_gate.csv": gate_frame,
        "path_level_policy_seed_results.csv": replications,
        "path_level_seed_aggregated.csv": path_level,
        "resource_trajectories.csv": resources,
        "trajectory_contract_checks.csv": contracts,
        "policy_summary.csv": summary,
        "paired_parameter_effects.csv": effects,
        "policy_regret.csv": regret,
        "policy_confidence_set.csv": confidence,
        "clearance_tolerance_diagnostic.csv": clearance_diagnostic,
        "parameter_registry_5_3_4.csv": parameter_registry(experiment),
        "formula_to_code_5_3_4.csv": formula_registry(),
        "independent_recalculation_checks.csv": independent,
        "runtime_summary.csv": runtime,
        "figure_5_3_4a_data.csv": effects.loc[effects["policy"] == "Reactive"],
        "figure_5_3_4b_data.csv": confidence,
        "figure_5_3_4c_data.csv": summary.loc[
            (summary["cell_id"] == "reference") | (summary["cell_type"] == "interaction")
        ],
    }
    for filename, frame in tables.items():
        write_csv(frame, staging / filename)
    (staging / "frozen_config_5_3_4.json").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )
    (staging / "acceptance_5_3_4.json").write_text(
        json.dumps(acceptance, indent=2) + "\n", encoding="utf-8"
    )
    report_directory = PROJECT_ROOT / "report - 8.4" / "5.3.4"
    write_reports(report_directory, acceptance, summary, effects, confidence, clearance_diagnostic, runtime)
    elapsed = time.perf_counter() - started_clock
    write_manifest(
        staging / "run_manifest.json",
        config_hash=config_hash,
        source_hash=source_hash,
        upstream=upstream,
        output_directory=staging,
        figures=figures,
        started_utc=started_utc,
        elapsed_seconds=elapsed,
        formal_paths=formal_count,
    )
    _publish(staging, output_final)
    print(
        json.dumps(
            {
                "status": acceptance["run_status"],
                "overall_evidence_acceptance": acceptance["OVERALL_EVIDENCE_ACCEPTANCE"],
                "formal_paths": formal_count,
                "elapsed_seconds": elapsed,
                "output": str(output_final),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if acceptance["run_status"] == "complete" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("gate", "formal", "all", "status"), default="all")
    args = parser.parse_args()
    return run(args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
