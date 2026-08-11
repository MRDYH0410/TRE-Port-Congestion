"""Single reproducible command for experiment 5.2.3.

Run from the code root:
    python experiments/5.2-3/run_5_2_3.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Prevent process-level parallelism from multiplying BLAS thread pools.  This is
# an execution constraint only; it does not alter any model or policy input.
_THREAD_CONFIG = json.loads(
    (Path(__file__).resolve().parent / "config_5_2_3.json").read_text(encoding="utf-8")
)
_MATH_THREADS = str(
    int(_THREAD_CONFIG["execution"]["math_library_threads_per_worker"])
)
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = _MATH_THREADS

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_DIR.parents[1]
SRC_ROOT = CODE_ROOT / "src"
EXPERIMENT_5_2_2 = CODE_ROOT / "experiments" / "5.2-2"
sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(EXPERIMENT_5_2_2))
sys.path.insert(0, str(SRC_ROOT))

from features import LinearActor  # noqa: E402
from mechanism import MechanismArtifacts, RESTRICTIONS, run_mechanism_replication  # noqa: E402
from mechanism_worker import evaluate_task, initialise_worker  # noqa: E402
from model import build_model  # noqa: E402
from paths import build_test_paths, load_frozen_5_2_1_inputs  # noqa: E402
from policies import (  # noqa: E402
    ActorPolicy,
    MPCPolicy,
    ModelGuidedPolicy,
    PassivePolicy,
    ReactivePolicy,
)
from reporting_5_2_3 import (  # noqa: E402
    acceptance_payload,
    chart_map,
    create_figures,
    parameter_registry,
    sha256_file,
    write_reports,
    write_run_manifest,
)
from statistics_5_2_3 import (  # noqa: E402
    aggregate_full_policy_mechanisms,
    aggregate_weekly_policy_mechanisms,
    proposed_policy_activation_audit,
    restricted_action_effects,
    select_mechanism_policy_set,
    select_physical_path_medoid,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_5_2_2(
    output: Path, config: dict[str, Any]
) -> tuple[bool, list[Path]]:
    acceptance_path = output / "acceptance_5_2_2.json"
    manifest_path = output / "run_manifest.json"
    checkpoint_manifest_path = output / "checkpoint_manifest.csv"
    locked = {
        str(name): str(value).lower()
        for name, value in config["input_locks"].items()
    }
    for name, path in {
        "acceptance_5_2_2.json": acceptance_path,
        "run_manifest.json": manifest_path,
        "checkpoint_manifest.csv": checkpoint_manifest_path,
    }.items():
        if not path.exists() or sha256_file(path).lower() != locked[name]:
            raise RuntimeError(f"Locked 5.2.2 input hash mismatch: {name}")
    acceptance = _read_json(acceptance_path)
    complete = acceptance.get("status") == "complete" and not acceptance.get("blocking_failures")
    manifest = _read_json(manifest_path)
    inputs: list[Path] = [acceptance_path, manifest_path, checkpoint_manifest_path]
    mismatches = []
    # 5.2.2 uses the ``outputs`` schema (paths relative to its output
    # directory); newer experiment manifests use ``artifacts`` (paths
    # relative to the code root).  Accept and verify both explicitly.
    frozen_items = [
        *(dict(item, _base="output") for item in manifest.get("outputs", [])),
        *(dict(item, _base="code_root") for item in manifest.get("artifacts", [])),
        *(dict(item, _base="experiment") for item in manifest.get("code_files", [])),
    ]
    for item in frozen_items:
        relative = item.get("path")
        expected = item.get("sha256")
        if not relative or not expected:
            continue
        if item["_base"] == "output":
            path = output / relative
        elif item["_base"] == "experiment":
            path = EXPERIMENT_5_2_2 / relative
        else:
            path = CODE_ROOT / relative
        if not path.exists():
            mismatches.append(f"{relative} (missing)")
            continue
        inputs.append(path)
        if sha256_file(path) != expected:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(f"5.2.2 artifact hashes changed: {mismatches}")
    manifest_config = manifest.get("config", {})
    manifest_config_path = EXPERIMENT_5_2_2 / str(manifest_config.get("path", ""))
    if (
        not manifest_config_path.exists()
        or sha256_file(manifest_config_path) != str(manifest_config.get("sha256", ""))
    ):
        raise RuntimeError("The frozen 5.2.2 configuration no longer matches its manifest")
    inputs.append(manifest_config_path)
    benchmark_config = _read_json(manifest_config_path)
    frozen_directory = CODE_ROOT / str(benchmark_config["input_5_2_1"])
    frozen_contract = manifest.get("frozen_5_2_1_inputs", {})
    frozen_paths = {
        "historical_information_event_path_sha256": frozen_directory
        / "historical_information_event_path.csv",
        "counterfactual_residual_library_sha256": frozen_directory
        / "counterfactual_residual_library.csv",
        "run_manifest_sha256": frozen_directory / "run_manifest.json",
    }
    for key, path in frozen_paths.items():
        if not path.exists() or sha256_file(path) != str(frozen_contract.get(key, "")):
            raise RuntimeError(f"Accepted 5.2.2 transitive path input changed: {path.name}")
        inputs.append(path)
    if not complete:
        raise RuntimeError("5.2.2 must be accepted before 5.2.3")
    return complete, list(dict.fromkeys(inputs))


def _load_actor(checkpoint: Path, expected_hash: str) -> tuple[LinearActor, int]:
    if sha256_file(checkpoint) != expected_hash:
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint.name}")
    with np.load(checkpoint, allow_pickle=False) as payload:
        actor = LinearActor(
            np.asarray(payload["weights"], dtype=float),
            np.asarray(payload["log_standard_deviation"], dtype=float),
        )
        seed = int(payload["training_seed"])
    return actor, seed


def _policy_instances(
    *,
    policy_name: str,
    model: Any,
    benchmark_output: Path,
    checkpoint_manifest: pd.DataFrame,
) -> list[Any]:
    if policy_name == "Passive":
        return [PassivePolicy(model)]
    if policy_name == "Reactive":
        return [ReactivePolicy(model)]
    if policy_name == "Projected stochastic MPC":
        return [MPCPolicy(model)]
    if policy_name in {"Behaviour cloning", "PPO", "Vanilla SAC", "Constrained SAC"}:
        rows = checkpoint_manifest.loc[checkpoint_manifest["policy"].eq(policy_name)].sort_values("seed_index")
        policies = []
        for row in rows.itertuples(index=False):
            actor, seed = _load_actor(
                benchmark_output / row.checkpoint_path,
                str(row.checkpoint_sha256),
            )
            policies.append(ActorPolicy(policy_name, model, actor, seed))
        return policies
    if policy_name == "Model-guided constrained SAC":
        policies = []
        bc_rows = checkpoint_manifest.loc[
            checkpoint_manifest["policy"].eq("Behaviour cloning")
        ].set_index("seed_index")
        sac_rows = checkpoint_manifest.loc[
            checkpoint_manifest["policy"].eq("Constrained SAC")
        ].set_index("seed_index")
        for seed_index in sorted(set(bc_rows.index) & set(sac_rows.index)):
            bc_row = bc_rows.loc[seed_index]
            sac_row = sac_rows.loc[seed_index]
            bc, _ = _load_actor(
                benchmark_output / str(bc_row["checkpoint_path"]),
                str(bc_row["checkpoint_sha256"]),
            )
            sac, seed = _load_actor(
                benchmark_output / str(sac_row["checkpoint_path"]),
                str(sac_row["checkpoint_sha256"]),
            )
            policies.append(
                ModelGuidedPolicy(
                    model=model,
                    bc_actor=bc,
                    sac_actor=sac,
                    training_seed=seed,
                )
            )
        return policies
    raise ValueError(f"Unsupported frozen 5.2.2 policy: {policy_name}")


def _policy_spec(policy: Any) -> dict[str, Any]:
    if isinstance(policy, PassivePolicy):
        return {"kind": "passive"}
    if isinstance(policy, ReactivePolicy):
        return {"kind": "reactive"}
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
    raise TypeError(f"Unsupported 5.2.3 replay policy: {policy.__class__.__name__}")


def _evaluate_compact(
    *,
    model: Any,
    benchmark_config: dict[str, Any],
    experiment_config: dict[str, Any],
    policies: Sequence[Any],
    tasks: Sequence[tuple[Any, int, str]],
    label: str,
) -> list[MechanismArtifacts]:
    workers = int(experiment_config["execution"]["parallel_workers"])
    specs = [_policy_spec(policy) for policy in policies]
    no_pacing = float(
        experiment_config["restricted_action_diagnostic"][
            "no_release_pacing_baseline"
        ]
    )
    artifacts: list[MechanismArtifacts] = []
    if workers <= 1:
        for path, policy_index, restriction in tasks:
            artifacts.append(
                run_mechanism_replication(
                    model=model,
                    base_policy=policies[int(policy_index)],
                    path=path,
                    restriction=restriction,
                    no_release_pacing_baseline=no_pacing,
                    store_detail=False,
                )
            )
        return artifacts
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialise_worker,
        initargs=(benchmark_config, specs, no_pacing),
    ) as executor:
        for completed, artifact in enumerate(
            executor.map(evaluate_task, tasks, chunksize=1), start=1
        ):
            artifacts.append(artifact)
            if completed % max(1, len(tasks) // 20) == 0 or completed == len(tasks):
                print(
                    f"[5.2.3] {label}: {completed}/{len(tasks)} compact replays",
                    flush=True,
                )
    return artifacts


def _frames(artifacts: Sequence[MechanismArtifacts]) -> dict[str, pd.DataFrame]:
    return {
        "replications": pd.DataFrame([item.replication for item in artifacts]),
        "actions": pd.DataFrame([row for item in artifacts for row in item.actions]),
        "behavior": pd.DataFrame([row for item in artifacts for row in item.behavior]),
        "physical": pd.DataFrame([row for item in artifacts for row in item.physical]),
        "capacity": pd.DataFrame([row for item in artifacts for row in item.capacity]),
        "losses": pd.DataFrame([row for item in artifacts for row in item.losses]),
        "proposals": pd.DataFrame([row for item in artifacts for row in item.proposals]),
        "contracts": pd.DataFrame([item.contract for item in artifacts]),
    }


def _aggregate_restricted_seeds(replications: pd.DataFrame) -> pd.DataFrame:
    if replications["training_seed"].isna().all():
        return replications.copy()
    numeric = replications.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {"training_seed"}
    numeric = [column for column in numeric if column not in excluded]
    group = replications.groupby(
        ["base_policy", "restriction", "path_id", "path_content_sha256", "released_information_path_sha256"],
        as_index=False,
    )
    averaged = group[numeric].mean()
    boolean_columns = [
        "right_censored",
        "numerical_failure",
        "transition_audits_passed",
        "accepted",
        "all_step_acceptance_passed",
    ]
    for column in boolean_columns:
        if column in replications:
            values = group[column].mean().rename(columns={column: f"seed_mean__{column}"})
            averaged = averaged.merge(values, on=["base_policy", "restriction", "path_id", "path_content_sha256", "released_information_path_sha256"])
            if column == "right_censored":
                averaged[column] = averaged[f"seed_mean__{column}"] > 0.5
    averaged["training_seed"] = np.nan
    averaged["learning_seeds_averaged_within_path_first"] = True
    return averaged


def _reproduction_checks(
    *,
    detailed_replications: pd.DataFrame,
    benchmark_replications: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    metrics = [
        "total_operational_objective",
        "loss_queue",
        "loss_waiting",
        "loss_exit",
        "loss_overflow",
        "loss_route_resource",
        "loss_action",
        "terminal_correction",
        "waiting_model_unit_weeks",
        "direct_sue_exit",
        "duration_attrition",
        "delivered_landbridge",
        "ending_outstanding_mass",
        "restricted_clearance_time_contribution",
    ]
    full = detailed_replications.loc[detailed_replications["restriction"].eq("full_action")]
    rows: list[dict[str, Any]] = []
    for actual in full.itertuples(index=False):
        expected = benchmark_replications.loc[
            benchmark_replications["policy"].eq(actual.base_policy)
            & benchmark_replications["path_id"].eq(actual.path_id)
        ]
        if pd.isna(actual.training_seed):
            expected = expected.loc[expected["training_seed"].isna()]
        else:
            expected = expected.loc[
                pd.to_numeric(expected["training_seed"], errors="coerce").eq(float(actual.training_seed))
            ]
        if len(expected) != 1:
            raise RuntimeError(
                f"Cannot identify frozen 5.2.2 replication for {actual.base_policy}/{actual.path_id}/{actual.training_seed}"
            )
        expected_row = expected.iloc[0]
        for metric in metrics:
            observed = float(getattr(actual, metric))
            frozen = float(expected_row[metric])
            difference = observed - frozen
            rows.append(
                {
                    "policy": actual.base_policy,
                    "path_id": actual.path_id,
                    "training_seed": actual.training_seed,
                    "metric": metric,
                    "frozen_5_2_2_value": frozen,
                    "replayed_5_2_3_full_action_value": observed,
                    "absolute_difference": abs(difference),
                    "tolerance": tolerance,
                    "passed": abs(difference) <= tolerance,
                }
            )
    return pd.DataFrame(rows)


def _formula_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("action -> projection", "apply_restriction; ActionProjector.project", "original, restricted and implemented actions"),
            ("rho -> oldest first", "tre84.behavior.oldest_first", "released waiting by vintage"),
            ("route-wait-exit SUE", "tre84.behavior.RCMSASolver.solve", "source simplex, selected start, step trace, residual"),
            ("waiting transition", "tre84.transition.TaggedTransition.step", "renewal, attrition and next vintage identity"),
            ("maritime provenance", "PipelineLot plus diagnostic shadow ledger", "committed/adaptive dispatch, due and holding"),
            ("four-stage transition", "TaggedTransition.step plus proportional shadow", "route-stage before/service/after/next"),
            ("capacity feedback", "CapacityDynamics.transition", "base, direct, readiness, effective capacity and pressure"),
            ("operational loss", "OperationalLoss.compute", "stage queue, waiting, two exits, overload, route, action"),
            ("terminal correction", "TerminalMassCorrection.compute", "right-censored final outstanding mass"),
            ("paired mechanism difference", "restricted_action_effects", "physical-path paired means, simultaneous intervals and Holm p"),
        ],
        columns=["formula_or_chain", "code_function", "saved_evidence"],
    )


def run() -> int:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    config_path = EXPERIMENT_DIR / "config_5_2_3.json"
    config = _read_json(config_path)
    benchmark_config_path = EXPERIMENT_5_2_2 / "config_5_2_2.json"
    benchmark_config = _read_json(benchmark_config_path)
    benchmark_output = CODE_ROOT / str(config["input_5_2_2"])
    input_complete, manifest_inputs = _verify_5_2_2(benchmark_output, config)
    output_final = CODE_ROOT / str(config["output_directory"])
    staging = output_final.parent / f".{output_final.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    confidence = pd.read_csv(benchmark_output / "policy_confidence_set.csv")
    path_level = pd.read_csv(benchmark_output / "path_level_seed_aggregated.csv")
    benchmark_periods = pd.read_csv(benchmark_output / "benchmark_period_paths.csv")
    benchmark_actions = pd.read_csv(benchmark_output / "requested_and_implemented_actions.csv")
    benchmark_replications = pd.read_csv(benchmark_output / "benchmark_replications.csv")
    benchmark_proposals = pd.read_csv(benchmark_output / "proposal_selection_log.csv")
    checkpoint_manifest = pd.read_csv(benchmark_output / "checkpoint_manifest.csv")
    policy_set, leader, figure_policies = select_mechanism_policy_set(confidence)
    print(f"[5.2.3] Benchmark leader selected from frozen 5.2.2: {leader}", flush=True)

    model = build_model(benchmark_config)
    frozen = load_frozen_5_2_1_inputs(benchmark_config)
    executed_paths = int(pd.read_csv(benchmark_output / "selected_path_count.csv").iloc[0]["executed_paths"])
    expected_paths = int(config["execution"]["expected_physical_paths"])
    if executed_paths != expected_paths:
        raise RuntimeError(
            f"5.2.3 requires the accepted {expected_paths}-path benchmark; 5.2.2 reports {executed_paths}"
        )
    test_paths = build_test_paths(config=benchmark_config, frozen=frozen, count=executed_paths)
    test_manifest = pd.read_csv(benchmark_output / "test_path_manifest.csv").set_index("path_id")
    if len(test_paths) != expected_paths or len(test_manifest) != expected_paths:
        raise RuntimeError("The accepted 5.2.2 test manifest is not the locked 88-path design")
    for path in test_paths:
        if path.path_id not in test_manifest.index or path.path_hash != test_manifest.loc[path.path_id, "path_content_sha256"]:
            raise RuntimeError("Reconstructed 5.2.3 test path differs from frozen 5.2.2")
    medoid, medoid_path_id = select_physical_path_medoid(test_paths)
    medoid_path = next(path for path in test_paths if path.path_id == medoid_path_id)
    print(f"[5.2.3] External physical-path medoid: {medoid_path_id}", flush=True)

    full_summary, full_paired = aggregate_full_policy_mechanisms(path_level)
    block_denominators = {
        "readiness_order": float(model.action_upper[: len(model.layout.readiness_order)].sum()),
        "direct_order": float(model.action_upper[len(model.layout.readiness_order) : len(model.layout.readiness_order) + len(model.layout.direct_order)].sum()),
        "readiness_exercise": float(
            model.action_upper[
                len(model.layout.readiness_order) + len(model.layout.direct_order) :
                len(model.layout.readiness_order) + len(model.layout.direct_order) + len(model.layout.readiness_exercise)
            ].sum()
        ),
        "release": float(model.action_upper[model.layout.keys.index(model.layout.release[0])]),
        "disclosure": float(sum(model.action_upper[model.layout.keys.index(key)] for key in model.layout.disclosure)),
    }
    weekly = aggregate_weekly_policy_mechanisms(
        periods=benchmark_periods,
        actions=benchmark_actions,
        policies=figure_policies,
        action_block_denominators=block_denominators,
    )
    activation = proposed_policy_activation_audit(
        actions=benchmark_actions,
        proposals=benchmark_proposals,
        projection_tolerance=float(benchmark_config["action"]["projection_tolerance"]),
    )

    full_policy_instances: list[Any] = []
    for policy_name in config["execution"]["full_action_replay_policies"]:
        full_policy_instances.extend(
            _policy_instances(
                policy_name=str(policy_name),
                model=model,
                benchmark_output=benchmark_output,
                checkpoint_manifest=checkpoint_manifest,
            )
        )
    full_tasks = [
        (path, policy_index, "full_action")
        for policy_index in range(len(full_policy_instances))
        for path in test_paths
    ]
    full_artifacts = _evaluate_compact(
        model=model,
        benchmark_config=benchmark_config,
        experiment_config=config,
        policies=full_policy_instances,
        tasks=full_tasks,
        label="full-action reproduction",
    )

    leader_policies = _policy_instances(
        policy_name=leader,
        model=model,
        benchmark_output=benchmark_output,
        checkpoint_manifest=checkpoint_manifest,
    )
    nonfull_restrictions = [value for value in RESTRICTIONS if value != "full_action"]
    restricted_tasks = [
        (path, policy_index, restriction)
        for restriction in nonfull_restrictions
        for path in test_paths
        for policy_index in range(len(leader_policies))
    ]
    restricted_nonfull_artifacts = _evaluate_compact(
        model=model,
        benchmark_config=benchmark_config,
        experiment_config=config,
        policies=leader_policies,
        tasks=restricted_tasks,
        label="Reactive restricted-action diagnostics",
    )
    leader_full_artifacts = [
        item
        for item in full_artifacts
        if item.replication["base_policy"] == leader
    ]
    restricted_artifacts = leader_full_artifacts + restricted_nonfull_artifacts

    detailed_artifacts: list[MechanismArtifacts] = []
    for policy_name in figure_policies:
        for policy in _policy_instances(
            policy_name=policy_name,
            model=model,
            benchmark_output=benchmark_output,
            checkpoint_manifest=checkpoint_manifest,
        ):
            print(
                f"[5.2.3] Detailed medoid full trace: {policy_name}, seed={policy.training_seed}",
                flush=True,
            )
            detailed_artifacts.append(
                run_mechanism_replication(
                    model=model,
                    base_policy=policy,
                    path=medoid_path,
                    restriction="full_action",
                    no_release_pacing_baseline=1.0,
                    store_detail=True,
                )
            )
    for restriction in nonfull_restrictions:
        for policy in leader_policies:
            print(
                f"[5.2.3] Detailed medoid restriction: {restriction}, seed={policy.training_seed}",
                flush=True,
            )
            detailed_artifacts.append(
                run_mechanism_replication(
                    model=model,
                    base_policy=policy,
                    path=medoid_path,
                    restriction=restriction,
                    no_release_pacing_baseline=float(
                        config["restricted_action_diagnostic"]["no_release_pacing_baseline"]
                    ),
                    store_detail=True,
                )
            )

    frames = _frames(detailed_artifacts)
    full_replay_raw = _frames(full_artifacts)["replications"]
    restricted_raw = _frames(restricted_artifacts)["replications"]
    all_contracts = _frames(full_artifacts + restricted_nonfull_artifacts)["contracts"]
    restricted_inference = _aggregate_restricted_seeds(restricted_raw)
    path_differences, restricted_effects = restricted_action_effects(
        restricted_inference,
        confidence_level=float(config["statistics"]["confidence_level"]),
    )
    reproduction = _reproduction_checks(
        detailed_replications=full_replay_raw,
        benchmark_replications=benchmark_replications,
        tolerance=float(config["acceptance"]["full_action_reproduction_tolerance"]),
    )
    reproduction_summary = reproduction.groupby(["policy", "path_id", "training_seed"], dropna=False, as_index=False).agg(
        maximum_absolute_difference=("absolute_difference", "max"),
        all_metrics_passed=("passed", "all"),
    )

    input_hashes = {
        "acceptance_5_2_2": sha256_file(benchmark_output / "acceptance_5_2_2.json"),
        "run_manifest_5_2_2": sha256_file(benchmark_output / "run_manifest.json"),
        "checkpoint_manifest_5_2_2": sha256_file(benchmark_output / "checkpoint_manifest.csv"),
        "path_level_seed_aggregated": sha256_file(benchmark_output / "path_level_seed_aggregated.csv"),
        "benchmark_period_paths": sha256_file(benchmark_output / "benchmark_period_paths.csv"),
        "requested_and_implemented_actions": sha256_file(benchmark_output / "requested_and_implemented_actions.csv"),
        "test_path_manifest": sha256_file(benchmark_output / "test_path_manifest.csv"),
    }
    registry = parameter_registry(
        config=config,
        benchmark_config=benchmark_config,
        input_hashes=input_hashes,
        benchmark_leader=leader,
        medoid_path_id=medoid_path_id,
    )
    formulas = _formula_map()
    charts = chart_map()
    tables = {
        "mechanism_policy_set.csv": policy_set,
        "path_medoid_selection.csv": medoid,
        "full_policy_mechanism_summary.csv": full_summary,
        "full_policy_paired_mechanism_differences.csv": full_paired,
        "weekly_policy_mechanisms.csv": weekly,
        "proposed_policy_activation_audit.csv": activation,
        "full_action_replay_replications.csv": full_replay_raw,
        "restricted_action_replications.csv": restricted_inference,
        "restricted_action_path_differences.csv": path_differences,
        "restricted_action_paired_effects.csv": restricted_effects,
        "full_action_reproduction.csv": reproduction,
        "full_action_reproduction_summary.csv": reproduction_summary,
        "mechanism_action_trajectory.csv": frames["actions"],
        "behavior_source_vintage.csv": frames["behavior"],
        "physical_tagged_trajectory.csv": frames["physical"],
        "capacity_feedback_trajectory.csv": frames["capacity"],
        "loss_mechanism_trajectory.csv": frames["losses"],
        "mechanism_proposal_log.csv": frames["proposals"],
        "mechanism_contract_checks.csv": all_contracts,
        "parameter_registry_5_2_3.csv": registry,
        "formula_to_code_5_2_3.csv": formulas,
        "chart_map_5_2_3.csv": charts,
    }
    for name, frame in tables.items():
        frame.to_csv(staging / name, index=False, lineterminator="\n")

    figure_paths = create_figures(
        weekly=weekly,
        physical=frames["physical"],
        restricted_effects=restricted_effects,
        figure_policies=figure_policies,
        medoid_path_id=medoid_path_id,
        output_directory=staging / "figures",
        dpi=int(config["figures"]["dpi"]),
    )
    public_figures = CODE_ROOT / str(config["figure_directory"])
    public_figures.mkdir(parents=True, exist_ok=True)
    for path in figure_paths:
        shutil.copy2(path, public_figures / path.name)

    acceptance = acceptance_payload(
        config=config,
        input_acceptance_complete=input_complete,
        policy_set=policy_set,
        figure_policies=figure_policies,
        medoid=medoid,
        restricted_replications=restricted_inference,
        reproduction=reproduction,
        contracts=all_contracts,
        activation=activation,
        restricted_effects=restricted_effects,
        figures=figure_paths,
    )
    (staging / "acceptance_5_2_3.json").write_text(
        json.dumps(acceptance, indent=2) + "\n", encoding="utf-8"
    )
    report_directory = (CODE_ROOT / str(config["report_directory"])).resolve()
    report_paths = write_reports(
        report_directory=report_directory,
        policy_set=policy_set,
        benchmark_leader=leader,
        figure_policies=figure_policies,
        medoid_path_id=medoid_path_id,
        full_summary=full_summary,
        activation=activation,
        restricted_effects=restricted_effects,
        acceptance=acceptance,
        weekly=weekly,
        physical=frames["physical"],
    )
    if output_final.exists():
        backup = output_final.parent / f".{output_final.name}.previous"
        if backup.exists():
            shutil.rmtree(backup)
        output_final.rename(backup)
        staging.rename(output_final)
        shutil.rmtree(backup)
    else:
        staging.rename(output_final)

    output_files = [
        *(output_final / name for name in tables),
        output_final / "acceptance_5_2_3.json",
        *(output_final / "figures" / path.name for path in figure_paths),
        *(public_figures / path.name for path in figure_paths),
        *report_paths,
    ]
    manifest_inputs.extend(
        [
            benchmark_config_path,
            benchmark_output / "policy_confidence_set.csv",
            benchmark_output / "path_level_seed_aggregated.csv",
            benchmark_output / "benchmark_period_paths.csv",
            benchmark_output / "requested_and_implemented_actions.csv",
            benchmark_output / "benchmark_replications.csv",
            benchmark_output / "proposal_selection_log.csv",
            benchmark_output / "checkpoint_manifest.csv",
            benchmark_output / "selected_path_count.csv",
            benchmark_output / "test_path_manifest.csv",
        ]
    )
    manifest_path = write_run_manifest(
        output_directory=output_final,
        code_root=CODE_ROOT,
        config_path=config_path,
        input_files=list(dict.fromkeys(manifest_inputs)),
        output_files=output_files,
        started_at=started_at,
        elapsed_seconds=time.perf_counter() - started,
        status=str(acceptance["status"]),
    )
    if acceptance["status"] != "complete":
        print(json.dumps(acceptance, indent=2), flush=True)
        raise RuntimeError("5.2.3 blocking acceptance failed")
    print(
        json.dumps(
            {
                "status": acceptance["status"],
                "benchmark_leader": leader,
                "medoid_path": medoid_path_id,
                "elapsed_seconds": time.perf_counter() - started,
                "warnings": acceptance["warnings"],
                "manifest": str(manifest_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
