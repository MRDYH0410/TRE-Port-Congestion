"""Single reproducible command for Experiment 5.2.4.

Run from the code root:
    python experiments/5.2-4/run_5_2_4.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

# The production benchmark showed that four process-isolated workers with four
# numerical-library threads each avoid the severe nested-MPC oversubscription
# observed under one process per logical CPU.  The same values are registered
# in config_5_2_4.json and are set before NumPy is imported in child processes.
_math_threads = os.environ.get("TRE524_MATH_THREADS", "4")
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = _math_threads

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_DIR.parents[1]
EXPERIMENT_5_2_2 = CODE_ROOT / "experiments" / "5.2-2"
EXPERIMENT_5_2_3 = CODE_ROOT / "experiments" / "5.2-3"
SRC_ROOT = CODE_ROOT / "src"
for path in (EXPERIMENT_DIR, EXPERIMENT_5_2_2, EXPERIMENT_5_2_3, SRC_ROOT):
    sys.path.insert(0, str(path))

from controller_factory import (  # noqa: E402
    capacity_rights_registry,
    controller_id,
    copy_frozen_il_rd_controller,
    train_condition,
)
from evaluation_5_2_4 import evaluation_worker  # noqa: E402
from information_design import (  # noqa: E402
    InformationProvider,
    ReleaseTimingScenarioBuilder,
    information_regime_registry,
    load_hmm_inputs,
)
from model import build_model  # noqa: E402
from paths import (  # noqa: E402
    build_test_paths,
    build_training_validation_paths,
    load_frozen_5_2_1_inputs,
    manifest_frame,
    sha256_file,
)
from reporting_5_2_4 import (  # noqa: E402
    acceptance_payload,
    create_figure_a,
    create_figure_b,
    create_figure_c,
    evidence_classification,
    parameter_registry,
    select_medoid,
    write_manifest,
    write_reports,
)
from statistics_5_2_4 import (  # noqa: E402
    aggregate_learning_seeds,
    capacity_effects,
    clearance_summary,
    false_warning_costs,
    information_effects,
    loss_decomposition,
    precision_audit,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    """Convert NumPy/pandas scalar audit results without weakening JSON validity."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _runtime_controller_cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    expected = {"I0_RD", "IF_RD", "IL_RD", "ORACLE_RD", "IL_R", "IL_D", "IL_NONE"}
    if set(frame["controller_id"]) != expected or not frame.groupby("controller_id")["seed_index"].nunique().eq(3).all():
        return False
    for row in frame.itertuples(index=False):
        for path_column, hash_column in (
            ("bc_checkpoint_path", "bc_checkpoint_sha256"),
            ("sac_checkpoint_path", "sac_checkpoint_sha256"),
        ):
            checkpoint = CODE_ROOT / str(getattr(row, path_column))
            if not checkpoint.exists() or sha256_file(checkpoint) != str(getattr(row, hash_column)):
                return False
    for name in (
        "training_curves_runtime.csv",
        "validation_curves_runtime.csv",
        "teacher_action_manifest_runtime.csv",
    ):
        if not (path.parent / name).exists():
            return False
    return True


def _evaluation_cache_key(index: int, spec: Mapping[str, Any]) -> str:
    """Bind a resumable evaluation artifact to its complete deterministic specification."""
    payload = json.dumps(dict(spec), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return f"{index:04d}_{hashlib.sha256(payload).hexdigest()[:16]}.pkl"


def _upstream_complete(path: Path, acceptance_name: str) -> bool:
    acceptance = _read_json(path / acceptance_name)
    return acceptance.get("status") == "complete" and not acceptance.get("blocking_failures")


def _verify_upstream_locks(
    *,
    config: Mapping[str, Any],
    input_5_2_1: Path,
    input_5_2_2: Path,
    input_5_2_3: Path,
) -> dict[str, str]:
    paths = {
        "historical_information_event_path_5_2_1": input_5_2_1 / "historical_information_event_path.csv",
        "acceptance_5_2_2": input_5_2_2 / "acceptance_5_2_2.json",
        "run_manifest_5_2_2": input_5_2_2 / "run_manifest.json",
        "checkpoint_manifest_5_2_2": input_5_2_2 / "checkpoint_manifest.csv",
        "acceptance_5_2_3": input_5_2_3 / "acceptance_5_2_3.json",
        "run_manifest_5_2_3": input_5_2_3 / "run_manifest.json",
        "path_medoid_selection_5_2_3": input_5_2_3 / "path_medoid_selection.csv",
    }
    expected = {key: str(value).lower() for key, value in config["upstream_locks"].items()}
    if set(paths) != set(expected):
        raise RuntimeError("The 5.2.4 upstream lock registry is incomplete")
    actual: dict[str, str] = {}
    for key, path in paths.items():
        if not path.exists():
            raise RuntimeError(f"Missing locked upstream input: {path}")
        actual[key] = sha256_file(path).lower()
        if actual[key] != expected[key]:
            raise RuntimeError(
                f"Upstream lock mismatch for {key}: expected {expected[key]}, observed {actual[key]}"
            )
    return actual


def _validate_test_path_reconstruction(base_paths: list[Any], benchmark_output: Path) -> None:
    frozen = pd.read_csv(benchmark_output / "test_path_manifest.csv")
    if len(frozen) != 88 or len(base_paths) != 88:
        raise RuntimeError("The accepted 5.2.4 inference set must contain exactly 88 physical paths")
    expected = frozen.set_index("path_id")["path_content_sha256"].astype(str).to_dict()
    actual = {path.path_id: path.path_hash for path in base_paths}
    if actual != expected:
        raise RuntimeError("Reconstructed test paths do not match the accepted 5.2.2 path manifest")


def _load_locked_medoid(input_5_2_3: Path, base_paths: list[Any]) -> tuple[str, pd.DataFrame]:
    audit = pd.read_csv(input_5_2_3 / "path_medoid_selection.csv")
    selected_flag = audit["selected_physical_path_medoid"].map(
        lambda value: str(value).strip().lower() == "true"
    )
    outcome_flag = audit["selection_uses_policy_outcomes"].map(
        lambda value: str(value).strip().lower() == "true"
    )
    selected = audit.loc[selected_flag]
    if len(selected) != 1 or bool(outcome_flag.any()):
        raise RuntimeError("The accepted 5.2.3 medoid registry is invalid")
    medoid_id = str(selected.iloc[0]["path_id"])
    medoid_hash = str(selected.iloc[0]["path_content_sha256"])
    matching = [path for path in base_paths if path.path_id == medoid_id]
    if len(matching) != 1 or matching[0].path_hash != medoid_hash:
        raise RuntimeError("The accepted 5.2.3 medoid does not match the 88-path test manifest")
    copied = audit.copy()
    copied["source_experiment"] = "accepted 5.2.3"
    copied["source_registry_sha256"] = sha256_file(input_5_2_3 / "path_medoid_selection.csv")
    return medoid_id, copied


def _replace_staging_paths(frame: pd.DataFrame, staging: Path, final: Path) -> pd.DataFrame:
    result = frame.copy()
    staging_text = staging.relative_to(CODE_ROOT).as_posix()
    final_text = final.relative_to(CODE_ROOT).as_posix()
    for column in ("bc_checkpoint_path", "sac_checkpoint_path", "controller_bundle_path"):
        result[column] = result[column].astype(str).str.replace(staging_text, final_text, regex=False)
    return result


def _path_panel(builder: ReleaseTimingScenarioBuilder, base_paths: list[Any]) -> pd.DataFrame:
    rows = []
    for scenario in ("GH", "GT", "GL", "GFW"):
        for path in base_paths:
            designed = builder.build(path, scenario)
            frame = designed.frame
            blocked = frame["normal_model_units"].to_numpy(dtype=float) * (
                1.0 - frame["serviceability"].to_numpy(dtype=float)
            )
            service = frame["serviceability"].to_numpy(dtype=float)
            recovery = np.maximum(np.diff(service), 0.0)
            rows.append(
                {
                    "warning_scenario": scenario,
                    "base_path_id": path.path_id,
                    "base_physical_path_sha256": frame["base_physical_path_sha256"].iloc[0],
                    "total_blocked": float(blocked.sum()),
                    "peak_blocked": float(blocked.max()),
                    "mean_serviceability": float(service.mean()),
                    "minimum_serviceability": float(service.min()),
                    "recovery_rate": float(recovery.mean()) if len(recovery) else 0.0,
                    "weeks": len(frame),
                }
            )
    return pd.DataFrame(rows)


def _training_manifests(
    *,
    benchmark_config: Mapping[str, Any],
    provider: InformationProvider,
    frozen: Any,
    reference_normal: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training, validation = build_training_validation_paths(
        config=benchmark_config,
        residuals=frozen.residuals,
        reference_normal_model_units=reference_normal,
    )
    controllers = [
        ("I0_RD", "I0", "RD"),
        ("IF_RD", "IF", "RD"),
        ("IL_RD", "IL", "RD"),
        ("ORACLE_RD", "ORACLE", "RD"),
        ("IL_R", "IL", "R"),
        ("IL_D", "IL", "D"),
        ("IL_NONE", "IL", "NONE"),
    ]
    outputs = []
    for paths, split in ((training, "training"), (validation, "validation")):
        rows = []
        for cid, regime, rights in controllers:
            for path in paths:
                transformed = provider.apply_training_regime(path, regime)
                row = transformed.manifest_record()
                row.update(
                    {
                        "controller_id": cid,
                        "information_regime": regime,
                        "capacity_rights": rights,
                        "scenario_label_in_controller_observation": False,
                        "split": split,
                    }
                )
                rows.append(row)
        outputs.append(pd.DataFrame(rows))
    return outputs[0], outputs[1]


def _evaluation_specs(
    *,
    controller_manifest_path: Path,
    benchmark_config_path: Path,
    executed_paths: int,
    event_onset: str,
) -> list[dict[str, Any]]:
    base = {
        "code_root": str(CODE_ROOT),
        "benchmark_config_path": str(benchmark_config_path),
        "controller_manifest_path": str(controller_manifest_path),
        "executed_paths": executed_paths,
        "event_onset": event_onset,
        "storage_schema_version": "weekly-action-and-resource-json-v2",
    }
    specs: list[dict[str, Any]] = []

    def add(layer: str, cid: str, train_regime: str, eval_regime: str, rights: str) -> None:
        for scenario in ("GH", "GT", "GL", "GFW"):
            for path_index in range(executed_paths):
                for seed_index in range(3):
                    specs.append(
                        {
                            **base,
                            "layer": layer,
                            "controller_id": cid,
                            "training_information_regime": train_regime,
                            "evaluation_information_regime": eval_regime,
                            "capacity_rights": rights,
                            "warning_scenario": scenario,
                            "path_index": path_index,
                            "seed_index": seed_index,
                        }
                    )

    for regime in ("I0", "IF", "IL", "ORACLE"):
        add("reoptimized_information_value", controller_id(regime, "RD"), regime, regime, "RD")
    # IL is copied from the primary layer after evaluation; only the two substitutions are additional runs.
    for regime in ("I0", "IF"):
        add("fixed_policy_information_responsiveness", "IL_RD", "IL", regime, "RD")
    # RD is copied from the primary IL layer; the other rights are newly optimized controllers.
    for rights in ("R", "D", "NONE"):
        add("reoptimized_capacity_rights", controller_id("IL", rights), "IL", "IL", rights)
    for path_index in range(executed_paths):
        for seed_index in range(3):
            specs.append(
                {
                    **base,
                    "layer": "5.2.2_event_window_anchor",
                    "controller_id": "IL_RD",
                    "training_information_regime": "IL",
                    "evaluation_information_regime": "IL",
                    "capacity_rights": "RD",
                    "warning_scenario": "GH_ANCHOR",
                    "path_index": path_index,
                    "seed_index": seed_index,
                    "anchor": True,
                }
            )
    return specs


def _copy_layer(records: list[dict[str, Any]], source_layer: str, target_layer: str) -> list[dict[str, Any]]:
    copied = []
    for record in records:
        if record.get("evidence_layer") != source_layer:
            continue
        item = copy.deepcopy(record)
        item["evidence_layer"] = target_layer
        copied.append(item)
    return copied


def _stream_evaluation_outputs(
    *,
    cache_paths: Iterable[Path],
    staging: Path,
    medoid_path_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    """Materialise full weekly traces without retaining millions of dicts in RAM."""

    action_path = staging / "capacity_action_trace.csv"
    capacity_path = staging / "readiness_stock_flow.csv"
    direct_path = staging / "direct_capacity_pipeline.csv"
    loss_path = staging / "period_loss_trace.csv"
    release_path = staging / "release_information_panel.csv"
    files = {
        "actions": action_path.open("w", encoding="utf-8", newline=""),
        "capacity": capacity_path.open("w", encoding="utf-8", newline=""),
        "losses": loss_path.open("w", encoding="utf-8", newline=""),
        "release": release_path.open("w", encoding="utf-8", newline=""),
    }
    writers: dict[str, csv.DictWriter] = {}
    replication_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    medoid_capacity_rows: list[dict[str, Any]] = []
    release_columns = [
        "evidence_layer",
        "controller_id",
        "training_information_regime",
        "evaluation_information_regime",
        "capacity_rights",
        "warning_scenario",
        "base_path_id",
        "training_seed",
        "decision_week",
        "source_observation_month",
        "actual_public_release_date",
        "scenario_release_date",
        "decision_availability_week",
        "event_onset",
        "g_R_weeks",
        "g_D_weeks",
        "released_filtered_high_risk_probability",
        "released_lead_high_risk_probability",
        "controller_current_high_risk_probability",
        "controller_lead_high_risk_probability",
        "monthly_transitions_to_readiness_maturity",
        "weekly_transition_matrix_applications",
    ]

    def write_rows(name: str, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            if name not in writers:
                fieldnames = release_columns if name == "release" else list(row.keys())
                writers[name] = csv.DictWriter(files[name], fieldnames=fieldnames, extrasaction="ignore")
                writers[name].writeheader()
            writers[name].writerow(row)

    try:
        for cache_path in sorted(cache_paths):
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            result = cached["result"]
            base_layer = str(result["replication"]["evidence_layer"])
            variants = [base_layer]
            if base_layer == "reoptimized_information_value" and result["replication"]["controller_id"] == "IL_RD":
                variants.extend(["fixed_policy_information_responsiveness", "reoptimized_capacity_rights"])
            for layer in variants:
                replication = dict(result["replication"])
                replication["evidence_layer"] = layer
                replication_rows.append(replication)
                if base_layer == "5.2.2_event_window_anchor":
                    continue
                contract = dict(result["contract"])
                contract["evidence_layer"] = layer
                contract_rows.append(contract)
                actions = [{**row, "evidence_layer": layer} for row in result["actions"]]
                capacity = [{**row, "evidence_layer": layer} for row in result["capacity"]]
                losses = [{**row, "evidence_layer": layer} for row in result["losses"]]
                write_rows("actions", actions)
                write_rows("capacity", capacity)
                write_rows("losses", losses)
                write_rows("release", capacity)
                if replication["base_path_id"] == medoid_path_id:
                    medoid_capacity_rows.extend(capacity)
    finally:
        for handle in files.values():
            handle.close()
    shutil.copy2(capacity_path, direct_path)
    return (
        pd.DataFrame(replication_rows),
        pd.DataFrame(contract_rows),
        pd.DataFrame(medoid_capacity_rows),
        [action_path, capacity_path, direct_path, loss_path, release_path],
    )


def _anchor_reproduction(anchor: pd.DataFrame, benchmark_output: Path) -> pd.DataFrame:
    frozen = pd.read_csv(benchmark_output / "benchmark_replications.csv")
    frozen = frozen.loc[frozen["policy"].eq("Model-guided constrained SAC")]
    metrics = [
        "total_operational_objective",
        "loss_queue",
        "loss_waiting",
        "loss_exit",
        "loss_overflow",
        "loss_route_resource",
        "loss_action",
        "terminal_correction",
        "ending_outstanding_mass",
    ]
    rows = []
    for actual in anchor.itertuples(index=False):
        expected = frozen.loc[
            frozen["path_id"].eq(actual.base_path_id)
            & pd.to_numeric(frozen["training_seed"], errors="coerce").eq(float(actual.training_seed))
        ]
        if len(expected) != 1:
            raise RuntimeError("Cannot bind the 5.2.4 anchor to the accepted 5.2.2 proposed-policy path")
        expected = expected.iloc[0]
        for metric in metrics:
            difference = float(getattr(actual, metric)) - float(expected[metric])
            rows.append(
                {
                    "base_path_id": actual.base_path_id,
                    "training_seed": actual.training_seed,
                    "metric": metric,
                    "accepted_5_2_2_value": float(expected[metric]),
                    "replayed_5_2_4_anchor_value": float(getattr(actual, metric)),
                    "absolute_difference": abs(difference),
                }
            )
    return pd.DataFrame(rows)


def _release_panel(capacity: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "evidence_layer",
        "controller_id",
        "training_information_regime",
        "evaluation_information_regime",
        "capacity_rights",
        "warning_scenario",
        "base_path_id",
        "training_seed",
        "decision_week",
        "source_observation_month",
        "actual_public_release_date",
        "scenario_release_date",
        "decision_availability_week",
        "event_onset",
        "g_R_weeks",
        "g_D_weeks",
        "released_filtered_high_risk_probability",
        "released_lead_high_risk_probability",
        "controller_current_high_risk_probability",
        "controller_lead_high_risk_probability",
        "monthly_transitions_to_readiness_maturity",
        "weekly_transition_matrix_applications",
    ]
    return capacity.loc[:, columns].drop_duplicates().sort_values(
        ["evidence_layer", "controller_id", "warning_scenario", "base_path_id", "training_seed", "decision_week"]
    )


def run() -> int:
    started = time.perf_counter()
    config_path = EXPERIMENT_DIR / "config_5_2_4.json"
    config = _read_json(config_path)
    config_hash = sha256_file(config_path)
    benchmark_config_path = EXPERIMENT_5_2_2 / "config_5_2_2.json"
    benchmark_config = _read_json(benchmark_config_path)
    input_5_2_1 = CODE_ROOT / str(config["input_5_2_1"])
    input_5_2_2 = CODE_ROOT / str(config["input_5_2_2"])
    input_5_2_3 = CODE_ROOT / str(config["input_5_2_3"])
    locked_hashes = _verify_upstream_locks(
        config=config,
        input_5_2_1=input_5_2_1,
        input_5_2_2=input_5_2_2,
        input_5_2_3=input_5_2_3,
    )
    complete_521 = _upstream_complete(input_5_2_1, "acceptance_5_2_1.json")
    complete_522 = _upstream_complete(input_5_2_2, "acceptance_5_2_2.json")
    complete_523 = _upstream_complete(input_5_2_3, "acceptance_5_2_3.json")
    if not complete_521 or not complete_522 or not complete_523:
        raise RuntimeError("5.2.1, 5.2.2, and 5.2.3 must be accepted before 5.2.4")
    output_final = CODE_ROOT / str(config["output_directory"])
    staging = output_final.parent / f".{output_final.name}.staging"
    runtime_manifest_path = staging / "controller_manifest_runtime.csv"
    provenance_path = staging / "current_run_provenance.json"
    provenance_payload = {
        "config_sha256": config_hash,
        "upstream_locks": locked_hashes,
        "old_5_2_4_artifacts_used": False,
    }
    resume_requested = os.environ.get("TRE524_RESUME_CURRENT_RUN", "0") == "1"
    resume_provenance_valid = False
    if resume_requested and provenance_path.exists():
        try:
            resume_provenance_valid = _read_json(provenance_path) == provenance_payload
        except Exception:
            resume_provenance_valid = False
    # A prior 5.2.4 checkpoint or partial output is never a legal input to this
    # repair run.  Keep the accepted public output untouched until the new run
    # passes, but always start the staging area from a clean directory.
    if staging.exists() and not resume_provenance_valid:
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance_payload, indent=2) + "\n", encoding="utf-8"
    )
    resume_controller_cache = bool(
        resume_provenance_valid and _runtime_controller_cache_valid(runtime_manifest_path)
    )
    checkpoint_root = staging / "checkpoints"
    print("[5.2.4] Verified all seven accepted 5.2.1-5.2.3 upstream locks.", flush=True)

    frozen = load_frozen_5_2_1_inputs(benchmark_config)
    hmm = load_hmm_inputs(input_5_2_1)
    model = build_model(benchmark_config)
    provider = InformationProvider(
        hmm=hmm,
        readiness_lead_weeks=int(benchmark_config["action"]["readiness_lead_weeks"]),
    )
    selected_522 = pd.read_csv(input_5_2_2 / "selected_path_count.csv").iloc[0]
    executed_paths = int(selected_522["executed_paths"])
    base_paths = build_test_paths(config=benchmark_config, frozen=frozen, count=executed_paths)
    _validate_test_path_reconstruction(base_paths, input_5_2_2)
    builder = ReleaseTimingScenarioBuilder(
        hmm=hmm,
        event_onset=pd.Timestamp(config["timing"]["event_onset"]),
        readiness_lead_weeks=int(benchmark_config["action"]["readiness_lead_weeks"]),
        reference_normal_model_units=float(sum(model.gateway_scales.values())),
    )
    warning_registry = builder.scenario_registry()
    information_registry = information_regime_registry(hmm)
    rights_registry = capacity_rights_registry()
    path_panel = _path_panel(builder, base_paths)
    medoid_path_id, medoid_audit = _load_locked_medoid(input_5_2_3, base_paths)
    training_manifest, validation_manifest = _training_manifests(
        benchmark_config=benchmark_config,
        provider=provider,
        frozen=frozen,
        reference_normal=float(sum(model.gateway_scales.values())),
    )
    print(f"[5.2.4] Using the hash-locked 5.2.3 physical-path medoid: {medoid_path_id}", flush=True)

    training_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    teacher_rows: list[dict[str, Any]] = []
    if resume_controller_cache:
        runtime_manifest = pd.read_csv(runtime_manifest_path).sort_values(["controller_id", "seed_index"])
        training_rows = pd.read_csv(staging / "training_curves_runtime.csv").to_dict("records")
        validation_rows = pd.read_csv(staging / "validation_curves_runtime.csv").to_dict("records")
        teacher_rows = pd.read_csv(staging / "teacher_action_manifest_runtime.csv").to_dict("records")
        print("[5.2.4] Resuming the current provenance-bound run with complete training and validation traces.", flush=True)
    else:
        manifest_rows = copy_frozen_il_rd_controller(
            code_root=CODE_ROOT,
            benchmark_output=input_5_2_2,
            checkpoint_root=checkpoint_root,
        )
        new_conditions = [
            ("I0", "RD"),
            ("IF", "RD"),
            ("ORACLE", "RD"),
            ("IL", "R"),
            ("IL", "D"),
            ("IL", "NONE"),
        ]
        print(f"[5.2.4] Training {len(new_conditions)} condition-specific controller bundles...", flush=True)
        with ProcessPoolExecutor(max_workers=int(config["computation"]["parallel_training_workers"])) as executor:
            futures = {
                executor.submit(
                    train_condition,
                    code_root=CODE_ROOT,
                    benchmark_config_path=benchmark_config_path,
                    information_regime=regime,
                    capacity_rights=rights,
                    checkpoint_root=checkpoint_root,
                    config_hash=config_hash,
                ): (regime, rights)
                for regime, rights in new_conditions
            }
            for future in as_completed(futures):
                regime, rights = futures[future]
                result = future.result()
                manifest_rows.extend(result["manifest"])
                training_rows.extend(result["training"])
                validation_rows.extend(result["validation"])
                teacher_rows.extend(result["teacher"])
                print(f"[5.2.4] Controller training complete: {regime}_{rights}", flush=True)
        runtime_manifest = pd.DataFrame(manifest_rows).sort_values(["controller_id", "seed_index"])
        runtime_manifest.to_csv(runtime_manifest_path, index=False, lineterminator="\n")
        pd.DataFrame(training_rows).to_csv(staging / "training_curves_runtime.csv", index=False, lineterminator="\n")
        pd.DataFrame(validation_rows).to_csv(staging / "validation_curves_runtime.csv", index=False, lineterminator="\n")
        pd.DataFrame(teacher_rows).to_csv(staging / "teacher_action_manifest_runtime.csv", index=False, lineterminator="\n")

    specs = _evaluation_specs(
        controller_manifest_path=runtime_manifest_path,
        benchmark_config_path=benchmark_config_path,
        executed_paths=executed_paths,
        event_onset=str(config["timing"]["event_onset"]),
    )
    completed_cache_paths: list[Path] = []
    evaluation_cache = staging / "evaluation_cache"
    evaluation_cache.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[dict[str, Any], Path]] = []
    for index, spec in enumerate(specs):
        cache_path = evaluation_cache / _evaluation_cache_key(index, spec)
        if cache_path.exists():
            try:
                with cache_path.open("rb") as handle:
                    cached = pickle.load(handle)
                if cached.get("cache_key") != cache_path.stem:
                    raise ValueError("evaluation cache key mismatch")
                completed_cache_paths.append(cache_path)
                continue
            except Exception:
                cache_path.unlink(missing_ok=True)
        pending.append((spec, cache_path))
    print(
        f"[5.2.4] Running {len(pending)} matched evaluations; "
        f"{len(completed_cache_paths)} restored from hash-bound cache ({len(specs)} total)...",
        flush=True,
    )
    os.environ["TRE524_MATH_THREADS"] = str(
        config["computation"]["evaluation_math_threads_per_worker"]
    )
    for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[thread_variable] = os.environ["TRE524_MATH_THREADS"]
    with ProcessPoolExecutor(max_workers=int(config["computation"]["parallel_evaluation_workers"])) as executor:
        futures = {executor.submit(evaluation_worker, spec): cache_path for spec, cache_path in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            cache_path = futures[future]
            temporary = cache_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                pickle.dump({"cache_key": cache_path.stem, "result": result}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(cache_path)
            completed_cache_paths.append(cache_path)
            completed = len(completed_cache_paths)
            if index % 12 == 0 or completed == len(specs):
                print(f"[5.2.4] Matched evaluations complete: {completed}/{len(specs)}", flush=True)

    raw, contract_frame, capacity_frame, streamed_trace_paths = _stream_evaluation_outputs(
        cache_paths=completed_cache_paths,
        staging=staging,
        medoid_path_id=medoid_path_id,
    )
    anchor_raw = raw.loc[raw["evidence_layer"].eq("5.2.2_event_window_anchor")].copy()
    raw = raw.loc[~raw["evidence_layer"].eq("5.2.2_event_window_anchor")].copy()
    path_level = aggregate_learning_seeds(raw)
    primary_path = path_level.loc[path_level["evidence_layer"].eq("reoptimized_information_value")].copy()
    fixed_path = path_level.loc[path_level["evidence_layer"].eq("fixed_policy_information_responsiveness")].copy()
    capacity_path = path_level.loc[path_level["evidence_layer"].eq("reoptimized_capacity_rights")].copy()
    primary_effects = information_effects(
        primary_path,
        confidence=float(config["statistics"]["confidence_level"]),
        bootstrap_resamples=int(config["statistics"]["bootstrap_resamples"]),
        fixed=False,
    )
    fixed_effect_rows = information_effects(
        fixed_path,
        confidence=float(config["statistics"]["confidence_level"]),
        bootstrap_resamples=int(config["statistics"]["bootstrap_resamples"]),
        fixed=True,
    )
    rights_effects = capacity_effects(
        capacity_path,
        confidence=float(config["statistics"]["confidence_level"]),
        bootstrap_resamples=int(config["statistics"]["bootstrap_resamples"]),
    )
    paired = pd.concat([primary_effects, fixed_effect_rows, rights_effects], ignore_index=True)
    false_warning = false_warning_costs(primary_path)
    loss_summary = loss_decomposition(path_level)
    clearance = clearance_summary(path_level, int(benchmark_config["clearance"]["maximum_weeks"]))
    precision, selected_path_count = precision_audit(
        paired,
        pilot_paths=int(config["statistics"]["pilot_paths"]),
        executed_paths=executed_paths,
        target_halfwidth=float(selected_522["target_halfwidth"]),
        confidence=float(config["statistics"]["confidence_level"]),
        maximum_paths=int(selected_522["computational_cap"]),
    )
    anchor = _anchor_reproduction(anchor_raw, input_5_2_2)
    input_hashes = {
        **locked_hashes,
        "acceptance_5_2_1": sha256_file(input_5_2_1 / "acceptance_5_2_1.json"),
        "hmm_parameter_manifest": sha256_file(input_5_2_1 / "hmm_parameter_manifest.csv"),
        "released_hmm_filter": sha256_file(input_5_2_1 / "released_hmm_filter.csv"),
        "release_clock": sha256_file(input_5_2_1 / "release_clock.csv"),
        "historical_information_event_path": sha256_file(input_5_2_1 / "historical_information_event_path.csv"),
        "acceptance_5_2_2": sha256_file(input_5_2_2 / "acceptance_5_2_2.json"),
        "checkpoint_manifest_5_2_2": sha256_file(input_5_2_2 / "checkpoint_manifest.csv"),
        "test_path_manifest_5_2_2": sha256_file(input_5_2_2 / "test_path_manifest.csv"),
        "acceptance_5_2_3_current": sha256_file(input_5_2_3 / "acceptance_5_2_3.json"),
        "run_manifest_5_2_3_current": sha256_file(input_5_2_3 / "run_manifest.json"),
        "path_medoid_selection_5_2_3_current": sha256_file(input_5_2_3 / "path_medoid_selection.csv"),
    }
    registry = parameter_registry(
        config=config,
        benchmark_config=benchmark_config,
        hmm=hmm,
        input_hashes=input_hashes,
    )
    evidence = evidence_classification()
    final_controller_manifest = _replace_staging_paths(runtime_manifest, staging, output_final)
    path_pairing = (
        raw.groupby(["evidence_layer", "warning_scenario", "base_path_id"], as_index=False)
        .agg(
            unique_physical_hashes=("base_physical_path_sha256", "nunique"),
            controllers=("controller_id", "nunique"),
            evaluation_regimes=("evaluation_information_regime", "nunique"),
            capacity_right_sets=("capacity_rights", "nunique"),
            seeds=("training_seed", "nunique"),
        )
    )
    path_pairing["matched_physical_path_passed"] = path_pairing["unique_physical_hashes"].eq(1)

    figure_output = staging / "figures"
    figure_a_paths, figure_a_data = create_figure_a(
        information_effects=primary_effects,
        fixed_effects=fixed_effect_rows,
        output_directory=figure_output,
        dpi=int(config["figures"]["dpi"]),
    )
    figure_b_paths, figure_b_data = create_figure_b(
        capacity_trace=capacity_frame,
        medoid_path_id=medoid_path_id,
        output_directory=figure_output,
        dpi=int(config["figures"]["dpi"]),
    )
    figure_c_paths, figure_c_data = create_figure_c(
        capacity_path_level=capacity_path,
        capacity_effects=rights_effects,
        loss_components=loss_summary,
        output_directory=figure_output,
        dpi=int(config["figures"]["dpi"]),
    )
    all_figure_paths = [*figure_a_paths, *figure_b_paths, *figure_c_paths]
    acceptance = acceptance_payload(
        upstream_5_2_1_complete=complete_521,
        upstream_5_2_2_complete=complete_522,
        upstream_5_2_3_complete=complete_523,
        upstream_locks_match=True,
        parameter_registry_frame=registry,
        information_registry=information_registry,
        warning_registry=warning_registry,
        controller_manifest=final_controller_manifest,
        raw_replications=raw,
        contracts=contract_frame,
        capacity_trace=capacity_frame,
        anchor=anchor,
        precision=precision,
        information_effects=information_effects,
        medoid_audit=medoid_audit,
        expected_physical_paths=executed_paths,
        figure_paths=all_figure_paths,
        tolerance=float(config["acceptance"]["anchor_tolerance"]),
    )
    if acceptance["status"] != "complete":
        (staging / "acceptance_5_2_4.json").write_text(
            json.dumps(acceptance, indent=2, default=_json_default) + "\n", encoding="utf-8"
        )
        print(json.dumps(acceptance, indent=2, default=_json_default), flush=True)
        raise RuntimeError("5.2.4 blocking acceptance failed")

    tables: dict[str, pd.DataFrame] = {
        "information_regime_registry.csv": information_registry,
        "warning_case_registry.csv": warning_registry,
        "capacity_rights_registry.csv": rights_registry,
        "controller_manifest.csv": final_controller_manifest,
        "training_path_manifest.csv": training_manifest,
        "validation_path_manifest.csv": validation_manifest,
        "training_curves.csv": pd.DataFrame(training_rows),
        "validation_curves.csv": pd.DataFrame(validation_rows),
        "teacher_action_manifest.csv": pd.DataFrame(teacher_rows),
        "training_validation_summary.csv": final_controller_manifest[["controller_id", "information_regime", "capacity_rights", "seed_index", "training_seed", "bc_selected_episode", "sac_selected_episode", "bc_validation_loss", "sac_validation_loss", "generated_from_scratch", "reused_5_2_2_anchor", "test_paths_seen_before_selection"]],
        "path_pairing_audit.csv": path_pairing,
        "primary_information_value_path_results.csv": primary_path,
        "fixed_policy_information_diagnostic.csv": fixed_path,
        "capacity_rights_path_results.csv": capacity_path,
        "controller_replications.csv": raw,
        "loss_decomposition.csv": loss_summary,
        "clearance_and_censoring.csv": clearance,
        "paired_effects.csv": paired,
        "false_warning_costs.csv": false_warning,
        "pilot_precision.csv": precision,
        "selected_path_count.csv": selected_path_count,
        "trajectory_contract_checks.csv": contract_frame,
        "historical_anchor_reproduction.csv": anchor,
        "physical_path_medoid_audit.csv": medoid_audit,
        "physical_path_panel.csv": path_panel,
        "parameter_registry_5_2_4.csv": registry,
        "evidence_classification.csv": evidence,
        "figure_5_2_4a_data.csv": figure_a_data,
        "figure_5_2_4b_data.csv": figure_b_data,
        "figure_5_2_4c_data.csv": figure_c_data,
    }
    runtime_manifest_path.unlink(missing_ok=True)
    for runtime_name in (
        "training_curves_runtime.csv",
        "validation_curves_runtime.csv",
        "teacher_action_manifest_runtime.csv",
    ):
        (staging / runtime_name).unlink(missing_ok=True)
    for name, frame in tables.items():
        frame.to_csv(staging / name, index=False, lineterminator="\n")
    (staging / "acceptance_5_2_4.json").write_text(
        json.dumps(acceptance, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    report_directory = (CODE_ROOT / str(config["report_directory"])).resolve()
    report_paths = write_reports(
        report_directory=report_directory,
        registry=registry,
        information_effects=primary_effects,
        fixed_effects=fixed_effect_rows,
        capacity_effects=rights_effects,
        false_warning=false_warning,
        acceptance=acceptance,
        precision=precision,
        clearance=clearance,
    )
    # The cache is a crash-recovery aid, not a scientific output.
    shutil.rmtree(evaluation_cache)
    if output_final.exists():
        backup = output_final.parent / f".{output_final.name}.previous"
        if backup.exists():
            shutil.rmtree(backup)
        output_final.rename(backup)
        staging.rename(output_final)
        shutil.rmtree(backup)
    else:
        staging.rename(output_final)
    public_figures = CODE_ROOT / str(config["figure_directory"])
    public_figures.mkdir(parents=True, exist_ok=True)
    final_figure_paths = []
    for path in all_figure_paths:
        source = output_final / "figures" / path.name
        target = public_figures / path.name
        shutil.copy2(source, target)
        final_figure_paths.extend([source, target])
    output_files = [
        *(output_final / name for name in tables),
        *(output_final / path.name for path in streamed_trace_paths),
        output_final / "acceptance_5_2_4.json",
        output_final / "current_run_provenance.json",
        *final_figure_paths,
        *report_paths,
        *(path for path in (output_final / "checkpoints").rglob("*.*")),
    ]
    input_files = [
        benchmark_config_path,
        input_5_2_1 / "acceptance_5_2_1.json",
        input_5_2_1 / "run_manifest.json",
        input_5_2_1 / "hmm_parameter_manifest.csv",
        input_5_2_1 / "released_hmm_filter.csv",
        input_5_2_1 / "release_clock.csv",
        input_5_2_1 / "historical_information_event_path.csv",
        input_5_2_1 / "counterfactual_residual_library.csv",
        input_5_2_2 / "acceptance_5_2_2.json",
        input_5_2_2 / "run_manifest.json",
        input_5_2_2 / "checkpoint_manifest.csv",
        input_5_2_2 / "benchmark_replications.csv",
        input_5_2_2 / "test_path_manifest.csv",
        input_5_2_2 / "selected_path_count.csv",
        input_5_2_3 / "acceptance_5_2_3.json",
        input_5_2_3 / "run_manifest.json",
        input_5_2_3 / "path_medoid_selection.csv",
    ]
    manifest_path = write_manifest(
        output_directory=output_final,
        config_path=config_path,
        input_files=input_files,
        output_files=output_files,
        elapsed_seconds=time.perf_counter() - started,
    )
    print(
        json.dumps(
            {
                "status": acceptance["status"],
                "elapsed_seconds": time.perf_counter() - started,
                "evaluations": len(specs),
                "medoid_path": medoid_path_id,
                "precision_targets_met": int(precision["precision_target_met"].sum()),
                "precision_contrasts": len(precision),
                "manifest": str(manifest_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
