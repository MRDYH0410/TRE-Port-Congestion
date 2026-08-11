r"""Resumable layered-coverage runner for Experiment 5.3.2.

Run from the code root:
    .\.venv\Scripts\python.exe experiments\5.3-2\run_5_3_2.py --phase all

The one-path and eight-path gates write no formal statistical result.  The
formal phase reuses only cache partitions with the same config, source,
upstream-input and checkpoint signature.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_DIR.parents[1]
PROJECT_ROOT = CODE_ROOT.parent
SRC_ROOT = CODE_ROOT / "src"
BENCHMARK_DIR = CODE_ROOT / "experiments" / "5.2-2"
MECHANISM_DIR = CODE_ROOT / "experiments" / "5.2-3"
for entry in (EXPERIMENT_DIR, BENCHMARK_DIR, MECHANISM_DIR, SRC_ROOT):
    sys.path.insert(0, str(entry))

from absorption_5_3_2 import absorption_certificate  # noqa: E402
from model import build_model  # noqa: E402
from paths import build_test_paths, load_frozen_5_2_1_inputs, manifest_frame  # noqa: E402
from reclosure_worker import (  # noqa: E402
    GridCell,
    build_cell_path,
    evaluate_path_policy_task,
    initialise_worker,
)
from reporting_5_3_2 import (  # noqa: E402
    acceptance_payload,
    create_figures,
    formula_registry,
    independent_recalculation,
    parameter_registry,
    sha256_file,
    write_csv,
    write_manifest,
    write_reports,
)
from statistics_5_3_2 import (  # noqa: E402
    aggregate_learning_seeds,
    confidence_sets_and_regret,
    paired_effects,
    precision_requirements,
    summaries,
    update_precision,
)


SOURCE_FILES = [
    "src/tre84/actions.py", "src/tre84/acceptance.py", "src/tre84/behavior.py",
    "src/tre84/capacity.py", "src/tre84/clearance.py", "src/tre84/control.py",
    "src/tre84/diagnostics.py", "src/tre84/engine.py", "src/tre84/loss.py",
    "src/tre84/scenarios.py", "src/tre84/state.py", "src/tre84/transition.py",
    "experiments/5.2-2/features.py", "experiments/5.2-2/model.py",
    "experiments/5.2-2/paths.py", "experiments/5.2-2/policies.py",
    "experiments/5.2-2/preparation.py", "experiments/5.2-2/simulator.py",
    "experiments/5.2-3/mechanism.py", "experiments/5.3-2/absorption_5_3_2.py",
    "experiments/5.3-2/reclosure_worker.py", "experiments/5.3-2/statistics_5_3_2.py",
    "experiments/5.3-2/reporting_5_3_2.py", "experiments/5.3-2/run_5_3_2.py",
]
SIMULATION_SOURCE_FILES = [
    item for item in SOURCE_FILES
    if item not in {
        "experiments/5.3-2/absorption_5_3_2.py",
        "experiments/5.3-2/statistics_5_3_2.py",
        "experiments/5.3-2/reporting_5_3_2.py",
        "experiments/5.3-2/run_5_3_2.py",
    }
]

CACHE_DIR = EXPERIMENT_DIR / "cache_layered"
GATE_DIR = EXPERIMENT_DIR / "gate_results"


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_hash(files: Sequence[str] = SOURCE_FILES) -> str:
    digest = hashlib.sha256()
    for relative in files:
        path = CODE_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_configs() -> tuple[dict[str, Any], dict[str, Any], str]:
    path = EXPERIMENT_DIR / "config_5_3_2.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    base = json.loads((CODE_ROOT / config["base_model_config"]).read_text(encoding="utf-8"))
    return config, base, sha256_file(path)


def _verify_upstream(config: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for lock in config["upstream_locks"]:
        path = CODE_ROOT / lock["path"]
        observed = sha256_file(path) if path.exists() else "MISSING"
        rows.append({
            "relative_path": lock["path"], "expected_sha256": lock["sha256"],
            "observed_sha256": observed, "bytes": path.stat().st_size if path.exists() else 0,
            "matched": observed == lock["sha256"],
        })
    frame = pd.DataFrame(rows)
    if not frame["matched"].all():
        raise RuntimeError(f"Upstream lock mismatch: {frame.loc[~frame['matched'], 'relative_path'].tolist()}")
    acceptance_525 = json.loads((CODE_ROOT / "output/5.2.5_computational_methodological_acceptance/acceptance_5_2_5.json").read_text(encoding="utf-8"))
    if acceptance_525.get("OVERALL_ACCEPTANCE") != "PASS":
        raise RuntimeError("The refreshed 5.2.5 methodology gate is not PASS")
    acceptance_531 = json.loads((CODE_ROOT / "output/5.3.1_commitment_sensitivity/acceptance_5_3_1.json").read_text(encoding="utf-8"))
    if acceptance_531.get("overall_evidence_acceptance") != "PASS":
        raise RuntimeError("5.3.1 is not an accepted upstream sensitivity experiment")
    compatibility = json.loads((CODE_ROOT / "output/5.3.1_core_repair_compatibility/acceptance_5_3_1_core_repair_compatibility.json").read_text(encoding="utf-8"))
    if compatibility.get("overall_compatibility_acceptance") != "PASS" and compatibility.get("status") != "PASS":
        raise RuntimeError("5.3.1 core-repair compatibility is not PASS")
    return frame


def _model_config(base: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    result["experiment_id"] = config["experiment_id"]
    result["committed_fraction_reference"] = float(config["commitment_fraction"])
    result["main_policies"] = list(config["main_policies"])
    result["learning_policies"] = ["Behaviour cloning", "Model-guided constrained SAC"]
    return result


def _cell(row: Mapping[str, Any]) -> GridCell:
    return GridCell(int(row["open_interval_weeks"]), float(row["reclosure_intensity"]), int(row["reclosure_duration_weeks"]))


def _certificate_cells(config: Mapping[str, Any]) -> list[GridCell]:
    grid = config["grid"]
    cells = [
        GridCell(int(open_weeks), float(intensity), int(duration))
        for duration in grid["reclosure_duration_weeks"]
        for intensity in grid["reclosure_intensity"]
        for open_weeks in grid["open_interval_weeks"]
    ]
    if len(cells) != 150 or len({cell.cell_id for cell in cells}) != 150:
        raise RuntimeError("The physical certificate grid is not exactly 150 unique cells")
    return cells


def _policy_cells(config: Mapping[str, Any]) -> tuple[list[GridCell], list[GridCell]]:
    grid = config["grid"]
    coverage = config["layered_policy_coverage"]
    reference = _cell(coverage["reference_cell"])
    cells = {
        GridCell(int(value), reference.intensity, reference.duration_weeks)
        for value in grid["open_interval_weeks"]
    }
    cells.update(
        GridCell(reference.open_weeks, float(value), reference.duration_weeks)
        for value in grid["reclosure_intensity"]
    )
    cells.update(
        GridCell(reference.open_weeks, reference.intensity, int(value))
        for value in grid["reclosure_duration_weeks"]
    )
    cells.add(_cell(coverage["mild_corner"]))
    cells.add(_cell(coverage["severe_corner"]))
    policy_cells = sorted(cells, key=lambda item: (item.open_weeks, item.intensity, item.duration_weeks))
    anchors = sorted(
        [_cell(coverage["reference_cell"]), _cell(coverage["mild_corner"]), _cell(coverage["severe_corner"])],
        key=lambda item: (item.open_weeks, item.intensity, item.duration_weeks),
    )
    if len(policy_cells) != 16 or len({item.cell_id for item in anchors}) != 3:
        raise RuntimeError("Layered policy coverage must contain 16 cells and three unique anchors")
    if not set(anchors).issubset(policy_cells):
        raise RuntimeError("Every full-policy anchor must belong to the 16 policy cells")
    return policy_cells, anchors


def _checkpoint_specs() -> tuple[list[dict[str, Any]], pd.DataFrame]:
    output = CODE_ROOT / "output/5.2.2_common_authority_benchmark"
    manifest = pd.read_csv(output / "checkpoint_manifest.csv")
    specs: list[dict[str, Any]] = [
        {"kind": "passive", "policy": "Passive", "seed_index": -1},
        {"kind": "reactive", "policy": "Reactive", "seed_index": -1},
        {"kind": "mpc", "policy": "Projected stochastic MPC", "seed_index": -1},
    ]
    audit_rows = []
    for seed_index in range(3):
        bc_row = manifest.loc[(manifest["policy"] == "Behaviour cloning") & (manifest["seed_index"] == seed_index)].iloc[0]
        sac_row = manifest.loc[(manifest["policy"] == "Constrained SAC") & (manifest["seed_index"] == seed_index)].iloc[0]
        bc_path = output / str(bc_row["checkpoint_path"])
        sac_path = output / str(sac_row["checkpoint_path"])
        for row, path in ((bc_row, bc_path), (sac_row, sac_path)):
            digest = sha256_file(path)
            if digest != row["checkpoint_sha256"]:
                raise RuntimeError(f"Frozen checkpoint hash mismatch: {path}")
            audit_rows.append({
                "policy": row["policy"], "seed_index": seed_index,
                "training_seed": int(row["training_seed"]),
                "checkpoint_path": path.relative_to(CODE_ROOT).as_posix(),
                "checkpoint_sha256": digest, "source_experiment": "accepted 5.2.2",
                "retrained_for_5_3_2": False,
            })
        specs.append({
            "kind": "actor", "name": "Behaviour cloning", "policy": "Behaviour cloning",
            "seed_index": seed_index, "training_seed": int(bc_row["training_seed"]),
            "checkpoint": str(bc_path),
        })
        specs.append({
            "kind": "model_guided", "policy": "Model-guided constrained SAC",
            "seed_index": seed_index, "training_seed": int(sac_row["training_seed"]),
            "bc_checkpoint": str(bc_path), "sac_checkpoint": str(sac_path),
        })
    if len(specs) != 9:
        raise RuntimeError("5.3.2 requires exactly nine policy/seed task types")
    return specs, pd.DataFrame(audit_rows)


def _cells_by_spec(specs: Sequence[Mapping[str, Any]], policy_cells: Sequence[GridCell], anchors: Sequence[GridCell]) -> list[list[GridCell]]:
    axial = {"Passive", "Reactive", "Behaviour cloning"}
    result = [list(policy_cells) if str(spec["policy"]) in axial else list(anchors) for spec in specs]
    if sum(len(cells) for cells in result) != 5 * 16 + 4 * 3:
        raise RuntimeError("Layered task coverage must equal 92 policy-seed cell evaluations per path")
    return result


def _assert_first_88(paths: Sequence[Any]) -> None:
    accepted = pd.read_csv(CODE_ROOT / "output/5.2.2_common_authority_benchmark/test_path_manifest.csv")
    current = manifest_frame(paths)
    merged = current.merge(
        accepted[["path_id", "path_content_sha256"]].rename(columns={"path_content_sha256": "accepted_hash"}),
        on="path_id", how="left", validate="one_to_one",
    )
    if len(paths) < 88 or merged.iloc[:88]["accepted_hash"].isna().any() or not (merged.iloc[:88]["path_content_sha256"] == merged.iloc[:88]["accepted_hash"]).all():
        raise RuntimeError("The first 88 test paths do not reproduce the accepted 5.2.2 manifest")


def _run_signature(config_hash: str, source_hash: str, upstream: pd.DataFrame, checkpoints: pd.DataFrame) -> str:
    return _json_hash({
        "config_hash": config_hash, "source_hash": source_hash,
        "upstream": upstream[["relative_path", "observed_sha256"]].to_dict(orient="records"),
        "checkpoints": checkpoints[["checkpoint_path", "checkpoint_sha256"]].to_dict(orient="records"),
    })


def _cell_payload(cells_by_policy: Sequence[Sequence[GridCell]]) -> list[list[dict[str, Any]]]:
    return [[{"open_weeks": c.open_weeks, "intensity": c.intensity, "duration_weeks": c.duration_weeks} for c in cells] for cells in cells_by_policy]


def _run_tasks(*, paths: Sequence[Any], model_config: Mapping[str, Any], specs: Sequence[Mapping[str, Any]], cells_by_policy: Sequence[Sequence[GridCell]], config: Mapping[str, Any], run_signature: str, workers: int) -> list[dict[str, Any]]:
    signature_cache = CACHE_DIR / run_signature
    signature_cache.mkdir(parents=True, exist_ok=True)
    tasks = [(path, policy_index) for path in paths for policy_index in range(len(specs))]
    expected_evaluations = len(paths) * sum(len(cells) for cells in cells_by_policy)
    print(f"[5.3.2] Layered evaluation: {len(paths)} paths, {len(tasks)} resumable tasks, {expected_evaluations} policy-seed-cell evaluations, {workers} workers", flush=True)
    results = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialise_worker,
        initargs=(dict(model_config), list(specs), _cell_payload(cells_by_policy), int(config["event_aligned_constructor"]["post_reclosure_recovery_weeks"]), str(signature_cache), run_signature),
    ) as executor:
        futures = {executor.submit(evaluate_path_policy_task, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 5 == 0 or completed == len(tasks):
                reused = sum(bool(item["reused"]) for item in results)
                print(f"[5.3.2] Completed resumable tasks {completed}/{len(tasks)} (cache reused {reused})", flush=True)
    return results


def _collect(task_results: Sequence[Mapping[str, Any]], allowed_paths: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_parts, contract_parts, partition_rows = [], [], []
    for item in task_results:
        raw = pd.read_csv(item["path_file"])
        raw_parts.append(raw.loc[raw["path_id"].isin(allowed_paths)])
        contracts = pd.read_csv(item["contract_file"])
        contract_parts.append(contracts.loc[contracts["path_id"].isin(allowed_paths)])
        weekly_path = Path(item["weekly_file"])
        marker = json.loads((weekly_path.parent / "complete.json").read_text(encoding="utf-8"))
        partition_rows.append({
            "partition_id": item["tag"], "source_path": str(weekly_path),
            "bytes": weekly_path.stat().st_size, "sha256": sha256_file(weekly_path),
            "weekly_rows": int(item["weekly_rows"]), "policy": marker["policy"],
            "training_seed": marker["training_seed"], "run_signature": marker["run_signature"],
            "policy_cell_count": len(marker["cell_ids"]), "cell_ids": "|".join(marker["cell_ids"]),
        })
    return pd.concat(raw_parts, ignore_index=True), pd.concat(contract_parts, ignore_index=True), pd.DataFrame(partition_rows).drop_duplicates("partition_id")


def _contract_gate(contracts: pd.DataFrame, raw: pd.DataFrame, expected_paths: int) -> dict[str, Any]:
    true_columns = [
        "all_step_acceptance_passed", "all_transition_audits_passed",
        "sue_residual_within_tolerance", "projection_feasible",
        "loss_components_reconstruct_total", "right_censoring_not_observed_clearance",
        "shared_prefix_execution", "frozen_checkpoint_or_rule",
        "provenance_shadow_conservation", "committed_mass_reconciliation",
    ]
    missing = [column for column in true_columns if column not in contracts]
    failed_rows = int((~contracts[true_columns].astype(bool).all(axis=1)).sum()) if not missing else len(contracts)
    false_contract_ok = bool((~contracts["branching_changes_scientific_logic"].astype(bool)).all() and (~contracts["future_information_used"].astype(bool)).all())
    result = {
        "status": "PASS" if not missing and failed_rows == 0 and false_contract_ok else "FAIL",
        "physical_paths": int(raw["path_id"].nunique()),
        "expected_physical_paths": expected_paths,
        "policy_seed_cell_rows": len(raw),
        "trajectory_contract_rows": len(contracts),
        "failed_contract_rows": failed_rows,
        "missing_contract_columns": missing,
        "branch_factorisation_is_numerically_neutral": false_contract_ok,
        "maximum_sue_residual": float(raw["maximum_sue_residual"].max()),
        "maximum_transition_residual": float(raw["maximum_transition_residual"].max()),
        "maximum_projection_violation": float(raw["maximum_projection_violation"].max()),
        "maximum_loss_reconciliation_error": float((raw["loss_component_sum_with_terminal"] - raw["total_operational_objective"]).abs().max()),
        "numerical_failure_rate": failed_rows / max(len(contracts), 1),
        "evidence_use": "computational gate only; not formal statistical evidence",
    }
    if result["physical_paths"] != expected_paths:
        result["status"] = "FAIL"
    return result


def _write_gate(name: str, payload: Mapping[str, Any]) -> Path:
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    path = GATE_DIR / name
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _run_gate(*, count: int, name: str, base: Mapping[str, Any], config: Mapping[str, Any], model_config: Mapping[str, Any], frozen: Any, specs: Sequence[Mapping[str, Any]], cells_by_policy: Sequence[Sequence[GridCell]], run_signature: str, workers: int) -> dict[str, Any]:
    started = time.perf_counter()
    paths = build_test_paths(config=base, frozen=frozen, count=max(88, count))[:count]
    results = _run_tasks(paths=paths, model_config=model_config, specs=specs, cells_by_policy=cells_by_policy, config=config, run_signature=run_signature, workers=workers)
    raw, contracts, _ = _collect(results, {path.path_id for path in paths})
    payload = _contract_gate(contracts, raw, count)
    payload.update({"gate": name, "elapsed_seconds": time.perf_counter() - started, "run_signature": run_signature})
    _write_gate(f"{name}.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    if payload["status"] != "PASS":
        raise RuntimeError(f"{name} failed")
    return payload


def _path_manifest(paths: Sequence[Any], cells: Sequence[GridCell]) -> pd.DataFrame:
    rows = []
    for base_path in paths:
        for cell in cells:
            path = build_cell_path(base_path, cell)
            rows.append({
                "cell_id": cell.cell_id, "open_interval_weeks": cell.open_weeks,
                "reclosure_intensity": cell.intensity, "reclosure_serviceability": cell.serviceability,
                "reclosure_duration_weeks": cell.duration_weeks, "path_id": base_path.path_id,
                "reclosure_path_id": path.path_id, "path_content_sha256": path.path_hash,
                "weeks": len(path.frame), "construction": path.construction,
                "historical_prefix_hash": base_path.path_hash,
                "historical_reclosure_duration_is_right_censored": True,
            })
    return pd.DataFrame(rows)


def _coverage_registry(config: Mapping[str, Any], certificate_cells: Sequence[GridCell], policy_cells: Sequence[GridCell], anchors: Sequence[GridCell]) -> pd.DataFrame:
    policies = list(config["main_policies"])
    axial = set(config["layered_policy_coverage"]["axial_policies"])
    policy_ids, anchor_ids = {cell.cell_id for cell in policy_cells}, {cell.cell_id for cell in anchors}
    reference_id = _cell(config["layered_policy_coverage"]["reference_cell"]).cell_id
    mild_id = _cell(config["layered_policy_coverage"]["mild_corner"]).cell_id
    severe_id = _cell(config["layered_policy_coverage"]["severe_corner"]).cell_id
    rows = []
    for cell in certificate_cells:
        for policy in policies:
            evaluated = cell.cell_id in policy_ids and (policy in axial or cell.cell_id in anchor_ids)
            rows.append({
                "cell_id": cell.cell_id, "open_interval_weeks": cell.open_weeks,
                "reclosure_intensity": cell.intensity, "reclosure_duration_weeks": cell.duration_weeks,
                "policy": policy, "policy_evaluated": evaluated,
                "evaluation_status": "EVALUATED" if evaluated else config["layered_policy_coverage"]["nonexecuted_status"],
                "comparison_family": "five_policy_anchor" if cell.cell_id in anchor_ids else "three_policy_axial_corner" if cell.cell_id in policy_ids else "physical_certificate_only",
                "is_reference_cell": cell.cell_id == reference_id, "is_mild_corner": cell.cell_id == mild_id,
                "is_severe_corner": cell.cell_id == severe_id,
                "full_150_cell_policy_leader_claim_permitted": False,
            })
    return pd.DataFrame(rows)


def _annotate_scope(frame: pd.DataFrame, anchor_ids: set[str]) -> pd.DataFrame:
    output = frame.copy()
    output["comparison_family"] = output["cell_id"].map(lambda value: "five_policy_anchor" if value in anchor_ids else "three_policy_axial_corner")
    output["full_five_policy_comparison"] = output["cell_id"].isin(anchor_ids)
    return output


def _safe_staging(final: Path) -> Path:
    staging = final.parent / f".{final.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def _publish(staging: Path, final: Path) -> None:
    if final.exists():
        previous = final.parent / f"{final.name}.previous_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        final.rename(previous)
    staging.rename(final)


def _status(run_signature: str, expected_paths: int = 88) -> int:
    complete = []
    signature_dir = CACHE_DIR / run_signature
    if signature_dir.exists():
        for marker in signature_dir.glob("*/complete.json"):
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("run_signature") == run_signature:
                complete.append(payload)
    tasks = len(complete)
    paths = len({str(item["tag"]).split("__policy_")[0] for item in complete})
    evaluations = sum(len(item.get("cell_ids", [])) for item in complete)
    expected_tasks = expected_paths * 9
    expected_evaluations = expected_paths * 92
    print(json.dumps({
        "run_signature": run_signature, "completed_resumable_tasks": tasks,
        "expected_tasks_at_88_paths": expected_tasks, "paths_with_any_completed_task": paths,
        "completed_policy_seed_cell_evaluations": evaluations,
        "expected_policy_seed_cell_evaluations_at_88_paths": expected_evaluations,
        "task_progress_percent": 100.0 * tasks / expected_tasks,
    }, indent=2), flush=True)
    return 0


def _formal(*, started: float, started_utc: str, config: Mapping[str, Any], base: Mapping[str, Any], config_hash: str, source_hash: str, upstream: pd.DataFrame, checkpoint_audit: pd.DataFrame, model_config: Mapping[str, Any], frozen: Any, specs: Sequence[Mapping[str, Any]], cells_by_policy: Sequence[Sequence[GridCell]], policy_cells: Sequence[GridCell], anchors: Sequence[GridCell], certificate_cells: Sequence[GridCell], run_signature: str, workers: int) -> int:
    for required in ("single_path_gate.json", "eight_path_computational_gate.json"):
        path = GATE_DIR / required
        if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("status") != "PASS" or json.loads(path.read_text(encoding="utf-8")).get("run_signature") != run_signature:
            raise RuntimeError(f"Formal execution requires the current PASS gate: {required}")
    minimum = int(config["path_design"]["minimum_common_physical_paths"])
    cap = int(config["path_design"]["maximum_physical_paths"])
    paths_88 = build_test_paths(config=base, frozen=frozen, count=minimum)
    _assert_first_88(paths_88)
    results_88 = _run_tasks(paths=paths_88, model_config=model_config, specs=specs, cells_by_policy=cells_by_policy, config=config, run_signature=run_signature, workers=workers)
    raw_88, contracts_88, partitions_88 = _collect(results_88, {path.path_id for path in paths_88})
    anchor_ids = {cell.cell_id for cell in anchors}
    raw_88, contracts_88 = _annotate_scope(raw_88, anchor_ids), _annotate_scope(contracts_88, anchor_ids)
    path_level_88 = _annotate_scope(aggregate_learning_seeds(raw_88), anchor_ids)
    requirements, selected = precision_requirements(path_level_88, config, config["main_policies"])
    print(f"[5.3.2] Anchor precision rule selected {selected} physical paths (cap {cap})", flush=True)
    if selected > minimum:
        all_paths = build_test_paths(config=base, frozen=frozen, count=selected)
        _assert_first_88(all_paths)
        all_results = _run_tasks(paths=all_paths, model_config=model_config, specs=specs, cells_by_policy=cells_by_policy, config=config, run_signature=run_signature, workers=workers)
        raw, contracts, partitions = _collect(all_results, {path.path_id for path in all_paths})
        raw, contracts = _annotate_scope(raw, anchor_ids), _annotate_scope(contracts, anchor_ids)
    else:
        all_paths, raw, contracts, partitions = paths_88, raw_88, contracts_88, partitions_88
    path_level = _annotate_scope(aggregate_learning_seeds(raw), anchor_ids)
    precision = update_precision(requirements, path_level, len(all_paths))
    paired = paired_effects(path_level, config["main_policies"], float(config["path_design"]["confidence_level"]))
    confidence, regret = confidence_sets_and_regret(path_level, config["main_policies"], float(config["path_design"]["confidence_level"]))
    mechanism, clearance = summaries(path_level, int(base["clearance"]["maximum_weeks"]))
    absorption_path, absorption_summary, absorption_envelope = absorption_certificate(
        model=build_model(model_config), base_paths=all_paths, cells=certificate_cells,
        tolerance=float(base["numerics"]["mass_tolerance"]),
    )
    path_manifest = _path_manifest(all_paths, policy_cells)
    coverage = _coverage_registry(config, certificate_cells, policy_cells, anchors)
    pairing = raw.groupby(["cell_id", "path_id"]).agg(
        policy_seed_rows=("policy", "size"), policy_count=("policy", "nunique"),
        exogenous_hash_count=("path_content_sha256", "nunique"),
    ).reset_index()
    pairing["expected_policy_count"] = pairing["cell_id"].isin(anchor_ids).map({True: 5, False: 3})
    pairing["expected_policy_seed_rows"] = pairing["cell_id"].isin(anchor_ids).map({True: 9, False: 5})
    pairing["matched_across_preregistered_policy_family"] = (
        (pairing["policy_count"] == pairing["expected_policy_count"])
        & (pairing["policy_seed_rows"] == pairing["expected_policy_seed_rows"])
        & (pairing["exogenous_hash_count"] == 1)
    )
    independent = independent_recalculation(path_level, paired, confidence)

    output_final = CODE_ROOT / config["output_directory"]
    staging = _safe_staging(output_final)
    (staging / "frozen_config_5_3_2.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(upstream, staging / "upstream_input_locks.csv")
    write_csv(checkpoint_audit, staging / "checkpoint_manifest_5_3_2.csv")
    write_csv(coverage, staging / "cell_policy_coverage_registry.csv")
    write_csv(path_manifest, staging / "reclosure_policy_path_manifest.csv")
    write_csv(pairing, staging / "path_pairing_audit.csv")
    write_csv(raw, staging / "raw_policy_path_seed_results.csv")
    write_csv(raw, staging / "path_level_results.csv")
    write_csv(path_level, staging / "path_level_seed_aggregated.csv")
    write_csv(contracts, staging / "trajectory_contract_checks.csv")
    write_csv(requirements, staging / "anchor_precision_requirements_initial.csv")
    write_csv(precision, staging / "anchor_precision_requirements.csv")
    write_csv(pd.DataFrame([{
        "selected_physical_paths": len(all_paths), "minimum": minimum, "cap": cap,
        "maximum_required": int(requirements["required_paths"].max()),
        "precision_targets_met": int(precision["precision_target_met"].sum()),
        "precision_contrasts": len(precision),
    }]), staging / "selected_path_count.csv")
    write_csv(paired, staging / "paired_effects.csv")
    write_csv(confidence, staging / "policy_confidence_set.csv")
    write_csv(regret, staging / "policy_regret.csv")
    write_csv(mechanism, staging / "mechanism_summary.csv")
    write_csv(clearance, staging / "clearance_and_censoring.csv")
    write_csv(absorption_path, staging / "absorption_certificate_path_results.csv")
    write_csv(absorption_summary, staging / "absorption_certificate_summary.csv")
    write_csv(absorption_envelope, staging / "absorption_capacity_envelope.csv")
    write_csv(parameter_registry(config, base), staging / "parameter_registry_5_3_2.csv")
    write_csv(formula_registry(), staging / "formula_to_code_5_3_2.csv")
    write_csv(independent, staging / "independent_recalculation_checks.csv")
    for gate_name in ("single_path_gate.json", "eight_path_computational_gate.json"):
        shutil.copy2(GATE_DIR / gate_name, staging / gate_name)

    weekly_dir = staging / "weekly_reclosure_trajectories"
    weekly_dir.mkdir()
    published_partitions = []
    for row in partitions.itertuples(index=False):
        target = weekly_dir / f"{row.partition_id}.csv.gz"
        shutil.copy2(row.source_path, target)
        published_partitions.append({
            "partition_id": row.partition_id, "relative_path": target.relative_to(staging).as_posix(),
            "bytes": target.stat().st_size, "sha256": sha256_file(target),
            "weekly_rows": row.weekly_rows, "policy": row.policy, "training_seed": row.training_seed,
            "policy_cell_count": row.policy_cell_count, "cell_ids": row.cell_ids,
            "run_signature": row.run_signature,
        })
    partition_manifest = pd.DataFrame(published_partitions)
    write_csv(partition_manifest, staging / "weekly_trajectory_partitions.csv")

    figures = create_figures(
        path_level=path_level, confidence=confidence, regret=regret, paired=paired,
        mechanism=mechanism, clearance=clearance, absorption=absorption_summary,
        coverage=coverage, figures_dir=EXPERIMENT_DIR / "figures", output_dir=staging,
        dpi=int(config["execution"]["figure_dpi"]), historical_marker=config["historical_marker"],
        config=config,
    )
    for path in figures.values():
        shutil.copy2(path, staging / path.name)
    acceptance = acceptance_payload(
        config=config, path_level=path_level, raw=raw, contracts=contracts,
        confidence=confidence, precision=precision, absorption=absorption_summary,
        upstream=upstream, figures=figures, independent_checks=independent,
        coverage=coverage, pairing=pairing,
    )
    (staging / "acceptance_5_3_2.json").write_text(json.dumps(acceptance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_reports(
        report_dir=PROJECT_ROOT / "report - 8.4" / "5.3.2", acceptance=acceptance,
        confidence=confidence, paired=paired, clearance=clearance,
        absorption=absorption_summary, precision=precision, config=config,
    )
    elapsed = time.perf_counter() - started
    manifest = write_manifest(
        output_dir=staging, config_hash=config_hash, source_hash=source_hash,
        upstream=upstream, started_utc=started_utc, elapsed_seconds=elapsed,
        figures=figures, weekly_partitions=partition_manifest,
    )
    (staging / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _publish(staging, output_final)
    print(json.dumps({
        "status": acceptance["run_status"],
        "overall_evidence_acceptance": acceptance["overall_evidence_acceptance"],
        "elapsed_seconds": elapsed, "certificate_grid_cells": 150,
        "policy_cells": 16, "full_policy_anchors": 3, "physical_paths": len(all_paths),
        "policy_path_seed_cell_results": len(raw),
        "precision_targets_met": acceptance["precision_targets_met"],
        "precision_contrasts": acceptance["precision_contrasts"],
        "manifest": str(output_final / "run_manifest.json"),
    }, indent=2), flush=True)
    return 0 if acceptance["run_status"] == "complete" else 2


def run(phase: str, workers_override: int | None = None) -> int:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    config, base, config_hash = _load_configs()
    upstream = _verify_upstream(config)
    source_hash = _source_hash()
    simulation_source_hash = _source_hash(SIMULATION_SOURCE_FILES)
    specs, checkpoint_audit = _checkpoint_specs()
    signature = _run_signature(config_hash, simulation_source_hash, upstream, checkpoint_audit)
    print(f"[5.3.2] Run signature: {signature}", flush=True)
    if phase == "status":
        return _status(signature)
    certificate_cells = _certificate_cells(config)
    policy_cells, anchors = _policy_cells(config)
    cells_by_policy = _cells_by_spec(specs, policy_cells, anchors)
    model_config = _model_config(base, config)
    frozen = load_frozen_5_2_1_inputs(base)
    workers = int(workers_override or config["execution"]["parallel_workers"])
    if workers < 1:
        raise ValueError("workers must be positive")
    if phase in {"all", "gate1"}:
        _run_gate(count=1, name="single_path_gate", base=base, config=config, model_config=model_config, frozen=frozen, specs=specs, cells_by_policy=cells_by_policy, run_signature=signature, workers=min(workers, 9))
        if phase == "gate1":
            return 0
    if phase in {"all", "gate8"}:
        _run_gate(count=8, name="eight_path_computational_gate", base=base, config=config, model_config=model_config, frozen=frozen, specs=specs, cells_by_policy=cells_by_policy, run_signature=signature, workers=workers)
        if phase == "gate8":
            return 0
    if phase in {"all", "formal"}:
        return _formal(
            started=started, started_utc=started_utc, config=config, base=base,
            config_hash=config_hash, source_hash=source_hash, upstream=upstream,
            checkpoint_audit=checkpoint_audit, model_config=model_config, frozen=frozen,
            specs=specs, cells_by_policy=cells_by_policy, policy_cells=policy_cells,
            anchors=anchors, certificate_cells=certificate_cells,
            run_signature=signature, workers=workers,
        )
    raise ValueError(phase)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "gate1", "gate8", "formal", "status"), default="all")
    parser.add_argument("--workers", type=int, default=None, help="Override the frozen parallel worker count without changing scientific parameters")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(run(args.phase, args.workers))
