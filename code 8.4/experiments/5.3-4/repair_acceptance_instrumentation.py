"""Targeted repair audit for the 5.3.4 source-mass acceptance instrument.

This script never reruns the experimental grid.  It replays only the six
previously flagged trajectories, requires scientific numerical equivalence,
then republishes acceptance metadata and hashes.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_DIR.parents[1]
PROJECT_ROOT = CODE_ROOT.parent
for entry in (
    EXPERIMENT_DIR,
    CODE_ROOT / "experiments" / "5.2-2",
    CODE_ROOT / "experiments" / "5.2-3",
    CODE_ROOT / "experiments" / "5.3-3",
    CODE_ROOT / "src",
):
    sys.path.insert(0, str(entry))

import run_5_3_4 as runner  # noqa: E402
from model import build_model  # noqa: E402
from paths import build_test_paths, load_frozen_5_2_1_inputs  # noqa: E402
from reporting_5_3_4 import (  # noqa: E402
    acceptance_payload,
    sha256_file,
    write_manifest,
    write_reports,
)
from robustness_5_3_4 import build_cells, model_config, transform_paths  # noqa: E402
from robustness_worker import evaluate_task, initialise_worker  # noqa: E402


KEYS = ("path_id", "policy", "training_seed")
TARGETS = (
    ("test_000_4ae657b9eeec", "Model-guided constrained SAC", 2635471077),
    ("test_003_eb3106001035", "Behaviour cloning", 540181960),
    ("test_005_981c73a17708", "Model-guided constrained SAC", 2029871371),
    ("test_005_981c73a17708", "Model-guided constrained SAC", 2255582880),
    ("test_005_981c73a17708", "Model-guided constrained SAC", 2635471077),
    ("test_006_daba4b1e9814", "Model-guided constrained SAC", 2635471077),
)
SCIENTIFIC_NUMERIC = (
    "decision_operational_loss",
    "clearance_operational_loss",
    "terminal_correction",
    "total_operational_objective",
    "loss_queue",
    "loss_waiting",
    "loss_exit",
    "loss_overflow",
    "loss_route_resource",
    "loss_action",
    "ending_outstanding_mass",
    "maximum_transition_residual",
    "loss_component_sum_with_terminal",
)


def _key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["path_id"].astype(str)
        + "|"
        + frame["policy"].astype(str)
        + "|"
        + frame["training_seed"].fillna(-1).astype(float).astype(np.int64).astype(str)
    )


def main() -> int:
    experiment, base, config_hash = runner._load()
    cells, _ = build_cells(experiment)
    cell = next(
        item
        for item in cells
        if item.cell_id == "interaction__long_lag__severe_reclosure"
    )
    config = model_config(base, experiment, cell)
    model = build_model(config)
    frozen = load_frozen_5_2_1_inputs(base)
    paths = transform_paths(
        build_test_paths(config=base, frozen=frozen, count=10), cell, experiment
    )
    path_lookup = {path.path_id: path for path in paths}
    bundle, _ = runner._checkpoint_bundle(
        manifest_path=EXPERIMENT_DIR / "checkpoints" / cell.cell_id / "checkpoint_manifest.csv",
        checkpoint_base=CODE_ROOT,
        source_experiment="5.3.4 targeted acceptance repair audit",
    )
    policies = runner._policies(
        model,
        bundle,
        ["Behaviour cloning", "Model-guided constrained SAC"],
    )
    policy_lookup = {
        (policy.name, int(policy.training_seed)): index
        for index, policy in enumerate(policies)
    }
    specs = [runner._spec(policy) for policy in policies]
    tasks = [
        (path_lookup[path_id], policy_lookup[(policy, seed)])
        for path_id, policy, seed in TARGETS
    ]
    with ProcessPoolExecutor(
        max_workers=6,
        initializer=initialise_worker,
        initargs=(config, specs, runner._cell_dict(cell)),
    ) as executor:
        artifacts = list(executor.map(evaluate_task, tasks, chunksize=1))

    new_rep = pd.DataFrame([artifact.replication for artifact in artifacts])
    new_con = pd.DataFrame([artifact.contract for artifact in artifacts])
    if len(new_rep) != 6 or not new_rep["all_step_acceptance_passed"].astype(bool).all():
        raise RuntimeError("The six targeted local acceptance replays did not all pass")
    if not new_rep["accepted"].astype(bool).all():
        raise RuntimeError("A targeted complete trajectory acceptance failed")

    output = CODE_ROOT / experiment["output_directory"]
    old_rep = pd.read_csv(output / "path_level_policy_seed_results.csv")
    old_con = pd.read_csv(output / "trajectory_contract_checks.csv")
    old_bad = old_rep.loc[~old_rep["all_step_acceptance_passed"].astype(bool)].copy()
    if set(_key(old_bad)) != set(_key(new_rep)):
        raise RuntimeError("Targeted repair keys differ from the six frozen failures")
    paired = old_bad.merge(new_rep, on=list(KEYS), suffixes=("_old", "_new"))
    audit_rows = []
    tolerance = float(base["numerics"]["mass_tolerance"])
    for row in paired.itertuples(index=False):
        differences = {
            name: abs(float(getattr(row, name + "_old")) - float(getattr(row, name + "_new")))
            for name in SCIENTIFIC_NUMERIC
        }
        maximum = max(differences.values())
        audit_rows.append(
            {
                "path_id": row.path_id,
                "policy": row.policy,
                "training_seed": int(row.training_seed),
                "old_local_acceptance": bool(row.all_step_acceptance_passed_old),
                "old_complete_trajectory_acceptance": bool(row.accepted_old),
                "new_local_acceptance": bool(row.all_step_acceptance_passed_new),
                "new_complete_trajectory_acceptance": bool(row.accepted_new),
                "maximum_scientific_numeric_difference": maximum,
                "equivalence_tolerance": tolerance,
                "numerically_equivalent": maximum <= tolerance,
                "repair_scope": "acceptance instrumentation only; no action, transition, loss, or terminal-state change",
            }
        )
    audit = pd.DataFrame(audit_rows)
    if not audit["numerically_equivalent"].astype(bool).all():
        raise RuntimeError("Targeted repair changed a scientific numeric result")

    target_keys = set(_key(new_rep))
    old_rep.loc[_key(old_rep).isin(target_keys), "all_step_acceptance_passed"] = True
    old_con.loc[_key(old_con).isin(target_keys), "all_step_acceptance_passed"] = True
    old_rep.to_csv(output / "path_level_policy_seed_results.csv", index=False, lineterminator="\n")
    old_con.to_csv(output / "trajectory_contract_checks.csv", index=False, lineterminator="\n")
    audit.to_csv(output / "acceptance_instrumentation_repair.csv", index=False, lineterminator="\n")

    # Keep the resumable severe-cell cache consistent with the republished
    # instrumentation flag; scientific quantities remain byte-for-byte values
    # from the original formal run.
    old_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    old_signature = str(old_manifest["source_bundle_sha256"])
    cache_candidates = list((EXPERIMENT_DIR / "cache" / "formal").glob("*/" + cell.cell_id))
    if len(cache_candidates) != 1:
        raise RuntimeError("Expected exactly one formal severe-cell cache")
    cache = cache_candidates[0]
    cache_rep = pd.read_csv(cache / "replications.csv.gz")
    cache_con = pd.read_csv(cache / "contracts.csv.gz")
    cache_rep.loc[_key(cache_rep).isin(target_keys), "all_step_acceptance_passed"] = True
    cache_con.loc[_key(cache_con).isin(target_keys), "all_step_acceptance_passed"] = True
    cache_rep.to_csv(cache / "replications.csv.gz", index=False, compression="gzip")
    cache_con.to_csv(cache / "contracts.csv.gz", index=False, compression="gzip")

    upstream = pd.read_csv(output / "upstream_input_locks.csv")
    registry = pd.read_csv(output / "cell_policy_coverage_registry.csv")
    effects = pd.read_csv(output / "paired_parameter_effects.csv")
    diagnostics = pd.read_csv(output / "clearance_tolerance_diagnostic.csv")
    independent = pd.read_csv(output / "independent_recalculation_checks.csv")
    path_level = pd.read_csv(output / "path_level_seed_aggregated.csv")
    figures = {
        "figure_a": output / "figure_5_3_4a_parameter_effect_forest.png",
        "figure_b": output / "figure_5_3_4b_policy_stability.png",
        "figure_c": output / "figure_5_3_4c_interaction_mechanisms.png",
    }
    acceptance = acceptance_payload(
        upstream=upstream,
        replications=old_rep,
        path_level=path_level,
        contracts=old_con,
        registry=registry,
        effects=effects,
        diagnostics=diagnostics,
        independent=independent,
        figures=figures,
        expected_paths=10,
        target_halfwidth=float(experiment["path_design"]["target_halfwidth"]),
        tolerance=tolerance,
    )
    gradient = pd.read_csv(output / "matched_sac_actor_gradient_check.csv")
    jacobian = pd.read_csv(output / "matched_projection_jacobian_check.csv")
    runtime = pd.read_csv(output / "runtime_summary.csv")
    acceptance["checks"]["matched_sac_gradient_check_pass"] = bool(gradient["passed"].astype(bool).all())
    acceptance["checks"]["matched_projection_jacobian_check_pass"] = bool(jacobian["status"].eq("PASS").all())
    acceptance["checks"]["eight_hour_wall_clock_respected_before_publication"] = bool(runtime.iloc[0]["eight_hour_wall_clock_respected"])
    acceptance["checks"]["targeted_acceptance_instrumentation_repair_pass"] = bool(audit["numerically_equivalent"].all())
    if not all(acceptance["checks"].values()):
        acceptance["run_status"] = "failed"
        acceptance["ENGINEERING_ACCEPTANCE"] = "FAIL"
        acceptance["OVERALL_EVIDENCE_ACCEPTANCE"] = "FAIL"
    (output / "acceptance_5_3_4.json").write_text(
        json.dumps(acceptance, indent=2) + "\n", encoding="utf-8"
    )
    report_directory = PROJECT_ROOT / "report - 8.4" / "5.3.4"
    write_reports(
        report_directory,
        acceptance,
        pd.read_csv(output / "policy_summary.csv"),
        effects,
        pd.read_csv(output / "policy_confidence_set.csv"),
        diagnostics,
        runtime,
    )
    (report_directory / "ACCEPTANCE_INSTRUMENTATION_REPAIR.md").write_text(
        "# Acceptance Instrumentation Repair\n\n"
        "Six trajectories at the long-lag severe-reclosure anchor were flagged only by a duplicated local period check. "
        "That check reconstructed source mass from alternative-level flow sums, which erased a strictly positive subnormal released vintage. "
        "The Chapter 4 complete-trajectory acceptance already used the demand-split and release ledger and passed all 1,610 trajectories. "
        "After routing the local check through the same formal ledger, all six targeted replays passed and every registered scientific numeric quantity was unchanged within the model tolerance.\n",
        encoding="utf-8",
    )
    source_hash = runner._source_hash()
    write_manifest(
        output / "run_manifest.json",
        config_hash=config_hash,
        source_hash=source_hash,
        upstream=upstream,
        output_directory=output,
        figures=figures,
        started_utc=str(old_manifest["started_utc"]),
        elapsed_seconds=float(old_manifest["elapsed_seconds"]),
        formal_paths=int(old_manifest["formal_paths"]),
    )
    status = {
        "status": "FINISHED_5_3_4_AFTER_ACCEPTANCE_INSTRUMENTATION_REPAIR",
        "message": "Formal grid complete; six targeted underflow-certificate replays passed without scientific numeric changes.",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "formal_paths": 10,
        "formal_cells": 31,
        "full_grid_rerun": False,
        "overall_evidence_acceptance": acceptance["OVERALL_EVIDENCE_ACCEPTANCE"],
    }
    (EXPERIMENT_DIR / "logs" / "continuation_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"acceptance": acceptance, "repair_rows": len(audit)}, indent=2))
    return 0 if acceptance["OVERALL_EVIDENCE_ACCEPTANCE"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
