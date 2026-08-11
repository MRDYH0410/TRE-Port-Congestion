"""Single-command runner for Experiment 5.2.5."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from acceptance_core import (
    CODE_ROOT,
    EXPERIMENT_DIR,
    acceptance_summary,
    diagnostic_contract_rows,
    hash_json,
    load_inputs,
    loss_reconciliation_audit,
    matched_scenario_audit,
    method_contract_registry,
    mpc_audit,
    nonanticipativity_audit,
    parameter_registry,
    projection_audit,
    rcmsa_audit,
    release_information_audit,
    runtime_profile,
    selector_audit,
    sha256_file,
    tagged_mass_audit,
    training_audit,
    travel_lag_audit,
    upstream_acceptance_audit,
)
from extended_audits import (
    capacity_timing_audit,
    chapter4_contract_reinforcement_audit,
    clearance_terminal_audit,
    core_repair_numerical_equivalence_audit,
    experimental_precision_audit,
    rcmsa_start_certification_audit,
    unavailable_route_hold_audit,
    upstream_anchor_replay_audit,
    verify_upstream_locks,
)
from sac_contracts import sac_learning_contract_audit
from reporting_5_2_5 import figure_a, figure_b, figure_c, write_reports


def _write(frame: pd.DataFrame, output: Path, name: str) -> None:
    frame.to_csv(output / name, index=False)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "NOT_AVAILABLE"


def main() -> int:
    started = time.time()
    config_path = EXPERIMENT_DIR / "config_5_2_5.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = CODE_ROOT / config["output_directory"]
    figures = CODE_ROOT / config["figure_directory"]
    report_dir = CODE_ROOT / config["report_directory"]
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    upstream, config_522, model, event = load_inputs(config)
    lock_audit = verify_upstream_locks(config, upstream, sha256_file)
    if not lock_audit["status"].eq("PASS").all():
        output.mkdir(parents=True, exist_ok=True)
        _write(lock_audit, output, "upstream_lock_audit.csv")
        raise RuntimeError("5.2.5 blocked: an authorized upstream SHA256 lock does not match")
    mass_tol = float(config_522["numerics"]["mass_tolerance"])
    loss_tol = float(config_522["numerics"]["loss_identity_tolerance"])

    print("[5.2.5] Auditing released information, matched paths, and projection...")
    upstream_trace = upstream_acceptance_audit(upstream, lock_audit)
    release, cross = release_information_audit(upstream, model)
    scenarios = matched_scenario_audit(upstream)
    projection, kkt = projection_audit(model, event, upstream)

    print("[5.2.5] Running controlled production RC-MSA and nested MPC cases...")
    rc_trace, rc_summary = rcmsa_audit(model, event, int(config["controlled_rcmsa_cases"]))
    rc_start = rcmsa_start_certification_audit(model, event, int(config["controlled_rcmsa_cases"]))
    mpc_rollouts, mpc_recalc, mpc_exact, mpc_precision = mpc_audit(model, event)
    nonant = nonanticipativity_audit(model, event, upstream)

    print("[5.2.5] Recalculating tag, lag, capacity, loss, selector, and replay contracts...")
    tagged = tagged_mass_audit(upstream, mass_tol)
    travel = travel_lag_audit(model, event)
    capacity = capacity_timing_audit(model, event)
    unavailable = unavailable_route_hold_audit(model, event)
    loss = loss_reconciliation_audit(upstream, loss_tol)
    selector, regret = selector_audit(upstream)
    bc, bc_summary, sac, sac_summary = training_audit(upstream)
    validation = pd.read_csv(upstream["5.2.2"] / "validation_curves.csv")
    reproducibility = upstream_anchor_replay_audit(
        upstream=upstream, config=config_522, tolerance=loss_tol
    )
    clearance = clearance_terminal_audit(upstream, config_522, loss_tol)
    precision, precision_summary = experimental_precision_audit(
        upstream=upstream, config=config_522
    )
    sac_outputs = sac_learning_contract_audit(
        model=model, upstream=upstream, config=config_522
    )
    reinforcement = chapter4_contract_reinforcement_audit(model, event)
    numerical_equivalence = core_repair_numerical_equivalence_audit(
        rc_summary=rc_summary,
        mpc_recalculation=mpc_recalc,
        precision_summary=precision_summary,
        reproducibility=reproducibility,
        baseline_path=EXPERIMENT_DIR / "pre_repair_numerical_baseline_5_2_5.json",
    )
    runtime = runtime_profile(upstream, rc_summary, mpc_rollouts)
    parameters = parameter_registry(config_522, config, model)

    registry = method_contract_registry(
        config=config, model=model, lock_audit=lock_audit,
        release=release, cross=cross, scenarios=scenarios,
        projection=projection, rc_summary=rc_summary, rc_start=rc_start,
        tagged=tagged, loss=loss,
        nonant=nonant, mpc_recalc=mpc_recalc, mpc_exact=mpc_exact,
        selector=selector, travel=travel, capacity=capacity,
        reproducibility=reproducibility, unavailable=unavailable,
        clearance=clearance, sac_contracts=sac_outputs["contract_status"],
        reinforcement=reinforcement,
        numerical_equivalence=numerical_equivalence,
    )
    registry = pd.concat([registry, diagnostic_contract_rows(model)], ignore_index=True, sort=False)
    summary = acceptance_summary(registry, lock_audit, precision_summary)

    acceptance_cases = pd.concat([
        upstream_trace,
        lock_audit.rename(columns={"experiment": "module"}).assign(trace_type="upstream_hash_lock"),
        scenarios.rename(columns={"audit": "trace_type", "maximum_residual": "observed_residual"}),
        pd.DataFrame([
            {"trace_type": "controlled_projection", "module": "M3", "status": "PASS" if projection["status"].eq("PASS").all() else "FAIL", "detail": f"{len(projection)} controlled and production projection rows"},
            {"trace_type": "controlled_rcmsa", "module": "M4", "status": "PASS" if rc_summary.loc[rc_summary['algorithm'].eq('RC-MSA'),'converged'].all() else "FAIL", "detail": f"{config['controlled_rcmsa_cases']} production lower-level problems"},
            {"trace_type": "rcmsa_start_and_final_trial", "module": "M4", "status": "PASS" if rc_start["status"].eq("PASS").all() else "FAIL", "detail": "true zero-loading start, actual start provenance and final generated trial certification"},
            {"trace_type": "controlled_mpc", "module": "M8", "status": str(mpc_exact["status"].iloc[0]), "detail": "exhaustive registered candidate lattice"},
            {"trace_type": "travel_impulse", "module": "M10", "status": "PASS" if travel["status"].eq("PASS").all() else "FAIL", "detail": "one-unit pulse on every declared route"},
            {"trace_type": "capacity_impulse", "module": "M11", "status": "PASS" if capacity["status"].eq("PASS").all() else "FAIL", "detail": "one-unit readiness and direct order through production CapacityDynamics"},
            {"trace_type": "unavailable_route_hold", "module": "M25", "status": "PASS" if unavailable["status"].eq("PASS").all() else "FAIL", "detail": "existing route-tagged mass remains in place with zero service"},
            {"trace_type": "clearance_terminal", "module": "M26", "status": "PASS" if clearance["status"].eq("PASS").all() else "FAIL", "detail": "right censoring, two exit channels and exactly one terminal correction"},
            {"trace_type": "chapter4_contract_reinforcement", "module": "M27--M30", "status": "PASS" if reinforcement["status"].eq("PASS").all() else "FAIL", "detail": "master-choice history distance, a_t^{-I}, per-vintage no-reset and complete MPC/selector certificates"},
            {"trace_type": "core_repair_numerical_equivalence", "module": "M31", "status": "PASS" if numerical_equivalence["status"].eq("PASS").all() else "FAIL", "detail": "pre/post repair fixed-point, action, nested objective, upstream-anchor and precision equivalence"},
        ])
    ], ignore_index=True, sort=False)

    _write(registry, output, "method_contract_registry.csv")
    _write(lock_audit, output, "upstream_lock_audit.csv")
    _write(registry[["methodological_claim", "equation_label", "implementation_file", "implementation_function", "output_file", "tolerance", "status", "failure_reason"]], output, "methodological_contract_table.csv")
    _write(acceptance_cases, output, "acceptance_case_registry.csv")
    _write(pd.concat([upstream_trace, cross], ignore_index=True, sort=False), output, "cross_module_trace.csv")
    _write(release, output, "release_nonanticipativity.csv")
    _write(nonant, output, "policy_nonanticipativity.csv")
    _write(projection, output, "projection_feasibility.csv")
    _write(kkt, output, "projection_kkt_trace.csv")
    _write(rc_trace, output, "rcmsa_iteration_trace.csv")
    _write(rc_summary, output, "rcmsa_comparison_summary.csv")
    _write(rc_start, output, "rcmsa_start_certification.csv")
    _write(mpc_rollouts, output, "mpc_candidate_rollouts.csv")
    _write(mpc_recalc, output, "mpc_objective_recalculation.csv")
    _write(mpc_exact, output, "mpc_exact_case_check.csv")
    _write(mpc_precision, output, "mpc_scenario_precision.csv")
    _write(bc, output, "bc_training_trace.csv")
    _write(bc_summary, output, "bc_validation_summary.csv")
    _write(sac_outputs["training_trace"], output, "sac_training_trace.csv")
    _write(sac_summary, output, "sac_validation_summary.csv")
    _write(sac_outputs["update_recalculation"], output, "sac_update_recalculation.csv")
    _write(sac_outputs["episode_replay_summary"], output, "sac_episode_replay_summary.csv")
    _write(sac_outputs["gradient_recalculation"], output, "sac_actor_gradient_recalculation.csv")
    _write(sac_outputs["projection_jacobian"], output, "sac_projection_jacobian.csv")
    _write(sac_outputs["checkpoint_replay"], output, "sac_checkpoint_replay.csv")
    _write(sac_outputs["contract_status"], output, "sac_contract_status.csv")
    _write(selector, output, "selector_decision_trace.csv")
    _write(regret, output, "selector_regret.csv")
    _write(tagged, output, "tagged_mass_balance.csv")
    _write(travel, output, "travel_lag_acceptance.csv")
    _write(capacity, output, "capacity_pipeline_acceptance.csv")
    _write(unavailable, output, "unavailable_route_acceptance.csv")
    _write(clearance, output, "clearance_terminal_acceptance.csv")
    _write(reinforcement, output, "chapter4_contract_reinforcement.csv")
    _write(numerical_equivalence, output, "core_repair_numerical_equivalence.csv")
    _write(loss, output, "loss_reconciliation.csv")
    _write(reproducibility, output, "reproducibility_audit.csv")
    _write(precision, output, "experimental_precision_recalculation.csv")
    _write(precision_summary, output, "experimental_precision_summary.csv")
    _write(runtime, output, "runtime_profile.csv")
    _write(runtime, output, "computational_profile_table.csv")
    _write(parameters, output, "parameter_registry_5_2_5.csv")

    print("[5.2.5] Rendering 300 dpi PNG and vector PDF figures...")
    figure_files = []
    figure_files += figure_a(rc_trace, rc_summary, mpc_precision, output, figures)
    figure_files += figure_b(
        bc, validation, sac_outputs["training_trace"],
        sac_outputs["gradient_recalculation"], selector, regret, output, figures
    )
    figure_files += figure_c(registry, reproducibility, runtime, output, figures)

    summary["generated_at_utc"] = pd.Timestamp.now("UTC").isoformat()
    summary["figure_files"] = figure_files
    acceptance_text = json.dumps(summary, indent=2, ensure_ascii=False)
    (output / "acceptance_summary.json").write_text(acceptance_text, encoding="utf-8")
    (output / "acceptance_5_2_5.json").write_text(acceptance_text, encoding="utf-8")
    claim_scope = {
        "supported": ["production method connectivity", "registered numerical tolerances", "matched replay reproducibility", "computational profile"],
        "not_supported": ["universal policy superiority", "global optimality", "causal real-port effect", "real-time deployability"],
        "evidence_status": summary["OVERALL_ACCEPTANCE"],
        "binding_limitation": "; ".join(summary["failure_reasons"]) if summary["failure_reasons"] else "none",
    }
    (output / "claim_scope.json").write_text(json.dumps(claim_scope, indent=2, ensure_ascii=False), encoding="utf-8")
    write_reports(
        report_dir, summary, registry, parameters, runtime,
        sac_outputs["contract_status"], reproducibility, precision_summary,
        rc_summary, regret,
    )

    generated = sorted(
        [path for path in output.iterdir() if path.is_file() and path.name != "run_manifest.json"]
        + [path for path in figures.iterdir() if path.is_file()]
        + [path for path in report_dir.iterdir() if path.is_file()]
        + [EXPERIMENT_DIR / "FORMULA_TO_CODE.md", EXPERIMENT_DIR / "README.md"]
    )
    production_sources = [
        CODE_ROOT / "src" / "tre84" / name
        for name in (
            "acceptance.py", "actions.py", "behavior.py", "capacity.py",
            "control.py", "engine.py", "factory.py", "loss.py", "transition.py",
        )
    ] + [
        CODE_ROOT / "experiments" / "5.2-2" / name
        for name in ("features.py", "model.py", "paths.py", "policies.py", "simulator.py", "training.py")
    ] + [
        EXPERIMENT_DIR / name
        for name in (
            "acceptance_core.py", "extended_audits.py", "sac_contracts.py",
            "reporting_5_2_5.py", "run_5_2_5.py", "config_5_2_5.json",
            "pre_repair_numerical_baseline_5_2_5.json",
        )
    ]
    source_hashes = {str(path.relative_to(CODE_ROOT)).replace("\\", "/"): sha256_file(path) for path in production_sources}
    manifest = {
        "experiment_id": config["experiment_id"],
        "schema_version": config["schema_version"],
        "command": "python experiments/5.2-5/run_5_2_5.py",
        "config_path": config_path.relative_to(CODE_ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "code_commit": _git_commit(),
        "production_source_hashes": source_hashes,
        "production_source_bundle_sha256": hash_json(source_hashes),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "elapsed_seconds": time.time() - started,
        "upstream_manifests": {key: {"path": str((path / "run_manifest.json").relative_to(CODE_ROOT)).replace("\\", "/"), "sha256": sha256_file(path / "run_manifest.json")} for key, path in upstream.items()},
        "outputs": [{"path": str(path.relative_to(CODE_ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in generated],
        "acceptance": {key: value for key, value in summary.items() if key.endswith("ACCEPTANCE")},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["acceptance"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
