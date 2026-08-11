"""Finalize a fully evaluated 5.2.2 staging run after postprocessing repair.

This command is intentionally fail-closed: it accepts only the staging tree
created by the current config and authorised 5.2.1 input.  It never reads the
previous formal 5.2.2 output or any checkpoint outside that staging tree.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_DIR.parents[1]
sys.path[:0] = [str(EXPERIMENT_DIR), str(CODE_ROOT / "src")]

from model import build_model, route_resource_cost_register  # noqa: E402
from paths import load_frozen_5_2_1_inputs, sha256_file  # noqa: E402
from reporting import (  # noqa: E402
    acceptance_payload,
    create_figures,
    write_acceptance_report,
    write_run_manifest,
)


def run() -> int:
    config_path = EXPERIMENT_DIR / "config_5_2_2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256_file(config_path)
    output_final = CODE_ROOT / str(config["output_directory"])
    staging = output_final.parent / f".{output_final.name}.staging"
    stage_record_path = staging / "training_stage_complete.json"
    if not stage_record_path.exists():
        raise RuntimeError("No current fully trained staging run is available")
    stage_record = json.loads(stage_record_path.read_text(encoding="utf-8"))
    if stage_record.get("config_sha256") != config_hash:
        raise RuntimeError("Staging config hash differs from the current frozen config")
    if stage_record.get("test_event_replayed") is not False:
        raise RuntimeError("The training boundary record is invalid")

    frozen = load_frozen_5_2_1_inputs(config)
    if stage_record.get("historical_interface_sha256") != frozen.interface_hash:
        raise RuntimeError("Staging and authorised 5.2.1 interface hashes disagree")
    model = build_model(config)

    def read(name: str) -> pd.DataFrame:
        path = staging / name
        if not path.exists():
            raise RuntimeError(f"Incomplete evaluated staging output: {name}")
        return pd.read_csv(path)

    training_manifest = read("training_path_manifest.csv")
    validation_manifest = read("validation_path_manifest.csv")
    test_manifest = read("test_path_manifest.csv")
    training_curves = read("training_curves.csv")
    checkpoint_manifest = read("checkpoint_manifest.csv")
    replications = read("benchmark_replications.csv")
    path_level = read("path_level_seed_aggregated.csv")
    actions = read("requested_and_implemented_actions.csv")
    diagnostics = read("solver_diagnostics.csv")
    contracts = read("trajectory_contract_checks.csv")
    paired_effects = read("paired_policy_effects.csv")
    pilot_precision = read("pilot_precision.csv")
    selected_path_count = read("selected_path_count.csv")
    loss_summary = read("loss_component_summary.csv")
    clearance = read("clearance_summary.csv")
    authority = read("policy_authority_register.csv")
    route_costs = read("route_resource_cost_register.csv")
    parameters = read("parameter_registry_5_2_2.csv")
    scientific_trace = read("scientific_parameter_traceability.csv")
    waiting_calibration = read("waiting_forecast_error_calibration.csv")
    sac_gradient_checks = read("sac_actor_gradient_check.csv")
    nonanticipativity_checks = read("policy_nonanticipativity_checks.csv")
    confidence = read("policy_confidence_set.csv")

    executed = int(selected_path_count.loc[0, "executed_paths"])
    if executed != 88 or len(test_manifest) != executed:
        raise RuntimeError("The fully evaluated 88-path staging result is incomplete")
    expected_replications = executed * (3 + 5 * int(config["training"]["seeds"]))
    if len(replications) != expected_replications:
        raise RuntimeError("A policy-path-seed replication is missing from staging")

    figure_output = staging / "figures"
    if figure_output.exists():
        shutil.rmtree(figure_output)
    figures = create_figures(
        path_level=path_level,
        replications=replications,
        paired_effects=paired_effects,
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
        training_manifest=training_manifest,
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        replications=replications,
        path_level=path_level,
        actions=actions,
        diagnostics=diagnostics,
        contracts=contracts,
        paired_effects=paired_effects,
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
        json.dumps(acceptance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_acceptance_report(
        acceptance=acceptance,
        confidence_set=confidence,
        paired_effects=paired_effects,
        clearance=clearance,
        output_path=staging / "ACCEPTANCE_REPORT.md",
    )
    provenance = {
        "5.2.1 historical information event path": frozen.interface_hash,
        "5.2.1 pre-event residual library": frozen.residual_hash,
        "5.2.2 frozen config": config_hash,
        "chapter_3_4_common_model": "current src/tre84; package hashes recorded in run_manifest code register",
        "postprocessing_recovery": "same-run staging finalized after replacing unsupported matplotlib boxplot orientation keyword with vert=False",
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
    write_run_manifest(
        output_directory=staging,
        experiment_directory=EXPERIMENT_DIR,
        config_path=config_path,
        frozen=frozen,
        figures_published=published,
        command=(
            "python experiments/5.2-2/run_5_2_2.py; "
            "python experiments/5.2-2/finalize_staging_5_2_2.py"
        ),
    )
    if output_final.exists():
        shutil.rmtree(output_final)
    staging.replace(output_final)
    print(f"[5.2.2] Staging finalized. Acceptance: {acceptance['status']}")
    return 0 if acceptance["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(run())

