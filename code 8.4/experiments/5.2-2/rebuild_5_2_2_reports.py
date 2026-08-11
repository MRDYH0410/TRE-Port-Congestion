"""Rebuild 5.2.2 statistics and figures from frozen raw replications only."""

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
    parameter_registry,
    policy_authority_register,
    scientific_parameter_traceability,
    write_acceptance_report,
    write_run_manifest,
)
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


def main() -> int:
    config_path = EXPERIMENT_DIR / "config_5_2_2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = CODE_ROOT / config["output_directory"]
    figures_public = CODE_ROOT / config["figure_directory"]
    model = build_model(config)
    frozen = load_frozen_5_2_1_inputs(config)

    replications = pd.read_csv(output / "benchmark_replications.csv")
    actions = pd.read_csv(output / "requested_and_implemented_actions.csv")
    diagnostics = pd.read_csv(output / "solver_diagnostics.csv")
    contracts = pd.read_csv(output / "trajectory_contract_checks.csv")
    train_manifest = pd.read_csv(output / "training_path_manifest.csv")
    validation_manifest = pd.read_csv(output / "validation_path_manifest.csv")
    test_manifest = pd.read_csv(output / "test_path_manifest.csv")
    checkpoint_manifest = pd.read_csv(output / "checkpoint_manifest.csv")
    training_curves = pd.read_csv(output / "training_curves.csv")
    sac_gradient_checks = pd.read_csv(output / "sac_actor_gradient_check.csv")
    nonanticipativity_checks = pd.read_csv(
        output / "policy_nonanticipativity_checks.csv"
    )

    path_level = aggregate_learning_seeds(
        replications, learning_policies=config["learning_policies"]
    )
    pilot_ids = test_manifest["path_id"].iloc[: int(config["paths"]["pilot_count"])].tolist()
    pilot_level = path_level[path_level["path_id"].isin(pilot_ids)].copy()
    reference_failure = (
        sum(model.gateway_scales.values())
        * int(config["event_weeks"])
        * float(config["behavior"]["exit_failure_cost_per_unit"])
    )
    pilot, selected = select_path_count(
        pilot_path_level=pilot_level,
        config=config,
        reference_failure_loss=reference_failure,
    )
    paired = paired_policy_effects(
        path_level,
        policies=config["main_policies"],
        confidence_level=float(config["paths"]["confidence_level"]),
    )
    pilot, selected = update_precision_achievement(
        pilot_precision=pilot, selection=selected, paired_effects=paired
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
        actions,
        action_names=model.layout.names,
        learning_policies=config["learning_policies"],
    )
    decision_times = decision_time_summary(actions)
    authority = policy_authority_register(config)
    route_costs = route_resource_cost_register(config)
    parameters = parameter_registry(config)
    scientific_trace = scientific_parameter_traceability(config)
    waiting_calibration = pd.read_csv(
        EXPERIMENT_DIR / "waiting_forecast_error_calibration.csv"
    )
    waiting_residuals = pd.read_csv(
        EXPERIMENT_DIR / "waiting_forecast_error_residuals.csv"
    )

    tables = {
        "path_level_seed_aggregated.csv": path_level,
        "pilot_precision.csv": pilot,
        "selected_path_count.csv": selected,
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
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False, lineterminator="\n")

    figure_paths = create_figures(
        path_level=path_level,
        replications=replications,
        paired_effects=paired,
        confidence_set=confidence,
        loss_summary=loss_summary,
        clearance=clearance,
        policies=config["main_policies"],
        output_directory=output / "figures",
        dpi=int(config["numerics"]["figure_dpi"]),
    )
    figures_public.mkdir(parents=True, exist_ok=True)
    published = []
    for figure in figure_paths:
        target = figures_public / figure.name
        shutil.copy2(figure, target)
        published.append(target)

    acceptance = acceptance_payload(
        config=config,
        frozen=frozen,
        model=model,
        training_manifest=train_manifest,
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        replications=replications,
        path_level=path_level,
        actions=actions,
        diagnostics=diagnostics,
        contracts=contracts,
        paired_effects=paired,
        pilot_precision=pilot,
        selected_path_count=selected,
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
        figures=figure_paths,
    )
    (output / "acceptance_5_2_2.json").write_text(
        json.dumps(acceptance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_acceptance_report(
        acceptance=acceptance,
        confidence_set=confidence,
        paired_effects=paired,
        clearance=clearance,
        output_path=output / "ACCEPTANCE_REPORT.md",
    )
    write_run_manifest(
        output_directory=output,
        experiment_directory=EXPERIMENT_DIR,
        config_path=config_path,
        frozen=frozen,
        figures_published=published,
        command=(
            "python experiments/5.2-2/run_5_2_2.py; "
            "python experiments/5.2-2/rebuild_5_2_2_reports.py"
        ),
    )
    print(json.dumps({"status": acceptance["status"], "warnings": acceptance["honest_result_warnings"]}, indent=2))
    return 0 if acceptance["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
