"""Regenerate every 5.2.1--5.3.4 figure from completed tabular outputs.

This is a presentation-only entry point.  It imports the nine reporting
modules, reads frozen CSV/JSON results, and redraws figures.  It never calls a
simulator, trainer, optimizer, or statistical estimator, and it verifies that
all non-image experiment outputs remain byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pandas as pd
from PIL import Image


CODE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = CODE_ROOT / "output"
OVERLEAF_FIGURES = CODE_ROOT.parent / "overleaf - 8.4" / "figures"
MANIFEST_PATH = EXPERIMENTS_ROOT / "figure_style_refresh_manifest.json"

OUTPUT_DIRECTORIES = {
    "5.2.1": OUTPUT_ROOT / "5.2.1_data_event_information_validity",
    "5.2.2": OUTPUT_ROOT / "5.2.2_common_authority_benchmark",
    "5.2.3": OUTPUT_ROOT / "5.2.3_action_and_congestion_mechanisms",
    "5.2.4": OUTPUT_ROOT / "5.2.4_released_risk_information_capacity_preparation",
    "5.2.5": OUTPUT_ROOT / "5.2.5_computational_methodological_acceptance",
    "5.3.1": OUTPUT_ROOT / "5.3.1_commitment_sensitivity",
    "5.3.2": OUTPUT_ROOT / "5.3.2_reclosure_sensitivity",
    "5.3.3": OUTPUT_ROOT / "5.3.3_gateway_network_sensitivity",
    "5.3.4": OUTPUT_ROOT / "5.3.4_parameter_robustness",
}

REPORTING_FILES = {
    "5.2.1": EXPERIMENTS_ROOT / "5.2-1" / "reporting.py",
    "5.2.2": EXPERIMENTS_ROOT / "5.2-2" / "reporting.py",
    "5.2.3": EXPERIMENTS_ROOT / "5.2-3" / "reporting_5_2_3.py",
    "5.2.4": EXPERIMENTS_ROOT / "5.2-4" / "reporting_5_2_4.py",
    "5.2.5": EXPERIMENTS_ROOT / "5.2-5" / "reporting_5_2_5.py",
    "5.3.1": EXPERIMENTS_ROOT / "5.3-1" / "reporting_5_3_1.py",
    "5.3.2": EXPERIMENTS_ROOT / "5.3-2" / "reporting_5_3_2.py",
    "5.3.3": EXPERIMENTS_ROOT / "5.3-3" / "reporting_5_3_3.py",
    "5.3.4": EXPERIMENTS_ROOT / "5.3-4" / "reporting_5_3_4.py",
}

MANUSCRIPT_EXCLUSIONS = {"figure_5_2_2c_clearance_censoring.png"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(directory: Path, name: str) -> pd.DataFrame:
    path = directory / name
    if not path.exists():
        raise FileNotFoundError(f"Missing completed experiment output: {path}")
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_reporting(experiment_id: str) -> ModuleType:
    path = REPORTING_FILES[experiment_id]
    module_name = "chapter5_plot_" + experiment_id.replace(".", "_")
    for local_name in ("model", "paths", "information_design"):
        sys.modules.pop(local_name, None)
    additions = [
        str(path.parent),
        str(EXPERIMENTS_ROOT / "5.2-2"),
        str(EXPERIMENTS_ROOT),
        str(CODE_ROOT / "src"),
    ]
    for addition in reversed(additions):
        if addition not in sys.path:
            sys.path.insert(0, addition)
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import reporting module: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def non_image_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for directory in OUTPUT_DIRECTORIES.values():
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() in {".png", ".pdf"}:
                continue
            hashes[path.relative_to(CODE_ROOT).as_posix()] = sha256_file(path)
    return hashes


def figure_directory(experiment_id: str) -> Path:
    return REPORTING_FILES[experiment_id].parent / "figures"


def render_5_2_1(module: ModuleType, scratch: Path, dpi: int) -> None:
    output = OUTPUT_DIRECTORIES["5.2.1"]
    figures = figure_directory("5.2.1")
    inputs = {
        "summary": read_csv(output, "counterfactual_rolling_origin_summary.csv"),
        "acf": read_csv(output, "counterfactual_one_step_residual_acf.csv"),
        "selection": read_csv(output, "counterfactual_model_selection.csv"),
        "density": read_csv(output, "heldout_density_summary.csv"),
        "filtered": read_csv(output, "released_hmm_filter.csv"),
        "interface": read_csv(output, "historical_information_event_path.csv"),
    }
    stems = {
        "a": "figure_5_2_1a_counterfactual_predictive_validity",
        "b": "figure_5_2_1b_released_hmm_validity",
        "c": "figure_5_2_1c_event_input_release_clock",
    }
    for suffix in (".png", ".pdf"):
        module.plot_counterfactual_validity(inputs["summary"], inputs["acf"], inputs["selection"], figures / f"{stems['a']}{suffix}", dpi=dpi)
        module.plot_hmm_validity(inputs["density"], inputs["filtered"], figures / f"{stems['b']}{suffix}", dpi=dpi)
        module.plot_event_and_release(inputs["interface"], figures / f"{stems['c']}{suffix}", dpi=dpi)


def render_5_2_2(module: ModuleType, scratch: Path, dpi: int) -> None:
    output = OUTPUT_DIRECTORIES["5.2.2"]
    config = read_json(REPORTING_FILES["5.2.2"].parent / "config_5_2_2.json")
    module.create_figures(
        path_level=read_csv(output, "path_level_seed_aggregated.csv"),
        replications=read_csv(output, "benchmark_replications.csv"),
        paired_effects=read_csv(output, "paired_policy_effects.csv"),
        confidence_set=read_csv(output, "policy_confidence_set.csv"),
        loss_summary=read_csv(output, "loss_component_summary.csv"),
        clearance=read_csv(output, "clearance_summary.csv"),
        policies=config["main_policies"],
        output_directory=figure_directory("5.2.2"),
        dpi=dpi,
    )


def render_5_2_3(module: ModuleType, scratch: Path, dpi: int) -> None:
    output = OUTPUT_DIRECTORIES["5.2.3"]
    policy_set = read_csv(output, "mechanism_policy_set.csv")
    figure_policies = policy_set.loc[policy_set["formal_figure_policy"].astype(bool), "policy"].tolist()
    medoid = read_csv(output, "path_medoid_selection.csv")
    medoid_id = str(medoid.loc[medoid["selected_physical_path_medoid"].astype(bool), "path_id"].iloc[0])
    module.create_figures(
        weekly=read_csv(output, "weekly_policy_mechanisms.csv"),
        physical=read_csv(output, "physical_tagged_trajectory.csv"),
        restricted_effects=read_csv(output, "restricted_action_paired_effects.csv"),
        figure_policies=figure_policies,
        medoid_path_id=medoid_id,
        output_directory=figure_directory("5.2.3"),
        dpi=dpi,
    )


def render_5_2_4(module: ModuleType, scratch: Path, dpi: int) -> None:
    output = OUTPUT_DIRECTORIES["5.2.4"]
    effects = read_csv(output, "paired_effects.csv")
    medoid = read_csv(output, "physical_path_medoid_audit.csv")
    medoid_id = str(medoid.loc[medoid["selected_physical_path_medoid"].astype(bool), "path_id"].iloc[0])
    figures = figure_directory("5.2.4")
    module.create_figure_a(
        information_effects=effects.loc[effects["evidence_layer"].eq("reoptimized_information_value")],
        fixed_effects=effects.loc[effects["evidence_layer"].eq("fixed_policy_information_responsiveness")],
        output_directory=figures,
        dpi=dpi,
    )
    timing = read_csv(output, "figure_5_2_4b_data.csv")
    capacity_plot_trace = pd.DataFrame(
        {
            "evidence_layer": "reoptimized_information_value",
            "controller_id": "IL_RD",
            "warning_scenario": "GH",
            "base_path_id": medoid_id,
            "training_seed": 0,
            "decision_week": timing["decision_week"],
            "controller_current_high_risk_probability": timing["current_risk"],
            "controller_lead_high_risk_probability": timing["lead_risk"],
            "implemented_readiness_order": timing["readiness_order"],
            "mature_readiness_stock_before": timing["readiness_stock"],
            "implemented_readiness_exercise": timing["readiness_exercise"],
            "readiness_matured_this_week": timing["readiness_matured"],
            "readiness_expiry_or_decay": timing["readiness_expiry"],
            "implemented_direct_order": timing["direct_order"],
            "direct_capacity_pipeline_before": timing["direct_pipeline"],
            "direct_capacity_arrival": timing["direct_arrival"],
            "usable_temporary_capacity": timing["temporary_capacity"],
            "blocked_model_units": timing["blocked"],
            "berth_queue_after": timing["queues"],
            "yard_queue_after": 0.0,
            "gate_queue_after": 0.0,
            "landbridge_queue_after": 0.0,
            "scenario_release_date": timing["scenario_release_date"],
            "event_onset": timing["event_onset"],
        }
    )
    module.create_figure_b(
        capacity_trace=capacity_plot_trace,
        medoid_path_id=medoid_id,
        output_directory=figures,
        dpi=dpi,
    )
    module.create_figure_c(
        capacity_path_level=read_csv(output, "capacity_rights_path_results.csv"),
        capacity_effects=effects.loc[effects["evidence_layer"].eq("reoptimized_capacity_rights")],
        loss_components=read_csv(output, "loss_decomposition.csv"),
        output_directory=figures,
        dpi=dpi,
    )


def render_5_2_5(module: ModuleType, scratch: Path, dpi: int) -> None:
    output = OUTPUT_DIRECTORIES["5.2.5"]
    benchmark = OUTPUT_DIRECTORIES["5.2.2"]
    figures = figure_directory("5.2.5")
    module.figure_a(
        read_csv(output, "rcmsa_iteration_trace.csv"),
        read_csv(output, "rcmsa_comparison_summary.csv"),
        read_csv(output, "mpc_scenario_precision.csv"),
        scratch,
        figures,
    )
    module.figure_b(
        read_csv(output, "bc_training_trace.csv"),
        read_csv(benchmark, "validation_curves.csv"),
        read_csv(output, "sac_training_trace.csv"),
        read_csv(output, "sac_actor_gradient_recalculation.csv"),
        read_csv(output, "selector_decision_trace.csv"),
        read_csv(output, "selector_regret.csv"),
        scratch,
        figures,
    )
    module.figure_c(
        read_csv(output, "method_contract_registry.csv"),
        read_csv(output, "reproducibility_audit.csv"),
        read_csv(output, "runtime_profile.csv"),
        scratch,
        figures,
    )


def render_5_3_1(module: ModuleType, scratch: Path, dpi: int) -> None:
    output = OUTPUT_DIRECTORIES["5.3.1"]
    module.create_figures(
        path_level=read_csv(output, "path_level_seed_aggregated.csv"),
        mechanism=read_csv(output, "mechanism_summary.csv"),
        confidence=read_csv(output, "policy_confidence_set.csv"),
        regret=read_csv(output, "policy_regret.csv"),
        clearance=read_csv(output, "clearance_and_censoring.csv"),
        figures_directory=figure_directory("5.3.1"),
        output_directory=scratch,
        dpi=dpi,
    )


def render_5_3_2(module: ModuleType, scratch: Path, dpi: int) -> None:
    output = OUTPUT_DIRECTORIES["5.3.2"]
    config = read_json(REPORTING_FILES["5.3.2"].parent / "config_5_3_2.json")
    module.create_figures(
        path_level=read_csv(output, "path_level_seed_aggregated.csv"),
        confidence=read_csv(output, "policy_confidence_set.csv"),
        regret=read_csv(output, "policy_regret.csv"),
        paired=read_csv(output, "paired_effects.csv"),
        mechanism=read_csv(output, "mechanism_summary.csv"),
        clearance=read_csv(output, "clearance_and_censoring.csv"),
        absorption=read_csv(output, "absorption_certificate_summary.csv"),
        coverage=read_csv(output, "cell_policy_coverage_registry.csv"),
        figures_dir=figure_directory("5.3.2"),
        output_dir=scratch,
        dpi=dpi,
        historical_marker=config["historical_marker"],
        config=config,
    )


def render_5_3_3(module: ModuleType, scratch: Path, dpi: int) -> None:
    output = OUTPUT_DIRECTORIES["5.3.3"]
    module.create_figures(
        read_csv(output, "policy_summary.csv"),
        read_csv(output, "policy_regret.csv"),
        figure_directory("5.3.3"),
        scratch,
        dpi,
    )


def render_5_3_4(module: ModuleType, scratch: Path, dpi: int) -> None:
    output = OUTPUT_DIRECTORIES["5.3.4"]
    module.create_figures(
        effects=read_csv(output, "paired_parameter_effects.csv"),
        confidence=read_csv(output, "policy_confidence_set.csv"),
        summary=read_csv(output, "policy_summary.csv"),
        figure_directory=figure_directory("5.3.4"),
        output_directory=scratch,
        dpi=dpi,
    )


RENDERERS: dict[str, Callable[[ModuleType, Path, int], None]] = {
    "5.2.1": render_5_2_1,
    "5.2.2": render_5_2_2,
    "5.2.3": render_5_2_3,
    "5.2.4": render_5_2_4,
    "5.2.5": render_5_2_5,
    "5.3.1": render_5_3_1,
    "5.3.2": render_5_3_2,
    "5.3.3": render_5_3_3,
    "5.3.4": render_5_3_4,
}


def publish_figures(experiment_id: str, publish_overleaf: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_directory = figure_directory(experiment_id)
    output_directory = OUTPUT_DIRECTORIES[experiment_id]
    for source in sorted(source_directory.glob("figure_5_*")):
        if source.suffix.lower() not in {".png", ".pdf"}:
            continue
        output_target = output_directory / source.name
        shutil.copy2(source, output_target)
        manuscript_included = source.name not in MANUSCRIPT_EXCLUSIONS
        if publish_overleaf and manuscript_included:
            OVERLEAF_FIGURES.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, OVERLEAF_FIGURES / source.name)
        record: dict[str, Any] = {
            "experiment": experiment_id,
            "file": source.name,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "manuscript_included": manuscript_included,
        }
        if source.suffix.lower() == ".png":
            with Image.open(source) as image:
                record["pixel_width"], record["pixel_height"] = image.size
                record["dpi"] = list(image.info.get("dpi", (None, None)))
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=350, help="Raster export resolution (default: 350)")
    parser.add_argument("--no-overleaf", action="store_true", help="Do not synchronize manuscript figure assets")
    parser.add_argument("--experiments", nargs="*", choices=tuple(RENDERERS), default=list(RENDERERS))
    args = parser.parse_args()

    selected = list(args.experiments)
    before = non_image_hashes()
    records: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory(prefix="chapter5_figure_refresh_") as temporary:
        temporary_root = Path(temporary)
        for experiment_id in selected:
            print(f"[{experiment_id}] loading completed outputs", flush=True)
            module = load_reporting(experiment_id)
            scratch = temporary_root / experiment_id
            scratch.mkdir(parents=True, exist_ok=True)
            RENDERERS[experiment_id](module, scratch, args.dpi)
            records.extend(publish_figures(experiment_id, not args.no_overleaf))
            print(f"[{experiment_id}] figures refreshed", flush=True)

    after = non_image_hashes()
    if before != after:
        changed = sorted(set(before) | set(after))
        changed = [path for path in changed if before.get(path) != after.get(path)]
        raise RuntimeError("Presentation refresh changed non-image outputs: " + ", ".join(changed))

    manifest = {
        "status": "complete",
        "scope": "presentation only; completed tabular outputs were read without simulation, training, optimization, or re-estimation",
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "dpi": args.dpi,
        "experiments": selected,
        "scientific_non_image_outputs_byte_identical": True,
        "manuscript_exclusions": sorted(MANUSCRIPT_EXCLUSIONS),
        "figures": records,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {MANIFEST_PATH}", flush=True)
    print(f"Completed {len(records)} figure assets.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
