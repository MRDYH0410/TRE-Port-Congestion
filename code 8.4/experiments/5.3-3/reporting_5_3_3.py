"""Figures, reports, acceptance, and manifest for Experiment 5.3.3."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from figure_style import (  # noqa: E402
    POLICY_COLOURS,
    TEXT_WIDTH,
    apply_publication_style,
    panel_title,
    policy_label,
    save_figure,
)


POLICY_ORDER = ["Passive", "Reactive", "Projected stochastic MPC", "Behaviour cloning", "Model-guided constrained SAC"]
POLICY_LABEL = {policy: policy_label(policy) for policy in POLICY_ORDER}
COLORS = {policy: POLICY_COLOURS[policy] for policy in POLICY_ORDER}
ARCH = ["capacity_neutral", "port_only", "end_to_end"]
ARCH_LABEL = {"capacity_neutral": "Capacity neutral", "port_only": "Port only", "end_to_end": "End to end"}
ARCH_COLORS = {"capacity_neutral": "#6B7280", "port_only": "#0072B2", "end_to_end": "#009E73"}
ELIG_STYLE = {"emergency_only": "-", "precontracted": "--", "observed_reference": ":"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _with_reference(frame: pd.DataFrame, architecture: str, eligibility: str) -> pd.DataFrame:
    subset = frame.loc[(frame["architecture"] == architecture) & (frame["eligibility"] == eligibility)].copy()
    reference = frame.loc[frame["cell_id"] == "n03_reference"].copy()
    reference["architecture"] = architecture
    reference["eligibility"] = eligibility
    return pd.concat([reference, subset], ignore_index=True)


def create_figures(
    summary: pd.DataFrame,
    regret: pd.DataFrame,
    figures_directory: Path,
    output_directory: Path,
    dpi: int,
) -> dict[str, Path]:
    figures_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    apply_publication_style()

    fig, axes = plt.subplots(3, 2, figsize=(TEXT_WIDTH, 7.35), constrained_layout=True)
    for row, architecture in enumerate(ARCH):
        ax_loss, ax_regret = axes[row, 0], axes[row, 1]
        for eligibility in ("emergency_only", "precontracted"):
            data = _with_reference(summary, architecture, eligibility)
            reg = _with_reference(regret.rename(columns={"regret_mean": "mean_regret"}), architecture, eligibility)
            for policy in POLICY_ORDER:
                current = data.loc[data["policy"] == policy].sort_values("gateway_count")
                if not current.empty:
                    ax_loss.plot(current["gateway_count"], current["mean_total_operational_loss"], color=COLORS[policy], linestyle=ELIG_STYLE[eligibility], marker="o", linewidth=1.4)
                current_reg = reg.loc[reg["policy"] == policy].sort_values("gateway_count")
                if not current_reg.empty:
                    ax_regret.plot(current_reg["gateway_count"], current_reg["mean_regret"], color=COLORS[policy], linestyle=ELIG_STYLE[eligibility], marker="o", linewidth=1.4)
        panel_title(ax_loss, chr(ord("A") + 2 * row), f"{ARCH_LABEL[architecture]}: loss")
        panel_title(ax_regret, chr(ord("B") + 2 * row), f"{ARCH_LABEL[architecture]}: regret")
        ax_loss.set_ylabel("Mean total loss")
        ax_regret.set_ylabel("Matched-path regret")
        ax_loss.set_xlabel("Gateway count")
        ax_regret.set_xlabel("Gateway count")
        ax_loss.set_xticks([3, 4, 5, 7, 9])
        ax_regret.set_xticks([3, 4, 5, 7, 9])
        ax_loss.grid(alpha=0.2)
        ax_regret.grid(alpha=0.2)
    policy_handles = [plt.Line2D([], [], color=COLORS[p], marker="o", label=POLICY_LABEL[p]) for p in POLICY_ORDER]
    style_handles = [plt.Line2D([], [], color="black", linestyle=ELIG_STYLE[e], label="Emergency only" if e == "emergency_only" else "Precontracted") for e in ("emergency_only", "precontracted")]
    fig.legend(handles=policy_handles + style_handles, loc="outside upper center", ncol=4)
    path = figures_directory / "figure_5_3_3a_loss_regret_network.png"
    save_figure(fig, path, dpi=dpi)
    plt.close(fig)
    paths["figure_a"] = path

    reactive = summary.loc[summary["policy"] == "Reactive"]
    metrics = [
        ("mean_corridor_overload_exposure", "Corridor overload exposure"),
        ("mean_overloaded_gateway_incidence", "Overloaded-gateway incidence"),
        ("mean_resource_week_overload", "Resource-week overload"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(TEXT_WIDTH, 6.8), constrained_layout=True)
    for letter, axis, (metric, label) in zip("ABC", axes, metrics):
        for architecture in ARCH:
            for eligibility in ("emergency_only", "precontracted"):
                current = _with_reference(reactive, architecture, eligibility).sort_values("gateway_count")
                axis.plot(current["gateway_count"], current[metric], color=ARCH_COLORS[architecture], marker="o", linewidth=1.5, linestyle=ELIG_STYLE[eligibility])
        panel_title(axis, letter, label)
        axis.set_xlabel("Gateway count")
        axis.set_xticks([3, 4, 5, 7, 9])
        axis.grid(alpha=0.2)
    architecture_handles = [plt.Line2D([], [], color=ARCH_COLORS[item], marker="o", label=ARCH_LABEL[item]) for item in ARCH]
    eligibility_handles = [plt.Line2D([], [], color="black", linestyle=ELIG_STYLE[item], label="Emergency only" if item == "emergency_only" else "Precontracted") for item in ("emergency_only", "precontracted")]
    fig.legend(handles=architecture_handles + eligibility_handles, loc="outside upper center", ncol=5)
    path = figures_directory / "figure_5_3_3b_network_overload.png"
    save_figure(fig, path, dpi=dpi)
    plt.close(fig)
    paths["figure_b"] = path

    metrics = [
        ("mean_waiting_exposure", "Waiting exposure"),
        ("mean_delivery", "Delivered cargo"),
        ("clearance_probability", "Clearance probability"),
        ("mean_terminal_outstanding", "Terminal outstanding mass"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 5.6), constrained_layout=True)
    for letter, axis, (metric, label) in zip("ABCD", axes.flat, metrics):
        for architecture in ARCH:
            for eligibility in ("emergency_only", "precontracted"):
                current = _with_reference(reactive, architecture, eligibility).sort_values("gateway_count")
                axis.plot(current["gateway_count"], current[metric], color=ARCH_COLORS[architecture], marker="o", linewidth=1.5, linestyle=ELIG_STYLE[eligibility])
        panel_title(axis, letter, label)
        axis.set_xlabel("Gateway count")
        axis.set_xticks([3, 4, 5, 7, 9])
        axis.grid(alpha=0.2)
    fig.legend(handles=architecture_handles + eligibility_handles, loc="outside upper center", ncol=5)
    path = figures_directory / "figure_5_3_3c_flow_clearance.png"
    save_figure(fig, path, dpi=dpi)
    plt.close(fig)
    paths["figure_c"] = path

    for path in paths.values():
        shutil.copy2(path, output_directory / path.name)
    return paths


def acceptance_payload(
    *, upstream: pd.DataFrame, cells: pd.DataFrame, path_level: pd.DataFrame,
    contracts: pd.DataFrame, checkpoints: pd.DataFrame, gradient: pd.DataFrame,
    jacobian: pd.DataFrame, precision: pd.DataFrame, figures: Mapping[str, Path],
    runtime: pd.DataFrame, expected_paths: int,
) -> dict[str, Any]:
    checks = {
        "all_upstream_hash_locks_match": bool(upstream["matched"].all()),
        "exactly_25_unique_structural_cells": cells["cell_id"].nunique() == 25 and len(cells) == 25,
        "all_gateway_counts_present": set(cells["gateway_count"]) == {3, 4, 5, 7, 9},
        "all_formal_cells_executed": path_level["cell_id"].nunique() == 25,
        "matched_formal_path_count": bool(path_level.groupby(["cell_id", "policy"])["path_id"].nunique().eq(expected_paths).all()),
        "dynamic_action_dimensions_hold": bool(contracts["dynamic_action_dimension_passed"].astype(bool).all()),
        "semi_synthetic_names_generic": bool(contracts["semi_synthetic_names_are_generic"].astype(bool).all()),
        "all_trajectory_contracts_pass": bool(contracts["all_step_acceptance_passed"].astype(bool).all() and contracts["loss_components_reconstruct_total"].astype(bool).all()),
        "five_size_specific_training_bundles": checkpoints["gateway_count"].nunique() == 5,
        "no_34d_checkpoint_padding": bool((checkpoints["checkpoint_action_dimension"] == checkpoints["expected_action_dimension"]).all()),
        "all_sac_gradient_checks_pass": bool(gradient["passed"].astype(bool).all()),
        "all_projection_jacobian_checks_pass": bool(jacobian["status"].eq("PASS").all()),
        "all_figures_are_png_and_300dpi": all(
            path.suffix.lower() == ".png"
            and path.exists()
            and min(Image.open(path).info.get("dpi", (0.0, 0.0))) >= 299.0
            for path in figures.values()
        ),
        "runtime_budget_respected": float(runtime["total_elapsed_seconds"].iloc[-1]) <= float(runtime["runtime_budget_seconds"].iloc[-1]),
    }
    engineering = all(checks[key] for key in checks if key not in {"runtime_budget_respected"})
    numerical = checks["all_trajectory_contracts_pass"] and checks["all_sac_gradient_checks_pass"] and checks["all_projection_jacobian_checks_pass"]
    precision_pass = bool(not precision.empty and precision["precision_target_met"].all())
    methodology = checks["dynamic_action_dimensions_hold"] and checks["five_size_specific_training_bundles"] and checks["no_34d_checkpoint_padding"]
    return {
        "run_status": "complete" if engineering and numerical and methodology else "failed",
        "ENGINEERING_ACCEPTANCE": "PASS" if engineering else "FAIL",
        "NUMERICAL_ACCEPTANCE": "PASS" if numerical else "FAIL",
        "METHODOLOGY_CONTRACT_ACCEPTANCE": "PASS" if methodology else "FAIL",
        "EXPERIMENTAL_PRECISION_ACCEPTANCE": "PASS" if precision_pass else "FAIL",
        "RUNTIME_BUDGET_ACCEPTANCE": "PASS" if checks["runtime_budget_respected"] else "FAIL",
        "OVERALL_EVIDENCE_ACCEPTANCE": "PASS" if engineering and numerical and methodology and precision_pass and checks["runtime_budget_respected"] else "FAIL",
        "checks": checks,
        "formal_paths_per_cell": expected_paths,
        "precision_contrasts": len(precision),
        "precision_targets_met": int(precision["precision_target_met"].sum()) if not precision.empty else 0,
        "maximum_achieved_halfwidth": float(precision["halfwidth"].max()) if not precision.empty else None,
        "maximum_transition_residual": float(contracts["maximum_transition_residual"].max()),
        "maximum_loss_identity_residual": float(contracts["maximum_period_loss_identity_residual"].max()),
        "boundary": "semi-synthetic network structural stress, not a named-port expansion forecast; unexecuted MPC/MG cell-policy pairs are NOT_EVALUATED_BY_DESIGN",
    }


def write_reports(
    report_directory: Path, acceptance: Mapping[str, Any], summary: pd.DataFrame,
    components: pd.DataFrame, confidence: pd.DataFrame, runtime: pd.DataFrame,
    cells: pd.DataFrame,
) -> None:
    report_directory.mkdir(parents=True, exist_ok=True)
    leaders = confidence.loc[confidence["policy"] == confidence["sample_leader"], ["cell_id", "sample_leader"]]
    best_e2e = components.loc[(components["component"] == "end_to_end_capacity_value") & (components["policy"] == "Reactive")].sort_values("mean", ascending=False).head(1)
    best_text = "not available" if best_e2e.empty else f"{best_e2e.iloc[0]['mean']:.2f} loss units at {best_e2e.iloc[0]['right_cell']}"
    analysis = f"""# Experiment 5.3.3: Gateway Network Sensitivity

## Answer first

The run completed {cells['cell_id'].nunique()} declared network cells with {acceptance['formal_paths_per_cell']} matched physical paths per cell. Engineering acceptance was **{acceptance['ENGINEERING_ACCEPTANCE']}**, numerical acceptance was **{acceptance['NUMERICAL_ACCEPTANCE']}**, methodology-contract acceptance was **{acceptance['METHODOLOGY_CONTRACT_ACCEPTANCE']}**, and precision acceptance was **{acceptance['EXPERIMENTAL_PRECISION_ACCEPTANCE']}**. A precision failure is evidence of limited resolution, not permission to widen claims.

The largest matched Reactive end-to-end component value was {best_text}. Positive component values mean the right-hand architecture produced lower loss on the same exogenous paths. Capacity-neutral differences identify choice/reallocation under fixed total capacity; port-only differences diagnose bottleneck transfer with a fixed shared corridor; only the port-only-to-end-to-end contrast adds common-corridor capacity.

## Design and evidence boundary

- One common observed three-gateway reference and 24 expansion cells form 25 unique cells.
- Every new node is named `SemiSynthetic_Gateway_XX` and inherits median observed service, lag, and waiting-error inputs.
- A fresh teacher, BC, and full constrained SAC were generated for each network size. Size-specific checkpoints were frozen across architectures and eligibility rules; this is a controlled controller-transfer design, not per-cell retuning.
- Passive, Reactive, and BC were run in all 25 cells. MPC and model-guided SAC were run only at the observed reference and six nine-gateway endpoints. Other pairs are `NOT_EVALUATED_BY_DESIGN`, never zeros and never imputed.
- This experiment is a structural network stress, not a forecast for any named port or an empirical expansion estimate.

## Runtime and scalability

The formal path count was selected only from validation-path timing, before test outcomes. Total elapsed time was {runtime['total_elapsed_seconds'].iloc[-1] / 3600:.2f} hours against the six-hour cap. Mean and maximum policy decision times and training times are preserved in `runtime_and_scalability.csv`; changes combine a larger physical network, a higher-dimensional action vector, and policy computation.

## Policy evidence

The cellwise confidence table contains {len(leaders)} sample leaders. It must be read within each executed policy family: three policies in structural cells and five policies only at the seven anchors. Cross-cell changes in the BC results combine structural change with size-specific retraining, whereas Passive and Reactive architecture contrasts retain fixed analytical mappings.
"""
    (report_directory / "FIGURE_AND_RESULTS_ANALYSIS.md").write_text(analysis, encoding="utf-8")
    acceptance_text = "# Experiment 5.3.3 Acceptance Report\n\n" + "\n".join(f"- {key}: **{value}**" for key, value in acceptance.items() if isinstance(value, str)) + "\n\n## Blocking checks\n\n" + "\n".join(f"- {key}: {value}" for key, value in acceptance["checks"].items()) + "\n"
    (report_directory / "ACCEPTANCE_REPORT.md").write_text(acceptance_text, encoding="utf-8")
    additions = """# 5.1 Parameter and Metric Additions for Experiment 5.3.3

## Parameters to register

- Gateway count n in {3,4,5,7,9}; semi-synthetic status; architecture; commitment eligibility.
- Median-template stage capacity, maritime lag, and waiting-error scale, with the three observed inputs as their data source.
- End-to-end reference capacity: minimum of the three gateway-stage capacities and the gate-capacity-proportional share of the common corridor.
- Dynamic action dimension: dim(a(n))=3(3n+1)+1+n=10n+4.
- Runtime-gate path rule, formal path count, and six-hour computational cap.

## Metrics to register

- Choice, port-capacity, end-to-end-capacity, and precontracting matched component values.
- Corridor overload exposure, overloaded-gateway incidence, resource-week overload, waiting exposure, delivery, clearance probability, restricted mean clearance time, terminal outstanding mass, and training/decision time.

## Boundary

No new observed port data source is introduced. New nodes are semi-synthetic median templates and cannot be interpreted as forecasts for specific ports.
"""
    (report_directory / "5_1_PARAMETER_AND_METRIC_ADDITIONS.md").write_text(additions, encoding="utf-8")


def write_manifest(
    path: Path, *, config_hash: str, source_hash: str, upstream: pd.DataFrame,
    output_directory: Path, figures: Mapping[str, Path], started_utc: str,
    elapsed_seconds: float, formal_paths: int,
) -> None:
    outputs = []
    for item in sorted(output_directory.iterdir()):
        if item.is_file() and item.name != path.name:
            outputs.append({"path": item.name, "bytes": item.stat().st_size, "sha256": sha256_file(item)})
    payload = {
        "experiment_id": "5.3.3_gateway_network_sensitivity",
        "status": "complete",
        "started_utc": started_utc,
        "elapsed_seconds": elapsed_seconds,
        "formal_paths_per_cell": formal_paths,
        "config_sha256": config_hash,
        "production_source_bundle_sha256": source_hash,
        "upstream_locks": upstream.to_dict(orient="records"),
        "figures": {key: value.name for key, value in figures.items()},
        "outputs": outputs,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
