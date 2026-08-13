"""Tables, figures, manifest, acceptance, and Markdown reports for 5.3.1."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from figure_style import (  # noqa: E402
    POLICY_COLOURS,
    POLICY_MARKERS,
    TEXT_WIDTH,
    apply_publication_style,
    panel_title,
    policy_label,
    save_figure,
)


POLICY_ORDER = [
    "Passive",
    "Reactive",
    "Projected stochastic MPC",
    "Behaviour cloning",
    "Model-guided constrained SAC",
]

POLICY_LABEL = {policy: policy_label(policy) for policy in POLICY_ORDER}
COLORS = {policy: POLICY_COLOURS[policy] for policy in POLICY_ORDER}
MARKERS = {policy: POLICY_MARKERS[policy] for policy in POLICY_ORDER}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def parameter_registry(
    experiment: Mapping[str, Any], base: Mapping[str, Any]
) -> pd.DataFrame:
    design = experiment["path_design"]
    rows = [
        ("commitment_grid", "|".join(map(str, experiment["commitment_grid"])), "share", "designed structural grid", experiment["commitment_grid_basis"]),
        ("commitment_application", experiment["commitment_application"], "formula", "Chapter 3 production split", "Only each new blocked cohort is split; state stocks are not relabelled."),
        ("main_policy_set", "|".join(experiment["main_policies"]), "policy names", "accepted 5.2 policy architecture", "Passive, transparent reactive, formal MPC, BC, and proposed MG controller."),
        ("learning_seeds", base["training"]["seeds"], "seeds per learning policy", "accepted 5.2.2 training contract", "Seeds are averaged within physical paths before inference."),
        ("minimum_common_physical_paths", design["minimum_common_physical_paths"], "paths", "accepted 5.2 evidence design", "Minimum common grid size, not an automatic precision certificate."),
        ("maximum_physical_paths", design["maximum_physical_paths"], "paths", "accepted 5.2.2 residual support", "At most 196 unique contiguous event-free residual blocks."),
        ("precision_endpoints", "|".join(map(str, design["precision_endpoints"])), "chi", "preregistered structural endpoints", "The largest endpoint requirement selects one common grid count."),
        ("target_halfwidth", design["target_halfwidth"], "loss index units", "accepted 5.2.2 statistical target", design["target_basis"]),
        ("confidence_level", design["confidence_level"], "probability", "accepted 5.2.2 inference rule", "Paired path intervals."),
        ("multiplicity", experiment["inference"]["multiplicity"], "rule", "preregistered experiment rule", "Applied by metric and reference family."),
        ("clearance_cap", base["clearance"]["maximum_weeks"], "weeks", "accepted 5.2.2 clearance contract", "Uncleared trajectories are right censored and retain outstanding mass."),
        ("clearance_tolerance", base["clearance"]["empty_tolerance"], "model units", "accepted 5.2.2 clearance contract", "Used only to classify the frozen trajectory."),
        ("mass_tolerance", base["numerics"]["mass_tolerance"], "model units", "accepted 5.2.5 numerical contract", "Split, provenance, and transition identities."),
        ("loss_tolerance", base["numerics"]["loss_identity_tolerance"], "loss index units", "accepted 5.2.5 numerical contract", "Complete objective reconciliation."),
        ("parallel_workers", experiment["execution"]["parallel_workers"], "processes", "engineering setting", experiment["execution"]["parallel_workers_basis"]),
        ("figure_dpi", experiment["execution"]["figure_dpi"], "dpi", "reproducible figure contract", "PNG only as explicitly requested for Experiment 5.3.1."),
    ]
    return pd.DataFrame(
        rows,
        columns=["parameter", "value", "unit", "source_or_evidence_class", "basis_and_role"],
    )


def formula_code_registry() -> pd.DataFrame:
    rows = [
        ("F531-01", "qC=chi*qB; qD=(1-chi)*qB", "src/tre84/transition.py", "construct_demand_split", "weekly_commitment_trajectories.csv", "maximum split residual <= mass tolerance"),
        ("F531-02", "chi only affects the current new blocked cohort", "experiments/5.2-2/preparation.py", "build_realization", "trajectory_contract_checks.csv", "existing_state_relabelled_by_chi is false"),
        ("F531-03", "tagged maritime-to-landbridge conservation", "src/tre84/transition.py", "TaggedTransition.step", "trajectory_contract_checks.csv", "production and provenance audits pass"),
        ("F531-04", "QC=QC0+sum(qC); OC=QC-DC", "experiments/5.3-1/commitment_worker.py", "summarise_mechanism_artifact", "path_level_results.csv", "committed conservation residual <= tolerance"),
        ("F531-05", "seed mean within path", "experiments/5.3-1/statistics_5_3_1.py", "aggregate_learning_seeds", "path_level_seed_aggregated.csv", "three seeds for BC/MG and one row before inference"),
        ("F531-06", "paired policy-reference effect", "experiments/5.3-1/statistics_5_3_1.py", "paired_effects", "paired_effects.csv", "matched physical path IDs and simultaneous intervals"),
        ("F531-07", "endpoint required path count", "experiments/5.3-1/statistics_5_3_1.py", "endpoint_precision", "endpoint_precision_requirements.csv", "common N is max endpoint requirement capped only for execution"),
        ("F531-08", "right-censored clearance", "src/tre84/clearance.py", "ClearanceRunner.run", "clearance_and_censoring.csv", "censored clearance is never recorded as observed"),
    ]
    return pd.DataFrame(
        rows,
        columns=["contract_id", "formula_or_contract", "implementation_file", "implementation_function", "output_evidence", "acceptance_test"],
    )


def _path_intervals(path_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (chi, policy), group in path_level.groupby(["chi", "policy"], sort=True):
        values = group["total_operational_objective"].to_numpy(dtype=float)
        n = len(values)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(n))
        from scipy.stats import t as student_t

        half = float(student_t.ppf(0.975, n - 1) * se)
        rows.append(
            {
                "panel": "total_loss",
                "chi": float(chi),
                "policy": policy,
                "mean": mean,
                "lower": mean - half,
                "upper": mean + half,
                "physical_paths": n,
            }
        )
    return pd.DataFrame(rows)


def create_figures(
    *,
    path_level: pd.DataFrame,
    mechanism: pd.DataFrame,
    confidence: pd.DataFrame,
    regret: pd.DataFrame,
    clearance: pd.DataFrame,
    figures_directory: Path,
    output_directory: Path,
    dpi: int,
) -> dict[str, Path]:
    figures_directory.mkdir(parents=True, exist_ok=True)
    apply_publication_style()

    intervals = _path_intervals(path_level)
    confidence_data = confidence.copy()
    confidence_data["panel"] = "confidence_set"
    regret_data = regret.copy()
    regret_data["panel"] = "policy_regret"
    figure_a_data = pd.concat([intervals, regret_data, confidence_data], ignore_index=True, sort=False)
    write_csv(figure_a_data, output_directory / "figure_5_3_1a_data.csv")

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 3.55), constrained_layout=True)
    ax = axes[0]
    for policy in POLICY_ORDER:
        data = intervals.loc[intervals["policy"] == policy].sort_values("chi")
        ax.plot(data["chi"], data["mean"], color=COLORS[policy], marker=MARKERS[policy], label=POLICY_LABEL[policy], linewidth=1.8)
        ax.fill_between(data["chi"], data["lower"], data["upper"], color=COLORS[policy], alpha=0.10)
    panel_title(ax, "A", "Total operational loss")
    ax.set_xlabel("Committed share")
    ax.set_ylabel("Loss index units")
    legend_handles, legend_labels = ax.get_legend_handles_labels()

    ax = axes[1]
    for policy in POLICY_ORDER:
        data = regret.loc[regret["policy"] == policy].sort_values("chi")
        ax.errorbar(
            data["chi"],
            data["mean_path_paired_regret"],
            yerr=np.vstack(
                [
                    data["mean_path_paired_regret"] - data["simultaneous_lower"],
                    data["simultaneous_upper"] - data["mean_path_paired_regret"],
                ]
            ),
            color=COLORS[policy],
            marker=MARKERS[policy],
            capsize=2,
            linewidth=1.2,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    panel_title(ax, "B", "Matched-path policy regret")
    ax.set_xlabel("Committed share")
    ax.set_ylabel("Difference from pathwise policy minimum")

    ax = axes[2]
    matrix = (
        confidence.pivot(index="policy", columns="chi", values="in_simultaneous_confidence_set")
        .reindex(POLICY_ORDER)
        .astype(float)
    )
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(POLICY_ORDER)), [POLICY_LABEL[p] for p in POLICY_ORDER])
    ax.set_xticks(range(len(matrix.columns)), [f"{value:g}" for value in matrix.columns], rotation=45)
    ax.set_xlabel("Committed share")
    panel_title(ax, "C", "Best-policy confidence set")
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["Outside", "Inside"])
    colorbar.set_label("Confidence set")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="outside upper center",
        ncol=len(POLICY_ORDER),
        frameon=False,
        fontsize=8.5,
        handlelength=1.8,
        columnspacing=1.0,
    )
    figure_a = figures_directory / "figure_5_3_1a_loss_regret_confidence_set.png"
    save_figure(fig, figure_a, dpi=dpi)
    plt.close(fig)

    mechanism_columns = {
        "mean_waiting_exposure": "Waiting exposure",
        "mean_direct_sue_exit": "Route-choice exit",
        "mean_duration_attrition": "Attrition exit",
        "mean_committed_delivery": "Committed delivery",
        "mean_adaptive_delivery": "Adaptive delivery",
    }
    figure_b_data = mechanism.melt(
        id_vars=["chi", "policy"],
        value_vars=list(mechanism_columns),
        var_name="metric",
        value_name="mean_value",
    )
    figure_b_data["metric_label"] = figure_b_data["metric"].map(mechanism_columns)
    write_csv(figure_b_data, output_directory / "figure_5_3_1b_data.csv")
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH, 5.15), constrained_layout=True)
    axes_flat = axes.ravel()
    for letter, axis, (column, label) in zip("ABCDE", axes_flat, mechanism_columns.items()):
        for policy in POLICY_ORDER:
            data = mechanism.loc[mechanism["policy"] == policy].sort_values("chi")
            axis.plot(data["chi"], data[column], color=COLORS[policy], marker=MARKERS[policy], linewidth=1.5)
        panel_title(axis, letter, label)
        axis.set_xlabel("Committed share")
        axis.set_ylabel("Cargo units")
    legend_axis = axes_flat[-1]
    legend_axis.axis("off")
    handles = [
        plt.Line2D([], [], color=COLORS[p], marker=MARKERS[p], label=POLICY_LABEL[p])
        for p in POLICY_ORDER
    ]
    legend_axis.legend(handles=handles, loc="center", frameon=False, title="Policy")
    figure_b = figures_directory / "figure_5_3_1b_wait_exit_delivery_mechanisms.png"
    save_figure(fig, figure_b, dpi=dpi)
    plt.close(fig)

    figure_c_data = clearance.copy()
    write_csv(figure_c_data, output_directory / "figure_5_3_1c_data.csv")
    fig = plt.figure(figsize=(TEXT_WIDTH, 3.65), constrained_layout=True)
    grid = fig.add_gridspec(1, 3)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])
    for policy in POLICY_ORDER:
        data = clearance.loc[clearance["policy"] == policy].sort_values("chi")
        ax0.step(data["chi"], data["clearance_probability"], where="mid", color=COLORS[policy], linewidth=1.8, label=POLICY_LABEL[policy])
        ax1.plot(data["chi"], data["restricted_mean_clearance_time"], color=COLORS[policy], marker=MARKERS[policy], linewidth=1.5)
    ax0.set_ylim(-0.02, 1.02)
    panel_title(ax0, "A", "Clearance probability")
    ax0.set_xlabel("Committed share")
    ax0.set_ylabel("Probability")
    legend_handles, legend_labels = ax0.get_legend_handles_labels()
    panel_title(ax1, "B", "Restricted clearance time")
    ax1.set_xlabel("Committed share")
    ax1.set_ylabel("Weeks")
    matrix = (
        clearance.pivot(index="policy", columns="chi", values="mean_terminal_outstanding_mass")
        .reindex(POLICY_ORDER)
    )
    image = ax2.imshow(matrix.to_numpy(), aspect="auto", cmap="magma_r")
    ax2.set_yticks(range(len(POLICY_ORDER)), [POLICY_LABEL[p] for p in POLICY_ORDER])
    ax2.set_xticks(range(len(matrix.columns)), [f"{value:g}" for value in matrix.columns], rotation=45)
    ax2.set_xlabel("Committed share")
    panel_title(ax2, "C", "Terminal outstanding mass")
    ax2.grid(False)
    fig.colorbar(image, ax=ax2, label="Model units")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="outside upper center",
        ncol=len(POLICY_ORDER),
        frameon=False,
        fontsize=8.5,
        handlelength=1.8,
        columnspacing=1.0,
    )
    figure_c = figures_directory / "figure_5_3_1c_clearance_terminal_mass.png"
    save_figure(fig, figure_c, dpi=dpi)
    plt.close(fig)
    return {"figure_a": figure_a, "figure_b": figure_b, "figure_c": figure_c}


def independent_checks(
    *,
    path_level: pd.DataFrame,
    paired: pd.DataFrame,
    figures: Mapping[str, Path],
    tolerance: float,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    component_sum = (
        path_level["loss_queue"]
        + path_level["loss_waiting"]
        + path_level["loss_exit"]
        + path_level["loss_overflow"]
        + path_level["loss_route_resource"]
        + path_level["loss_action"]
        + path_level["terminal_correction"]
    )
    residual = float((component_sum - path_level["total_operational_objective"]).abs().max())
    checks.append({"check": "independent_loss_reconciliation", "observed": residual, "tolerance": tolerance, "passed": residual <= tolerance})
    committed = float(path_level["committed_conservation_residual"].abs().max())
    checks.append({"check": "independent_committed_conservation", "observed": committed, "tolerance": tolerance, "passed": committed <= tolerance})
    split = float(max(path_level["maximum_committed_split_residual"].max(), path_level["maximum_adaptive_split_residual"].max()))
    checks.append({"check": "independent_new_cohort_split", "observed": split, "tolerance": tolerance, "passed": split <= tolerance})
    example = paired.loc[
        (paired["metric"] == "total_operational_loss")
        & (paired["reference_policy"] == "Passive")
    ].iloc[0]
    cell = path_level.loc[np.isclose(path_level["chi"], example["chi"])]
    reference = cell.loc[cell["policy"] == "Passive", ["path_id", "total_operational_objective"]].rename(columns={"total_operational_objective": "reference"})
    current = cell.loc[cell["policy"] == example["policy"], ["path_id", "total_operational_objective"]].merge(reference, on="path_id", validate="one_to_one")
    recomputed = float((current["total_operational_objective"] - current["reference"]).mean())
    paired_residual = abs(recomputed - float(example["mean_paired_difference"]))
    checks.append({"check": "independent_paired_effect", "observed": paired_residual, "tolerance": tolerance, "passed": paired_residual <= tolerance})
    for key, path in figures.items():
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))[0]
        checks.append({"check": f"{key}_png_dpi", "observed": dpi, "tolerance": 299.0, "passed": dpi >= 299.0})
    return pd.DataFrame(checks)


def acceptance_payload(
    *,
    upstream_locks: pd.DataFrame,
    replications: pd.DataFrame,
    path_level: pd.DataFrame,
    contracts: pd.DataFrame,
    checkpoints: pd.DataFrame,
    requirements: pd.DataFrame,
    selected: pd.DataFrame,
    independent: pd.DataFrame,
    figures: Mapping[str, Path],
    expected_grid: Sequence[float],
    expected_policies: Sequence[str],
    tolerance: float,
) -> dict[str, Any]:
    cells = path_level.groupby(["chi", "policy"])["path_id"].nunique()
    executed = int(selected.loc[0, "executed_paths"])
    expected_cells = len(expected_grid) * len(expected_policies)
    same_hash = (
        path_level.groupby(["chi", "path_id"])["path_content_sha256"].nunique().max()
        == 1
    )
    learning_counts = path_level.loc[path_level["policy"].isin(["Behaviour cloning", "Model-guided constrained SAC"]), "training_seed_count"]
    blocking = {
        "all_accepted_5_2_hash_locks_match": bool(upstream_locks["matched"].all()),
        "exact_preregistered_nine_point_grid": set(np.round(path_level["chi"].unique(), 6)) == set(np.round(expected_grid, 6)),
        "exact_main_policy_set_in_every_grid_cell": len(cells) == expected_cells and set(path_level["policy"].unique()) == set(expected_policies),
        "one_common_physical_path_count_used_in_all_cells": bool((cells == executed).all()),
        "matched_physical_path_hashes_across_chi_and_policy": bool(same_hash),
        "three_learning_seeds_aggregated_within_path_first": bool((learning_counts == 3).all()),
        "new_teacher_and_checkpoints_exist_for_every_chi": checkpoints.groupby("chi")["policy"].nunique().eq(3).all() and checkpoints["generated_for_5_3_1"].all(),
        "no_5_2_checkpoint_reused_as_grid_checkpoint": bool((~checkpoints["loaded_from_5_2_checkpoint"]).all()),
        "chi_split_identities_hold": bool(contracts["commitment_split_identity_passed"].all()),
        "chi_never_relabels_existing_state": bool((~contracts["existing_state_relabelled_by_chi"]).all()),
        "committed_provenance_conservation_holds": bool(contracts["committed_provenance_conservation_passed"].all()),
        "all_production_step_acceptance_passed": bool(contracts["all_step_acceptance_passed"].all()),
        "all_tagged_transition_and_capacity_contracts_passed": bool(contracts["loss_components_reconstruct_total"].all()) and float(contracts["maximum_transition_residual"].max()) <= tolerance,
        "complete_loss_reconciliation": float((path_level["loss_component_sum_with_terminal"] - path_level["total_operational_objective"]).abs().max()) <= tolerance,
        "right_censoring_not_recorded_as_clearance": not bool(path_level.loc[path_level["right_censored"].astype(bool), "clearance_weeks_observed"].notna().any()),
        "endpoint_precision_rule_executed": len(requirements) == 16,
        "path_count_respects_196_cap": executed <= 196,
        "independent_recalculation_passed": bool(independent["passed"].all()),
        "three_png_figures_generated_at_300dpi": len(figures) == 3 and all(path.exists() for path in figures.values()),
    }
    # Pandas reductions may return numpy.bool_, which is logically correct but
    # is not accepted by Python's JSON encoder.  Normalise only the scalar
    # representation here; this does not alter any acceptance calculation.
    blocking = {key: bool(value) for key, value in blocking.items()}
    failures = [key for key, value in blocking.items() if not value]
    precision = bool(requirements["precision_target_met"].all())
    return {
        "experiment_id": "5.3.1_commitment_sensitivity",
        "run_status": "complete" if not failures else "blocked",
        "experimental_precision_acceptance": "PASS" if precision else "FAIL",
        "overall_evidence_acceptance": "PASS" if not failures and precision else "FAIL",
        "blocking_checks": blocking,
        "blocking_failures": failures,
        "precision_contrasts": int(len(requirements)),
        "precision_targets_met": int(requirements["precision_target_met"].sum()),
        "executed_physical_paths_per_grid_cell": executed,
        "maximum_achieved_halfwidth": float(requirements["achieved_halfwidth"].max()),
        "target_halfwidth": float(requirements["target_halfwidth"].iloc[0]),
        "evidence_boundary": "Designed structural commitment sensitivity. Chi is not a historical Hormuz estimate; endpoints alter only newly blocked cohorts and do not erase existing system state.",
    }


def write_reports(
    *,
    report_directory: Path,
    acceptance: Mapping[str, Any],
    mechanism: pd.DataFrame,
    paired: pd.DataFrame,
    confidence: pd.DataFrame,
    clearance: pd.DataFrame,
    selected: pd.DataFrame,
    experiment: Mapping[str, Any],
) -> None:
    report_directory.mkdir(parents=True, exist_ok=True)
    leader_rows = confidence.loc[confidence["policy"] == confidence["sample_leader"]]
    leader_text = ", ".join(
        f"χ={row.chi:g}: {POLICY_LABEL.get(row.sample_leader, row.sample_leader)}"
        for row in leader_rows.itertuples(index=False)
    )
    endpoint = mechanism.loc[mechanism["chi"].isin([0.0, 1.0])]
    passive = endpoint.loc[endpoint["policy"] == "Passive"].set_index("chi")
    reactive = endpoint.loc[endpoint["policy"] == "Reactive"].set_index("chi")
    reactive_vs_passive = paired.loc[
        (paired["metric"] == "total_operational_loss")
        & (paired["policy"] == "Reactive")
        & (paired["reference_policy"] == "Passive")
        & (paired["chi"].isin([0.0, 1.0]))
    ].set_index("chi")
    clearance_endpoint = clearance.loc[
        clearance["chi"].isin([0.0, 1.0])
    ].set_index(["chi", "policy"])
    uniquely_resolved = int(
        confidence.loc[
            confidence["resolved_better_than_all_competitors"].astype(bool), "chi"
        ].nunique()
    )
    maximum_mean_sue_exit = float(mechanism["mean_direct_sue_exit"].max())
    precision_note = (
        "All preregistered endpoint precision targets were met."
        if acceptance["experimental_precision_acceptance"] == "PASS"
        else "At least one endpoint precision target remained unmet at the computational cap; inferential claims must remain precision-limited."
    )
    analysis = f"""# 5.3.1 Results and Figure Analysis

## Result first

The experiment completed on {int(selected.loc[0, 'executed_paths'])} matched physical paths per grid cell. {precision_note}

Benchmark leaders by commitment cell were: {leader_text}. Reactive was the only member of the simultaneous best-policy confidence set in {uniquely_resolved} of {confidence['chi'].nunique()} cells. This is a conditional benchmark result under the declared network, costs, paths, and retraining design; it is not a universal optimality claim.

At the structural endpoints, Passive mean loss changed from {passive.loc[0.0, 'mean_total_operational_objective']:.2f} at χ=0 to {passive.loc[1.0, 'mean_total_operational_objective']:.2f} at χ=1. Reactive changed from {reactive.loc[0.0, 'mean_total_operational_objective']:.2f} to {reactive.loc[1.0, 'mean_total_operational_objective']:.2f}. These are designed commitment contrasts, not estimates of historical committed cargo.

Relative to Passive, Reactive reduced matched total loss by {-reactive_vs_passive.loc[0.0, 'mean_paired_difference']:.2f} at χ=0 (simultaneous interval [{-reactive_vs_passive.loc[0.0, 'simultaneous_upper']:.2f}, {-reactive_vs_passive.loc[0.0, 'simultaneous_lower']:.2f}]) and by {-reactive_vs_passive.loc[1.0, 'mean_paired_difference']:.2f} at χ=1 (interval [{-reactive_vs_passive.loc[1.0, 'simultaneous_upper']:.2f}, {-reactive_vs_passive.loc[1.0, 'simultaneous_lower']:.2f}]).

The loss ordering does not imply a clearance ordering. At χ=0, Reactive cleared only {clearance_endpoint.loc[(0.0, 'Reactive'), 'clearance_probability']:.3f} of paths and had {int(clearance_endpoint.loc[(0.0, 'Reactive'), 'right_censored_paths'])} right-censored paths, whereas Passive cleared {clearance_endpoint.loc[(0.0, 'Passive'), 'clearance_probability']:.3f}. At χ=1, clearance probabilities were {clearance_endpoint.loc[(1.0, 'Passive'), 'clearance_probability']:.3f} for Passive, {clearance_endpoint.loc[(1.0, 'Reactive'), 'clearance_probability']:.3f} for Reactive, {clearance_endpoint.loc[(1.0, 'Projected stochastic MPC'), 'clearance_probability']:.3f} for MPC, {clearance_endpoint.loc[(1.0, 'Behaviour cloning'), 'clearance_probability']:.3f} for BC, and {clearance_endpoint.loc[(1.0, 'Model-guided constrained SAC'), 'clearance_probability']:.3f} for MG constrained SAC. These negative and right-censored results remain part of the evidence.

## Figure 5.3.1a

The first panel shows path-based mean total operational loss and 95% intervals. Mean loss declined as χ increased for every policy, most sharply for Passive. This pattern is generated by moving each new blocked cohort from adaptive route/wait/exit choice into committed tagged dispatch; it must not be read as a recommendation to increase the real committed share. The second panel reports matched-path regret relative to the lowest of the five declared policies on the same path and χ. The third is the simultaneous best-policy confidence set; an `IN` cell means the policy cannot be excluded from that set under the predeclared all-pair family.

## Figure 5.3.1b

The mechanism panels keep waiting exposure, SUE exit, duration attrition, committed delivery, and adaptive delivery separate. Waiting exposure and attrition fell toward zero as χ approached one, committed delivery rose, and adaptive delivery mechanically fell to zero. Mean direct SUE exit was negligible at this scale (maximum {maximum_mean_sue_exit:.6f} model units), so the exit mechanism is carried almost entirely by duration attrition in these paths. Committed delivery is computed from the accepted route-tag provenance ledger, not a new weighted absorption score. All total-loss comparisons retain route-resource and transport loss.

## Figure 5.3.1c

Clearance probability, restricted mean clearance time, and terminal outstanding mass are reported together. The figure exposes the key adverse result: low cumulative loss can coexist with poorer clearance and positive terminal mass. A right-censored path is not assigned an observed clearance week. The clearance tolerance only classifies a trajectory after simulation and does not change its actions, transition, loss, or terminal state.

## Evidence boundary

χ=0 means no newly blocked cargo is committed; it does not define an empty initial system. χ=1 commits every newly blocked cohort; it does not revoke formal choices from pre-existing waiting vintages. The evidence supports how coordination space changes under a designed itinerary-locking fraction only.
"""
    (report_directory / "FIGURE_AND_RESULTS_ANALYSIS.md").write_text(analysis, encoding="utf-8")

    additions = f"""# 5.1 Parameter and Metric Additions for Experiment 5.3.1

## Data

NO CHANGE. Experiment 5.3.1 reads the accepted 5.2.1 interface and the accepted 5.2 production path builder. It introduces no new observed data and does not reinterpret χ as an empirical estimate.

## Parameters to register

- Structural commitment grid: `{experiment['commitment_grid']}`.
- Grid role: endpoints and quartiles plus four preregistered eighth-grid resolution points.
- Application rule: χ splits only each period's newly blocked cohort through `qC=χqB` and `qD=(1-χ)qB`.
- Common path rule: 88 minimum, endpoint variance recalculation, common expansion up to 196.
- Precision target: {experiment['path_design']['target_halfwidth']} loss-index units, inherited unchanged from 5.2.2.
- Policy set: Passive, Reactive, projected stochastic MPC, BC, and MG constrained SAC.
- Retraining rule: teacher data, BC, and constrained SAC are regenerated at each χ; test paths never select checkpoints.

## Metrics to add

- Total new committed mass `QC`, committed landbridge delivery `DC`, terminal committed outstanding `OC`, committed delivery share, and terminal committed outstanding share.
- Waiting exposure, SUE exit, duration attrition, committed delivery, and adaptive delivery by policy and χ.
- Matched effects versus both Passive and Reactive, path-paired policy regret, and simultaneous confidence-set membership.
- Clearance probability, restricted mean clearance time, terminal outstanding mass, and right-censoring count.

No arbitrary absorption score is introduced.
"""
    (report_directory / "5_1_PARAMETER_AND_METRIC_ADDITIONS.md").write_text(additions, encoding="utf-8")

    acceptance_report = f"""# 5.3.1 Acceptance Report

## Outcome

- Run status: **{acceptance['run_status']}**
- Experimental precision acceptance: **{acceptance['experimental_precision_acceptance']}**
- Overall evidence acceptance: **{acceptance['overall_evidence_acceptance']}**
- Physical paths per χ-policy cell: **{acceptance['executed_physical_paths_per_grid_cell']}**
- Precision contrasts met: **{acceptance['precision_targets_met']} / {acceptance['precision_contrasts']}**
- Maximum achieved half-width: **{acceptance['maximum_achieved_halfwidth']:.4f}** against target **{acceptance['target_halfwidth']:.4f}**

## Blocking checks

""" + "\n".join(
        f"- {'PASS' if value else 'FAIL'} — `{key}`"
        for key, value in acceptance["blocking_checks"].items()
    ) + f"""

## Scope

{acceptance['evidence_boundary']}
"""
    (report_directory / "ACCEPTANCE_REPORT.md").write_text(acceptance_report, encoding="utf-8")


def write_manifest(
    *,
    path: Path,
    config_hash: str,
    source_bundle_hash: str,
    upstream_locks: pd.DataFrame,
    output_directory: Path,
    figures: Mapping[str, Path],
    started_utc: str,
    elapsed_seconds: float,
    executed_paths: int,
) -> None:
    files = []
    for item in sorted(output_directory.rglob("*")):
        if item.is_file() and item != path:
            files.append(
                {
                    "relative_path": item.relative_to(output_directory).as_posix(),
                    "bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
    manifest = {
        "experiment_id": "5.3.1_commitment_sensitivity",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "config_sha256": config_hash,
        "production_source_bundle_sha256": source_bundle_hash,
        "upstream_locks": upstream_locks.to_dict(orient="records"),
        "executed_physical_paths_per_grid_cell": executed_paths,
        "python": sys.version,
        "platform": platform.platform(),
        "figure_sources": {
            figures["figure_a"].name: ["figure_5_3_1a_data.csv", "path_level_seed_aggregated.csv", "policy_regret.csv", "policy_confidence_set.csv"],
            figures["figure_b"].name: ["figure_5_3_1b_data.csv", "mechanism_summary.csv"],
            figures["figure_c"].name: ["figure_5_3_1c_data.csv", "clearance_and_censoring.csv"],
        },
        "outputs": files,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
