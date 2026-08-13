"""Figures, acceptance, manifests, and reports for Experiment 5.3.4."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import Patch

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from figure_style import (  # noqa: E402
    POLICY_COLOURS,
    TEXT_WIDTH,
    apply_publication_style,
    factor_label,
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


def _compact_stability_label(cell_id: str, factor: str, level: Any) -> str:
    """Keep the dense confidence-set matrix readable at manuscript scale."""

    interaction_labels = {
        "interaction__n09__low_corridor": "Nine gateways + low corridor",
        "interaction__long_lag__severe_reclosure": "Long lag + severe reclosure",
        "interaction__lead16__low_credibility": "Long lead + low credibility",
        "interaction__convex_hazard__low_exit": "Convex waiting + low exit consequence",
    }
    if cell_id in interaction_labels:
        return interaction_labels[cell_id]
    label = factor_label(cell_id, factor, level)
    if label == "Waiting age sensitivity  power 0.5":
        return "Waiting-age response - mild"
    if label == "Waiting age sensitivity  power 1":
        return "Waiting-age response - linear"
    replacements = {
        "  0.25 times": " - quarter",
        "  0.5 times": " - half",
        "  1 times": " - standard",
        "  2 times": " - double",
        "  ": " - ",
    }
    for source, target in replacements.items():
        label = label.replace(source, target)
    return label


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def parameter_registry(experiment: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for item in experiment["one_factor_screen"]:
        rows.append(
            {
                "parameter_or_design": item["factor"],
                "family": item["family"],
                "declared_levels": "|".join(f"{float(value):g}" for value in item["levels"]),
                "reference": float(item["reference"]),
                "construction_or_operator": item["operator"],
                "evidence_status": "designed robustness transformation; not an empirical estimate",
            }
        )
    rows.append(
        {
            "parameter_or_design": "commitment_fraction",
            "family": "structural boundary handled upstream",
            "declared_levels": "0.5 only",
            "reference": 0.5,
            "construction_or_operator": experiment["commitment_contract"]["reason_not_varied"],
            "evidence_status": "main effect belongs exclusively to Commitment Sensitivity",
        }
    )
    return pd.DataFrame(rows)


def formula_registry() -> pd.DataFrame:
    rows = [
        ("F534-01", "reference-centred multiplicative level", "robustness_5_3_4.model_config", "parameter_registry_5_3_4.csv"),
        ("F534-02", "waiting hazard (j/Jbar)^p", "model.build_model", "parameter_registry_5_3_4.csv"),
        ("F534-03", "path exposure multiplication", "robustness_5_3_4.transform_paths", "test_path_cell_manifest.csv"),
        ("F534-04", "nonzero route-resource-cost multiplier", "model.route_resource_cost_register", "parameter_registry_5_3_4.csv"),
        ("F534-05", "validation-RMSE waiting-scale multiplier", "model.build_model", "parameter_registry_5_3_4.csv"),
        ("F534-06", "severe reclosure path (1,0.95,32)", "robustness_5_3_4._severe_reclosure_path", "test_path_cell_manifest.csv"),
        ("F534-07", "nine-gateway structural constructor", "network_5_3_3.build_cell_config", "test_path_cell_manifest.csv"),
        ("F534-08", "matched BC and full constrained-SAC training", "run_5_3_4._train_matched_bundle", "matched_training_curves.csv|matched_sac_actor_gradient_check.csv"),
        ("F534-09", "frozen or matched policy replay", "robustness_worker.evaluate_task", "path_level_policy_seed_results.csv"),
        ("F534-10", "learning seeds averaged within physical path", "statistics_5_3_4.aggregate_learning_seeds", "path_level_seed_aggregated.csv"),
        ("F534-11", "cell-minus-reference paired effect", "statistics_5_3_4.paired_cell_effects", "paired_parameter_effects.csv"),
        ("F534-12", "pathwise policy regret and confidence set", "statistics_5_3_4.policy_regret", "policy_regret.csv|policy_confidence_set.csv"),
        ("F534-13", "terminal-state clearance classification", "statistics_5_3_4.clearance_endpoint_diagnostic", "clearance_tolerance_diagnostic.csv"),
    ]
    return pd.DataFrame(rows, columns=["formula_id", "object", "production_code", "output"])


def _save(fig: Any, name: str, figure_directory: Path, output_directory: Path, dpi: int) -> Path:
    figure_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    target = figure_directory / name
    save_figure(fig, target, dpi=dpi)
    plt.close(fig)
    shutil.copy2(target, output_directory / name)
    return target


def create_figures(
    *,
    effects: pd.DataFrame,
    confidence: pd.DataFrame,
    summary: pd.DataFrame,
    figure_directory: Path,
    output_directory: Path,
    dpi: int,
) -> dict[str, Path]:
    apply_publication_style()
    reactive = effects.loc[effects["policy"] == "Reactive"].copy()
    reactive["label"] = [
        factor_label(row.cell_id, row.display_factor, row.display_level)
        for row in reactive.itertuples(index=False)
    ]
    reactive = reactive.sort_values(["cell_type", "family", "display_factor", "display_level"])
    y = np.arange(len(reactive))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, max(8.0, 0.30 * len(reactive))), constrained_layout=True)
    for index, row in enumerate(reactive.itertuples(index=False)):
        ax.errorbar(
            row.mean,
            index,
            xerr=[[row.mean - row.lower], [row.upper - row.mean]],
            fmt="o",
            color="#0072B2" if row.cell_type == "one_factor" else "#D55E00",
            capsize=2.5,
        )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y, reactive["label"])
    ax.set_xlabel("Paired loss difference from the common reference")
    ax.grid(axis="x", alpha=0.25)
    figure_a = _save(fig, "figure_5_3_4a_parameter_effect_forest.png", figure_directory, output_directory, dpi)

    cell_order = (
        confidence[["cell_id", "cell_type", "family", "display_factor", "display_level"]]
        .drop_duplicates()
        .sort_values(["cell_type", "family", "display_factor", "display_level"])
    )
    labels = [
        _compact_stability_label(row.cell_id, row.display_factor, row.display_level)
        for row in cell_order.itertuples(index=False)
    ]
    matrix = np.full((len(cell_order), len(POLICY_ORDER)), np.nan)
    for i, cell_id in enumerate(cell_order["cell_id"]):
        group = confidence.loc[confidence["cell_id"] == cell_id]
        for j, policy in enumerate(POLICY_ORDER):
            row = group.loc[group["policy"] == policy]
            if not row.empty:
                matrix[i, j] = 1.0 if bool(row.iloc[0]["in_simultaneous_confidence_set"]) else 0.0
    masked = np.ma.masked_invalid(matrix)
    outside_colour = "#E1E6EB"
    inside_colour = "#315B7D"
    missing_colour = "#FFFFFF"
    cmap = plt.matplotlib.colors.ListedColormap([outside_colour, inside_colour])
    cmap.set_bad(missing_colour)

    # Split the long matrix into two balanced semantic blocks.  This preserves
    # every cell while giving each row enough physical height at manuscript size.
    split_at = 16
    panel_slices = (slice(0, split_at), slice(split_at, len(cell_order)))
    panel_titles = (
        "Behavioural and joint stresses",
        "Information and physical stresses",
    )
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 6.35), constrained_layout=True)
    for panel_index, (ax, row_slice, title) in enumerate(zip(axes, panel_slices, panel_titles)):
        panel_matrix = masked[row_slice, :]
        panel_labels = labels[row_slice]
        ax.imshow(panel_matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(
            range(len(POLICY_ORDER)),
            [POLICY_LABEL[p] for p in POLICY_ORDER],
            rotation=34,
            ha="right",
            fontsize=9.2,
        )
        ax.set_yticks(range(len(panel_labels)), panel_labels, fontsize=9.6)
        ax.set_xticks(np.arange(-0.5, len(POLICY_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(panel_labels), 1), minor=True)
        ax.grid(which="minor", color="#FFFFFF", linewidth=1.1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(which="major", length=0, colors="#263238")
        panel_title(ax, chr(ord("A") + panel_index), title)
        ax.title.set_fontsize(10.5)
    fig.legend(
        handles=[
            Patch(facecolor=outside_colour, edgecolor="#7D8994", label="Outside confidence set"),
            Patch(facecolor=inside_colour, edgecolor=inside_colour, label="Inside confidence set"),
            Patch(facecolor=missing_colour, edgecolor="#7D8994", label="Not evaluated"),
        ],
        loc="outside upper center",
        ncol=3,
        fontsize=9.3,
    )
    figure_b = _save(fig, "figure_5_3_4b_policy_stability.png", figure_directory, output_directory, dpi)

    anchors = summary.loc[
        (summary["cell_id"] == "reference") | (summary["cell_type"] == "interaction")
    ].copy()
    anchor_order = anchors[["cell_id", "display_factor", "display_level"]].drop_duplicates()
    interaction_order = {
        "reference": 0,
        "interaction__long_lag__severe_reclosure": 1,
        "interaction__convex_hazard__low_exit": 2,
        "interaction__n09__low_corridor": 3,
        "interaction__lead16__low_credibility": 4,
    }
    anchor_order["plot_order"] = anchor_order["cell_id"].map(interaction_order)
    anchor_order = anchor_order.sort_values("plot_order")
    short_anchor_labels = {
        "reference": "Reference",
        "interaction__long_lag__severe_reclosure": "Long maritime lag\nand severe reclosure",
        "interaction__convex_hazard__low_exit": "Convex waiting hazard\nand low exit consequence",
        "interaction__n09__low_corridor": "Nine gateways\nand low corridor capacity",
        "interaction__lead16__low_credibility": "Long readiness lead\nand low information credibility",
    }
    anchor_labels = [short_anchor_labels[cell_id] for cell_id in anchor_order["cell_id"]]
    metrics = [
        ("mean_waiting_exposure", "Waiting exposure"),
        ("mean_duration_attrition", "Duration attrition"),
        ("mean_corridor_overload_exposure", "Corridor overload"),
        ("mean_terminal_outstanding", "Terminal outstanding"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(TEXT_WIDTH, 8.8), constrained_layout=True, sharex=True)
    for letter, ax, (column, title) in zip("ABCD", axes, metrics):
        for policy in POLICY_ORDER:
            current = anchors.loc[anchors["policy"] == policy].set_index("cell_id")
            values = [current.loc[cell_id, column] if cell_id in current.index else np.nan for cell_id in anchor_order["cell_id"]]
            ax.plot(range(len(values)), values, marker="o", linewidth=1.4, color=COLORS[policy], label=POLICY_LABEL[policy])
        ax.set_xticks(range(len(anchor_labels)), anchor_labels)
        panel_title(ax, letter, title)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].tick_params(axis="x", labelsize=7.2)
    fig.legend(handles=[plt.Line2D([], [], color=COLORS[p], marker="o", label=POLICY_LABEL[p]) for p in POLICY_ORDER], loc="outside upper center", ncol=5)
    figure_c = _save(fig, "figure_5_3_4c_interaction_mechanisms.png", figure_directory, output_directory, dpi)
    return {"figure_a": figure_a, "figure_b": figure_b, "figure_c": figure_c}


def independent_checks(
    *,
    path_level: pd.DataFrame,
    effects: pd.DataFrame,
    figures: Mapping[str, Path],
    tolerance: float,
) -> pd.DataFrame:
    rows = []
    example = effects.iloc[0]
    current = path_level.loc[
        (path_level["cell_id"] == example.cell_id) & (path_level["policy"] == example.policy),
        ["path_id", "total_operational_objective"],
    ].rename(columns={"total_operational_objective": "current"})
    reference = path_level.loc[
        (path_level["cell_id"] == "reference") & (path_level["policy"] == example.policy),
        ["path_id", "total_operational_objective"],
    ].rename(columns={"total_operational_objective": "reference"})
    recomputed = float((current.merge(reference, on="path_id")["current"] - current.merge(reference, on="path_id")["reference"]).mean())
    rows.append(
        {
            "check": "paired_effect_independent_mean",
            "observed": float(example["mean"]),
            "recomputed": recomputed,
            "absolute_error": abs(float(example["mean"]) - recomputed),
            "passed": abs(float(example["mean"]) - recomputed) <= tolerance,
        }
    )
    for name, path in figures.items():
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
            minimum = min(float(dpi[0]), float(dpi[1])) if dpi else 0.0
            rows.append(
                {
                    "check": f"{name}_dpi",
                    "observed": minimum,
                    "recomputed": 300.0,
                    "absolute_error": abs(minimum - 300.0),
                    "passed": minimum >= 299.0,
                }
            )
    return pd.DataFrame(rows)


def acceptance_payload(
    *,
    upstream: pd.DataFrame,
    replications: pd.DataFrame,
    path_level: pd.DataFrame,
    contracts: pd.DataFrame,
    registry: pd.DataFrame,
    effects: pd.DataFrame,
    diagnostics: pd.DataFrame,
    independent: pd.DataFrame,
    figures: Mapping[str, Path],
    expected_paths: int,
    target_halfwidth: float,
    tolerance: float,
) -> dict[str, Any]:
    cell_paths = path_level.groupby(["cell_id", "policy"])["path_id"].nunique()
    anchor = registry.loc[registry["comparison_family"] == "five_policy_anchor"]
    screen = registry.loc[registry["comparison_family"] == "three_policy_screen"]
    expected_rows_per_path = int(
        registry.loc[registry["policy_evaluated"]].assign(
            seed_rows=lambda x: x["policy"].isin(["Behaviour cloning", "Model-guided constrained SAC"]).map({True: 3, False: 1})
        )["seed_rows"].sum()
    )
    contract_flags = [
        column
        for column in contracts.columns
        if column.endswith("passed") or column in {"all_step_acceptance_passed", "loss_components_reconstruct_total", "commitment_fixed_at_reference", "declared_action_dimension_matched"}
    ]
    convex = path_level.loc[path_level["cell_id"] == "interaction__convex_hazard__low_exit", ["path_id", "policy", "total_operational_objective"]]
    low_exit = path_level.loc[path_level["cell_id"] == "exit_consequence__0p5", ["path_id", "policy", "total_operational_objective"]]
    identity = convex.merge(low_exit, on=["path_id", "policy"], suffixes=("_combined", "_main"))
    checks = {
        "all_upstream_hash_locks_match": bool(upstream["matched"].all()),
        "exact_31_simulated_cells": int(path_level["cell_id"].nunique()) == 31,
        "exact_four_five_policy_anchors": int(anchor["cell_id"].nunique()) == 4,
        "exact_twenty_two_three_policy_screen_cells": int(screen["cell_id"].nunique()) == 22,
        "exact_five_dimension_changed_rule_mpc_cells": int(registry.loc[registry["comparison_family"] == "dimension_changed_rule_mpc", "cell_id"].nunique()) == 5,
        "nonexecuted_policies_not_imputed": not set(registry.loc[~registry["policy_evaluated"], ["cell_id", "policy"]].itertuples(index=False, name=None)) & set(path_level[["cell_id", "policy"]].drop_duplicates().itertuples(index=False, name=None)),
        "matched_physical_path_count_in_every_evaluated_cell_policy": bool((cell_paths == expected_paths).all()),
        "policy_seed_rows_complete": len(replications) == expected_paths * expected_rows_per_path,
        "learning_seeds_averaged_within_path": bool(path_level.loc[path_level["policy"].isin(["Behaviour cloning", "Model-guided constrained SAC"]), "training_seed_count"].eq(3).all()),
        "commitment_not_varied": bool(contracts["commitment_fixed_at_reference"].astype(bool).all()),
        "declared_action_dimension_matched": bool(contracts["declared_action_dimension_matched"].astype(bool).all()),
        "trajectory_contracts_pass": bool(contracts[contract_flags].astype(bool).all().all()) if contract_flags else False,
        "transition_residual_within_tolerance": float(contracts["maximum_transition_residual"].abs().max()) <= tolerance,
        "loss_closure_within_tolerance": float(replications["loss_component_sum_with_terminal"].sub(replications["total_operational_objective"]).abs().max()) <= tolerance,
        "clearance_tolerance_is_diagnostic_only": bool((~diagnostics["changes_actions_transition_or_loss"].astype(bool)).all()) and diagnostics["tolerance"].nunique() == 3,
        "independent_checks_pass": bool(independent["passed"].astype(bool).all()),
        "three_png_figures_exist": len(figures) == 3 and all(path.exists() and path.suffix.lower() == ".png" for path in figures.values()),
        "convex_low_exit_combination_identity_audited": len(identity) > 0 and float(identity["total_operational_objective_combined"].sub(identity["total_operational_objective_main"]).abs().max()) <= tolerance,
    }
    precision = effects.loc[effects["policy"] == "Reactive"].copy()
    precision["target_met"] = precision["halfwidth"] <= float(target_halfwidth)
    engineering = all(value for key, value in checks.items() if key not in {"transition_residual_within_tolerance", "loss_closure_within_tolerance"})
    numerical = checks["transition_residual_within_tolerance"] and checks["loss_closure_within_tolerance"]
    intervals_complete = bool(np.isfinite(precision[["mean", "lower", "upper", "halfwidth"]].to_numpy(dtype=float)).all())
    return {
        "run_status": "complete" if engineering and numerical else "failed",
        "ENGINEERING_ACCEPTANCE": "PASS" if engineering else "FAIL",
        "NUMERICAL_ACCEPTANCE": "PASS" if numerical else "FAIL",
        "CONDITIONAL_ENSEMBLE_ACCEPTANCE": "PASS" if intervals_complete else "FAIL",
        "OVERALL_EVIDENCE_ACCEPTANCE": "PASS" if engineering and numerical and intervals_complete else "FAIL",
        "checks": checks,
        "simulated_cells": int(path_level["cell_id"].nunique()),
        "physical_paths_per_evaluated_cell_policy": int(expected_paths),
        "policy_path_seed_runs": int(len(replications)),
        "reactive_interval_contrasts": int(len(precision)),
        "reactive_reference_halfwidth_targets_met": int(precision["target_met"].sum()),
        "maximum_reactive_halfwidth": float(precision["halfwidth"].max()),
        "reference_halfwidth_target_not_used_as_acceptance_gate": float(target_halfwidth),
        "maximum_transition_residual": float(contracts["maximum_transition_residual"].abs().max()),
        "maximum_loss_closure_error": float(replications["loss_component_sum_with_terminal"].sub(replications["total_operational_objective"]).abs().max()),
        "commitment_main_effect_evaluated_here": False,
    }


def write_reports(
    report_directory: Path,
    acceptance: Mapping[str, Any],
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    confidence: pd.DataFrame,
    diagnostics: pd.DataFrame,
    runtime: pd.DataFrame,
) -> None:
    report_directory.mkdir(parents=True, exist_ok=True)
    reference = summary.loc[summary["cell_id"] == "reference"].set_index("policy")
    leader_rows = confidence.loc[confidence["policy"] == confidence["sample_leader"]]
    unique = int(
        confidence.groupby("cell_id")["in_simultaneous_confidence_set"].sum().eq(1).sum()
    )
    strongest = effects.loc[effects["policy"] == "Reactive"].iloc[
        effects.loc[effects["policy"] == "Reactive", "mean"].abs().argmax()
    ]
    text = f"""# 5.3.4 Figure and Results Analysis

## Evidence status

- Run status: **{acceptance['run_status']}**
- Engineering acceptance: **{acceptance['ENGINEERING_ACCEPTANCE']}**
- Numerical acceptance: **{acceptance['NUMERICAL_ACCEPTANCE']}**
- Conditional physical-path ensemble acceptance: **{acceptance['CONDITIONAL_ENSEMBLE_ACCEPTANCE']}**
- Simulated cells: **{acceptance['simulated_cells']}**
- Physical paths per evaluated cell-policy: **{acceptance['physical_paths_per_evaluated_cell_policy']}**

## Reference result

At the unchanged reference, mean total operational loss is {reference.loc['Reactive', 'mean_total_operational_loss']:.2f} for Reactive and {reference.loc['Passive', 'mean_total_operational_loss']:.2f} for Passive. These are deployment anchors, not newly estimated historical parameters.

## Conditional robustness

Across the 31 declared cells, {unique} cells have a single policy in the simultaneous confidence set within their preregistered policy family. The largest absolute Reactive cell-minus-reference effect is {strongest['mean']:.2f} in `{strongest['cell_id']}`, with simultaneous interval [{strongest['lower']:.2f}, {strongest['upper']:.2f}]. This result is conditional on the time-budget-selected matched physical-path ensemble and is not an empirical calibration.

## Figure interpretation

- Figure 5.3.4a reports paired Reactive loss changes from the common reference. No mean-only tornado ranking is used.
- Figure 5.3.4b reports confidence-set membership. White cells are policies not evaluated by design and are never interpreted as failures.
- Figure 5.3.4c traces waiting, duration attrition, shared-corridor overload, and terminal outstanding at the reference and four theory-driven interaction anchors.

## Commitment boundary

Commitment is fixed at chi=0.5. Its main effect is not tested here because the complete domain was already evaluated in Commitment Sensitivity. This experiment must not be used to revise that experiment's commitment conclusions.

## Clearance diagnostic

The clearance-tolerance table reclassifies the same terminal states at 1e-8, 1e-6, and 1e-4. It does not change a policy action, transition, loss, or terminal state and is not a reoptimised clearance experiment.

## Evidence boundary

Dimension-compatible one-factor cells use frozen accepted checkpoints and therefore measure deployment robustness. The severe-reclosure/long-lag endpoint uses separately matched BC and full constrained-SAC training. The nine-gateway interaction uses the frozen size-specific checkpoint, while the lead-16 interaction excludes incompatible learned checkpoints rather than padding them. Missing policies are never imputed. The convex-hazard/low-exit combined anchor is algebraically identical to the low-exit main-effect cell because p=2 is the reference; it is not interpreted as an identified non-additive interaction.
"""
    (report_directory / "FIGURE_AND_RESULTS_ANALYSIS.md").write_text(text, encoding="utf-8")
    acceptance_text = f"""# 5.3.4 Acceptance Report

- ENGINEERING_ACCEPTANCE: **{acceptance['ENGINEERING_ACCEPTANCE']}**
- NUMERICAL_ACCEPTANCE: **{acceptance['NUMERICAL_ACCEPTANCE']}**
- CONDITIONAL_ENSEMBLE_ACCEPTANCE: **{acceptance['CONDITIONAL_ENSEMBLE_ACCEPTANCE']}**
- OVERALL_EVIDENCE_ACCEPTANCE: **{acceptance['OVERALL_EVIDENCE_ACCEPTANCE']}**
- Policy-path-seed runs: **{acceptance['policy_path_seed_runs']}**
- Maximum transition residual: **{acceptance['maximum_transition_residual']:.6g}**
- Maximum loss closure error: **{acceptance['maximum_loss_closure_error']:.6g}**

The full machine-readable check set is stored in `acceptance_5_3_4.json`. All estimates and simultaneous intervals are reported conditionally on the path ensemble selected by the preregistered eight-hour computational rule; outcomes do not change that ensemble.
"""
    (report_directory / "ACCEPTANCE_REPORT.md").write_text(acceptance_text, encoding="utf-8")
    additions = """# 5.1 Parameter and Metric Additions for 5.3.4

## Data source section

NO CHANGE. The experiment uses the frozen event interface, residual paths, parameter register, accepted historical checkpoints, the accepted severe-reclosure constructor, and the accepted semi-synthetic gateway constructor.

## Parameter table additions

Add the following registered structural transformations to the experiment-parameter table:

- route sensitivity multiplier `m_theta in {0.5,1,2}`;
- waiting-hazard exponent `p in {0.5,1,2}`;
- exit consequence `c^E_k=m_E T_D`, `m_E in {0.5,1,2}`;
- network exposure `m_Q varrho^(Q,ref)`, `m_Q in {0.5,1,2}`;
- positive rounded maritime lag multiplier `m_tau in {0.5,1,2}`;
- port service and shared-corridor capacity multipliers `m_mu,m_kappa in {0.5,1,2}`;
- feedback strength `eta in {0,1,2}`;
- nonzero route-resource-cost and action-cost multipliers `m_R,m_A in {0.5,1,2}`;
- authority-budget fraction `b in {0.25,0.5,1}`;
- readiness lead `Lambda^R in {4,8,16}` weeks;
- waiting-error RMSE multiplier `m_sigma in {0.5,1,2}`;
- clearance tolerances `{1e-8,1e-6,1e-4}`, used only to reclassify unchanged terminal states;
- the severe reclosure anchor `(D_open,1-a_reclose,D_reclose)=(1,0.95,32)` and the eight-week common recovery constructor.

State explicitly that the multiplicative levels are dimensionless designed structural stresses, not confidence intervals or newly estimated historical parameters. `gamma_I=0.5` appears only in the preregistered long-readiness/low-credibility combined stress and is the midpoint of the existing formal `[0,1]` domain.

## Metric additions

Add cell-minus-reference paired loss, within-cell policy regret, simultaneous confidence-set membership, policy-identity stability, port-stage and corridor overload exposure, resource-week overload, waiting exposure, both exit channels, delivery, clearance, terminal outstanding, numerical-failure incidence, and checkpoint-support status. Learning seeds are averaged within physical path before all paired summaries.

## Experimental design statement

Record the evidence hierarchy: frozen-policy one-factor screening measures deployment robustness; the historical reference is matched by the accepted benchmark checkpoints; the long-lag/severe-reclosure endpoint receives matched BC and full constrained-SAC retraining; the nine-gateway/low-corridor cell uses a size-matched but stress-frozen checkpoint; dimension-incompatible learned policies are `NOT_RUN_DIMENSION_MISMATCH_OR_PREREGISTERED_SCOPE` and are never padded or imputed. The physical-path count is selected by the preregistered eight-hour timing gate without using test outcomes.

The convex-hazard/low-exit combined anchor uses `p=2`, which is already the reference hazard. It therefore duplicates the low-exit main-effect environment and must be described as a combined anchor with an identity audit, not as an independently identified non-additive interaction.

## Commitment

NO ADDITION. Commitment remains governed exclusively by the existing Commitment Sensitivity registration.
"""
    (report_directory / "5_1_PARAMETER_AND_METRIC_ADDITIONS.md").write_text(additions, encoding="utf-8")


def write_manifest(
    path: Path,
    *,
    config_hash: str,
    source_hash: str,
    upstream: pd.DataFrame,
    output_directory: Path,
    figures: Mapping[str, Path],
    started_utc: str,
    elapsed_seconds: float,
    formal_paths: int,
) -> None:
    outputs = []
    for item in sorted(output_directory.iterdir()):
        if item.is_file() and item.name != path.name:
            outputs.append({"path": item.name, "bytes": item.stat().st_size, "sha256": sha256_file(item)})
    payload = {
        "experiment_id": "5.3.4_parameter_robustness",
        "status": "complete",
        "started_utc": started_utc,
        "elapsed_seconds": elapsed_seconds,
        "formal_paths": formal_paths,
        "config_sha256": config_hash,
        "source_bundle_sha256": source_hash,
        "upstream_locks": upstream.to_dict(orient="records"),
        "figures": {key: {"filename": value.name, "sha256": sha256_file(value)} for key, value in figures.items()},
        "outputs": outputs,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
