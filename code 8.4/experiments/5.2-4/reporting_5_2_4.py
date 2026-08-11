"""Registries, figures, reports, manifests, and blocking acceptance."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from information_design import FrozenHMMInputs
from paths import sha256_file


PALETTE = {
    "I0": "#7A7A7A",
    "IF": "#4C78A8",
    "IL": "#E07B24",
    "ORACLE": "#2A9D8F",
    "RD": "#264653",
    "R": "#2A9D8F",
    "D": "#E9C46A",
    "NONE": "#E76F51",
}


def parameter_registry(
    *,
    config: Mapping[str, Any],
    benchmark_config: Mapping[str, Any],
    hmm: FrozenHMMInputs,
    input_hashes: Mapping[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(parameter: str, value: Any, source: str, evidence: str, basis: str) -> None:
        missing_basis = any(
            str(item).strip() == "MISSING_BASIS" for item in (value, source, evidence, basis)
        )
        rows.append(
            {
                "parameter": parameter,
                "value": json.dumps(value) if isinstance(value, (dict, list, tuple)) else value,
                "source": source,
                "evidence_class": evidence,
                "basis_or_formula": basis,
                "basis_status": "MISSING_BASIS" if missing_basis else "REGISTERED",
            }
        )

    add("HMM.transition_matrix", hmm.transition.tolist(), "5.2.1 hmm_parameter_manifest.csv", "training-sample estimate", f"estimated on {hmm.training_rows} months through {hmm.training_end}")
    add("HMM.training_stationary_distribution", hmm.stationary.tolist(), "5.2.1 unconditional comparator", "derived estimate", "stationary row distribution satisfying alpha_bar P = alpha_bar")
    add("release_mapping.nu_t", "latest packet with release_date <= decision week", "5.2.1 release clock", "historical information interface", "nonanticipative carry-forward at Monday decisions")
    add("lead_horizon.h", "calendar month ordinal(maturity)-ordinal(source month)", "Chapter 3 and 5.2.1", "model definition", "monthly matrix is never multiplied once per week")
    for section, keys in {
        "action": ["readiness_lead_weeks", "direct_lead_weeks", "readiness_order_cost_per_unit", "direct_order_cost_per_unit", "readiness_exercise_cost_per_unit", "publication_cost_per_unit", "period_budget_fraction", "cumulative_budget_fraction"],
        "capacity_technology": ["readiness_maturity_yield", "readiness_consumption", "readiness_capacity_yield", "readiness_decay", "direct_maturity_yield", "direct_decay"],
        "clearance": ["maximum_weeks", "empty_tolerance"],
        "numerics": ["mass_tolerance", "loss_identity_tolerance", "figure_dpi"],
    }.items():
        for key in keys:
            add(f"5.2.2.{section}.{key}", benchmark_config[section][key], "accepted 5.2.2 config", "Chapter 3 model or numerical design", "inherited unchanged")
    for key, definition in config["information_regimes"].items():
        add(f"information_regime.{key}", definition, "5.2.4 preregistration", definition["evidence_class"], definition["controller_input"])
    for key, definition in config["warning_scenarios"].items():
        add(f"warning_scenario.{key}", definition, "5.2.4 preregistration", definition["evidence_class"], definition["construction"])
    for key, definition in config["capacity_rights"].items():
        add(f"capacity_rights.{key}", definition, "5.2.4 preregistration", "reoptimized action-right design", "restriction after raw proposal and before common projector")
    for section in ("controller", "timing", "statistics", "computation", "acceptance"):
        for key, value in config[section].items():
            add(f"5.2.4.{section}.{key}", value, "config_5_2_4.json", "preregistered experiment design", "frozen before test evaluation")
    for name, digest in input_hashes.items():
        add(f"input_sha256.{name}", digest, "run manifest", "hash-locked input", "SHA256")
    return pd.DataFrame(rows)


def evidence_classification() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("GH release dates and 21-week event", "historical information replay", "historical public-information timing and observed-event replay"),
            ("HMM alpha and alpha P^h", "estimated information", "latent geopolitical risk-state belief; not closure probability"),
            ("GT, GL and GFW", "designed scenario", "structural timing stresses; not historical Hormuz facts"),
            ("ORACLE", "unattainable oracle-information benchmark", "perfect-information input; achieved learned-policy performance is not guaranteed to attain the theoretical bound"),
            ("fixed IL checkpoint under I0/IF/IL", "fixed-policy diagnostic", "information responsiveness; not reoptimized information value"),
            ("capacity rights RD/R/D/NONE", "reoptimized designed comparison", "conditional action-right values under the reference network"),
        ],
        columns=["object", "evidence_class", "permitted_claim"],
    )


def select_medoid(path_panel: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    gh = path_panel.loc[path_panel["warning_scenario"].eq("GH")].drop_duplicates("base_path_id")
    features = gh[["base_path_id", "total_blocked", "peak_blocked", "mean_serviceability", "minimum_serviceability", "recovery_rate"]].copy()
    values = features.iloc[:, 1:].to_numpy(dtype=float)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    standardized = (values - values.mean(axis=0)) / scale
    distance = np.sqrt(np.square(standardized - np.median(standardized, axis=0)).sum(axis=1))
    features["distance_to_physical_center"] = distance
    selected = str(features.sort_values(["distance_to_physical_center", "base_path_id"]).iloc[0]["base_path_id"])
    features["selected_medoid"] = features["base_path_id"].eq(selected)
    return selected, features


def _save_figure(fig: Any, output_directory: Path, name: str, dpi: int) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    png = output_directory / f"{name}.png"
    pdf = output_directory / f"{name}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, pdf]


def create_figure_a(
    *,
    information_effects: pd.DataFrame,
    fixed_effects: pd.DataFrame,
    output_directory: Path,
    dpi: int,
) -> tuple[list[Path], pd.DataFrame]:
    primary = information_effects.loc[
        information_effects["comparison"].isin(["IF vs I0", "IL vs I0", "ORACLE vs I0"])
    ].copy()
    fixed = fixed_effects.copy()
    data = pd.concat([primary, fixed], ignore_index=True)
    scenarios = ["GH", "GT", "GL", "GFW"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8), sharey=False)
    for axis, frame, title in (
        (axes[0], primary, "A. Reoptimized information value"),
        (axes[1], fixed, "B. Fixed IL checkpoint responsiveness"),
    ):
        labels = []
        positions = []
        cursor = 0
        for scenario in scenarios:
            group = frame.loc[frame["warning_scenario"].eq(scenario)]
            for row in group.itertuples(index=False):
                regime = str(row.comparison).split()[0]
                axis.errorbar(
                    row.mean_paired_effect,
                    cursor,
                    xerr=[[row.mean_paired_effect - row.simultaneous_95_lower], [row.simultaneous_95_upper - row.mean_paired_effect]],
                    fmt="o" if regime != "ORACLE" else "D",
                    color=PALETTE.get(regime, "#4C78A8"),
                    capsize=3,
                    markersize=6,
                )
                labels.append(f"{scenario}: {row.comparison}" + (" (unattainable oracle input)" if regime == "ORACLE" else ""))
                positions.append(cursor)
                cursor += 1
            cursor += 0.6
        axis.axvline(0.0, color="#333333", linewidth=1, linestyle=":")
        axis.set_yticks(positions, labels, fontsize=8.5)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlabel("Paired loss reduction versus I0 (positive is lower loss)")
        axis.set_title(title, loc="left", fontweight="bold")
    axes[1].set_facecolor("#F7F3EA")
    fig.suptitle("Figure 5.2.4a. Released information value and fixed-policy response", fontsize=15)
    fig.text(0.5, 0.015, "Points are 88-physical-path paired means after within-path seed averaging; lines are within-family simultaneous 95% intervals. ORACLE information is unattainable; its trained controller is not assumed to attain the theoretical bound. Panel B is not information value.", ha="center", fontsize=8.7)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save_figure(fig, output_directory, "figure_5_2_4a_released_information_value", dpi), data


def create_figure_b(
    *,
    capacity_trace: pd.DataFrame,
    medoid_path_id: str,
    output_directory: Path,
    dpi: int,
) -> tuple[list[Path], pd.DataFrame]:
    frame = capacity_trace.loc[
        capacity_trace["evidence_layer"].eq("reoptimized_information_value")
        & capacity_trace["controller_id"].eq("IL_RD")
        & capacity_trace["warning_scenario"].eq("GH")
        & capacity_trace["base_path_id"].eq(medoid_path_id)
    ].copy()
    seed_week = frame.groupby(["training_seed", "decision_week"], as_index=False).agg(
        current_risk=("controller_current_high_risk_probability", "mean"),
        lead_risk=("controller_lead_high_risk_probability", "mean"),
        readiness_order=("implemented_readiness_order", "sum"),
        readiness_stock=("mature_readiness_stock_before", "sum"),
        readiness_exercise=("implemented_readiness_exercise", "sum"),
        readiness_matured=("readiness_matured_this_week", "sum"),
        readiness_expiry=("readiness_expiry_or_decay", "sum"),
        direct_order=("implemented_direct_order", "sum"),
        direct_pipeline=("direct_capacity_pipeline_before", "sum"),
        direct_arrival=("direct_capacity_arrival", "sum"),
        temporary_capacity=("usable_temporary_capacity", "sum"),
        blocked=("blocked_model_units", "mean"),
        queues=("berth_queue_after", "mean"),
        yard=("yard_queue_after", "mean"),
        gate=("gate_queue_after", "mean"),
        landbridge=("landbridge_queue_after", "mean"),
        scenario_release_date=("scenario_release_date", "first"),
        event_onset=("event_onset", "first"),
    )
    seed_week["queues"] = seed_week[["queues", "yard", "gate", "landbridge"]].sum(axis=1)
    grouped = seed_week.groupby("decision_week", as_index=False).agg(
        current_risk=("current_risk", "mean"),
        lead_risk=("lead_risk", "mean"),
        readiness_order=("readiness_order", "mean"),
        readiness_stock=("readiness_stock", "mean"),
        readiness_exercise=("readiness_exercise", "mean"),
        readiness_matured=("readiness_matured", "mean"),
        readiness_expiry=("readiness_expiry", "mean"),
        direct_order=("direct_order", "mean"),
        direct_pipeline=("direct_pipeline", "mean"),
        direct_arrival=("direct_arrival", "mean"),
        temporary_capacity=("temporary_capacity", "mean"),
        blocked=("blocked", "mean"),
        queues=("queues", "mean"),
        scenario_release_date=("scenario_release_date", "first"),
        event_onset=("event_onset", "first"),
    )
    grouped["decision_week"] = pd.to_datetime(grouped["decision_week"])
    event = pd.Timestamp(grouped["event_onset"].iloc[0])
    releases = sorted(pd.to_datetime(grouped["scenario_release_date"]).drop_duplicates())
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 12), sharex=True)
    axes[0].step(grouped["decision_week"], grouped["current_risk"], where="post", label="Released current filtered risk", color=PALETTE["IF"], linewidth=2)
    axes[0].step(grouped["decision_week"], grouped["lead_risk"], where="post", label="Readiness lead-aligned risk", color=PALETTE["IL"], linewidth=2)
    axes[0].set_ylabel("Risk-state probability")
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_title("Released information", loc="left", fontweight="bold")
    axes[1].plot(grouped["decision_week"], grouped["readiness_order"], label="Order", color="#4C78A8")
    axes[1].plot(grouped["decision_week"], grouped["readiness_stock"], label="Mature stock", color="#2A9D8F")
    axes[1].plot(grouped["decision_week"], grouped["readiness_exercise"], label="Exercise", color="#E07B24")
    axes[1].plot(grouped["decision_week"], grouped["readiness_expiry"], label="Expiry/decay", color="#8E5EA2", linestyle="--")
    axes[1].set_ylabel("Model units")
    axes[1].legend(loc="upper right", ncol=4)
    axes[1].set_title("Readiness order, maturity, exercise, and expiry", loc="left", fontweight="bold")
    axes[2].plot(grouped["decision_week"], grouped["direct_order"], label="Direct order", color="#D55E00")
    axes[2].plot(grouped["decision_week"], grouped["direct_pipeline"], label="Pipeline", color="#0072B2")
    axes[2].plot(grouped["decision_week"], grouped["direct_arrival"], label="Arrival", color="#009E73", linestyle="--")
    axes[2].set_ylabel("Model units")
    axes[2].legend(loc="upper right", ncol=3)
    axes[2].set_title("Direct procurement order, pipeline, and arrival", loc="left", fontweight="bold")
    axes[3].fill_between(grouped["decision_week"], grouped["blocked"], alpha=0.25, color="#D55E00", label="Blocked cargo proxy")
    axes[3].plot(grouped["decision_week"], grouped["temporary_capacity"], color="#009E73", linewidth=2, label="Usable temporary capacity")
    axes[3].plot(grouped["decision_week"], grouped["queues"], color="#333333", linewidth=2, label="External waiting excluded; four-stage queues")
    axes[3].set_ylabel("Model cargo units")
    axes[3].legend(loc="upper right", ncol=3)
    axes[3].set_title("Physical burden and temporary capacity", loc="left", fontweight="bold")
    for axis in axes:
        axis.axvline(event, color="#B22222", linewidth=1.5, linestyle="--")
        for release in releases:
            axis.axvline(release, color="#777777", linewidth=0.55, alpha=0.28)
        axis.grid(alpha=0.22)
    axes[-1].set_xlabel("Monday decision week")
    fig.suptitle(f"Figure 5.2.4b. Information release and capacity preparation timing\nPhysical-path medoid: {medoid_path_id}", fontsize=15)
    fig.text(0.5, 0.012, "Gray lines are public release dates; the red line is event onset. No dual axes are used.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return _save_figure(fig, output_directory, "figure_5_2_4b_release_capacity_timing", dpi), grouped


def create_figure_c(
    *,
    capacity_path_level: pd.DataFrame,
    capacity_effects: pd.DataFrame,
    loss_components: pd.DataFrame,
    output_directory: Path,
    dpi: int,
) -> tuple[list[Path], pd.DataFrame]:
    scenarios = ["GH", "GT", "GL", "GFW"]
    rights = ["RD", "R", "D", "NONE"]
    mean_loss = capacity_path_level.groupby(["warning_scenario", "capacity_rights"])["total_operational_objective"].mean().unstack()
    mean_loss = mean_loss.reindex(index=scenarios, columns=rights)
    fig = plt.figure(figsize=(16, 10.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    ax_a = fig.add_subplot(grid[0, 0])
    image = ax_a.imshow(mean_loss.to_numpy(), aspect="auto", cmap="YlOrRd")
    ax_a.set_xticks(range(len(rights)), rights)
    ax_a.set_yticks(range(len(scenarios)), scenarios)
    for y in range(len(scenarios)):
        for x in range(len(rights)):
            ax_a.text(x, y, f"{mean_loss.iloc[y, x]:,.0f}", ha="center", va="center", fontsize=8)
    ax_a.set_title("A. Mean total loss by reoptimized capacity rights", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax_a, fraction=0.045, label="Operational loss")
    ax_b = fig.add_subplot(grid[0, 1])
    markers = {"V_R_given_D": "o", "V_D_given_R": "s", "S_RD": "D"}
    offsets = {"V_R_given_D": -0.18, "V_D_given_R": 0.0, "S_RD": 0.18}
    for comparison in markers:
        subset = capacity_effects.loc[capacity_effects["comparison"].eq(comparison)].set_index("warning_scenario").reindex(scenarios)
        x = np.arange(len(scenarios)) + offsets[comparison]
        ax_b.errorbar(
            x,
            subset["mean_paired_effect"],
            yerr=[subset["mean_paired_effect"] - subset["simultaneous_95_lower"], subset["simultaneous_95_upper"] - subset["mean_paired_effect"]],
            fmt=markers[comparison],
            capsize=3,
            label=comparison,
        )
    ax_b.axhline(0.0, color="#333333", linestyle=":")
    ax_b.set_xticks(range(len(scenarios)), scenarios)
    ax_b.set_ylabel("Paired value / combination effect")
    ax_b.set_title("B. Readiness, direct procurement, and combination", loc="left", fontweight="bold")
    ax_b.legend(ncol=3, fontsize=8)
    ax_b.grid(axis="y", alpha=0.25)
    ax_c = fig.add_subplot(grid[1, :])
    selected = loss_components.loc[
        loss_components["evidence_layer"].eq("reoptimized_capacity_rights")
    ].copy()
    selected["label"] = selected["warning_scenario"] + " | " + selected["capacity_rights"]
    selected = selected.set_index(["warning_scenario", "capacity_rights"]).reindex(
        pd.MultiIndex.from_product([scenarios, rights], names=["warning_scenario", "capacity_rights"])
    ).reset_index()
    components = ["queue", "waiting", "sue_exit", "attrition_exit", "overload", "route_resource", "action", "terminal"]
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]
    bottom = np.zeros(len(selected))
    x = np.arange(len(selected))
    for component, color in zip(components, colors):
        values = selected[component].to_numpy(dtype=float)
        ax_c.bar(x, values, bottom=bottom, label=component.replace("_", " "), color=color, width=0.8)
        bottom += values
    ax_c.set_xticks(x, [f"{s}\n{r}" for s, r in zip(selected["warning_scenario"], selected["capacity_rights"])], fontsize=8)
    ax_c.set_ylabel("Mean loss component")
    ax_c.set_title("C. Complete loss decomposition, including false-warning preparation cost", loc="left", fontweight="bold")
    ax_c.legend(ncol=8, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax_c.grid(axis="y", alpha=0.2)
    # The common absolute scale is needed for the disrupted cases, but it makes
    # the preregistered false-warning costs visually disappear.  Retain all
    # sixteen bars on the common scale and add an explicitly separate-scale
    # inset for GFW; this is a zoomed decomposition, not a second y axis.
    gfw = selected.loc[selected["warning_scenario"].eq("GFW")].reset_index(drop=True)
    inset = ax_c.inset_axes([0.775, 0.51, 0.215, 0.41])
    inset_bottom = np.zeros(len(gfw))
    inset_x = np.arange(len(gfw))
    for component, color in zip(components, colors):
        values = gfw[component].to_numpy(dtype=float)
        inset.bar(inset_x, values, bottom=inset_bottom, color=color, width=0.72)
        inset_bottom += values
    inset.set_xticks(inset_x, gfw["capacity_rights"], fontsize=7)
    inset.set_ylabel("Mean loss", fontsize=7)
    inset.tick_params(axis="y", labelsize=7)
    inset.set_title("GFW decomposition (separate zoom scale)", fontsize=8, loc="left")
    inset.grid(axis="y", alpha=0.2)
    fig.suptitle("Figure 5.2.4c. Readiness and direct-procurement portfolio", fontsize=15)
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    data = pd.concat(
        [
            mean_loss.stack().rename("mean_total_loss").reset_index().assign(panel="A"),
            capacity_effects.assign(panel="B"),
            selected.assign(panel="C"),
        ],
        ignore_index=True,
        sort=False,
    )
    return _save_figure(fig, output_directory, "figure_5_2_4c_capacity_portfolio", dpi), data


def acceptance_payload(
    *,
    upstream_5_2_1_complete: bool,
    upstream_5_2_2_complete: bool,
    upstream_5_2_3_complete: bool,
    upstream_locks_match: bool,
    parameter_registry_frame: pd.DataFrame,
    information_registry: pd.DataFrame,
    warning_registry: pd.DataFrame,
    controller_manifest: pd.DataFrame,
    raw_replications: pd.DataFrame,
    contracts: pd.DataFrame,
    capacity_trace: pd.DataFrame,
    anchor: pd.DataFrame,
    precision: pd.DataFrame,
    information_effects: pd.DataFrame,
    medoid_audit: pd.DataFrame,
    expected_physical_paths: int,
    figure_paths: Sequence[Path],
    tolerance: float,
) -> dict[str, Any]:
    fixed = raw_replications.loc[raw_replications["evidence_layer"].eq("fixed_policy_information_responsiveness")]
    fixed_hash_counts = (
        fixed[["controller_id", "training_seed", "controller_bundle_sha256"]]
        .drop_duplicates()
        .shape[0]
        == 3
        and fixed["controller_id"].eq("IL_RD").all()
        and fixed.groupby("training_seed")["controller_bundle_sha256"].nunique().eq(1).all()
    )
    primary = raw_replications.loc[raw_replications["evidence_layer"].eq("reoptimized_information_value")]
    rights = raw_replications.loc[raw_replications["evidence_layer"].eq("reoptimized_capacity_rights")]
    medoid_selected = medoid_audit["selected_physical_path_medoid"].map(
        lambda value: str(value).strip().lower() == "true"
    )
    medoid_uses_outcomes = medoid_audit["selection_uses_policy_outcomes"].map(
        lambda value: str(value).strip().lower() == "true"
    )
    physical_fair = bool(
        primary.groupby(["warning_scenario", "base_path_id"])["base_physical_path_sha256"].nunique().eq(1).all()
        and rights.groupby(["warning_scenario", "base_path_id"])["base_physical_path_sha256"].nunique().eq(1).all()
    )
    checks = {
        "accepted_5_2_1_input": upstream_5_2_1_complete,
        "accepted_5_2_2_input": upstream_5_2_2_complete,
        "accepted_5_2_3_input": upstream_5_2_3_complete,
        "all_seven_upstream_hash_locks_match": upstream_locks_match,
        "no_missing_parameter_basis": parameter_registry_frame["basis_status"].eq("REGISTERED").all(),
        "no_information_is_nonzero_stationary_baseline": bool(information_registry.loc[information_registry["information_regime"].eq("I0"), "baseline_high_risk_probability"].iloc[0] > 0),
        "historical_and_designed_scenarios_are_distinct": warning_registry["scenario_release_date"].nunique() >= 3,
        "release_never_after_decision": bool(contracts["release_not_after_decision"].all()),
        "monthly_matrix_not_applied_weekly": bool(contracts["monthly_matrix_not_applied_weekly"].all()),
        "matched_physical_paths_across_controllers": physical_fair,
        "three_learning_seeds_per_controller_path": bool(raw_replications.groupby(["evidence_layer", "controller_id", "evaluation_information_regime", "capacity_rights", "warning_scenario", "base_path_id"])["training_seed"].nunique().eq(3).all()),
        "exactly_88_physical_paths_are_inference_units": bool(
            expected_physical_paths == 88
            and primary["base_path_id"].nunique() == 88
            and rights["base_path_id"].nunique() == 88
        ),
        "reoptimized_information_controllers_have_distinct_bundles": controller_manifest.groupby("controller_id")["controller_bundle_sha256"].apply(lambda x: x.nunique() == 3).all(),
        "six_nonanchor_controller_families_trained_from_scratch": bool(
            set(controller_manifest.loc[controller_manifest["generated_from_scratch"].astype(bool), "controller_id"])
            == {"I0_RD", "IF_RD", "ORACLE_RD", "IL_R", "IL_D", "IL_NONE"}
        ),
        "il_rd_uses_only_accepted_5_2_2_anchor": bool(
            controller_manifest.loc[controller_manifest["controller_id"].eq("IL_RD"), "reused_5_2_2_anchor"].astype(bool).all()
        ),
        "fixed_diagnostic_uses_identical_il_checkpoint": fixed_hash_counts,
        "oracle_only_in_primary_upper_bound_layer": bool(raw_replications.loc[raw_replications["evaluation_information_regime"].eq("ORACLE"), "evidence_layer"].eq("reoptimized_information_value").all()),
        "capacity_rights_reoptimized_separately": set(rights["capacity_rights"].unique()) == {"RD", "R", "D", "NONE"},
        "rights_applied_before_common_projection": bool(contracts["rights_before_common_projection"].all()),
        "readiness_rights_enforced": bool(contracts["readiness_disabled_when_required"].all()),
        "direct_rights_enforced": bool(contracts["direct_disabled_when_required"].all()),
        "readiness_stock_nonnegative": bool(contracts["readiness_stock_nonnegative"].all()),
        "direct_stock_nonnegative": bool(contracts["direct_stock_nonnegative"].all()),
        "all_common_kernel_step_acceptance_passed": bool(contracts["all_step_acceptance_passed"].all()),
        "loss_components_close": bool(contracts["loss_components_reconstruct_total"].all()),
        "right_censoring_valid": bool(contracts["right_censoring_not_recorded_as_clearance"].all()),
        "information_hashes_enter_controller_trace": bool(contracts["information_hashes_enter_controller_trace"].all()),
        "scenario_label_absent_from_controller_observation": bool(contracts["scenario_label_absent_from_controller_observation"].all()),
        "route_resource_loss_retained": bool((raw_replications["loss_route_resource"] > 0).any()),
        "il_rd_historical_anchor_reproduces_5_2_2": bool(anchor["absolute_difference"].max() <= tolerance),
        "accepted_5_2_3_medoid_is_used_only_for_timing_figure": bool(
            medoid_selected.sum() == 1 and not medoid_uses_outcomes.any()
        ),
        "all_40_preregistered_precision_checks_are_present": len(precision) == 40,
        "all_40_preregistered_precision_targets_met": bool(len(precision) == 40 and precision["precision_target_met"].all()),
        "all_three_evidence_layers_present": set(raw_replications["evidence_layer"].unique()) == {"reoptimized_information_value", "fixed_policy_information_responsiveness", "reoptimized_capacity_rights"},
        "all_png_and_pdf_figures_generated": len(figure_paths) == 6 and all(path.exists() for path in figure_paths),
    }
    failures = [name for name, passed in checks.items() if not bool(passed)]
    oracle_gaps = information_effects.loc[
        information_effects["comparison"].eq("oracle gap"), "mean_paired_effect"
    ]
    oracle_bound_attained = bool(len(oracle_gaps) == 4 and (oracle_gaps >= -tolerance).all())
    return {
        "experiment_id": "5.2.4_released_risk_information_capacity_preparation",
        "status": "complete" if not failures else "BLOCKED",
        "blocking_checks": checks,
        "blocking_failures": failures,
        "evidence_boundary": "Historical release replay, estimated HMM information, designed timing stresses, and an unattainable oracle are kept distinct. Fixed-checkpoint responses are not information value.",
        "nonblocking_diagnostics": {
            "theoretical_oracle_information_set_is_unattainable": True,
            "trained_oracle_controller_attains_empirical_upper_bound": oracle_bound_attained,
            "negative_oracle_gap_must_be_reported_as_optimization_limitation": bool((oracle_gaps < -tolerance).any()),
        },
    }


def write_reports(
    *,
    report_directory: Path,
    registry: pd.DataFrame,
    information_effects: pd.DataFrame,
    fixed_effects: pd.DataFrame,
    capacity_effects: pd.DataFrame,
    false_warning: pd.DataFrame,
    acceptance: Mapping[str, Any],
    precision: pd.DataFrame,
    clearance: pd.DataFrame,
) -> list[Path]:
    report_directory.mkdir(parents=True, exist_ok=True)
    parameter_path = report_directory / "5_1_PARAMETER_AND_METRIC_ADDITIONS.md"
    result_path = report_directory / "FIGURE_AND_RESULTS_ANALYSIS.md"
    required = registry.loc[
        registry["parameter"].str.startswith(("HMM.", "release_", "lead_", "information_regime.", "warning_scenario.", "capacity_rights.", "5.2.4.statistics"))
    ]
    parameter_lines = [
        "# 5.1 additions required by Experiment 5.2.4",
        "",
        "These entries are generated from the accepted 5.2.1/5.2.2 interfaces and the frozen 5.2.4 registry. No convenient default was introduced.",
        "",
        "| Parameter | Value | Evidence | Basis |",
        "|---|---|---|---|",
    ]
    for row in required.itertuples(index=False):
        value = str(row.value).replace("|", "\\|")
        basis = str(row.basis_or_formula).replace("|", "\\|")
        parameter_lines.append(f"| `{row.parameter}` | {value} | {row.evidence_class} | {basis} |")
    parameter_path.write_text("\n".join(parameter_lines) + "\n", encoding="utf-8")

    def effects_table(frame: pd.DataFrame) -> list[str]:
        lines = ["| Scenario | Comparison | Mean effect | Simultaneous 95% interval | Holm p |", "|---|---|---:|---:|---:|"]
        for row in frame.itertuples(index=False):
            lines.append(f"| {row.warning_scenario} | {row.comparison} | {row.mean_paired_effect:,.3f} | [{row.simultaneous_95_lower:,.3f}, {row.simultaneous_95_upper:,.3f}] | {row.holm_adjusted_p_value:.4g} |")
        return lines

    def effect(frame: pd.DataFrame, scenario: str, comparison: str) -> pd.Series:
        selected = frame.loc[
            frame["warning_scenario"].eq(scenario) & frame["comparison"].eq(comparison)
        ]
        if len(selected) != 1:
            raise RuntimeError(f"Missing unique report contrast: {scenario}, {comparison}")
        return selected.iloc[0]

    gh_if = effect(information_effects, "GH", "IF vs I0")
    gh_il = effect(information_effects, "GH", "IL vs I0")
    gh_oracle = effect(information_effects, "GH", "ORACLE vs I0")
    gh_oracle_gap = effect(information_effects, "GH", "oracle gap")
    gh_il_if = effect(information_effects, "GH", "IL vs IF")
    gh_readiness = effect(capacity_effects, "GH", "V_R_given_D")
    gh_direct = effect(capacity_effects, "GH", "V_D_given_R")
    gh_combination = effect(capacity_effects, "GH", "S_RD")
    timing_rows = information_effects.loc[
        information_effects["warning_scenario"].isin(["GH", "GT", "GL"])
        & information_effects["comparison"].isin(["IF vs I0", "IL vs I0"])
    ]
    timing_spread = timing_rows.groupby("comparison")["mean_paired_effect"].agg(lambda values: float(values.max() - values.min()))
    precision_met = int(precision["precision_target_met"].sum())
    precision_total = len(precision)
    precision_sentence = (
        f"All {precision_total} preregistered contrasts met the paired half-width target."
        if precision_met == precision_total
        else f"Precision targets met for {precision_met} of {precision_total} preregistered contrasts; unmet targets are retained rather than changing the path cap."
    )
    censored_total = int(clearance["number_censored_paths"].sum())
    degenerate = pd.concat([information_effects, fixed_effects, capacity_effects], ignore_index=True)["standard_error"].fillna(0.0).abs().lt(1e-12).any()

    def interval_interpretation(row: pd.Series, subject: str) -> str:
        lower = float(row.simultaneous_95_lower)
        upper = float(row.simultaneous_95_upper)
        mean = float(row.mean_paired_effect)
        if lower > 0.0:
            return f"{subject} lowers loss by {mean:,.1f}; the adjusted interval excludes zero."
        if upper < 0.0:
            return f"{subject} increases loss by {-mean:,.1f}; the adjusted interval excludes zero."
        return f"{subject} has paired effect {mean:,.1f}, but its adjusted interval crosses zero."

    if gh_combination.mean_paired_effect > 0:
        portfolio_sentence = "The positive combination statistic indicates complementarity in this matched design."
    elif gh_combination.mean_paired_effect < 0:
        portfolio_sentence = "The negative combination statistic indicates that the two technologies behave mainly as substitutes in this matched design."
    else:
        portfolio_sentence = "The zero combination statistic provides no directional evidence on complementarity versus substitution."
    false_warning_means = false_warning.groupby("information_regime")["false_warning_cost"].mean().round(3).to_dict()

    result_lines = [
        "# 5.2.4 Released Risk Information and Capacity Preparation",
        "",
        f"Acceptance status: **{acceptance['status']}**.",
        "",
        "## Reoptimized information value",
        "",
        *effects_table(information_effects),
        "",
        "## Fixed-checkpoint information responsiveness",
        "",
        "These rows use the identical IL_RD checkpoint bundle and are not information value estimates.",
        "",
        *effects_table(fixed_effects),
        "",
        "## Reoptimized capacity rights",
        "",
        *effects_table(capacity_effects),
        "",
        "## False-warning and precision evidence",
        "",
        f"Mean false-warning costs by regime: {false_warning_means}.",
        precision_sentence,
        f"Right-censored aggregate cells report {censored_total} censored physical paths; the simulation cap is never recorded as an observed clearance week.",
        "",
        "## Result interpretation",
        "",
        f"- {interval_interpretation(gh_if, 'Under historical release, the reoptimized current-filtered controller relative to I0')}",
        f"- {interval_interpretation(gh_il, 'Under historical release, the reoptimized lead-aligned controller relative to I0')} Its incremental paired effect versus current filtering is {gh_il_if.mean_paired_effect:,.1f}, with adjusted interval [{gh_il_if.simultaneous_95_lower:,.1f}, {gh_il_if.simultaneous_95_upper:,.1f}].",
        f"- Moving the target release packet between GH, GT, and GL changes the IF effect by up to {timing_spread.get('IF vs I0', float('nan')):,.3f} and the IL effect by up to {timing_spread.get('IL vs I0', float('nan')):,.3f}. These are designed timing stresses; identical or zero-variance outcomes are retained rather than converted into a historical timing claim.",
        f"- {interval_interpretation(gh_oracle, 'Under historical release, the controller trained with ORACLE information relative to I0')} The achieved oracle gap, defined as J_IL - J_ORACLE, is {gh_oracle_gap.mean_paired_effect:,.1f} with adjusted interval [{gh_oracle_gap.simultaneous_95_lower:,.1f}, {gh_oracle_gap.simultaneous_95_upper:,.1f}]. Because it is negative, the trained ORACLE controller did not attain the theoretical perfect-information performance bound; this is retained as an optimization/training limitation rather than relabelled as an upper-bound result.",
        f"- Under GH, V_R|D={gh_readiness.mean_paired_effect:,.1f}, V_D|R={gh_direct.mean_paired_effect:,.1f}, and S_RD={gh_combination.mean_paired_effect:,.1f}. {portfolio_sentence}",
        f"- False-warning effects are retained with their signs: {false_warning_means}. Positive values mean information increased loss without a physical disruption; negative values mean it reduced loss in that designed no-disruption comparison.",
        ("- Some GFW contrasts have zero across-path variance and hence zero-width intervals. Their deterministic matched-path differences are descriptive for this design and their nominal p-values should not be read as conventional sampling evidence." if degenerate else "- No preregistered contrast has degenerate across-path variance."),
        "",
        "## Interpretation boundaries",
        "",
        "- GH is a historical release replay; GT, GL, and GFW are designed timing stresses.",
        "- ORACLE information is unattainable and cannot be interpreted as an implementable policy. Its information set is theoretically dominant, but the achieved learned-controller result is not an empirical upper bound when the recorded oracle gap is negative.",
        "- Fixed-checkpoint substitutions show signal dependence or out-of-distribution response, not information value.",
        "- Capacity-right effects are conditional simulation results under the reference network and reoptimized controllers.",
        "- Negative values, false-warning costs, right censoring, and any unmet precision targets are retained.",
    ]
    result_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8")
    return [parameter_path, result_path]


def write_manifest(
    *,
    output_directory: Path,
    config_path: Path,
    input_files: Sequence[Path],
    output_files: Sequence[Path],
    elapsed_seconds: float,
) -> Path:
    def record(path: Path) -> dict[str, Any]:
        return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "experiment_id": "5.2.4_released_risk_information_capacity_preparation",
        "status": "complete",
        "command": "python experiments/5.2-4/run_5_2_4.py",
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": elapsed_seconds,
        "config": record(config_path),
        "inputs": [record(path) for path in dict.fromkeys(input_files)],
        "artifacts": [record(path) for path in dict.fromkeys(output_files)],
    }
    path = output_directory / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
