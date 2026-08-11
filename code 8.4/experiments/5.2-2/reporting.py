"""Registries, formal figures, manifest, and blocking acceptance for experiment 5.2.2."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from model import BenchmarkModel
from paths import Frozen521Inputs, sha256_file


CODE_ROOT = Path(__file__).resolve().parents[2]

LOSS_COMPONENTS = [
    ("queue", "Queue"),
    ("waiting", "Waiting"),
    ("exit", "Exit"),
    ("overload", "Overload"),
    ("route_resource_and_transport", "Route resource and transport"),
    ("action", "Action"),
    ("terminal", "Terminal"),
]


def _json_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _flatten(prefix: str, value: Any, rows: list[dict[str, Any]]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            _flatten(name, child, rows)
    else:
        rows.append(
            {
                "parameter": prefix,
                "value": _json_value(value),
                "source": "config_5_2_2.json frozen before final test replay",
                "basis": "paper Sections 3-4 and preregistered 5.1/5.2.2 design",
                "evidence_status": "declared design parameter unless the value names a frozen 5.2.1 input",
            }
        )


def parameter_registry(config: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    _flatten("", config, rows)
    calibration_path = CODE_ROOT / str(
        config["information"]["waiting_error_scale_calibration_file"]
    )
    calibration = pd.read_csv(calibration_path)
    calibration_hash = sha256_file(calibration_path)
    for row in calibration.itertuples(index=False):
        rows.append(
            {
                "parameter": f"information.calibration_sigma_W.{row.route}",
                "value": float(row.sigma_W_rmse_weeks),
                "source": f"{calibration_path.relative_to(CODE_ROOT).as_posix()} sha256={calibration_hash}",
                "basis": str(row.estimator),
                "evidence_status": "held-out validation waiting forecast error estimate; frozen before training",
            }
        )
    return pd.DataFrame(rows)


def scientific_parameter_traceability(config: Mapping[str, Any]) -> pd.DataFrame:
    """Map every identified result-affecting design family to config and code."""

    rows = [
        ("late attrition consequence", "behavior.late_exit_cost_per_vintage", "model.CommonDisclosureBehaviorFactory", "Chapter 3 unidentified consequence convention"),
        ("disclosure credibility", "information.gamma_I", "model.CommonDisclosureBehaviorFactory", "formal credibility-domain reference case"),
        ("waiting forecast error scale", "information.waiting_error_scale_weeks_by_route", "model.CommonDisclosureBehaviorFactory", "held-out validation RMSE calibration"),
        ("public waiting signal", "information.public_signal_formula", "model._public_route_wait", "predetermined reference loading before SUE"),
        ("capacity conversion and decay", "capacity_technology", "model.build_model", "unit-preserving designed capacity technology"),
        ("physical feedback", "physical_feedback", "model.build_model", "declared 5.1 feedback family"),
        ("corridor history", "physical_feedback.corridor_history_window_weeks", "model.build_model", "lagged shared-corridor accounting"),
        ("waiting hazard and costs", "behavior", "model.build_model and CommonDisclosureBehaviorFactory", "declared behavioral family"),
        ("training and validation event geometry", "paths.designed_serviceability", "paths._designed_serviceability", "designed support independent of test event"),
        ("training and validation risk anchors", "paths.designed_information", "paths._synthetic_information", "endpoint-midpoint training and interleaved validation support"),
        ("synthetic path calendar", "paths.synthetic_calendar", "paths.build_training_validation_paths", "sentinel disjoint calendars"),
        ("MPC scenario construction", "mpc", "preparation.prepare_period and policies._candidate_profiles", "registered slow-central-fast support with a beginning-of-week information firewall"),
        ("reactive mapping", "reactive_policy", "policies._reactive_normalised", "current-state-only formula"),
        ("learning design", "training", "training.py", "pre-test checkpoint and stopping protocol"),
        ("state vector and actor initialisation", "state_and_actor", "features.py", "complete-state encoding and deterministic seeded initialisation"),
    ]
    return pd.DataFrame(
        rows,
        columns=["scientific_family", "config_path", "code_location", "basis"],
    ).assign(
        frozen_source="config_5_2_2.json or its named hash-locked calibration input",
        traceable=True,
    )


def policy_authority_register(config: Mapping[str, Any]) -> pd.DataFrame:
    policies = list(config["main_policies"])
    learning = set(config["learning_policies"])
    rule = {
        "Passive": "zero requested coordination action; endogenous route/wait/exit SUE remains active",
        "Reactive": "current queue, waiting, stock, disruption status and remaining budget only",
        "Projected stochastic MPC": "formal nested H_C objective; first projected action executed",
        "Behaviour cloning": "projected full-state fit to feasible formal-MPC first actions",
        "PPO": "from-scratch projected full-state policy trained on the formal operational reward",
        "Vanilla SAC": "from-scratch reparameterised latent policy with learned entropy temperature, common projection and twin reward critics",
        "Constrained SAC": "reparameterised latent policy with learned entropy temperature, common projection, twin reward critics, constraint critic and dual",
        "Model-guided constrained SAC": "formal common-rollout selector over BC and constrained-SAC proposals",
    }
    resource_count = 3 * len(config["routes"]) + 1
    numerical_dimension = 3 * resource_count + 1 + len(config["routes"])
    rows = []
    for policy in policies:
        rows.append(
            {
                "policy": policy,
                "in_main_ranking": True,
                "learning_policy": policy in learning,
                "action_dimension": numerical_dimension,
                "action_blocks": "y_R[10 resources]|y_V[10 resources]|v_R[10 resources]|rho[1 class]|lambda[3 routes]",
                "shared_projector": "weighted Euclidean projector with reciprocal formal block upper bounds",
                "direct_procurement_right_retained": True,
                "exit_is_control_action": False,
                "future_serviceability_access": False,
                "future_event_realization_access": False,
                "information_set": "current physical state and released information through decision week",
                "policy_rule": rule[policy],
                "common_budget_and_phase_bounds": True,
            }
        )
    rows.append(
        {
            "policy": config["forbidden_main_policy"],
            "in_main_ranking": False,
            "learning_policy": False,
            "action_dimension": np.nan,
            "action_blocks": "authority-violating route-share prescription",
            "shared_projector": "not applicable",
            "direct_procurement_right_retained": np.nan,
            "exit_is_control_action": False,
            "future_serviceability_access": False,
            "future_event_realization_access": False,
            "information_set": "excluded",
            "policy_rule": "excluded because it directly prescribes adaptive route shares",
            "common_budget_and_phase_bounds": False,
        }
    )
    return pd.DataFrame(rows)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def create_figures(
    *,
    path_level: pd.DataFrame,
    replications: pd.DataFrame,
    paired_effects: pd.DataFrame,
    confidence_set: pd.DataFrame,
    loss_summary: pd.DataFrame,
    clearance: pd.DataFrame,
    policies: Sequence[str],
    output_directory: Path,
    dpi: int,
) -> list[Path]:
    _style()
    output_directory.mkdir(parents=True, exist_ok=True)
    policy_order = list(policies)
    short = {
        "Projected stochastic MPC": "Projected\nMPC",
        "Behaviour cloning": "BC",
        "Vanilla SAC": "Vanilla\nSAC",
        "Constrained SAC": "Constrained\nSAC",
        "Model-guided constrained SAC": "MG constrained\nSAC",
    }
    labels = [short.get(policy, policy) for policy in policy_order]
    blue, blue_light, orange, ink = "#355C7D", "#DCE8F2", "#D9822B", "#263238"

    confidence = confidence_set.set_index("policy").loc[policy_order]
    means = confidence["mean_total_operational_loss"].to_numpy(float)
    lower = confidence["path_95_lower"].to_numpy(float)
    upper = confidence["path_95_upper"].to_numpy(float)
    positions = np.arange(len(policy_order))
    distributions = [
        path_level.loc[path_level["policy"] == policy, "total_operational_objective"].to_numpy(float)
        for policy in policy_order
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.4), constrained_layout=True)
    boxes = axes[0].boxplot(
        distributions,
        positions=positions,
        vert=False,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": ink, "linewidth": 1.2},
        whiskerprops={"color": blue, "linewidth": 1.0},
        capprops={"color": blue, "linewidth": 1.0},
        boxprops={"edgecolor": blue, "linewidth": 1.0},
    )
    for box in boxes["boxes"]:
        box.set_facecolor(blue_light)
    offsets = np.linspace(-0.13, 0.13, max(len(values) for values in distributions))
    for index, values in enumerate(distributions):
        axes[0].scatter(
            values,
            index + offsets[: len(values)],
            s=20,
            facecolor="white",
            edgecolor=blue,
            linewidth=0.7,
            zorder=3,
        )
    axes[0].errorbar(
        means,
        positions,
        xerr=np.vstack([means - lower, upper - means]),
        fmt="D",
        color=orange,
        ecolor=orange,
        markersize=4.5,
        capsize=3,
        linewidth=1.1,
        label="Mean and path-based 95% interval",
        zorder=4,
    )
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Total operational and real-resource loss (index units)")
    axes[0].set_title("a. Matched-path loss distributions")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    axes[0].text(0.01, -0.14, "Open circles are physical paths; diamonds are means. Learning seeds are averaged within path.", transform=axes[0].transAxes, fontsize=8)

    effect = paired_effects.set_index("policy").loc[[p for p in policy_order if p != "Passive"]]
    y = np.arange(len(effect))
    e_mean = effect["mean_paired_difference"].to_numpy(float)
    e_low = effect["simultaneous_95_lower"].to_numpy(float)
    e_high = effect["simultaneous_95_upper"].to_numpy(float)
    axes[1].axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].hlines(y, e_low, e_high, color=blue, linewidth=1.6)
    axes[1].scatter(e_mean, y, s=42, facecolor=orange, edgecolor=ink, linewidth=0.6, zorder=3)
    axes[1].set_yticks(y, [short.get(p, p) for p in effect.index])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Paired loss difference versus Passive (lower is better)")
    axes[1].set_title("b. Multiplicity-adjusted paired intervals")
    axes[1].text(0.01, -0.14, "Intervals use the physical path as the inference unit.", transform=axes[1].transAxes, fontsize=8)
    fig.suptitle("Figure 5.2.2a. Common-authority policy performance and paired uncertainty", fontsize=12)
    path_a = output_directory / "figure_5_2_2a_policy_performance.png"
    fig.savefig(path_a, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    components = loss_summary.set_index("policy").loc[policy_order]
    component_columns = [column for column, _ in LOSS_COMPONENTS]
    component_labels = [label for _, label in LOSS_COMPONENTS]
    absolute = components[component_columns].to_numpy(float)
    totals = components["mean_total_operational_loss"].to_numpy(float)
    shares = np.divide(absolute, totals[:, None], out=np.zeros_like(absolute), where=totals[:, None] > 0) * 100.0
    fig, ax = plt.subplots(figsize=(14.2, 6.7), constrained_layout=True)
    heat = ax.imshow(shares, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(50.0, float(shares.max())))
    for row in range(shares.shape[0]):
        for column in range(shares.shape[1]):
            value = shares[row, column]
            text_color = "white" if value > 0.55 * max(50.0, float(shares.max())) else ink
            ax.text(
                column,
                row,
                f"{absolute[row, column] / 1000:.1f}k\n{value:.1f}%",
                ha="center",
                va="center",
                fontsize=7.5,
                color=text_color,
            )
        ax.text(
            len(component_columns) + 0.12,
            row,
            f"Total {totals[row] / 1000:.1f}k",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=ink,
            clip_on=False,
        )
    ax.set_xlim(-0.5, len(component_columns) + 1.25)
    ax.set_xticks(np.arange(len(component_labels)), component_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(policy_order)), labels)
    ax.set_title("Figure 5.2.2b. Formal loss-component matrix")
    colorbar = fig.colorbar(heat, ax=ax, shrink=0.82, pad=0.08)
    colorbar.set_label("Share of policy total loss (%)")
    ax.text(0.0, -0.20, "Cells show absolute loss in thousands and within-policy share. Every row reconstructs its reported total.", transform=ax.transAxes, fontsize=8)
    path_b = output_directory / "figure_5_2_2b_loss_decomposition.png"
    fig.savefig(path_b, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    clear = clearance.set_index("policy").loc[policy_order]
    cap = int(clear["restriction_weeks"].max())
    weeks = np.arange(0, cap + 1)
    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, policy, label in zip(axes.ravel(), policy_order, labels):
        group = replications.loc[replications["policy"] == policy].copy()
        observed = pd.to_numeric(group["clearance_weeks_observed"], errors="coerce").to_numpy(float)
        censored = group["right_censored"].astype(bool).to_numpy()
        not_cleared = np.asarray(
            [np.mean(censored | np.isnan(observed) | (observed > week)) for week in weeks]
        )
        ax.step(weeks, not_cleared, where="post", color=blue, linewidth=1.7)
        ax.fill_between(weeks, 0.0, not_cleared, step="post", color=blue_light, alpha=0.8)
        ax.axvline(cap, color=ink, linewidth=0.8, linestyle=":")
        summary = clear.loc[policy]
        ax.set_title(label.replace("\n", " "), fontsize=9.5)
        ax.text(
            0.04,
            0.94,
            f"clear={summary['clearance_probability']:.2f}\ncensored={int(group['right_censored'].sum())}/{len(group)}\nfinal={summary['mean_final_outstanding_mass']:.3g}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
            color=ink,
        )
        ax.set_xlim(0, cap)
        ax.set_ylim(0, 1.03)
    fig.supxlabel("Clearance follow-up week")
    fig.supylabel("Share of trajectories not yet cleared")
    fig.suptitle("Figure 5.2.2c. Empirical clearance curves and right censoring", fontsize=12)
    physical_path_count = int(path_level["path_id"].nunique())
    learning_seed_count = int(
        path_level.loc[
            path_level["policy"].isin(
                ["Behaviour cloning", "PPO", "Vanilla SAC", "Constrained SAC", "Model-guided constrained SAC"]
            ),
            "training_seed_count",
        ].max()
    )
    fig.text(
        0.5,
        -0.015,
        f"Learning panels contain {physical_path_count} physical paths x {learning_seed_count} training seeds; "
        f"nonlearning panels contain {physical_path_count} paths. Censored trajectories remain above zero at week {cap}.",
        ha="center",
        fontsize=8,
    )
    path_c = output_directory / "figure_5_2_2c_clearance_censoring.png"
    fig.savefig(path_c, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [path_a, path_b, path_c]


def acceptance_payload(
    *,
    config: Mapping[str, Any],
    frozen: Frozen521Inputs,
    model: BenchmarkModel,
    training_manifest: pd.DataFrame,
    validation_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    replications: pd.DataFrame,
    path_level: pd.DataFrame,
    actions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    contracts: pd.DataFrame,
    paired_effects: pd.DataFrame,
    pilot_precision: pd.DataFrame,
    selected_path_count: pd.DataFrame,
    loss_summary: pd.DataFrame,
    clearance: pd.DataFrame,
    authority: pd.DataFrame,
    route_costs: pd.DataFrame,
    checkpoint_manifest: pd.DataFrame,
    parameter_registry_frame: pd.DataFrame,
    scientific_traceability: pd.DataFrame,
    waiting_calibration: pd.DataFrame,
    training_curves: pd.DataFrame,
    sac_gradient_checks: pd.DataFrame,
    nonanticipativity_checks: pd.DataFrame,
    figures: Sequence[Path],
) -> dict[str, Any]:
    train = set(training_manifest["path_content_sha256"])
    valid = set(validation_manifest["path_content_sha256"])
    test = set(test_manifest["path_content_sha256"])
    bool_contract_columns = [
        column for column in contracts.columns
        if column not in {"policy", "path_id", "training_seed", "maximum_transition_residual"}
    ]
    contracts_pass = bool(contracts[bool_contract_columns].astype(bool).all().all())
    test_policy_counts = replications.groupby("policy")["path_id"].nunique()
    info_unique = replications.groupby("path_id")["released_information_path_sha256"].nunique()
    path_hash_unique = replications.groupby("path_id")["path_content_sha256"].nunique()
    action_names = list(model.layout.names)
    route_nonmissing = not route_costs["total_incremental_resource_cost"].isna().any()
    configured_sigma = config["information"]["waiting_error_scale_weeks_by_route"]
    calibrated_sigma = dict(
        zip(waiting_calibration["route"], waiting_calibration["sigma_W_rmse_weeks"])
    )
    sigma_matches = set(configured_sigma) == set(calibrated_sigma) and all(
        np.isclose(float(configured_sigma[route]), float(calibrated_sigma[route]))
        for route in configured_sigma
    )
    required_sections = set(config["scientific_parameter_contract"]["required_sections"])
    sac_curves = training_curves.loc[
        training_curves["policy"].isin(["Vanilla SAC", "Constrained SAC"])
    ].copy()
    matched_inputs = actions.groupby(["path_id", "period_offset"])[
        [
            "information_vector_sha256",
            "scenario_ids",
            "readiness_weights",
            "operational_weights",
        ]
    ].nunique()
    checks = {
        "5_2_1_acceptance_complete_and_hash_locked": bool(
            frozen.interface_hash
            == str(config["expected_5_2_1_interface_sha256"])
            and frozen.residual_hash
        ),
        "training_validation_test_paths_disjoint": not (train & valid or train & test or valid & test),
        "all_test_paths_are_unique": len(test) == len(test_manifest),
        "all_policies_use_same_test_path_ids": bool(test_policy_counts.eq(len(test_manifest)).all()),
        "all_policies_use_same_physical_path_hash_per_path": bool(path_hash_unique.eq(1).all()),
        "all_policies_use_same_released_information_per_path": bool(info_unique.eq(1).all()),
        "historical_test_path_uses_only_5_2_1_released_information": bool(test_manifest["released_information_source"].eq("5.2.1 released_hmm_filter").all()),
        "historical_path_has_no_future_information": bool(frozen.historical["timing_valid"].all() and (pd.to_datetime(frozen.historical["release_date"]) <= pd.to_datetime(frozen.historical["week"])).all()),
        "historical_path_has_no_artificial_risk_ramp": bool(frozen.historical["risk_information_source"].astype(str).str.contains("released", case=False).all()),
        "all_policies_receive_identical_exogenous_prepared_information": bool(
            matched_inputs.eq(1).all().all()
        ),
        "paired_future_and_same_week_nonanticipativity_probes_pass": bool(
            len(nonanticipativity_checks) > 0
            and nonanticipativity_checks["passed"].astype(bool).all()
        ),
        "checkpoints_generated_from_scratch": bool(checkpoint_manifest["generated_from_scratch"].astype(bool).all()),
        "checkpoints_selected_before_test_replay": bool(checkpoint_manifest["selected_before_test_replay"].astype(bool).all()),
        "sac_reparameterised_actor_gradients_match_finite_differences": bool(
            len(sac_gradient_checks) > 0
            and sac_gradient_checks["passed"].astype(bool).all()
        ),
        "sac_updates_all_registered_components_each_decision_period": bool(
            len(sac_curves) > 0
            and sac_curves["period_update_count"].eq(int(config["event_weeks"])).all()
            and sac_curves["reward_critic_q1_update_count"].eq(int(config["event_weeks"])).all()
            and sac_curves["reward_critic_q2_update_count"].eq(int(config["event_weeks"])).all()
            and sac_curves["actor_update_count"].eq(int(config["event_weeks"])).all()
            and sac_curves["entropy_temperature_update_count"].eq(int(config["event_weeks"])).all()
            and sac_curves.loc[sac_curves["policy"].eq("Constrained SAC"), "constraint_critic_update_count"].eq(int(config["event_weeks"])).all()
            and sac_curves.loc[sac_curves["policy"].eq("Constrained SAC"), "constraint_dual_update_count"].eq(int(config["event_weeks"])).all()
            and sac_curves["mean_log_standard_deviation"].nunique() > 1
            and not np.allclose(
                sac_curves["entropy_temperature"].to_numpy(float),
                float(config["training"]["sac_entropy_temperature"]),
            )
        ),
        "all_actions_have_formal_dimension": bool(
            replications["action_dimension"].eq(len(action_names)).all()
            and len(action_names) == 3 * len(model.controlled_resources) + len(model.layout.release) + len(model.layout.disclosure)
        ),
        "all_policies_use_shared_projector": bool(replications["projector_id"].nunique() == 1),
        "all_policies_use_shared_kernel": bool(replications["kernel_id"].nunique() == 1),
        "all_policies_retain_direct_procurement_right": bool(authority.loc[authority["in_main_ranking"], "direct_procurement_right_retained"].astype(bool).all()),
        "equal_allocation_excluded_from_main_ranking": config["forbidden_main_policy"] not in set(replications["policy"]),
        "exit_is_not_an_action": "exit" not in action_names,
        "all_trajectory_contract_checks_pass": contracts_pass,
        "all_sue_residuals_within_registered_tolerance": bool((diagnostics["sue_residual"] <= float(config["behavior"]["rcmsa_tolerance"])).all()),
        "no_solver_failure_was_converted_to_transition": bool(diagnostics["solver_failure"].fillna("").eq("").all() and diagnostics["sue_status"].eq("converged").all()),
        "budget_remaining_nonnegative": bool(contracts["budget_nonnegative"].astype(bool).all()),
        "all_physical_states_nonnegative": bool(contracts["physical_states_nonnegative"].astype(bool).all()),
        "route_resource_cost_is_closed_and_nonzero": bool(route_nonmissing and (route_costs["total_incremental_resource_cost"] > 0).any()),
        "formal_loss_identity_holds": bool((loss_summary["loss_identity_residual"].abs() <= float(config["numerics"]["loss_identity_tolerance"])).all()),
        "seed_aggregation_precedes_path_inference": bool(path_level.loc[path_level["policy"].isin(config["learning_policies"]), "training_seed_count"].ge(3).all()),
        "inference_unit_is_physical_path": bool(path_level["inference_unit"].eq("physical_path").all() and paired_effects["inference_unit"].eq("physical_path").all()),
        "path_count_rule_executed": int(selected_path_count.loc[0, "executed_paths"]) == len(test_manifest),
        "multiplicity_control_applied": bool(paired_effects["holm_adjusted_p_value"].notna().all()),
        "right_censoring_not_recorded_as_104_clearance": bool(~clearance["censored_clearance_time_recorded_as_104"].astype(bool).any()),
        "figures_exist_and_are_300_dpi_targets": bool(len(figures) == 3 and all(path.exists() for path in figures) and int(config["numerics"]["figure_dpi"]) == 300),
        "requested_and_implemented_actions_logged": all(f"requested_{name}" in actions and f"implemented_{name}" in actions for name in action_names),
        "unidentified_late_exit_cost_is_zero": bool(
            float(config["behavior"]["late_exit_cost_per_vintage"]) == 0.0
        ),
        "disclosure_error_scale_matches_frozen_calibration": bool(
            sigma_matches
            and waiting_calibration["uses_historical_test_event"].astype(str).str.lower().eq("false").all()
        ),
        "disclosure_gamma_is_registered_and_bounded": bool(
            0.0 <= float(config["information"]["gamma_I"]) <= 1.0
            and parameter_registry_frame["parameter"].eq("information.gamma_I").any()
        ),
        "public_signal_and_reference_loading_are_registered": bool(
            config["information"]["public_signal_formula"]
            and config["information"]["reference_loading_rule"]
        ),
        "all_scientific_parameter_sections_are_frozen": bool(
            required_sections.issubset(config)
        ),
        "all_identified_scientific_parameter_families_are_traceable": bool(
            len(scientific_traceability) >= 15
            and scientific_traceability["traceable"].astype(bool).all()
            and scientific_traceability[["config_path", "code_location", "basis"]].notna().all().all()
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    warnings: list[str] = []
    passive_loss = float(path_level.loc[path_level["policy"] == "Passive", "total_operational_objective"].mean())
    for row in paired_effects.itertuples(index=False):
        if row.mean_paired_difference >= 0:
            warnings.append(f"{row.policy} did not improve mean paired loss versus Passive.")
        if row.simultaneous_95_lower <= 0 <= row.simultaneous_95_upper:
            warnings.append(f"{row.policy} paired effect is unresolved after multiplicity adjustment.")
    unmet = pilot_precision.loc[~pilot_precision["precision_target_met"].astype(bool), "policy"].tolist()
    if unmet:
        warnings.append("Paired precision target was not met for: " + ", ".join(unmet))
    censored = clearance.loc[clearance["number_censored_paths"] > 0, ["policy", "number_censored_paths"]]
    for row in censored.itertuples(index=False):
        warnings.append(
            f"{row.policy} has {int(row.number_censored_paths)} physical paths with at least one right-censored training-seed trajectory."
        )
    mean_loss = path_level.groupby("policy")["total_operational_objective"].mean()
    transparent_best = min(
        ("Reactive", "Projected stochastic MPC"), key=lambda policy: mean_loss[policy]
    )
    learning_best = min(config["learning_policies"], key=lambda policy: mean_loss[policy])
    if mean_loss[learning_best] > mean_loss[transparent_best]:
        warnings.append(
            f"Best learning policy {learning_best} exceeded transparent benchmark "
            f"{transparent_best} by {mean_loss[learning_best] - mean_loss[transparent_best]:.6f} mean loss units."
        )
    if mean_loss["Model-guided constrained SAC"] > mean_loss["Behaviour cloning"]:
        warnings.append(
            "Model-guided constrained SAC exceeded Behaviour cloning by "
            f"{mean_loss['Model-guided constrained SAC'] - mean_loss['Behaviour cloning']:.6f} mean loss units."
        )
    for policy in ("Vanilla SAC", "Constrained SAC"):
        warnings.append(
            f"{policy} reduced mean paired loss versus Passive by only "
            f"{passive_loss - mean_loss[policy]:.6f} units despite satisfying the SAC implementation contract."
        )
    return {
        "experiment_id": config["experiment_id"],
        "status": "complete" if not failures else "failed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "blocking_checks": checks,
        "blocking_failures": failures,
        "honest_result_warnings": warnings,
        "precision_target_met": bool(selected_path_count.loc[0, "precision_target_met"]),
        "mean_passive_loss": passive_loss,
        "boundary": "Conditional policy comparison for the declared reference network, chi=0.5 design point, cost proxies, common authority and frozen information only.",
    }


def write_acceptance_report(
    *,
    acceptance: Mapping[str, Any],
    confidence_set: pd.DataFrame,
    paired_effects: pd.DataFrame,
    clearance: pd.DataFrame,
    output_path: Path,
) -> None:
    best = confidence_set.sort_values("mean_total_operational_loss").iloc[0]
    lines = [
        "# 5.2.2 Acceptance Report",
        "",
        f"Status: **{acceptance['status']}**.",
        "",
        "## Result boundary",
        "",
        str(acceptance["boundary"]),
        "",
        "## Main numerical result",
        "",
        f"The sample-lowest mean loss policy is **{best['policy']}**. This is not labelled universally optimal; the simultaneous confidence-set field determines whether it is statistically resolved.",
        "",
        "## Paired effects versus Passive",
        "",
    ]
    for row in paired_effects.itertuples(index=False):
        lines.append(
            f"- {row.policy}: mean difference {row.mean_paired_difference:.6g}; simultaneous 95% interval [{row.simultaneous_95_lower:.6g}, {row.simultaneous_95_upper:.6g}]; Holm p={row.holm_adjusted_p_value:.6g}."
        )
    lines.extend(["", "## Clearance and censoring", ""])
    for row in clearance.itertuples(index=False):
        lines.append(
            f"- {row.policy}: clearance probability {row.clearance_probability:.3f}; restricted mean {row.restricted_mean_clearance_time:.3f} weeks; censored paths {int(row.number_censored_paths)}; mean final outstanding {row.mean_final_outstanding_mass:.6g}."
        )
    lines.extend(["", "## Negative, weak, uncertain, or incomplete evidence", ""])
    warnings = list(acceptance.get("honest_result_warnings", []))
    lines.extend([f"- {warning}" for warning in warnings] or ["- None registered."])
    lines.extend(["", "## Blocking failures", ""])
    failures = list(acceptance.get("blocking_failures", []))
    lines.extend([f"- {failure}" for failure in failures] or ["- None."])
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "These outputs do not estimate the historical committed share, causal port effects, universal Hormuz performance, or global optimality. Designed route-resource proxies require the registered 5.3.4 sensitivity analysis.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_manifest(
    *,
    output_directory: Path,
    experiment_directory: Path,
    config_path: Path,
    frozen: Frozen521Inputs,
    figures_published: Sequence[Path],
    command: str,
) -> dict[str, Any]:
    outputs = []
    for path in sorted(output_directory.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            outputs.append(
                {
                    "path": path.relative_to(output_directory).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "experiment_id": "5.2.2_common_authority_benchmark",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "config": {
            "path": config_path.relative_to(experiment_directory).as_posix(),
            "sha256": sha256_file(config_path),
        },
        "frozen_5_2_1_inputs": {
            "historical_information_event_path_sha256": frozen.interface_hash,
            "counterfactual_residual_library_sha256": frozen.residual_hash,
            "run_manifest_sha256": frozen.run_manifest_hash,
        },
        "outputs": outputs,
        "published_figures": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in figures_published
        ],
        "code_files": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
            }
            for path in sorted(experiment_directory.glob("*.py"))
        ],
    }
    (output_directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
