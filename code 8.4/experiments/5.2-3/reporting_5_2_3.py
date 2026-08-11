"""Static figures, manifests, acceptance and technical Markdown reports for 5.2.3."""

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
from matplotlib.colors import PowerNorm


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def parameter_registry(
    *,
    config: Mapping[str, Any],
    benchmark_config: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    benchmark_leader: str,
    medoid_path_id: str,
) -> pd.DataFrame:
    rows = [
        ("experiment_id", config["experiment_id"], "5.2.3 config", "unique experiment namespace"),
        ("input_5_2_2", config["input_5_2_2"], "5.2.3 config", "accepted frozen benchmark outputs"),
        ("benchmark_leader", benchmark_leader, "5.2.2 path-level mean loss", "automatic selection; not universal optimality"),
        ("restricted_action_base_policy", benchmark_leader, "5.2.3 preregistered rule", "same frozen transparent policy across restrictions"),
        ("no_release_pacing_baseline", 1.0, "endpoint of formal rho in [0,1]", "all available waiting enters oldest-first release"),
        ("no_readiness_initial_stock", 0.0, "Chapter 3 initial state", "no historic readiness stock and no new readiness orders"),
        ("direct_procurement_retained_without_readiness", True, "Chapter 3/5.1 common authority", "no-readiness is not no-capacity"),
        ("representative_path", medoid_path_id, "external physical-path medoid", "outcomes excluded from path selection"),
        ("medoid_variables", "|".join(config["representative_path"]["variables"]), "5.2.3 config", "blocked mass, serviceability and recovery geometry"),
        ("confidence_level", config["statistics"]["confidence_level"], "5.2.3 config", "paired physical-path intervals"),
        ("multiplicity_adjustment", config["statistics"]["multiplicity_adjustment"], "5.2.3 config", "four restrictions within each outcome"),
        ("full_action_reproduction_tolerance", config["acceptance"]["full_action_reproduction_tolerance"], "5.2.2 loss identity tolerance", "blocking no-difference replay check"),
        ("source_simplex_tolerance", config["acceptance"]["source_simplex_tolerance"], "5.2.2 mass tolerance", "route-wait-exit conservation"),
        ("waiting_identity_tolerance", config["acceptance"]["waiting_identity_tolerance"], "5.2.2 mass tolerance", "vintage release-renewal-attrition balance"),
        ("provenance_shadow_tolerance", config["acceptance"]["provenance_shadow_tolerance"], "5.2.2 mass tolerance", "diagnostic decomposition must reconstruct formal route-stage state"),
        ("readiness_lead_weeks", benchmark_config["action"]["readiness_lead_weeks"], "5.1/5.2.2", "unchanged capacity pipeline"),
        ("clearance_cap_weeks", benchmark_config["clearance"]["maximum_weeks"], "5.1/5.2.2", "right-censoring cap"),
        ("clearance_empty_tolerance", benchmark_config["clearance"]["empty_tolerance"], "5.1/5.2.2", "unchanged empty-state rule"),
        ("rcmsa_tolerance", benchmark_config["behavior"]["rcmsa_tolerance"], "5.1/5.2.2", "same behavior solver certificate"),
        ("action_projection_tolerance", benchmark_config["action"]["projection_tolerance"], "5.1/5.2.2", "same common projector"),
        ("learning_seed_aggregation", config["statistics"]["learning_seed_rule"], "5.1 comparison protocol", "physical path remains inference unit"),
        ("physical_path_count", config["execution"]["expected_physical_paths"], "accepted 5.2.2 selected_path_count", "all-path mechanism inference"),
        ("math_library_threads_per_worker", config["execution"]["math_library_threads_per_worker"], "5.2.3 execution control", "prevents nested BLAS oversubscription; no scientific parameter change"),
        ("detailed_trace_scope", config["execution"]["detailed_trace_scope"], "5.2.3 execution protocol", "high-dimensional evidence without reducing the inference sample"),
    ]
    for name, value in input_hashes.items():
        rows.append((f"input_sha256__{name}", value, "runtime hash audit", "frozen input identity"))
    return pd.DataFrame(rows, columns=["parameter", "value", "source", "basis"])


def chart_map() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "figure": "Figure 5.2.3a",
                "question": "When do the five formal action blocks activate and where does burden accumulate?",
                "family": "Trend",
                "chart_type": "faceted line with path interquartile ribbons",
                "fields": "weekly action reference ratios and waiting-plus-four-stage queue burden",
                "supported_claim": "timing and activation mechanism, not causal action value",
            },
            {
                "figure": "Figure 5.2.3b",
                "question": "How do committed and adaptive cohorts occupy routes, maritime pipelines and four stages?",
                "family": "Matrix and cohort",
                "chart_type": "policy-provenance by route-stage heatmap",
                "fields": "cumulative preservice exposure on the physical-path medoid",
                "supported_claim": "tag/provenance process evidence on an externally selected representative path",
            },
            {
                "figure": "Figure 5.2.3c",
                "question": "How do fixed-policy restrictions change loss, congestion transfer and clearance?",
                "family": "Uncertainty and benchmark",
                "chart_type": "small-multiple paired forest plot",
                "fields": "mean paired differences and simultaneous 95% intervals",
                "supported_claim": "restricted-action diagnostic; not reoptimised or causal marginal value",
            },
        ]
    )


def _policy_label(policy: str) -> str:
    return {
        "Model-guided constrained SAC": "MG constrained SAC",
        "Projected stochastic MPC": "Projected MPC",
    }.get(policy, policy)


def create_figures(
    *,
    weekly: pd.DataFrame,
    physical: pd.DataFrame,
    restricted_effects: pd.DataFrame,
    figure_policies: Sequence[str],
    medoid_path_id: str,
    output_directory: Path,
    dpi: int,
) -> list[Path]:
    _style()
    output_directory.mkdir(parents=True, exist_ok=True)
    palette = {
        "Passive": "#6B7280",
        "Reactive": "#D9822B",
        "Model-guided constrained SAC": "#355C7D",
    }
    line_styles = {"Passive": ":", "Reactive": "-", "Model-guided constrained SAC": "--"}

    decision = weekly.loc[(weekly["scope"] == "decision") & weekly["policy"].isin(figure_policies)]
    panels = [
        ("action_readiness_order_reference_ratio", "Readiness order", "Mean implemented share of block bound"),
        ("action_direct_order_reference_ratio", "Direct capacity order", "Mean implemented share of block bound"),
        ("action_readiness_exercise_reference_ratio", "Readiness exercise", "Mean implemented share of block bound"),
        ("action_release_reference_ratio", "Waiting release", "Implemented release fraction"),
        ("action_disclosure_reference_ratio", "Disclosure", "Mean implemented disclosure intensity"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6), sharex=True, constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes.ravel()[:5], panels):
        for policy in figure_policies:
            group = decision.loc[decision["policy"] == policy].sort_values("period_offset")
            x = group["period_offset"].to_numpy(float) + 1
            mean = group[f"mean__{metric}"].to_numpy(float)
            q25 = group[f"q25__{metric}"].to_numpy(float)
            q75 = group[f"q75__{metric}"].to_numpy(float)
            color = palette.get(policy, "#355C7D")
            ax.plot(x, mean, color=color, linestyle=line_styles.get(policy, "-"), linewidth=1.6, label=_policy_label(policy))
            ax.fill_between(x, q25, q75, color=color, alpha=0.10)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    burden_ax = axes.ravel()[5]
    for policy in figure_policies:
        group = decision.loc[decision["policy"] == policy].sort_values("period_offset")
        x = group["period_offset"].to_numpy(float) + 1
        means = (
            group["mean__waiting_before"]
            + group["mean__queue_berth"]
            + group["mean__queue_yard"]
            + group["mean__queue_gate"]
            + group["mean__queue_landbridge"]
        ).to_numpy(float)
        lower = (
            group["q25__waiting_before"]
            + group["q25__queue_berth"]
            + group["q25__queue_yard"]
            + group["q25__queue_gate"]
            + group["q25__queue_landbridge"]
        ).to_numpy(float)
        upper = (
            group["q75__waiting_before"]
            + group["q75__queue_berth"]
            + group["q75__queue_yard"]
            + group["q75__queue_gate"]
            + group["q75__queue_landbridge"]
        ).to_numpy(float)
        color = palette.get(policy, "#355C7D")
        burden_ax.plot(x, means, color=color, linestyle=line_styles.get(policy, "-"), linewidth=1.6, label=_policy_label(policy))
        burden_ax.fill_between(x, lower, upper, color=color, alpha=0.10)
    burden_ax.set_title("External waiting plus four-stage queues")
    burden_ax.set_ylabel("Model cargo units")
    burden_ax.set_ylim(bottom=0)
    for ax in axes[-1, :]:
        ax.set_xlabel("Event week")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", ncol=1, frameon=False, bbox_to_anchor=(1.005, 0.5))
    fig.suptitle("Figure 5.2.3a. Formal action activation and congestion burden", fontsize=12, y=1.02)
    fig.text(0.5, -0.02, "Lines are path means; ribbons are path interquartile bands. Learning seeds are averaged within path first.", ha="center", fontsize=8)
    path_a = output_directory / "figure_5_2_3a_action_congestion_trajectories.png"
    fig.savefig(path_a, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    trace = physical.loc[
        physical["path_id"].eq(medoid_path_id)
        & physical["restriction"].eq("full_action")
        & physical["scope"].eq("decision")
        & physical["base_policy"].isin(figure_policies)
    ].copy()
    trace["route_stage"] = trace["route"].str.replace("Khor_Fakkan", "KF") + " | " + trace["stage"].replace(
        {"maritime_pipeline": "maritime", "corridor": "landbridge"}
    )
    stage_order = ["maritime", "berth", "yard", "gate", "landbridge"]
    route_order = ["Khor_Fakkan", "Fujairah", "Sohar"]
    columns = [
        ("KF" if route == "Khor_Fakkan" else route) + " | " + stage
        for route in route_order
        for stage in stage_order
    ]
    seed_aggregated = trace.groupby(
        ["base_policy", "path_id", "training_seed", "provenance", "route_stage"],
        dropna=False,
        as_index=False,
    )["preservice_workload"].sum()
    path_aggregated = seed_aggregated.groupby(
        ["base_policy", "path_id", "provenance", "route_stage"], as_index=False
    )["preservice_workload"].mean()
    row_labels = []
    matrix_rows = []
    for policy in figure_policies:
        for provenance in ("committed", "adaptive"):
            group = path_aggregated.loc[
                path_aggregated["base_policy"].eq(policy)
                & path_aggregated["provenance"].eq(provenance)
            ].set_index("route_stage")
            matrix_rows.append([float(group["preservice_workload"].get(column, 0.0)) for column in columns])
            row_labels.append(f"{_policy_label(policy)} | {provenance}")
    matrix = np.asarray(matrix_rows, dtype=float)
    fig, ax = plt.subplots(figsize=(15.5, 5.5), constrained_layout=True)
    vmax = max(float(matrix.max()), 1.0)
    heat = ax.imshow(matrix, aspect="auto", cmap="Blues", norm=PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=6.5, color="white" if value > 0.55 * vmax else "#263238")
    ax.set_xticks(np.arange(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title("Figure 5.2.3b. Route-stage exposure by policy and dispatch provenance")
    colorbar = fig.colorbar(heat, ax=ax, pad=0.02, shrink=0.85)
    colorbar.set_label("Cumulative preservice exposure (model-unit weeks; square-root color scale)")
    fig.text(0.5, -0.035, f"Physical-path medoid: {medoid_path_id}. Values use the 21-week decision window; MG seeds are averaged before display.", ha="center", fontsize=8)
    path_b = output_directory / "figure_5_2_3b_tagged_route_stage_heatmap.png"
    fig.savefig(path_b, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    outcome_order = [
        "total_loss",
        "waiting_exposure",
        "sue_exit",
        "attrition_exit",
        "overload",
        "route_resource_loss",
        "action_loss",
        "clearance_probability",
        "restricted_mean_clearance_time",
        "final_outstanding",
    ]
    outcome_titles = {
        "total_loss": "Total loss",
        "waiting_exposure": "Waiting exposure",
        "sue_exit": "SUE exit",
        "attrition_exit": "Attrition exit",
        "overload": "Overload",
        "route_resource_loss": "Route resource loss",
        "action_loss": "Action loss",
        "clearance_probability": "Clearance probability",
        "restricted_mean_clearance_time": "Restricted clearance time",
        "final_outstanding": "Final outstanding",
    }
    restriction_order = [
        "no_readiness",
        "no_direct_capacity",
        "no_release_pacing_authority",
        "no_disclosure",
    ]
    restriction_labels = ["No readiness", "No direct", "Immediate release", "No disclosure"]
    inference_paths = int(restricted_effects["physical_paths"].max())
    fig, axes = plt.subplots(2, 5, figsize=(15.8, 7.0), constrained_layout=True)
    for ax, outcome in zip(axes.ravel(), outcome_order):
        group = restricted_effects.loc[restricted_effects["outcome"] == outcome].set_index("restriction").loc[restriction_order]
        y = np.arange(len(group))
        mean = group["mean_paired_difference"].to_numpy(float)
        lower = group["simultaneous_95_lower"].to_numpy(float)
        upper = group["simultaneous_95_upper"].to_numpy(float)
        ax.axvline(0.0, color="#263238", linewidth=0.9, linestyle=":")
        ax.hlines(y, lower, upper, color="#355C7D", linewidth=1.5)
        ax.scatter(mean, y, s=30, color="#D9822B", edgecolor="#263238", linewidth=0.5, zorder=3)
        ax.set_yticks(y, restriction_labels if ax in axes[:, 0] else ["", "", "", ""])
        ax.invert_yaxis()
        ax.set_title(outcome_titles[outcome], fontsize=9.5)
        ax.set_xlabel("Difference vs full")
    fig.suptitle("Figure 5.2.3c. Fixed-policy restricted-action paired diagnostics", fontsize=12)
    fig.text(0.5, -0.015, f"Points are {inference_paths}-path paired means; lines are within-outcome simultaneous 95% intervals. These are not reoptimised or causal action values.", ha="center", fontsize=8)
    path_c = output_directory / "figure_5_2_3c_restricted_action_forest.png"
    fig.savefig(path_c, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return [path_a, path_b, path_c]


def acceptance_payload(
    *,
    config: Mapping[str, Any],
    input_acceptance_complete: bool,
    policy_set: pd.DataFrame,
    figure_policies: Sequence[str],
    medoid: pd.DataFrame,
    restricted_replications: pd.DataFrame,
    reproduction: pd.DataFrame,
    contracts: pd.DataFrame,
    activation: pd.DataFrame,
    restricted_effects: pd.DataFrame,
    figures: Sequence[Path],
) -> dict[str, Any]:
    tol = config["acceptance"]
    expected_paths = int(config["execution"]["expected_physical_paths"])
    required_full_policies = set(config["execution"]["full_action_replay_policies"])
    reproduced_policies = set(reproduction["policy"].astype(str))
    reproduced_path_counts = reproduction.groupby("policy")["path_id"].nunique()
    blocking = {
        "accepted_5_2_2_input": bool(input_acceptance_complete),
        "passive_leader_and_proposed_retained": bool(
            {"Passive", "Model-guided constrained SAC"}.issubset(set(figure_policies))
            and policy_set["is_benchmark_leader"].any()
        ),
        "medoid_uses_no_policy_outcome": bool(~medoid["selection_uses_policy_outcomes"].astype(bool).any()),
        "all_88_paths_used_for_each_restriction": bool(
            restricted_replications.groupby("restriction")["path_id"].nunique().eq(expected_paths).all()
        ),
        "required_full_action_policies_replayed": bool(
            required_full_policies.issubset(reproduced_policies)
            and all(int(reproduced_path_counts.get(policy, 0)) == expected_paths for policy in required_full_policies)
        ),
        "full_action_reproduces_5_2_2_within_1e_6": bool(
            len(reproduction) > 0 and reproduction["passed"].astype(bool).all()
        ),
        "all_period_production_acceptance_passed": bool(
            contracts["all_step_acceptance_passed"].astype(bool).all()
        ),
        "all_restrictions_change_only_declared_action_blocks": bool(
            contracts["restriction_changes_only_declared_action_block"].astype(bool).all()
        ),
        "no_readiness_initial_stock_is_zero": bool(
            contracts["no_readiness_starts_without_readiness_stock"].astype(bool).all()
        ),
        "tagged_transition_and_capacity_pipeline_close": bool(
            contracts["maximum_transition_residual"].max() <= float(tol["source_simplex_tolerance"])
        ),
        "source_simplex_conserved": bool(
            contracts["maximum_source_simplex_residual"].max() <= float(tol["source_simplex_tolerance"])
        ),
        "waiting_vintage_identity_conserved": bool(
            contracts["maximum_waiting_identity_residual"].max() <= float(tol["waiting_identity_tolerance"])
        ),
        "provenance_shadow_reconstructs_formal_state": bool(
            contracts["maximum_provenance_shadow_residual"].max() <= float(tol["provenance_shadow_tolerance"])
        ),
        "period_losses_reconstruct": bool(
            contracts["maximum_period_loss_identity_residual"].max() <= float(tol["loss_identity_tolerance"])
            and contracts["loss_components_reconstruct_total"].astype(bool).all()
        ),
        "no_readiness_retains_direct_procurement": bool(
            contracts.loc[
                contracts["restriction"].eq("no_readiness"),
                "direct_procurement_retained_under_no_readiness",
            ].astype(bool).all()
        ),
        "no_pacing_uses_rho_one": bool(
            contracts.loc[
                contracts["restriction"].eq("no_release_pacing_authority"),
                "no_release_pacing_uses_immediate_release_baseline",
            ].astype(bool).all()
        ),
        "no_disclosure_preserves_nonaction_information": bool(
            contracts.loc[
                contracts["restriction"].eq("no_disclosure"),
                "no_disclosure_preserves_information_system",
            ].astype(bool).all()
        ),
        "all_figures_generated": len(figures) == 3 and all(path.exists() for path in figures),
    }
    warnings: list[str] = []
    for metric in ("BC_proposal_selected_count", "SAC_proposal_selected_count"):
        value = activation.loc[activation["metric"].eq(metric), "value"]
        if len(value) and float(value.iloc[0]) == 0.0:
            warnings.append(f"Proposed selector never selected {metric.split('_')[0]} on the held-out replay.")
    for module in ("readiness order", "direct order", "readiness exercise", "release", "disclosure"):
        value = activation.loc[
            activation["module"].eq(module) & activation["metric"].eq("activation_count"),
            "value",
        ]
        if len(value) and float(value.iloc[0]) == 0.0:
            warnings.append(f"Proposed policy {module} block did not activate on held-out decisions.")
    improved = restricted_effects.loc[
        restricted_effects["outcome"].eq("total_loss")
        & (restricted_effects["mean_paired_difference"] < 0.0)
    ]
    for row in improved.itertuples(index=False):
        warnings.append(
            f"Fixed Reactive restriction {row.restriction} reduced mean loss by "
            f"{-float(row.mean_paired_difference):.6f}; retain as a negative coordination diagnostic, not causal value."
        )
    failures = [name for name, passed in blocking.items() if not passed]
    return {
        "experiment_id": config["experiment_id"],
        "status": "complete" if not failures else "blocked",
        "blocking_checks": blocking,
        "blocking_failures": failures,
        "warnings": warnings,
        "evidence_boundary": (
            "Mechanism and fixed-policy restricted-action evidence under the 5.2.2 reference network. "
            "No restricted comparison is a causal marginal value or a reoptimised policy value."
        ),
    }


def write_run_manifest(
    *,
    output_directory: Path,
    code_root: Path,
    config_path: Path,
    input_files: Sequence[Path],
    output_files: Sequence[Path],
    started_at: str,
    elapsed_seconds: float,
    status: str,
) -> Path:
    artifacts = []
    for path in sorted(set(output_files)):
        if not path.exists():
            continue
        item: dict[str, Any] = {
            "path": path.relative_to(code_root).as_posix() if path.is_relative_to(code_root) else str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix.lower() == ".csv":
            item["rows"] = len(pd.read_csv(path))
        artifacts.append(item)
    payload = {
        "experiment_id": "5.2.3_action_and_congestion_mechanisms",
        "status": status,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "command": "python experiments/5.2-3/run_5_2_3.py",
        "python": sys.version,
        "platform": platform.platform(),
        "config": {"path": config_path.relative_to(code_root).as_posix(), "sha256": sha256_file(config_path)},
        "inputs": [
            {
                "path": path.relative_to(code_root).as_posix() if path.is_relative_to(code_root) else str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in input_files
        ],
        "artifacts": artifacts,
    }
    target = output_directory / "run_manifest.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def _mean_value(summary: pd.DataFrame, policy: str, metric: str) -> float:
    return float(summary.loc[summary["policy"].eq(policy), f"mean__{metric}"].iloc[0])


def _write_reports_legacy(
    *,
    report_directory: Path,
    policy_set: pd.DataFrame,
    benchmark_leader: str,
    figure_policies: Sequence[str],
    medoid_path_id: str,
    full_summary: pd.DataFrame,
    activation: pd.DataFrame,
    restricted_effects: pd.DataFrame,
    acceptance: Mapping[str, Any],
    weekly: pd.DataFrame | None = None,
    physical: pd.DataFrame | None = None,
) -> list[Path]:
    report_directory.mkdir(parents=True, exist_ok=True)
    additions = report_directory / "5_1_PARAMETER_AND_METRIC_ADDITIONS.md"
    additions.write_text(
        """# 5.2.3 对 5.1 参数与指标表的补充建议

## 技术结论

5.1 已经包含本实验所需的正式动作、oldest-first、waiting vintage、SUE、tagged transition、八周 readiness lead、104 周 clearance cap、共同损失与数值容差。5.2.3 不需要新增经济校准或改变网络参数。建议只把下列实验协议补入 5.1，以便后续写作能够复现机制证据。

## 建议补充的实验协议

1. **机制政策集合规则。** 必含 Passive、5.2.2 mean-loss benchmark leader、proposed model-guided policy，以及任何仍属于 sample-best confidence set 的政策。leader 只能称为 benchmark leader。
2. **代表路径规则。** 在 held-out physical paths 上，以 total blocked mass、peak blocked mass、mean/minimum serviceability 和 recovery rate 标准化后到共同中心的欧氏距离选择 medoid；不得使用政策损失或 proposed advantage。
3. **无 release pacing 权限基准。** 固定为 $\\rho_t^{base}=1$，即全部可释放 waiting cargo 继续通过同一 oldest-first operator 立即进入当期 SUE。$\\rho_t=0$ 不得代表无 pacing 权限。
4. **受限动作证据边界。** 限制在冻结策略原始输出之后、共同 projection 之前实施；结果命名为 fixed-policy restricted-action diagnostic，不命名为 causal marginal value、reoptimised value 或 action-right value。
5. **复现容差。** full-action 路径必须在 $10^{-6}$ 绝对容差内复现 5.2.2 对应路径的正式结果；该值继承共同质量与损失 identity tolerance，不是行为参数。
6. **统计单位。** 学习种子先在 physical path 内平均，限制相对 full action 的差异再以六条 physical paths 为配对单位；每个 outcome 内四项限制使用 Holm 调整和 Bonferroni-$t$ simultaneous interval。
7. **模块激活定义。** 动作激活阈值为共同 projection tolerance 的十倍，仅用于区分数值零；同时报告 requested/implemented 变化范围、BC/SAC proposal selection 和 fallback，而不是仅检查接口存在。

## 不应新增到 5.1 的内容

- 不新增根据本次结果倒推的阈值或动作比例。
- 不把 medoid 当作推断样本；正式结论仍使用全部 held-out paths。
- 不把 provenance shadow ledger 写成新的物理状态。它只是用正式比例服务规则分解 aggregate route-stage state 的审计账本。
""",
        encoding="utf-8",
    )

    leader_loss = _mean_value(full_summary, benchmark_leader, "total_operational_objective")
    mg_loss = _mean_value(full_summary, "Model-guided constrained SAC", "total_operational_objective")
    passive_loss = _mean_value(full_summary, "Passive", "total_operational_objective")
    activation_lines = []
    for row in activation.itertuples(index=False):
        if row.metric in {"BC_proposal_selected_count", "SAC_proposal_selected_count", "fallback_count", "activation_count", "range_across_test_decisions"}:
            activation_lines.append(f"- {row.module} / {row.metric}: {float(row.value):.6g} (denominator {int(row.denominator)}).")
    total_effects = restricted_effects.loc[restricted_effects["outcome"].eq("total_loss")]
    restriction_lines = [
        f"- {row.restriction}: total-loss difference {row.mean_paired_difference:,.2f}; simultaneous 95% interval [{row.simultaneous_95_lower:,.2f}, {row.simultaneous_95_upper:,.2f}]; Holm p={row.holm_adjusted_p_value:.4g}."
        for row in total_effects.itertuples(index=False)
    ]
    warning_lines = [f"- {value}" for value in acceptance.get("warnings", [])] or ["- No activation warning was triggered."]
    trajectory_lines: list[str] = []
    if weekly is not None:
        decision_weekly = weekly.loc[weekly["scope"].eq("decision")]
        for policy in figure_policies:
            group = decision_weekly.loc[decision_weekly["policy"].eq(policy)].sort_values("period_offset")
            if group.empty:
                continue
            final = group.iloc[-1]
            burden = float(
                final["mean__waiting_before"]
                + final["mean__queue_berth"]
                + final["mean__queue_yard"]
                + final["mean__queue_gate"]
                + final["mean__queue_landbridge"]
            )
            trajectory_lines.append(
                f"- {policy}: week-21 waiting-plus-queue burden {burden:,.2f} model units; "
                f"mean release fraction {group['mean__action_release_reference_ratio'].mean():.3f}; "
                f"mean disclosure intensity {group['mean__action_disclosure_reference_ratio'].mean():.3f}."
            )
    if not trajectory_lines:
        trajectory_lines = ["- Weekly mechanism values are preserved in `weekly_policy_mechanisms.csv`."]
    heatmap_lines: list[str] = []
    if physical is not None:
        trace = physical.loc[
            physical["path_id"].eq(medoid_path_id)
            & physical["restriction"].eq("full_action")
            & physical["scope"].eq("decision")
            & physical["base_policy"].isin(figure_policies)
        ]
        exposure = trace.groupby(
            ["base_policy", "training_seed", "provenance", "route", "stage"],
            dropna=False,
            as_index=False,
        )["preservice_workload"].sum()
        exposure = exposure.groupby(
            ["base_policy", "provenance", "route", "stage"], as_index=False
        )["preservice_workload"].mean()
        for policy in figure_policies:
            adaptive = exposure.loc[
                exposure["base_policy"].eq(policy) & exposure["provenance"].eq("adaptive")
            ]
            if adaptive.empty:
                continue
            maximum = adaptive.sort_values("preservice_workload", ascending=False).iloc[0]
            heatmap_lines.append(
                f"- {policy}: largest adaptive route-stage exposure is {maximum.route}/{maximum.stage} "
                f"at {maximum.preservice_workload:,.2f} model-unit weeks."
            )
    if not heatmap_lines:
        heatmap_lines = ["- The exact route-stage cells are preserved in `physical_tagged_trajectory.csv`."]
    clearance_effects = restricted_effects.loc[
        restricted_effects["outcome"].isin(["clearance_probability", "final_outstanding", "waiting_exposure"])
    ]
    diagnostic_highlights = []
    for restriction in ("no_release_pacing_authority", "no_disclosure"):
        values = clearance_effects.loc[clearance_effects["restriction"].eq(restriction)].set_index("outcome")
        diagnostic_highlights.append(
            f"- {restriction}: waiting difference {values.loc['waiting_exposure', 'mean_paired_difference']:,.2f}; "
            f"clearance-probability difference {values.loc['clearance_probability', 'mean_paired_difference']:+.3f}; "
            f"final-outstanding difference {values.loc['final_outstanding', 'mean_paired_difference']:+.3f}."
        )
    analysis = report_directory / "FIGURE_AND_RESULTS_ANALYSIS.md"
    analysis.write_text(
        f"""# 5.2.3 Action and Congestion Mechanisms：结果与图形分析

## 技术摘要

本实验在不重新训练政策的条件下解释 5.2.2 的动作—SUE—tagged transition—loss 链。全部 held-out paths 上，Passive、benchmark leader `{benchmark_leader}` 和 proposed MG policy 均被保留。5.2.2 的平均正式损失分别为 Passive {passive_loss:,.2f}、leader {leader_loss:,.2f} 和 MG {mg_loss:,.2f}。这些数值支持条件机制比较，但不证明 leader 是普遍最优政策。

受限动作部分以 `{benchmark_leader}` 为同一冻结策略，并在原始动作输出后、共同 projection 前施加限制。full action 已通过与 5.2.2 的无差异复现验收。其余比较只能解释 action channel activation，不能解释重优化后的 action-right value 或因果边际价值。

## Figure 5.2.3a：动作激活与拥堵负担的时间链

图文件：`figure_5_2_3a_action_congestion_trajectories.png`。

该图采用 faceted line-and-ribbon，而不是政策总损失柱状图。前五个 panel 分别对应 $y^R$、$y^V$、$v^R$、$\\rho$ 和 $\\lambda$；最后一个 panel 将外部 waiting 与 berth、yard、gate、landbridge 四阶段库存共同显示。线为 physical-path mean，带为 path IQR，学习种子已经先在 path 内平均。因此图形说明的是动作发生时间与负担位置，不把 21 个周观测当作独立样本。

{chr(10).join(trajectory_lines)}

## Figure 5.2.3b：route-stage 与 provenance 机制

图文件：`figure_5_2_3b_tagged_route_stage_heatmap.png`。

代表路径 `{medoid_path_id}` 由外生 blocked mass、serviceability 和 recovery geometry 选为 physical-path medoid，不使用任何政策结果。热力图按 policy × committed/adaptive provenance 展示三个 route 在 maritime pipeline、berth、yard、gate 和 landbridge 的累计 preservice exposure。MG 的三个训练种子先平均。正式 Chapter 3 队列状态保持 class-route tag；图中的 provenance 是一个不反馈到政策或损失的 proportional-service shadow ledger，并逐期重构正式 route-stage state。

{chr(10).join(heatmap_lines)}

## Figure 5.2.3c：固定策略受限动作诊断

图文件：`figure_5_2_3c_restricted_action_forest.png`。

每个 panel 使用不同 outcome 的原始单位，点为六条相同 physical paths 上相对 full action 的配对均值，线为 outcome 内四项限制的 simultaneous 95% interval。零线表示该机制 outcome 没有平均变化。图中 immediate release 指 $\\rho_t^{{base}}=1$，而不是永久阻止释放的 $\\rho_t=0$。

### Total-loss restricted diagnostics

{chr(10).join(restriction_lines)}

### Waiting、clearance 与 terminal transfer

{chr(10).join(diagnostic_highlights)}

这些结果必须与 waiting、两类 exit、overload、clearance 和 terminal mass 一起解释；即使某项限制降低某个 queue 指标，也可能只是把负担转移到 external waiting、exit 或右删失的 terminal mass。

## Proposed MG policy 模块激活审计

{chr(10).join(activation_lines)}

模块计数为描述性激活证据。某一动作块没有激活或某一 proposal 从未被 selector 选中，属于需要如实报告的负面结果，而不是通过改变阈值或路径来修正的代码失败。

## 机制接口、样本与度量

- 主政策机制比较：5.2.2 的全部六条 held-out physical paths；学习种子先在 path 内平均。
- 受限动作诊断：同一 `{benchmark_leader}` 策略、相同六条 paths、相同 released information、相同 projector、SUE、tagged transition、loss 和 clearance rule。
- 细粒度 source/vintage/provenance trace：外生 medoid path；用于过程解释，不用于正式推断。
- waiting exposure 单位为 model-unit weeks；delivered 只来自 landbridge discharge；PortWatch 活动仍是 AIS-derived proxy，不是 observed diversion cargo。

## 局限、不确定性与负面结果

{chr(10).join(warning_lines)}

- 六条 paths 的精度限制继承 5.2.2；本实验没有把周数当作独立样本扩大显著性。
- 固定策略限制后的状态会改变，但策略没有在缩小后的 action set 上重新训练或重新优化。
- provenance shadow ledger 是审计性分解，不增加新的政策信息或物理状态维度。
- benchmark leader 来自声明网络、$\\chi=0.5$、成本、预算和历史 replay，不支持所有 Hormuz 情景上的普遍排序。

## 下一步

5.2.4 信息实验应只改变 readiness information regime，并保持本实验已经验收的动作、behavior、physical transition、operational weights 和 path hashes。对任何未激活的 action block，应先报告 activation failure，再讨论其信息价值。
""",
        encoding="utf-8",
    )
    return [additions, analysis]


def write_reports(
    *,
    report_directory: Path,
    policy_set: pd.DataFrame,
    benchmark_leader: str,
    figure_policies: Sequence[str],
    medoid_path_id: str,
    full_summary: pd.DataFrame,
    activation: pd.DataFrame,
    restricted_effects: pd.DataFrame,
    acceptance: Mapping[str, Any],
    weekly: pd.DataFrame | None = None,
    physical: pd.DataFrame | None = None,
) -> list[Path]:
    """Write the current 88-path reports without reading any legacy 5.2.3 result."""

    report_directory.mkdir(parents=True, exist_ok=True)
    path_count = int(restricted_effects["physical_paths"].max())
    passive_loss = _mean_value(full_summary, "Passive", "total_operational_objective")
    leader_loss = _mean_value(full_summary, benchmark_leader, "total_operational_objective")
    mg_loss = _mean_value(
        full_summary, "Model-guided constrained SAC", "total_operational_objective"
    )
    total_effects = restricted_effects.loc[
        restricted_effects["outcome"].eq("total_loss")
    ].sort_values("restriction")
    restriction_lines = []
    for row in total_effects.itertuples(index=False):
        direction = "降低" if float(row.mean_paired_difference) < 0 else "增加"
        restriction_lines.append(
            f"- `{row.restriction}`：相对 full action {direction}总损失 "
            f"{abs(float(row.mean_paired_difference)):,.2f}；同时 95% 区间 "
            f"[{float(row.simultaneous_95_lower):,.2f}, "
            f"{float(row.simultaneous_95_upper):,.2f}]；Holm 调整 p="
            f"{float(row.holm_adjusted_p_value):.4g}。"
        )
    activation_lines = []
    for row in activation.itertuples(index=False):
        if row.metric in {
            "BC_proposal_selected_count",
            "SAC_proposal_selected_count",
            "fallback_count",
            "activation_count",
            "range_across_test_decisions",
        }:
            activation_lines.append(
                f"- {row.module} / {row.metric}: {float(row.value):.6g} "
                f"(denominator={int(row.denominator)})。"
            )
    warning_lines = [f"- {item}" for item in acceptance.get("warnings", [])]
    if not warning_lines:
        warning_lines = ["- 未触发额外的模块未激活或受限动作改善警告。"]

    additions = report_directory / "5_1_PARAMETER_AND_METRIC_ADDITIONS.md"
    additions.write_text(
        f"""# 5.2.3 对 5.1 参数与指标表的补充报告

## 结论

- 数据部分：**NO CHANGE**。本实验不引入新数据，唯一上游是已验收的 5.2.2 配置、88 条测试路径、轨迹和 checkpoint。
- 经济与行为参数：**NO ADDITION**。网络、损失、预算、SUE、能力交付和数值容差均继承 5.2.2，不根据 5.2.3 结果重新校准。
- 需要在 5.1 中保持可追踪的仅是实验协议和派生指标；这些不是新的经济参数。

## 实验协议

1. 正式推断单位为 physical path，共 {path_count} 条；学习种子必须先在路径内平均。
2. 机制政策集包含 Passive、5.2.2 的唯一置信集领先者 `{benchmark_leader}`、以及 Model-guided constrained SAC。
3. 代表路径 `{medoid_path_id}` 由 total blocked mass、peak blocked mass、mean/minimum serviceability 和 recovery rate 标准化后选择 physical-path medoid；政策损失不进入选择。
4. Full action 的 5.2.2 复现绝对容差为 $10^{{-6}}$。
5. No release pacing 使用非控制基准 $\rho_t^{{base}}=1$，即所有当前可释放 waiting cargo 通过同一 oldest-first operator 立即进入 SUE。
6. 所有限制在冻结 Reactive 原始动作之后、共同凸投影之前施加；策略不重新训练或重新优化。

## 派生指标

- 受限动作路径差：$D_{{r,m}}^Y=Y_m^{{(r)}}-Y_m^{{(full)}}$。
- 每个 outcome 内对四项限制使用 Holm 调整，并报告 Bonferroni-$t$ simultaneous 95% interval。
- waiting exposure 以 model-unit weeks 计量；clearance probability、restricted mean clearance time 和 final outstanding 单独报告右删失边界。
- route-stage exposure 是 medoid 路径上的累计 preservice workload，仅作过程解释，不作为统计推断样本。

不得把这些诊断写成因果边际价值、重新优化后的 action-right value，或管理者删除某项权利后的最优结果。
""",
        encoding="utf-8",
    )

    analysis = report_directory / "FIGURE_AND_RESULTS_ANALYSIS.md"
    analysis.write_text(
        f"""# 5.2.3 Action and Congestion Mechanisms：结果与图形分析

## 结果摘要

5.2.3 使用全部 {path_count} 条已验收 5.2.2 物理路径。Passive、`{benchmark_leader}` 和 MG constrained SAC 的 5.2.2 平均总损失分别为 {passive_loss:,.2f}、{leader_loss:,.2f} 和 {mg_loss:,.2f}。这些数值只用于确定需要解释的机制政策，不构成新的政策排名。

Full action 对三类机制政策和全部路径进行了生产引擎重放，并在 $10^{{-6}}$ 内逐项复现 5.2.2。受限动作部分冻结 `{benchmark_leader}` 的策略映射，只限制声明的动作块并重新投影；其结果属于 fixed-policy restricted-action diagnostic。

## Figure 5.2.3a：动作与拥堵轨迹

折线与路径四分位带分别显示 readiness order、direct order、readiness exercise、waiting release、disclosure，以及 external waiting 加四阶段队列负担。学习种子先在同一路径内平均，周数未被当作独立样本。该图回答动作何时激活以及负担积累在哪里。

## Figure 5.2.3b：route-stage 与 provenance 热力图

热力图使用外生 medoid `{medoid_path_id}`，按 policy、committed/adaptive provenance、route 和 maritime/berth/yard/gate/landbridge stage 展示累计 preservice exposure。代表路径不根据政策表现选择；该图仅解释 route-tagged 状态转移。

## Figure 5.2.3c：固定策略受限动作配对效应

森林图以 {path_count} 条 physical paths 为配对单位，报告相对 full action 的均值差和 outcome 内 simultaneous 95% interval。

{chr(10).join(restriction_lines)}

负差值必须保留为冻结策略不协调或训练不足的诊断，不能解释为删除动作权利具有因果收益。

## Proposed MG 模块激活审计

{chr(10).join(activation_lines)}

## 负面、较弱或不确定结果

{chr(10).join(warning_lines)}

区间跨零、动作块未激活、限制动作反而改善损失或 clearance 发生转移，均不是代码失败；它们限定了 5.2.3 可支持的机制结论。

## 证据边界

- 不重新训练或重新优化任何受限动作控制器。
- 不把 medoid 路径当作推断样本。
- 不将 queue 降低单独解释为系统改善，必须同时检查 waiting、两类 exit、overload、terminal mass 和 clearance。
- 本实验不修改 5.2.2 的外生路径、随机数、released information、checkpoint、共同 projector、RC-MSA、tagged transition 或 loss。
""",
        encoding="utf-8",
    )

    checks = acceptance.get("blocking_checks", {})
    check_lines = [
        f"- {'PASS' if bool(value) else 'FAIL'} — `{name}`"
        for name, value in checks.items()
    ]
    acceptance_report = report_directory / "ACCEPTANCE_REPORT.md"
    acceptance_report.write_text(
        f"""# 5.2.3 阻断验收报告

总体状态：**{str(acceptance.get('status', 'blocked')).upper()}**

## 阻断检查

{chr(10).join(check_lines)}

## 输入与样本合同

- 上游只允许锁定的 5.2.2 acceptance、run manifest 和 checkpoint manifest。
- 全动作复现与受限动作推断均使用 {path_count} 条物理路径。
- 学习 seed 先在路径内聚合；森林图区间的推断单位是 physical path。
- 详细高维轨迹只保存 medoid，用于机制解释，不减少正式推断样本。

## 警告与非阻断结果

{chr(10).join(warning_lines)}

## 解释限制

本验收只证明新 5.2.3 产物与已验收 5.2.2、共同生产转移和声明的固定策略诊断合同一致；它不证明动作限制的因果价值或重新优化价值。
""",
        encoding="utf-8",
    )
    return [additions, analysis, acceptance_report]
