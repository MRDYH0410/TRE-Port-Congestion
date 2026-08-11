"""Code-generated figures and reports for Experiment 5.2.5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"Conventional MSA": "#8c6d31", "RC-MSA": "#2166ac"}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 120,
        "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.22,
    })


def _save(fig: plt.Figure, figures: Path, stem: str) -> list[str]:
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / f"{stem}.png"
    pdf = figures / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png.name, pdf.name]


def figure_a(rc_trace: pd.DataFrame, rc_summary: pd.DataFrame, precision: pd.DataFrame, output: Path, figures: Path) -> list[str]:
    _style()
    data = []
    tmp = rc_trace.copy(); tmp.insert(0, "panel", "A_residual_trace"); data.append(tmp)
    tmp = rc_summary.copy(); tmp.insert(0, "panel", "B_terminal_and_iterations"); data.append(tmp)
    tmp = precision.copy(); tmp.insert(0, "panel", "C_mpc_scenario_prefix"); data.append(tmp)
    pd.concat(data, ignore_index=True, sort=False).to_csv(output / "figure_5_2_5a_data.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3))
    for algorithm, group in rc_trace.groupby("algorithm"):
        for _, case in group.groupby("case_id"):
            axes[0].plot(case["iteration"], case["equilibrium_residual"], color=COLORS[algorithm], alpha=.22, lw=.8)
        median = group.groupby("iteration")["equilibrium_residual"].median()
        axes[0].plot(median.index, median.values, color=COLORS[algorithm], lw=2.2, label=algorithm)
    tolerance = float(rc_trace["tolerance"].iloc[0])
    axes[0].axhline(tolerance, color="#b2182b", ls="--", lw=1.1, label=f"Tolerance {tolerance:.0e}")
    axes[0].set_yscale("log"); axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Equilibrium residual")
    axes[0].set_title("A. Same-problem fixed-point convergence"); axes[0].grid(True, which="both"); axes[0].legend(frameon=False)
    xmap = {name: i for i, name in enumerate(COLORS)}
    ax2 = axes[1].twinx()
    for algorithm, group in rc_summary.groupby("algorithm"):
        axes[1].scatter(np.full(len(group), xmap[algorithm])-.07, group["terminal_residual"], s=36, color=COLORS[algorithm], marker="o")
        ax2.scatter(np.full(len(group), xmap[algorithm])+.08, group["iterations"], s=32, facecolors="none", edgecolors=COLORS[algorithm], marker="s")
    axes[1].axhline(tolerance, color="#b2182b", ls="--", lw=1)
    axes[1].set_yscale("log"); axes[1].set_xticks(list(xmap.values()), list(xmap)); axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_ylabel("Terminal residual (filled circles)"); ax2.set_ylabel("Iterations (open squares)")
    axes[1].set_title("B. Terminal residual and work")
    x = precision["scenario_count"].to_numpy(float); y = precision["out_of_sample_objective"].to_numpy(float)
    half = precision["confidence_half_width"].to_numpy(float)
    finite = np.isfinite(half)
    axes[2].plot(x, y, "o-", color="#4d9221", lw=2, label="Out-of-sample objective")
    axes[2].errorbar(x[finite], y[finite], yerr=half[finite], fmt="none", color="#4d9221", capsize=3)
    for row in precision.itertuples(index=False):
        axes[2].annotate(str(row.selected_first_action_profile), (row.scenario_count, row.out_of_sample_objective), xytext=(3, 6), textcoords="offset points", fontsize=7, rotation=18)
    axes[2].set_xlabel("Nested scenario-prefix count"); axes[2].set_ylabel("Full-bundle objective")
    axes[2].set_title("C. MPC scenario-prefix stability"); axes[2].grid(True); axes[2].set_xticks(x)
    fig.suptitle("Figure 5.2.5a  Numerical convergence and solver acceptance", fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, figures, "figure_5_2_5a_numerical_convergence")


def figure_b(
    bc: pd.DataFrame, validation: pd.DataFrame, sac: pd.DataFrame,
    gradient: pd.DataFrame, selector: pd.DataFrame, regret: pd.DataFrame,
    output: Path, figures: Path,
) -> list[str]:
    _style()
    data = []
    for panel, frame in (("A_BC_training", bc), ("A_validation", validation), ("B_SAC_training", sac), ("C_gradient", gradient), ("D_selector", selector), ("E_regret", regret)):
        tmp = frame.copy(); tmp.insert(0, "panel", panel); data.append(tmp)
    pd.concat(data, ignore_index=True, sort=False).to_csv(output / "figure_5_2_5b_data.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.0))
    ax = axes[0, 0]
    for seed, group in bc.groupby("seed_index"):
        ax.plot(group["episode"], group["training_loss"], lw=1.4, alpha=.75, label=f"seed {seed}")
    ax.set_yscale("log"); ax.set_xlabel("Episode"); ax.set_ylabel("Imitation loss"); ax.set_title("A. BC training across all frozen seeds"); ax.grid(True); ax.legend(frameon=False, ncol=3)
    ax = axes[0, 1]
    sac_val = validation[validation["policy"].isin(["Vanilla SAC", "Constrained SAC"])]
    for (policy, seed), group in sac_val.groupby(["policy", "seed_index"]):
        ax.plot(group["episode"], group["validation_operational_loss"], alpha=.65, lw=1.2, label=f"{policy}, s{seed}")
    ax.set_xlabel("Checkpoint episode"); ax.set_ylabel("Validation system loss"); ax.set_title("B. Validation-only checkpoint evidence"); ax.grid(True); ax.legend(frameon=False, fontsize=6, ncol=2)
    ax = axes[0, 2]
    for policy, group in sac.groupby("policy"):
        trace = group.groupby("episode", as_index=False)[["critic_loss_q1", "critic_loss_q2", "constraint_critic_loss"]].median()
        ax.plot(trace["episode"], trace["critic_loss_q1"], lw=1.5, label=f"{policy}: Q1")
        ax.plot(trace["episode"], trace["critic_loss_q2"], lw=1.1, ls="--", label=f"{policy}: Q2")
        if policy == "Constrained SAC":
            ax.plot(trace["episode"], trace["constraint_critic_loss"], lw=1.2, ls=":", label="Constrained: Qg")
    ax.set_yscale("log"); ax.set_xlabel("Episode"); ax.set_ylabel("Squared critic loss")
    ax.set_title("C. Twin reward and constraint critics"); ax.grid(True); ax.legend(frameon=False, fontsize=6)
    ax = axes[1, 0]
    for policy, group in sac.groupby("policy"):
        trace = group.groupby("episode", as_index=False)[["entropy_temperature", "mean_log_standard_deviation", "constraint_dual"]].median()
        ax.plot(trace["episode"], trace["entropy_temperature"], lw=1.6, label=f"{policy}: temperature")
        ax.plot(trace["episode"], trace["mean_log_standard_deviation"], lw=1.1, ls="--", label=f"{policy}: mean log std")
        if policy == "Constrained SAC":
            ax.plot(trace["episode"], trace["constraint_dual"], lw=1.2, ls=":", label="Constrained: dual")
    ax.set_xlabel("Episode"); ax.set_ylabel("Recorded adaptive value"); ax.set_title("D. Stochasticity and dual updates"); ax.grid(True); ax.legend(frameon=False, fontsize=6)
    ax = axes[1, 1]
    x = gradient["finite_difference_gradient_recalculated"].to_numpy(float)
    y = gradient["analytic_gradient_recalculated"].to_numpy(float)
    ax.scatter(x, y, s=20, alpha=.75, color="#1b7837")
    low, high = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    ax.plot([low, high], [low, high], color="black", ls="--", lw=1)
    ax.set_xlabel("Central finite difference"); ax.set_ylabel("Analytic projected actor gradient")
    ax.set_title("E. Independent gradient reconstruction"); ax.grid(True)
    ax = axes[1, 2]
    pairs = selector.pivot_table(index=["evaluation_split", "path_id", "training_seed", "period_offset"], columns="proposal_source", values="nested_objective").dropna()
    if {"BC", "SAC"}.issubset(pairs.columns):
        ax.scatter(pairs["BC"], pairs["SAC"], s=8, alpha=.35, color="#762a83")
        low = float(min(pairs["BC"].min(), pairs["SAC"].min())); high = float(max(pairs["BC"].max(), pairs["SAC"].max()))
        ax.plot([low, high], [low, high], ls="--", color="black", lw=1)
    ax.set_xlabel("Formal nested objective: BC"); ax.set_ylabel("Formal nested objective: SAC"); ax.set_title("F. Mechanical BC-SAC nested selector"); ax.grid(True)
    regret_values = regret["selector_ex_post_regret"].dropna().to_numpy(float)
    if len(regret_values):
        inset = ax.inset_axes([0.64, 0.08, 0.32, 0.36])
        inset.boxplot(regret_values, vert=True, widths=.55, patch_artist=True,
                      boxprops={"facecolor": "#f4a582", "alpha": .7},
                      medianprops={"color": "#67001f"})
        inset.axhline(0, color="black", lw=.7, ls="--")
        inset.set_xticks([1], ["regret"]); inset.tick_params(labelsize=6)
        inset.set_ylabel("ex-post", fontsize=6)
    fig.suptitle("Figure 5.2.5b  Learning and selector acceptance", fontweight="bold", y=.995)
    fig.tight_layout(rect=[0, 0, 1, .98])
    return _save(fig, figures, "figure_5_2_5b_learning_selector")


def figure_c(registry: pd.DataFrame, reproducibility: pd.DataFrame, runtime: pd.DataFrame, output: Path, figures: Path) -> list[str]:
    _style()
    critical = registry[registry["critical"]].copy()
    critical["normalised_residual"] = critical["maximum_observed_residual"] / critical["tolerance"].replace(0, np.nan)
    zero_tol = critical["tolerance"].eq(0)
    critical.loc[zero_tol, "normalised_residual"] = np.where(critical.loc[zero_tol, "maximum_observed_residual"].eq(0), 0.0, np.inf)
    critical["plot_residual"] = critical["normalised_residual"].clip(lower=1e-12, upper=1e4)
    a = critical.copy(); a.insert(0, "panel", "A_normalised_contract_residual")
    b = reproducibility.copy(); b.insert(0, "panel", "B_anchor_reproduction")
    c = runtime.copy(); c.insert(0, "panel", "C_runtime_profile")
    pd.concat([a, b, c], ignore_index=True, sort=False).to_csv(output / "figure_5_2_5c_data.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.3), gridspec_kw={"width_ratios": [1.2, .8, 1]})
    y = np.arange(len(critical))
    colors = critical["status"].map({"PASS": "#1b7837", "FAIL": "#b2182b", "BLOCKED": "#d6604d", "NOT_TESTED": "#969696"})
    axes[0].hlines(y, 1e-12, critical["plot_residual"], color=colors, lw=2)
    axes[0].scatter(critical["plot_residual"], y, color=colors, s=42, zorder=3)
    axes[0].axvline(1, color="black", ls="--", lw=1.2, label="Acceptance boundary z=1")
    axes[0].set_xscale("log"); axes[0].set_yticks(y, critical["contract_id"]); axes[0].invert_yaxis(); axes[0].set_xlabel("Normalised maximum residual z")
    axes[0].set_title("A. Critical end-to-end contracts"); axes[0].grid(True, axis="x", which="both"); axes[0].legend(frameon=False)
    replay = reproducibility.groupby("experiment", as_index=False).agg(
        maximum_difference=("maximum_difference", "max"),
        failed=("status", lambda values: int((values != "PASS").sum())),
    )
    axes[1].stem(replay["experiment"], replay["maximum_difference"], basefmt=" ", linefmt="#762a83", markerfmt="o")
    axes[1].axhline(float(reproducibility["tolerance"].max()), color="#b2182b", ls="--", lw=1, label="registered tolerance")
    axes[1].set_yscale("symlog", linthresh=1e-12); axes[1].set_ylabel("Maximum path-level replay difference")
    axes[1].set_title("B. Accepted-anchor reproduction"); axes[1].grid(True, axis="y"); axes[1].legend(frameon=False)
    measured = runtime[runtime["runtime_p50_seconds"].notna()].sort_values("runtime_p50_seconds")
    y2 = np.arange(len(measured))
    axes[2].hlines(y2, measured["runtime_p50_seconds"], measured["runtime_p95_seconds"], color="#0571b0", lw=3)
    axes[2].scatter(measured["runtime_p50_seconds"], y2, marker="o", color="#0571b0", label="p50")
    axes[2].scatter(measured["runtime_p90_seconds"], y2, marker="s", color="#ca0020", label="p90")
    axes[2].scatter(measured["runtime_p95_seconds"], y2, marker="|", s=90, color="#ca0020", label="p95")
    axes[2].set_xscale("log"); axes[2].set_yticks(y2, measured["algorithm"]); axes[2].set_xlabel("Wall-clock seconds per recorded call")
    axes[2].set_title("C. Computational profile (no real-time claim)"); axes[2].grid(True, axis="x", which="both"); axes[2].legend(frameon=False)
    fig.suptitle("Figure 5.2.5c  End-to-end contract and computational profile", fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, figures, "figure_5_2_5c_contract_runtime")


def write_reports(
    report_dir: Path, summary: Mapping[str, Any], registry: pd.DataFrame,
    parameters: pd.DataFrame, runtime: pd.DataFrame,
    sac_contracts: pd.DataFrame, reproducibility: pd.DataFrame,
    precision_summary: pd.DataFrame, rc_summary: pd.DataFrame,
    selector_regret: pd.DataFrame,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    failed = registry[registry["status"].ne("PASS")]
    critical_failed = registry[registry["critical"] & registry["status"].ne("PASS")]
    lines = [
        "# 5.2.5 Computational and Methodological Acceptance Report", "",
        f"Overall outcome: **{summary['OVERALL_ACCEPTANCE']}**.", "",
        "## Independent acceptance layers", "",
    ]
    for key in ("ENGINEERING_ACCEPTANCE", "NUMERICAL_ACCEPTANCE", "METHODOLOGY_CONTRACT_ACCEPTANCE", "EXPERIMENTAL_EVIDENCE_ACCEPTANCE", "OVERALL_ACCEPTANCE"):
        lines.append(f"- {key}: **{summary[key]}**")
    precision = precision_summary.iloc[0]
    lines += ["", "## Interpretation", "", f"All {summary['critical_contracts_total']} critical production-method contracts were executed; {summary['critical_contracts_passed']} passed. The paired design independently recovers {int(precision.executed_paths)} physical paths, with three learning seeds aggregated within path. Its maximum half-width is {float(precision.maximum_achieved_halfwidth):,.3f} against the registered {float(precision.target_halfwidth):,.3f} target.", "", f"The complete SAC gate evaluates {len(sac_contracts)} critical learning contracts: latent Gaussian sampling, actor mean and log-standard-deviation updates, entropy in the actor objective, adaptive temperature, twin reward critics, the constraint critic and dual, the projection-gradient chain, finite differences, validation-only selection, and checkpoint replay. These are reconstructed from persisted production traces rather than inferred from checkpoint readability.", "", "The RC-MSA comparison is a numerical fixed-point diagnostic, not a policy comparison. MPC exactness is established only over the preregistered finite candidate lattice. Selector regret is relative only to the frozen BC and SAC candidates. Runtime results are a computational profile because no external operational response deadline is registered."]
    reinforced = registry[registry["contract_id"].isin([
        "M27_RCMSA_MASTER_CHOICE_DISTANCE",
        "M28_DISCLOSURE_REFERENCE_ACTION",
        "M29_WAITING_VINTAGE_NO_RESET",
        "M30_MPC_SELECTOR_MODULE_CERTIFICATES",
        "M31_CORE_REPAIR_NUMERICAL_EQUIVALENCE",
    ])]
    lines += [
        "",
        "## Chapter 4 contract reinforcement",
        "",
        f"The refresh adds five noncompensatory checks: complete-master RC-MSA history distance, interface-level $a_t^{{-I}}$, independent per-vintage no-reset identities, complete MPC/selector module certificates, and pre/post-repair numerical equivalence. {int(reinforced['status'].eq('PASS').sum())}/{len(reinforced)} passed.",
        "",
        "These are audit-contract reinforcements. They do not change the frozen network, costs, paths, checkpoints, policy definitions, training settings, or the accepted 5.2.1--5.2.4 results.",
    ]
    rc = rc_summary.groupby("algorithm").agg(converged=("converged", "sum"), cases=("case_id", "size"), median_iterations=("iterations", "median"), maximum_terminal_residual=("terminal_residual", "max"))
    selector_mean = float(selector_regret["selector_ex_post_regret"].mean())
    selector_interval = (float(selector_regret["selector_ex_post_regret"].min()), float(selector_regret["selector_ex_post_regret"].max()))
    lines += ["", "## Numerical and selector findings", "", f"RC-MSA converged in {int(rc.loc['RC-MSA','converged'])}/{int(rc.loc['RC-MSA','cases'])} controlled production problems, with median {float(rc.loc['RC-MSA','median_iterations']):.0f} iterations and maximum terminal residual {float(rc.loc['RC-MSA','maximum_terminal_residual']):.3e}. Conventional MSA converged in {int(rc.loc['Conventional MSA','converged'])}/{int(rc.loc['Conventional MSA','cases'])} within the common 500-iteration cap; this negative comparator result is retained and is not an acceptance failure for RC-MSA.", "", f"The BC-SAC selector was mechanically consistent in every recorded decision, but its path-level ex-post regret relative to the better frozen candidate averaged {selector_mean:,.3f} and ranged from {selector_interval[0]:,.3f} to {selector_interval[1]:,.3f}. This is negative candidate-set evidence and does not contradict mechanical selector acceptance.", "", "## Missing or weaker evidence", ""]
    lines += ["- SLSQP primal feasibility and an independent projection objective/Jacobian check are available; solver dual and complementarity multipliers were not persisted, so that noncritical KKT subdiagnostic remains `NOT_TESTED`.", "- BC action-coordinate validation errors were not persisted; aggregate training and validation imitation loss remain available.", "- Historical training wall time and peak memory were not persisted, and no retrospective values are fabricated.", "- The three-scenario MPC prefix analysis is diagnostic; the formal statistical evidence gate remains the 88-path paired precision calculation.", ""]
    if len(critical_failed):
        lines += ["## Failed critical method contracts", ""] + [f"- {row.contract_id}: {row.failure_reason}" for row in critical_failed.itertuples(index=False)] + [""]
    elif len(failed):
        lines += ["## Noncritical NOT_TESTED diagnostics", ""] + [f"- {row.contract_id}: {row.failure_reason}" for row in failed.itertuples(index=False)] + [""]
    lines += ["## Upstream reproduction", "", f"The audit contains {len(reproducibility):,} path/checkpoint replay rows across 5.2.2--5.2.4. Maximum accepted-anchor difference: {float(reproducibility['maximum_difference'].max()):.3e}.", ""]
    lines += ["## Evidence boundary", "", "This experiment verifies that the named methods are connected to the production chain and reports numerical acceptance. It does not prove global optimality, universal policy superiority, causal validity for real ports, or real-time deployability.", ""]
    (report_dir / "METHODOLOGICAL_ACCEPTANCE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    parameter_lines = ["# 5.1 Parameter and Diagnostic Additions from Experiment 5.2.5", "", "**DATA: NO CHANGE.** Experiment 5.2.5 introduces no new empirical data, event window, HMM estimate, or upstream policy result.", "", "**NUMERICAL PARAMETERS: NO ADDITION.** The refresh adds acceptance certificates and one machine-precision equivalence check; it does not change an economic, physical, statistical-design, training, or solver parameter.", "", "The following registry is code generated for later 5.1 updating. Values are not inserted into the manuscript automatically.", "", "| Parameter | Value | Basis category | Basis |", "|---|---:|---|---|"]
    for row in parameters.itertuples(index=False):
        parameter_lines.append(f"| {row.parameter} | {str(row.value).replace('|','/')} | {row.basis_category} | {row.basis} |")
    parameter_lines += ["", "New diagnostic scope: the RC-MSA selector distance now has a complete-master support certificate; disclosure-reference construction has an interface-level a_t^{-I} certificate; waiting transition has a per-vintage no-reset certificate; MPC and the two-proposal selector persist complete nested module certificates and mechanical selection logs; accepted scientific quantities are compared against the pre-repair baseline at 1e-12. These are metrics/contracts, not new model parameters.", "", "SAC critic, entropy, adaptive-temperature and dual traces remain audited directly; they are not parameter additions. Remaining noncritical instrumentation gaps are persisted solver KKT multipliers, BC action-coordinate validation errors, historical training wall time, and peak memory. No convenient default is introduced for any gap.", ""]
    (report_dir / "5_1_PARAMETER_ADDITIONS.md").write_text("\n".join(parameter_lines), encoding="utf-8")
