"""Seed-first aggregation and matched-path inference for experiment 5.2.4."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

from tre84.inference import holm_adjust, student_interval


GROUP_COLUMNS = [
    "evidence_layer",
    "controller_id",
    "training_information_regime",
    "evaluation_information_regime",
    "capacity_rights",
    "warning_scenario",
    "base_path_id",
    "base_physical_path_sha256",
]


def aggregate_learning_seeds(replications: pd.DataFrame) -> pd.DataFrame:
    if replications.empty:
        return replications.copy()
    numeric = [
        column
        for column in replications.columns
        if pd.api.types.is_numeric_dtype(replications[column])
        and not pd.api.types.is_bool_dtype(replications[column])
        and column not in {"training_seed"}
    ]
    boolean = [
        column for column in replications.columns if pd.api.types.is_bool_dtype(replications[column])
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in replications.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        if group["training_seed"].nunique() != 3:
            raise RuntimeError("Every learning controller/path comparison requires three seeds")
        row = dict(zip(GROUP_COLUMNS, keys))
        row["training_seed_count"] = 3
        row["learning_seeds_averaged_within_path_first"] = True
        row["inference_unit"] = "physical_path"
        for column in numeric:
            row[column] = float(group[column].mean())
        for column in boolean:
            row[column] = bool(group[column].all())
        censored = group["right_censored"].astype(bool)
        row["seed_clearance_probability"] = float(1.0 - censored.mean())
        row["censored_seed_trajectories"] = int(censored.sum())
        row["right_censored"] = bool(censored.any())
        row["clearance_weeks_observed"] = (
            np.nan if censored.any() else float(group["clearance_weeks_observed"].mean())
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_interval(values: np.ndarray, confidence: float, resamples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def _seed(namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode("utf-8")).digest()[:4], "big")


def _effect_row(
    *,
    differences: np.ndarray,
    layer: str,
    scenario: str,
    comparison: str,
    estimand: str,
    confidence: float,
    family_size: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    standard = student_interval(differences, confidence_level=confidence)
    simultaneous = student_interval(
        differences, confidence_level=confidence, family_size=family_size
    )
    boot_low, boot_high = _bootstrap_interval(
        differences,
        confidence,
        bootstrap_resamples,
        _seed(f"5.2.4|{layer}|{scenario}|{comparison}"),
    )
    p_value = float(ttest_1samp(differences, 0.0).pvalue)
    return {
        "evidence_layer": layer,
        "warning_scenario": scenario,
        "comparison": comparison,
        "estimand": estimand,
        "physical_paths": len(differences),
        "mean_paired_effect": standard.mean,
        "standard_error": standard.standard_error,
        "paired_95_lower": standard.lower,
        "paired_95_upper": standard.upper,
        "simultaneous_95_lower": simultaneous.lower,
        "simultaneous_95_upper": simultaneous.upper,
        "paired_bootstrap_95_lower": boot_low,
        "paired_bootstrap_95_upper": boot_high,
        "unadjusted_p_value": p_value,
        "holm_adjusted_p_value": np.nan,
        "multiplicity_family_size": family_size,
        "learning_seeds_aggregated_within_path_first": True,
        "inference_unit": "physical_path",
    }


def information_effects(
    path_level: pd.DataFrame,
    *,
    confidence: float,
    bootstrap_resamples: int,
    fixed: bool,
) -> pd.DataFrame:
    regimes = ["IF", "IL"] if fixed else ["IF", "IL", "ORACLE"]
    family_size = 4 * len(regimes)
    rows: list[dict[str, Any]] = []
    for scenario in ("GH", "GT", "GL", "GFW"):
        subset = path_level.loc[path_level["warning_scenario"].eq(scenario)]
        baseline = subset.loc[
            subset["evaluation_information_regime"].eq("I0"),
            ["base_path_id", "total_operational_objective"],
        ].rename(columns={"total_operational_objective": "baseline"})
        for regime in regimes:
            alternative = subset.loc[
                subset["evaluation_information_regime"].eq(regime),
                ["base_path_id", "total_operational_objective"],
            ].rename(columns={"total_operational_objective": "alternative"})
            merged = baseline.merge(alternative, on="base_path_id", validate="one_to_one")
            differences = (merged["baseline"] - merged["alternative"]).to_numpy(dtype=float)
            rows.append(
                _effect_row(
                    differences=differences,
                    layer="fixed_policy_information_responsiveness" if fixed else "reoptimized_information_value",
                    scenario=scenario,
                    comparison=f"{regime} vs I0",
                    estimand=(
                        "fixed IL checkpoint response; not information value"
                        if fixed
                        else "J_I0 - J_I; positive means lower loss than the baseline information controller"
                    ),
                    confidence=confidence,
                    family_size=family_size,
                    bootstrap_resamples=bootstrap_resamples,
                )
            )
        if not fixed:
            filtered = subset.loc[
                subset["evaluation_information_regime"].eq("IF"),
                ["base_path_id", "total_operational_objective"],
            ].rename(columns={"total_operational_objective": "filtered"})
            lead = subset.loc[
                subset["evaluation_information_regime"].eq("IL"),
                ["base_path_id", "total_operational_objective"],
            ].rename(columns={"total_operational_objective": "lead"})
            oracle = subset.loc[
                subset["evaluation_information_regime"].eq("ORACLE"),
                ["base_path_id", "total_operational_objective"],
            ].rename(columns={"total_operational_objective": "oracle"})
            delta = filtered.merge(lead, on="base_path_id").merge(oracle, on="base_path_id")
            for comparison, values, estimand in (
                ("IL vs IF", delta["filtered"] - delta["lead"], "J_IF - J_IL; lead forecast incremental value"),
                ("oracle gap", delta["lead"] - delta["oracle"], "J_IL - J_ORACLE; unattainable upper-bound gap"),
            ):
                rows.append(
                    _effect_row(
                        differences=values.to_numpy(dtype=float),
                        layer="reoptimized_information_value",
                        scenario=scenario,
                        comparison=comparison,
                        estimand=estimand,
                        confidence=confidence,
                        family_size=8,
                        bootstrap_resamples=bootstrap_resamples,
                    )
                )
    result = pd.DataFrame(rows)
    for layer, index in result.groupby("evidence_layer").groups.items():
        result.loc[index, "holm_adjusted_p_value"] = holm_adjust(
            result.loc[index, "unadjusted_p_value"].to_numpy(dtype=float)
        )
    return result


def capacity_effects(
    path_level: pd.DataFrame,
    *,
    confidence: float,
    bootstrap_resamples: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    family_size = 12
    for scenario in ("GH", "GT", "GL", "GFW"):
        subset = path_level.loc[path_level["warning_scenario"].eq(scenario)]
        pivot = subset.pivot(
            index="base_path_id",
            columns="capacity_rights",
            values="total_operational_objective",
        )
        comparisons = {
            "V_R_given_D": (pivot["D"] - pivot["RD"], "J_D - J_RD; readiness value when direct procurement exists"),
            "V_D_given_R": (pivot["R"] - pivot["RD"], "J_R - J_RD; direct procurement value when readiness exists"),
            "S_RD": (pivot["R"] + pivot["D"] - pivot["RD"] - pivot["NONE"], "positive complementarity; negative substitutability"),
        }
        for comparison, (values, estimand) in comparisons.items():
            rows.append(
                _effect_row(
                    differences=values.to_numpy(dtype=float),
                    layer="reoptimized_capacity_rights",
                    scenario=scenario,
                    comparison=comparison,
                    estimand=estimand,
                    confidence=confidence,
                    family_size=family_size,
                    bootstrap_resamples=bootstrap_resamples,
                )
            )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_value"] = holm_adjust(
        result["unadjusted_p_value"].to_numpy(dtype=float)
    )
    return result


def false_warning_costs(path_level: pd.DataFrame) -> pd.DataFrame:
    subset = path_level.loc[path_level["warning_scenario"].eq("GFW")]
    baseline = subset.loc[
        subset["evaluation_information_regime"].eq("I0"),
        ["base_path_id", "total_operational_objective"],
    ].rename(columns={"total_operational_objective": "baseline_loss"})
    rows = []
    for regime in ("IF", "IL", "ORACLE"):
        alternative = subset.loc[
            subset["evaluation_information_regime"].eq(regime),
            ["base_path_id", "total_operational_objective", "loss_action"],
        ].rename(
            columns={
                "total_operational_objective": "regime_loss",
                "loss_action": "regime_action_loss",
            }
        )
        merged = baseline.merge(alternative, on="base_path_id", validate="one_to_one")
        merged["information_regime"] = regime
        merged["false_warning_cost"] = merged["regime_loss"] - merged["baseline_loss"]
        merged["interpretation"] = "positive means information increased loss without a physical disruption"
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def loss_decomposition(path_level: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "evidence_layer",
        "controller_id",
        "evaluation_information_regime",
        "capacity_rights",
        "warning_scenario",
    ]
    columns = {
        "queue": "loss_queue",
        "waiting": "loss_waiting",
        "sue_exit": "loss_direct_sue_exit",
        "attrition_exit": "loss_duration_attrition",
        "overload": "loss_overflow",
        "route_resource": "loss_route_resource",
        "action": "loss_action",
        "terminal": "terminal_correction",
        "total": "total_operational_objective",
    }
    rows = []
    for keys, group in path_level.groupby(groups, sort=True):
        row = dict(zip(groups, keys))
        row["physical_paths"] = len(group)
        for output, source in columns.items():
            row[output] = float(group[source].mean())
        row["component_sum"] = sum(row[name] for name in columns if name != "total")
        row["loss_identity_residual"] = row["component_sum"] - row["total"]
        rows.append(row)
    return pd.DataFrame(rows)


def clearance_summary(path_level: pd.DataFrame, cap: int) -> pd.DataFrame:
    groups = [
        "evidence_layer",
        "controller_id",
        "evaluation_information_regime",
        "capacity_rights",
        "warning_scenario",
    ]
    rows = []
    for keys, group in path_level.groupby(groups, sort=True):
        censored = group["right_censored"].astype(bool)
        rows.append(
            {
                **dict(zip(groups, keys)),
                "physical_paths": len(group),
                "clearance_probability": float(group["seed_clearance_probability"].mean()),
                "restricted_mean_clearance_time": float(group["restricted_clearance_time_contribution"].mean()),
                "restriction_weeks": cap,
                "number_censored_paths": int(censored.sum()),
                "mean_final_outstanding": float(group["ending_outstanding_mass"].mean()),
                "mean_terminal_loss": float(group["terminal_correction"].mean()),
                "censored_clearance_recorded_as_cap": bool(
                    group.loc[censored, "clearance_weeks_observed"].notna().any()
                ),
            }
        )
    return pd.DataFrame(rows)


def precision_audit(
    effects: pd.DataFrame,
    *,
    pilot_paths: int,
    executed_paths: int,
    target_halfwidth: float,
    confidence: float,
    maximum_paths: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    critical = float(student_t.ppf(1.0 - (1.0 - confidence) / 2.0, df=pilot_paths - 1))
    rows = []
    # The saved effect table contains full-path uncertainty. Recover a conservative SD from its standard error.
    for row in effects.itertuples(index=False):
        sd = float(row.standard_error) * math.sqrt(int(row.physical_paths))
        required = 2 if sd == 0 else int(math.ceil((critical * sd / target_halfwidth) ** 2))
        achieved = (float(row.paired_95_upper) - float(row.paired_95_lower)) / 2.0
        rows.append(
            {
                "evidence_layer": row.evidence_layer,
                "warning_scenario": row.warning_scenario,
                "comparison": row.comparison,
                "pilot_paths": pilot_paths,
                "pilot_t_critical": critical,
                "paired_difference_standard_deviation": sd,
                "target_halfwidth": target_halfwidth,
                "raw_required_paths": required,
                "executed_paths": executed_paths,
                "maximum_computational_paths": maximum_paths,
                "achieved_halfwidth": achieved,
                "precision_target_met": achieved <= target_halfwidth,
                "required_within_computational_cap": required <= maximum_paths,
            }
        )
    audit = pd.DataFrame(rows)
    selection = pd.DataFrame(
        [
            {
                "selection_rule": "inherit 5.2.2 matched-path cap and audit every preregistered paired 5.2.4 contrast",
                "required_paths_maximum": int(audit["raw_required_paths"].max()),
                "executed_paths": executed_paths,
                "target_halfwidth": target_halfwidth,
                "maximum_computational_paths": maximum_paths,
                "maximum_achieved_halfwidth": float(audit["achieved_halfwidth"].max()),
                "all_precision_targets_met": bool(audit["precision_target_met"].all()),
            }
        ]
    )
    return audit, selection
