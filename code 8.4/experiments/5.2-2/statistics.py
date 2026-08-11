"""Path-count selection, seed aggregation, paired inference, and summaries."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

from tre84.inference import holm_adjust, student_interval


def aggregate_learning_seeds(
    replications: pd.DataFrame,
    *,
    learning_policies: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    learning = set(learning_policies)
    identifier_columns = {
        "policy",
        "path_id",
        "path_content_sha256",
        "released_information_path_sha256",
        "training_seed",
        "information_source",
        "projector_id",
        "kernel_id",
        "clearance_status",
        "physical_scope",
    }
    numeric_columns = [
        column
        for column in replications.columns
        if column not in identifier_columns
        and pd.api.types.is_numeric_dtype(replications[column])
        and not pd.api.types.is_bool_dtype(replications[column])
    ]
    bool_columns = [
        column for column in replications.columns if pd.api.types.is_bool_dtype(replications[column])
    ]
    for (policy, path_id), group in replications.groupby(["policy", "path_id"], sort=True):
        expected = group["training_seed"].nunique(dropna=True) if policy in learning else 1
        if policy in learning and expected < 3:
            raise ValueError(f"{policy}/{path_id} has fewer than three training seeds")
        if policy not in learning and len(group) != 1:
            raise ValueError(f"Non-learning policy {policy} was duplicated by a pseudo seed")
        record: dict[str, Any] = {
            "policy": policy,
            "path_id": path_id,
            "path_content_sha256": group["path_content_sha256"].iloc[0],
            "released_information_path_sha256": group[
                "released_information_path_sha256"
            ].iloc[0],
            "training_seed_count": int(expected),
            "seed_aggregation_applied_before_path_inference": policy in learning,
            "inference_unit": "physical_path",
            "information_source": group["information_source"].iloc[0],
            "projector_id": group["projector_id"].iloc[0],
            "kernel_id": group["kernel_id"].iloc[0],
        }
        for column in numeric_columns:
            record[column] = float(group[column].mean())
        for column in bool_columns:
            record[column] = bool(group[column].all())
        censored_count = int(group["right_censored"].astype(bool).sum())
        any_censored = censored_count > 0
        all_censored = censored_count == len(group)
        record["seed_clearance_probability"] = float(1.0 - censored_count / len(group))
        record["censored_seed_trajectories"] = censored_count
        record["any_seed_right_censored"] = any_censored
        record["all_seeds_right_censored"] = all_censored
        record["right_censored"] = any_censored
        record["clearance_status"] = (
            "right_censored"
            if all_censored
            else "mixed_across_training_seeds"
            if any_censored
            else "cleared"
        )
        record["clearance_weeks_observed"] = (
            np.nan if any_censored else float(group["clearance_weeks_observed"].mean())
        )
        rows.append(record)
    return pd.DataFrame(rows)


def select_path_count(
    *,
    pilot_path_level: pd.DataFrame,
    config: Mapping[str, Any],
    reference_failure_loss: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    passive = pilot_path_level.loc[
        pilot_path_level["policy"] == "Passive",
        ["path_id", "total_operational_objective"],
    ].rename(columns={"total_operational_objective": "passive_loss"})
    pilot_count = int(config["paths"]["pilot_count"])
    if len(passive) != pilot_count:
        raise ValueError("Pilot precision must use the prespecified physical-path count")
    confidence = float(config["paths"]["confidence_level"])
    alpha = 1.0 - confidence
    target = (
        float(config["paths"]["target_halfwidth_fraction_of_reference_failure"])
        * reference_failure_loss
    )
    critical = float(student_t.ppf(1.0 - alpha / 2.0, df=pilot_count - 1))
    minimum = int(config["paths"]["minimum_final_count"])
    maximum = int(config["paths"]["maximum_final_count"])
    rows = []
    for policy in config["main_policies"]:
        if policy == "Passive":
            continue
        frame = pilot_path_level.loc[
            pilot_path_level["policy"] == policy,
            ["path_id", "total_operational_objective"],
        ].merge(passive, on="path_id", validate="one_to_one")
        differences = frame["total_operational_objective"] - frame["passive_loss"]
        sd = float(differences.std(ddof=1))
        raw_required = 2 if sd == 0 else int(math.ceil((critical * sd / target) ** 2))
        bounded_required = max(minimum, raw_required)
        within_cap = bounded_required <= maximum
        rows.append(
            {
                "policy": policy,
                "pilot_paths": pilot_count,
                "paired_difference_standard_deviation": sd,
                "confidence_level": confidence,
                "pilot_t_critical": critical,
                "target_halfwidth": target,
                "reference_failure_loss": reference_failure_loss,
                "raw_formula_required_paths": raw_required,
                "minimum_final_paths": minimum,
                "maximum_final_paths": maximum,
                "required_paths": bounded_required,
                "required_within_computational_cap": within_cap,
            }
        )
    pilot = pd.DataFrame(rows)
    selected_required = int(pilot["required_paths"].max())
    executed = min(selected_required, maximum)
    selection = pd.DataFrame(
        [
            {
                "selection_rule": "maximum paired-precision requirement across all main policies",
                "pilot_paths": pilot_count,
                "required_paths": selected_required,
                "executed_paths": executed,
                "target_halfwidth": target,
                "computational_cap": maximum,
                "required_paths_within_cap": selected_required <= maximum,
                "precision_target_met": False,
                "maximum_achieved_halfwidth": np.nan,
            }
        ]
    )
    return pilot, selection


def paired_policy_effects(
    path_level: pd.DataFrame,
    *,
    policies: Sequence[str],
    confidence_level: float,
) -> pd.DataFrame:
    passive = path_level.loc[
        path_level["policy"] == "Passive", ["path_id", "total_operational_objective"]
    ].rename(columns={"total_operational_objective": "passive_loss"})
    family = [policy for policy in policies if policy != "Passive"]
    rows = []
    p_values = []
    for policy in family:
        frame = path_level.loc[
            path_level["policy"] == policy,
            ["path_id", "total_operational_objective"],
        ].merge(passive, on="path_id", validate="one_to_one")
        differences = (frame["total_operational_objective"] - frame["passive_loss"]).to_numpy()
        standard = student_interval(differences, confidence_level=confidence_level)
        simultaneous = student_interval(
            differences,
            confidence_level=confidence_level,
            family_size=len(family),
        )
        p_value = float(ttest_1samp(differences, popmean=0.0).pvalue)
        p_values.append(p_value)
        rows.append(
            {
                "policy": policy,
                "reference_policy": "Passive",
                "physical_paths": standard.count,
                "mean_paired_difference": standard.mean,
                "standard_error": standard.standard_error,
                "paired_95_lower": standard.lower,
                "paired_95_upper": standard.upper,
                "simultaneous_95_lower": simultaneous.lower,
                "simultaneous_95_upper": simultaneous.upper,
                "unadjusted_p_value": p_value,
                "holm_adjusted_p_value": np.nan,
                "multiplicity_family_size": len(family),
                "learning_seeds_aggregated_within_path_first": True,
                "inference_unit": "physical_path",
            }
        )
    adjusted = holm_adjust(p_values)
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p_value"] = float(value)
    return pd.DataFrame(rows)


def policy_confidence_set(
    path_level: pd.DataFrame,
    *,
    policies: Sequence[str],
    confidence_level: float,
) -> pd.DataFrame:
    performance: dict[str, Any] = {}
    for policy in policies:
        values = path_level.loc[
            path_level["policy"] == policy, "total_operational_objective"
        ].to_numpy()
        interval = student_interval(values, confidence_level=confidence_level)
        performance[policy] = interval
    sample_best = min(policies, key=lambda policy: (performance[policy].mean, policy))
    all_pairs = len(policies) * (len(policies) - 1) // 2
    best = path_level.loc[
        path_level["policy"] == sample_best,
        ["path_id", "total_operational_objective"],
    ].rename(columns={"total_operational_objective": "best_loss"})
    rows = []
    for policy in policies:
        values = path_level.loc[
            path_level["policy"] == policy,
            ["path_id", "total_operational_objective"],
        ].merge(best, on="path_id", validate="one_to_one")
        differences = values["total_operational_objective"] - values["best_loss"]
        if policy == sample_best:
            lower = upper = mean_difference = 0.0
        else:
            interval = student_interval(
                differences,
                confidence_level=confidence_level,
                family_size=all_pairs,
            )
            lower, upper, mean_difference = interval.lower, interval.upper, interval.mean
        in_set = lower <= 0.0
        rows.append(
            {
                "policy": policy,
                "mean_total_operational_loss": performance[policy].mean,
                "path_95_lower": performance[policy].lower,
                "path_95_upper": performance[policy].upper,
                "sample_best_policy": sample_best,
                "mean_difference_from_sample_best": mean_difference,
                "simultaneous_difference_lower": lower,
                "simultaneous_difference_upper": upper,
                "all_pair_family_size": all_pairs,
                "in_sample_best_confidence_set": in_set,
                "resolved_better_than_all_competitors": False,
            }
        )
    result = pd.DataFrame(rows)
    for policy in policies:
        if policy != sample_best:
            continue
        comparisons = result.loc[result["policy"] != policy]
        result.loc[result["policy"] == policy, "resolved_better_than_all_competitors"] = bool(
            (comparisons["simultaneous_difference_lower"] > 0).all()
        )
    return result


def loss_component_summary(path_level: pd.DataFrame) -> pd.DataFrame:
    component_columns = {
        "queue": "loss_queue",
        "waiting": "loss_waiting",
        "exit": "loss_exit",
        "overload": "loss_overflow",
        "route_resource_and_transport": "loss_route_resource",
        "action": "loss_action",
        "terminal": "terminal_correction",
    }
    rows = []
    for policy, group in path_level.groupby("policy", sort=False):
        record: dict[str, Any] = {"policy": policy, "physical_paths": len(group)}
        for output, source in component_columns.items():
            record[output] = float(group[source].mean())
        record["component_sum"] = float(sum(record[key] for key in component_columns))
        record["mean_total_operational_loss"] = float(
            group["total_operational_objective"].mean()
        )
        record["loss_identity_residual"] = (
            record["component_sum"] - record["mean_total_operational_loss"]
        )
        record["mean_direct_sue_exit_loss"] = float(group["loss_direct_sue_exit"].mean())
        record["mean_duration_attrition_loss"] = float(
            group["loss_duration_attrition"].mean()
        )
        rows.append(record)
    return pd.DataFrame(rows)


def clearance_summary(path_level: pd.DataFrame, cap: int) -> pd.DataFrame:
    rows = []
    for policy, group in path_level.groupby("policy", sort=False):
        censored = group["any_seed_right_censored"].astype(bool)
        all_censored = group["all_seeds_right_censored"].astype(bool)
        rows.append(
            {
                "policy": policy,
                "physical_paths": len(group),
                "clearance_probability": float(group["seed_clearance_probability"].mean()),
                "restricted_mean_clearance_time": float(
                    group["restricted_clearance_time_contribution"].mean()
                ),
                "restriction_weeks": cap,
                "mean_observed_clearance_time_among_cleared": (
                    float(group.loc[~censored, "clearance_weeks_observed"].mean())
                    if (~censored).any()
                    else np.nan
                ),
                "mean_final_outstanding_mass": float(group["ending_outstanding_mass"].mean()),
                "mean_final_outstanding_mass_censored": (
                    float(group.loc[censored, "ending_outstanding_mass"].mean())
                    if censored.any()
                    else 0.0
                ),
                "number_censored_paths": int(censored.sum()),
                "number_all_seed_censored_paths": int(all_censored.sum()),
                "number_censored_seed_trajectories": int(
                    group["censored_seed_trajectories"].sum()
                ),
                "mean_terminal_loss": float(group["terminal_correction"].mean()),
                "censored_clearance_time_recorded_as_104": bool(
                    group.loc[censored, "clearance_weeks_observed"].notna().any()
                ),
            }
        )
    return pd.DataFrame(rows)


def policy_activation_summary(
    actions: pd.DataFrame,
    *,
    action_names: Sequence[str],
    learning_policies: Sequence[str],
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    path_rows = []
    learning = set(learning_policies)
    for (policy, path_id), group in actions.groupby(["policy", "path_id"], sort=False):
        record: dict[str, Any] = {"policy": policy, "path_id": path_id}
        seed_count = group["training_seed"].nunique(dropna=True)
        record["training_seed_count"] = seed_count if policy in learning else 1
        for name in action_names:
            values = group[f"implemented_{name}"].to_numpy(dtype=float)
            record[f"mean_{name}"] = float(values.mean())
            record[f"activation_rate_{name}"] = float((values > tolerance).mean())
        path_rows.append(record)
    path_frame = pd.DataFrame(path_rows)
    return path_frame.groupby("policy", as_index=False).mean(numeric_only=True)


def decision_time_summary(actions: pd.DataFrame) -> pd.DataFrame:
    return (
        actions.groupby("policy", as_index=False)["decision_time_seconds"]
        .agg(
            mean_decision_time_seconds="mean",
            median_decision_time_seconds="median",
            maximum_decision_time_seconds="max",
            decisions="count",
        )
    )


def update_precision_achievement(
    *,
    pilot_precision: pd.DataFrame,
    selection: pd.DataFrame,
    paired_effects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    achieved = paired_effects.set_index("policy").apply(
        lambda row: (row["paired_95_upper"] - row["paired_95_lower"]) / 2.0,
        axis=1,
    )
    pilot = pilot_precision.copy()
    pilot["executed_paths"] = int(selection.loc[0, "executed_paths"])
    pilot["achieved_halfwidth"] = pilot["policy"].map(achieved)
    pilot["precision_target_met"] = pilot["achieved_halfwidth"] <= pilot["target_halfwidth"]
    selection = selection.copy()
    selection.loc[0, "maximum_achieved_halfwidth"] = float(pilot["achieved_halfwidth"].max())
    selection.loc[0, "precision_target_met"] = bool(pilot["precision_target_met"].all())
    return pilot, selection
