"""Seed aggregation, endpoint precision, paired inference, and summaries for 5.3.1."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from tre84.inference import holm_adjust


LEARNING_POLICIES = {
    "Behaviour cloning",
    "Model-guided constrained SAC",
}


PAIR_METRICS: dict[str, str] = {
    "total_operational_loss": "total_operational_objective",
    "queue_loss": "loss_queue",
    "waiting_loss": "loss_waiting",
    "exit_loss": "loss_exit",
    "overload_loss": "loss_overflow",
    "route_resource_loss": "loss_route_resource",
    "action_loss": "loss_action",
    "terminal_loss": "terminal_correction",
    "sue_exit_mass": "direct_sue_exit",
    "attrition_exit_mass": "duration_attrition",
    "waiting_exposure": "waiting_exposure",
    "committed_delivery": "committed_delivery",
    "adaptive_delivery": "adaptive_delivery",
    "clearance_probability": "seed_clearance_probability",
    "restricted_mean_clearance_time": "restricted_clearance_time_contribution",
    "terminal_outstanding_mass": "ending_outstanding_mass",
}


def _interval(
    values: np.ndarray,
    *,
    confidence_level: float,
    family_size: int = 1,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "se": np.nan, "lower": np.nan, "upper": np.nan}
    mean = float(values.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "se": np.nan, "lower": np.nan, "upper": np.nan}
    se = float(values.std(ddof=1) / np.sqrt(n))
    alpha = 1.0 - float(confidence_level)
    critical = float(student_t.ppf(1.0 - alpha / (2.0 * max(family_size, 1)), n - 1))
    half = critical * se
    return {"n": n, "mean": mean, "se": se, "lower": mean - half, "upper": mean + half}


def _one_sample_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    sd = float(values.std(ddof=1))
    mean = float(values.mean())
    if sd <= 1e-14:
        return 1.0 if abs(mean) <= 1e-14 else 0.0
    statistic = mean / (sd / np.sqrt(values.size))
    return float(2.0 * student_t.sf(abs(statistic), values.size - 1))


def aggregate_learning_seeds(replications: pd.DataFrame) -> pd.DataFrame:
    """Average learning seeds within each physical path before inference."""

    identifiers = {
        "chi",
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
        "restriction",
    }
    numeric = [
        column
        for column in replications.columns
        if column not in identifiers
        and pd.api.types.is_numeric_dtype(replications[column])
        and not pd.api.types.is_bool_dtype(replications[column])
    ]
    booleans = [
        column
        for column in replications.columns
        if pd.api.types.is_bool_dtype(replications[column])
    ]
    rows: list[dict[str, Any]] = []
    for (chi, policy, path_id), group in replications.groupby(
        ["chi", "policy", "path_id"], sort=True
    ):
        learning = policy in LEARNING_POLICIES
        if learning:
            seeds = int(group["training_seed"].nunique(dropna=True))
            if seeds != 3 or len(group) != 3:
                raise RuntimeError(
                    f"{policy}/{chi}/{path_id} must have exactly three learning seeds"
                )
        else:
            seeds = 1
            if len(group) != 1:
                raise RuntimeError(
                    f"Non-learning policy {policy}/{chi}/{path_id} was duplicated"
                )
        record: dict[str, Any] = {
            "chi": float(chi),
            "policy": policy,
            "path_id": path_id,
            "path_content_sha256": group["path_content_sha256"].iloc[0],
            "released_information_path_sha256": group[
                "released_information_path_sha256"
            ].iloc[0],
            "training_seed_count": seeds,
            "seed_aggregation_applied_before_path_inference": learning,
            "inference_unit": "physical_path",
            "information_source": group["information_source"].iloc[0],
            "projector_id": group["projector_id"].iloc[0],
            "kernel_id": group["kernel_id"].iloc[0],
        }
        for column in numeric:
            record[column] = float(group[column].mean())
        for column in booleans:
            record[column] = bool(group[column].all())
        censored = group["right_censored"].astype(bool)
        record["seed_clearance_probability"] = float(1.0 - censored.mean())
        record["censored_seed_trajectories"] = int(censored.sum())
        record["any_seed_right_censored"] = bool(censored.any())
        record["all_seeds_right_censored"] = bool(censored.all())
        record["right_censored"] = bool(censored.any())
        record["clearance_status"] = (
            "right_censored"
            if censored.all()
            else "mixed_across_training_seeds"
            if censored.any()
            else "cleared"
        )
        record["clearance_weeks_observed"] = (
            np.nan
            if censored.any()
            else float(group["clearance_weeks_observed"].mean())
        )
        rows.append(record)
    return pd.DataFrame(rows)


def endpoint_precision(
    path_level_88: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    policies: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    """Use both preregistered endpoints to select one common path count."""

    design = config["path_design"]
    minimum = int(design["minimum_common_physical_paths"])
    cap = int(design["maximum_physical_paths"])
    target = float(design["target_halfwidth"])
    confidence = float(design["confidence_level"])
    endpoints = [float(value) for value in design["precision_endpoints"]]
    rows: list[dict[str, Any]] = []
    for chi in endpoints:
        cell = path_level_88.loc[np.isclose(path_level_88["chi"], chi)]
        if cell["path_id"].nunique() != minimum:
            raise RuntimeError(f"Endpoint chi={chi} does not contain the required 88 paths")
        for reference in ("Passive", "Reactive"):
            ref = cell.loc[
                cell["policy"] == reference,
                ["path_id", "total_operational_objective"],
            ].rename(columns={"total_operational_objective": "reference_loss"})
            for policy in policies:
                if policy == reference:
                    continue
                current = cell.loc[
                    cell["policy"] == policy,
                    ["path_id", "total_operational_objective"],
                ].merge(ref, on="path_id", validate="one_to_one")
                differences = (
                    current["total_operational_objective"] - current["reference_loss"]
                ).to_numpy(dtype=float)
                sd = float(differences.std(ddof=1))
                critical = float(
                    student_t.ppf(0.5 + confidence / 2.0, len(differences) - 1)
                )
                raw = 2 if sd <= 1e-14 else int(math.ceil((critical * sd / target) ** 2))
                required = max(minimum, raw)
                rows.append(
                    {
                        "chi": chi,
                        "reference_policy": reference,
                        "policy": policy,
                        "variance_estimation_paths": len(differences),
                        "paired_difference_standard_deviation": sd,
                        "confidence_level": confidence,
                        "t_critical": critical,
                        "target_halfwidth": target,
                        "raw_formula_required_paths": raw,
                        "minimum_common_paths": minimum,
                        "maximum_path_cap": cap,
                        "required_paths": required,
                        "required_within_cap": required <= cap,
                    }
                )
    frame = pd.DataFrame(rows)
    return frame, int(min(max(int(frame["required_paths"].max()), minimum), cap))


def update_endpoint_precision(
    requirements: pd.DataFrame,
    final_path_level: pd.DataFrame,
    *,
    executed_paths: int,
) -> pd.DataFrame:
    output = requirements.copy()
    achieved: list[float] = []
    for row in output.itertuples(index=False):
        cell = final_path_level.loc[np.isclose(final_path_level["chi"], row.chi)]
        ref = cell.loc[
            cell["policy"] == row.reference_policy,
            ["path_id", "total_operational_objective"],
        ].rename(columns={"total_operational_objective": "reference_loss"})
        current = cell.loc[
            cell["policy"] == row.policy,
            ["path_id", "total_operational_objective"],
        ].merge(ref, on="path_id", validate="one_to_one")
        interval = _interval(
            (
                current["total_operational_objective"] - current["reference_loss"]
            ).to_numpy(dtype=float),
            confidence_level=float(row.confidence_level),
        )
        achieved.append((interval["upper"] - interval["lower"]) / 2.0)
    output["executed_paths"] = int(executed_paths)
    output["achieved_halfwidth"] = achieved
    output["precision_target_met"] = (
        output["achieved_halfwidth"] <= output["target_halfwidth"]
    )
    return output


def paired_effects(
    path_level: pd.DataFrame,
    *,
    policies: Sequence[str],
    confidence_level: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric_name, column in PAIR_METRICS.items():
        for reference in ("Passive", "Reactive"):
            family_rows: list[dict[str, Any]] = []
            for chi in sorted(path_level["chi"].unique()):
                cell = path_level.loc[np.isclose(path_level["chi"], chi)]
                ref = cell.loc[cell["policy"] == reference, ["path_id", column]].rename(
                    columns={column: "reference_value"}
                )
                for policy in policies:
                    if policy == reference:
                        continue
                    current = cell.loc[
                        cell["policy"] == policy, ["path_id", column]
                    ].merge(ref, on="path_id", validate="one_to_one")
                    differences = (
                        current[column] - current["reference_value"]
                    ).to_numpy(dtype=float)
                    standard = _interval(
                        differences, confidence_level=confidence_level
                    )
                    family_rows.append(
                        {
                            "metric": metric_name,
                            "source_column": column,
                            "chi": float(chi),
                            "policy": policy,
                            "reference_policy": reference,
                            "physical_paths": standard["n"],
                            "mean_paired_difference": standard["mean"],
                            "standard_error": standard["se"],
                            "paired_lower": standard["lower"],
                            "paired_upper": standard["upper"],
                            "unadjusted_p_value": _one_sample_p(differences),
                            "_differences": differences,
                        }
                    )
            family_size = len(family_rows)
            finite_p = [
                row["unadjusted_p_value"]
                if np.isfinite(row["unadjusted_p_value"])
                else 1.0
                for row in family_rows
            ]
            adjusted = holm_adjust(finite_p)
            for row, adjusted_p in zip(family_rows, adjusted):
                simultaneous = _interval(
                    row.pop("_differences"),
                    confidence_level=confidence_level,
                    family_size=family_size,
                )
                row["simultaneous_lower"] = simultaneous["lower"]
                row["simultaneous_upper"] = simultaneous["upper"]
                row["holm_adjusted_p_value"] = float(adjusted_p)
                row["multiplicity_family_size"] = family_size
                row["seed_aggregated_within_path_first"] = True
                row["inference_unit"] = "physical_path"
                rows.append(row)
    return pd.DataFrame(rows)


def confidence_sets_and_regret(
    path_level: pd.DataFrame,
    *,
    policies: Sequence[str],
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    pair_family = len(policies) * (len(policies) - 1) // 2
    for chi in sorted(path_level["chi"].unique()):
        cell = path_level.loc[np.isclose(path_level["chi"], chi)]
        pivot = cell.pivot(
            index="path_id", columns="policy", values="total_operational_objective"
        )
        if set(policies) - set(pivot.columns):
            raise RuntimeError(f"Missing policies at chi={chi}")
        means = pivot[list(policies)].mean()
        leader = min(policies, key=lambda policy: (means[policy], policy))
        for policy in policies:
            differences = (pivot[policy] - pivot[leader]).to_numpy(dtype=float)
            interval = _interval(
                differences,
                confidence_level=confidence_level,
                family_size=pair_family,
            )
            set_rows.append(
                {
                    "chi": float(chi),
                    "policy": policy,
                    "sample_leader": leader,
                    "mean_total_operational_loss": float(means[policy]),
                    "mean_difference_from_leader": interval["mean"],
                    "simultaneous_lower": interval["lower"],
                    "simultaneous_upper": interval["upper"],
                    "in_simultaneous_confidence_set": bool(
                        policy == leader or interval["lower"] <= 0.0
                    ),
                    "all_pair_family_size": pair_family,
                    "resolved_better_than_all_competitors": False,
                }
            )
        path_minimum = pivot[list(policies)].min(axis=1)
        for policy in policies:
            regret = (pivot[policy] - path_minimum).to_numpy(dtype=float)
            interval = _interval(
                regret,
                confidence_level=confidence_level,
                family_size=len(policies),
            )
            regret_rows.append(
                {
                    "chi": float(chi),
                    "policy": policy,
                    "physical_paths": interval["n"],
                    "mean_path_paired_regret": interval["mean"],
                    "simultaneous_lower": interval["lower"],
                    "simultaneous_upper": interval["upper"],
                    "regret_reference": "minimum loss among five main policies on the same path and chi",
                }
            )
    confidence = pd.DataFrame(set_rows)
    for chi, group in confidence.groupby("chi"):
        leader = group["sample_leader"].iloc[0]
        competitors = group.loc[group["policy"] != leader]
        unique = bool((competitors["simultaneous_lower"] > 0.0).all())
        confidence.loc[
            np.isclose(confidence["chi"], chi) & (confidence["policy"] == leader),
            "resolved_better_than_all_competitors",
        ] = unique
    return confidence, pd.DataFrame(regret_rows)


def mechanism_summary(path_level: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "total_operational_objective",
        "loss_queue",
        "loss_waiting",
        "loss_exit",
        "loss_overflow",
        "loss_route_resource",
        "loss_action",
        "terminal_correction",
        "direct_sue_exit",
        "duration_attrition",
        "waiting_exposure",
        "committed_delivery",
        "adaptive_delivery",
        "committed_delivery_share",
        "terminal_committed_outstanding",
        "terminal_committed_outstanding_share",
        "ending_outstanding_mass",
    ]
    return (
        path_level.groupby(["chi", "policy"], as_index=False)[columns]
        .mean(numeric_only=True)
        .rename(columns={column: f"mean_{column}" for column in columns})
    )


def clearance_summary(path_level: pd.DataFrame, cap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (chi, policy), group in path_level.groupby(["chi", "policy"], sort=True):
        censored = group["any_seed_right_censored"].astype(bool)
        rows.append(
            {
                "chi": float(chi),
                "policy": policy,
                "physical_paths": len(group),
                "clearance_probability": float(group["seed_clearance_probability"].mean()),
                "restricted_mean_clearance_time": float(
                    group["restricted_clearance_time_contribution"].mean()
                ),
                "restriction_weeks": int(cap),
                "mean_observed_clearance_time_among_cleared": (
                    float(group.loc[~censored, "clearance_weeks_observed"].mean())
                    if (~censored).any()
                    else np.nan
                ),
                "mean_terminal_outstanding_mass": float(
                    group["ending_outstanding_mass"].mean()
                ),
                "mean_terminal_committed_outstanding": float(
                    group["terminal_committed_outstanding"].mean()
                ),
                "right_censored_paths": int(censored.sum()),
                "right_censored_seed_trajectories": int(
                    group["censored_seed_trajectories"].sum()
                ),
                "terminal_loss": float(group["terminal_correction"].mean()),
                "censored_recorded_as_observed_clearance": bool(
                    group.loc[censored, "clearance_weeks_observed"].notna().any()
                ),
            }
        )
    return pd.DataFrame(rows)
