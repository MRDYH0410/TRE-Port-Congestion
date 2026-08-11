"""Path-level aggregation and paired mechanism inference for experiment 5.2.3."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from paths import PhysicalPath


MECHANISM_OUTCOMES = {
    "total_loss": "total_operational_objective",
    "waiting_exposure": "waiting_model_unit_weeks",
    "sue_exit": "direct_sue_exit",
    "attrition_exit": "duration_attrition",
    "overload": "overload",
    "route_resource_loss": "loss_route_resource",
    "action_loss": "loss_action",
    "clearance_probability": "clearance_probability_indicator",
    "restricted_mean_clearance_time": "restricted_clearance_time_contribution",
    "final_outstanding": "ending_outstanding_mass",
}


def select_mechanism_policy_set(
    confidence: pd.DataFrame,
    *,
    proposed_policy: str = "Model-guided constrained SAC",
) -> tuple[pd.DataFrame, str, list[str]]:
    ordered = confidence.sort_values(["mean_total_operational_loss", "policy"]).reset_index(drop=True)
    leader = str(ordered.iloc[0]["policy"])
    retained = set(
        ordered.loc[ordered["in_sample_best_confidence_set"].astype(bool), "policy"].astype(str)
    )
    retained.update({"Passive", leader, proposed_policy})
    rows = []
    for row in ordered.itertuples(index=False):
        policy = str(row.policy)
        rows.append(
            {
                "policy": policy,
                "mean_total_operational_loss_5_2_2": float(row.mean_total_operational_loss),
                "is_passive_reference": policy == "Passive",
                "is_benchmark_leader": policy == leader,
                "is_proposed_model_guided_policy": policy == proposed_policy,
                "in_5_2_2_sample_best_confidence_set": bool(row.in_sample_best_confidence_set),
                "retained_in_complete_mechanism_csv": policy in retained,
                "formal_figure_policy": policy in {"Passive", leader, proposed_policy},
                "evidence_label": (
                    "benchmark leader; not a universal optimum"
                    if policy == leader
                    else "proposed model-guided policy"
                    if policy == proposed_policy
                    else "passive reference"
                    if policy == "Passive"
                    else "5.2.2 confidence-set member"
                    if bool(row.in_sample_best_confidence_set)
                    else "complete benchmark policy"
                ),
            }
        )
    figure_policies = [
        policy
        for policy in ("Passive", leader, proposed_policy)
        if policy in retained and policy not in []
    ]
    figure_policies = list(dict.fromkeys(figure_policies))
    return pd.DataFrame(rows), leader, figure_policies


def _recovery_rate(serviceability: np.ndarray) -> float:
    minimum = int(np.argmin(serviceability))
    tail = np.asarray(serviceability[minimum:], dtype=float)
    if len(tail) <= 1:
        return 0.0
    x = np.arange(len(tail), dtype=float)
    return float(np.polyfit(x, tail, 1)[0])


def select_physical_path_medoid(paths: Sequence[PhysicalPath]) -> tuple[pd.DataFrame, str]:
    rows = []
    for path in paths:
        normal = path.frame["normal_model_units"].to_numpy(float)
        service = path.frame["serviceability"].to_numpy(float)
        blocked = normal * (1.0 - service)
        rows.append(
            {
                "path_id": path.path_id,
                "path_content_sha256": path.path_hash,
                "total_blocked_mass": float(blocked.sum()),
                "peak_weekly_blocked_mass": float(blocked.max()),
                "mean_serviceability": float(service.mean()),
                "minimum_serviceability": float(service.min()),
                "recovery_rate": _recovery_rate(service),
            }
        )
    frame = pd.DataFrame(rows)
    variables = [
        "total_blocked_mass",
        "peak_weekly_blocked_mass",
        "mean_serviceability",
        "minimum_serviceability",
        "recovery_rate",
    ]
    values = frame[variables].to_numpy(float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = (values - mean) / scale
    distance = np.sqrt(np.square(z).sum(axis=1))
    frame["standardised_distance_to_common_centroid"] = distance
    for index, variable in enumerate(variables):
        frame[f"z__{variable}"] = z[:, index]
    chosen = frame.sort_values(
        ["standardised_distance_to_common_centroid", "path_id"]
    ).iloc[0]["path_id"]
    frame["selected_physical_path_medoid"] = frame["path_id"].eq(chosen)
    frame["selection_uses_policy_outcomes"] = False
    return frame, str(chosen)


def aggregate_full_policy_mechanisms(path_level: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "total_operational_objective",
        "waiting_model_unit_weeks",
        "direct_sue_exit",
        "duration_attrition",
        "overload",
        "loss_route_resource",
        "loss_action",
        "ending_outstanding_mass",
        "restricted_clearance_time_contribution",
        "seed_clearance_probability",
    ]
    summary_rows: list[dict[str, Any]] = []
    for policy, group in path_level.groupby("policy", sort=False):
        row: dict[str, Any] = {"policy": policy, "physical_paths": group["path_id"].nunique()}
        for metric in metrics:
            values = group[metric].to_numpy(float)
            row[f"mean__{metric}"] = float(values.mean())
            row[f"path_sd__{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"path_q25__{metric}"] = float(np.quantile(values, 0.25))
            row[f"path_q75__{metric}"] = float(np.quantile(values, 0.75))
        summary_rows.append(row)
    paired_rows: list[dict[str, Any]] = []
    passive = path_level.loc[path_level["policy"] == "Passive"].set_index("path_id")
    for policy, group in path_level.groupby("policy", sort=False):
        if policy == "Passive":
            continue
        current = group.set_index("path_id")
        common = sorted(set(current.index) & set(passive.index))
        for metric in metrics:
            differences = (
                current.loc[common, metric].to_numpy(float)
                - passive.loc[common, metric].to_numpy(float)
            )
            n = len(differences)
            se = float(differences.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            critical = float(student_t.ppf(0.975, n - 1)) if n > 1 else np.nan
            paired_rows.append(
                {
                    "policy": policy,
                    "reference_policy": "Passive",
                    "outcome": metric,
                    "physical_paths": n,
                    "mean_paired_difference": float(differences.mean()),
                    "standard_error": se,
                    "paired_95_lower": float(differences.mean() - critical * se),
                    "paired_95_upper": float(differences.mean() + critical * se),
                    "inference_unit": "physical_path",
                    "learning_seeds_averaged_within_path_first": True,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def aggregate_weekly_policy_mechanisms(
    *,
    periods: pd.DataFrame,
    actions: pd.DataFrame,
    policies: Sequence[str],
    action_block_denominators: Mapping[str, float],
) -> pd.DataFrame:
    selected_periods = periods.loc[periods["policy"].isin(policies)].copy()
    stage_columns = {
        "queue_berth": [c for c in selected_periods if c.startswith("queue_before_berth_")],
        "queue_yard": [c for c in selected_periods if c.startswith("queue_before_yard_")],
        "queue_gate": [c for c in selected_periods if c.startswith("queue_before_gate_")],
        "queue_landbridge": [c for c in selected_periods if c.startswith("queue_before_corridor_")],
    }
    for name, columns in stage_columns.items():
        selected_periods[name] = selected_periods[columns].sum(axis=1)
    selected_actions = actions.loc[actions["policy"].isin(policies)].copy()
    prefixes = {
        "readiness_order": "implemented_readiness_order__",
        "direct_order": "implemented_direct_order__",
        "readiness_exercise": "implemented_readiness_exercise__",
        "release": "implemented_release__",
        "disclosure": "implemented_disclosure__",
    }
    for block, prefix in prefixes.items():
        columns = [column for column in selected_actions if column.startswith(prefix)]
        selected_actions[f"action_{block}_model_units"] = selected_actions[columns].sum(axis=1)
        selected_actions[f"action_{block}_reference_ratio"] = (
            selected_actions[f"action_{block}_model_units"]
            / float(action_block_denominators[block])
        )
    action_metrics = [
        column
        for column in selected_actions
        if column.startswith("action_") and column.endswith(("_model_units", "_reference_ratio"))
    ]
    action_by_seed_path = selected_actions.groupby(
        ["policy", "path_id", "training_seed", "period_offset"],
        dropna=False,
        as_index=False,
    )[action_metrics].mean()
    action_by_path = action_by_seed_path.groupby(
        ["policy", "path_id", "period_offset"], as_index=False
    )[action_metrics].mean()
    period_metrics = [
        "queue_loss",
        "waiting_loss",
        "direct_sue_exit_mass",
        "duration_attrition_mass",
        "overload_loss",
        "route_resource_loss",
        "action_loss",
        "delivered_landbridge",
        "waiting_before",
        "outstanding_after",
        *stage_columns,
    ]
    periods_by_seed_path = selected_periods.groupby(
        ["policy", "path_id", "training_seed", "scope", "period_offset"],
        dropna=False,
        as_index=False,
    )[period_metrics].mean()
    periods_by_path = periods_by_seed_path.groupby(
        ["policy", "path_id", "scope", "period_offset"], as_index=False
    )[period_metrics].mean()
    periods_by_path = periods_by_path.merge(
        action_by_path,
        on=["policy", "path_id", "period_offset"],
        how="left",
    )
    all_metrics = period_metrics + action_metrics
    rows: list[dict[str, Any]] = []
    for keys, group in periods_by_path.groupby(["policy", "scope", "period_offset"], sort=False):
        row: dict[str, Any] = {
            "policy": keys[0],
            "scope": keys[1],
            "period_offset": int(keys[2]),
            "physical_paths": group["path_id"].nunique(),
            "learning_seed_aggregation": "within_path_first",
        }
        for metric in all_metrics:
            values = group[metric].dropna().to_numpy(float)
            if len(values) == 0:
                row[f"mean__{metric}"] = np.nan
                row[f"q25__{metric}"] = np.nan
                row[f"q75__{metric}"] = np.nan
            else:
                row[f"mean__{metric}"] = float(values.mean())
                row[f"q25__{metric}"] = float(np.quantile(values, 0.25))
                row[f"q75__{metric}"] = float(np.quantile(values, 0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def proposed_policy_activation_audit(
    *,
    actions: pd.DataFrame,
    proposals: pd.DataFrame,
    projection_tolerance: float,
) -> pd.DataFrame:
    policy = "Model-guided constrained SAC"
    selected_actions = actions.loc[actions["policy"] == policy].copy()
    selected_proposals = proposals.loc[
        (proposals["policy"] == policy) & (proposals["evaluation_split"] == "test")
    ].copy()
    activation = 10.0 * float(projection_tolerance)
    rows: list[dict[str, Any]] = []
    decision_key = ["path_id", "training_seed", "period_offset"]
    for source in ("BC", "SAC"):
        count = int(
            selected_proposals.loc[
                selected_proposals["proposal_source"].eq(source)
                & selected_proposals["selected"].astype(bool)
            ].shape[0]
        )
        rows.append(
            {
                "module": "BC-SAC selector",
                "metric": f"{source}_proposal_selected_count",
                "value": count,
                "denominator": selected_proposals[decision_key].drop_duplicates().shape[0],
                "activation_rule": "selected by the formal common nested objective",
            }
        )
    fallback = int(
        selected_proposals.loc[selected_proposals["used_fallback"].astype(bool), decision_key]
        .drop_duplicates()
        .shape[0]
    )
    rows.append(
        {
            "module": "BC-SAC selector",
            "metric": "fallback_count",
            "value": fallback,
            "denominator": selected_proposals[decision_key].drop_duplicates().shape[0],
            "activation_rule": "fallback used after invalid proposals",
        }
    )
    block_prefixes = {
        "readiness order": "implemented_readiness_order__",
        "direct order": "implemented_direct_order__",
        "readiness exercise": "implemented_readiness_exercise__",
        "release": "implemented_release__",
        "disclosure": "implemented_disclosure__",
    }
    for module, prefix in block_prefixes.items():
        columns = [column for column in selected_actions if column.startswith(prefix)]
        block_values = selected_actions[columns].sum(axis=1).to_numpy(float)
        active_count = int(np.count_nonzero(block_values > activation))
        rows.extend(
            [
                {
                    "module": module,
                    "metric": "activation_count",
                    "value": active_count,
                    "denominator": len(block_values),
                    "activation_rule": f"implemented block total > 10 x projection tolerance ({activation:g})",
                },
                {
                    "module": module,
                    "metric": "range_across_test_decisions",
                    "value": float(block_values.max() - block_values.min()),
                    "denominator": len(block_values),
                    "activation_rule": "maximum minus minimum implemented block total",
                },
            ]
        )
    return pd.DataFrame(rows)


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def restricted_action_effects(
    replications: pd.DataFrame,
    *,
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = replications.copy()
    frame["clearance_probability_indicator"] = (~frame["right_censored"].astype(bool)).astype(float)
    baseline = frame.loc[frame["restriction"] == "full_action"].set_index("path_id")
    restrictions = [value for value in frame["restriction"].unique() if value != "full_action"]
    path_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    alpha = 1.0 - float(confidence_level)
    family_size = len(restrictions)
    for outcome, column in MECHANISM_OUTCOMES.items():
        outcome_effects: list[dict[str, Any]] = []
        for restriction in restrictions:
            current = frame.loc[frame["restriction"] == restriction].set_index("path_id")
            common = sorted(set(current.index) & set(baseline.index))
            differences = current.loc[common, column].to_numpy(float) - baseline.loc[common, column].to_numpy(float)
            for path_id, value in zip(common, differences):
                path_rows.append(
                    {
                        "restriction": restriction,
                        "reference": "full_action",
                        "outcome": outcome,
                        "path_id": path_id,
                        "paired_difference": float(value),
                        "inference_unit": "physical_path",
                    }
                )
            n = len(differences)
            mean = float(differences.mean())
            sd = float(differences.std(ddof=1)) if n > 1 else 0.0
            se = sd / np.sqrt(n) if n else np.nan
            critical = float(student_t.ppf(1.0 - alpha / 2.0, n - 1)) if n > 1 else np.nan
            simultaneous_critical = (
                float(student_t.ppf(1.0 - alpha / (2.0 * family_size), n - 1))
                if n > 1
                else np.nan
            )
            if n <= 1:
                p_value = np.nan
            elif sd == 0:
                p_value = 1.0 if mean == 0 else 0.0
            else:
                p_value = float(2.0 * student_t.sf(abs(mean / se), n - 1))
            outcome_effects.append(
                {
                    "restriction": restriction,
                    "reference": "full_action",
                    "outcome": outcome,
                    "physical_paths": n,
                    "mean_paired_difference": mean,
                    "paired_difference_sd": sd,
                    "standard_error": se,
                    "paired_95_lower": mean - critical * se,
                    "paired_95_upper": mean + critical * se,
                    "simultaneous_95_lower": mean - simultaneous_critical * se,
                    "simultaneous_95_upper": mean + simultaneous_critical * se,
                    "unadjusted_p_value": p_value,
                    "multiplicity_family": f"four restrictions within {outcome}",
                    "family_size": family_size,
                    "interpretation": "fixed-policy restricted-action diagnostic; not causal or reoptimised value",
                }
            )
        p = np.asarray([row["unadjusted_p_value"] for row in outcome_effects], dtype=float)
        adjusted = _holm_adjust(np.nan_to_num(p, nan=1.0))
        for row, value in zip(outcome_effects, adjusted):
            row["holm_adjusted_p_value"] = float(value)
            effect_rows.append(row)
    return pd.DataFrame(path_rows), pd.DataFrame(effect_rows)
