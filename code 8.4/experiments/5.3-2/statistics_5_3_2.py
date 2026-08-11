"""Path-level seed aggregation, paired inference and precision for 5.3.2."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from tre84.inference import holm_adjust


LEARNING_POLICIES = {"Behaviour cloning", "Model-guided constrained SAC"}
CELL_KEYS = ["cell_id", "open_interval_weeks", "reclosure_intensity", "reclosure_duration_weeks"]
PAIR_METRICS = {
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
    "waiting_exposure": "waiting_model_unit_weeks",
    "committed_delivery": "committed_delivery",
    "adaptive_delivery": "adaptive_delivery",
    "clearance_probability": "seed_clearance_probability",
    "restricted_mean_clearance_time": "restricted_clearance_time_contribution",
    "terminal_outstanding_mass": "ending_outstanding_mass",
}


def interval(values: np.ndarray, confidence: float, family_size: int = 1) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if not n:
        return {"n": 0, "mean": np.nan, "se": np.nan, "lower": np.nan, "upper": np.nan}
    mean = float(values.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "se": np.nan, "lower": np.nan, "upper": np.nan}
    se = float(values.std(ddof=1) / np.sqrt(n))
    alpha = 1.0 - confidence
    critical = float(student_t.ppf(1.0 - alpha / (2.0 * max(family_size, 1)), n - 1))
    return {"n": n, "mean": mean, "se": se, "lower": mean - critical * se, "upper": mean + critical * se}


def one_sample_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    sd = float(values.std(ddof=1))
    mean = float(values.mean())
    if sd <= 1e-14:
        return 1.0 if abs(mean) <= 1e-14 else 0.0
    return float(2.0 * student_t.sf(abs(mean / (sd / np.sqrt(len(values)))), len(values) - 1))


def aggregate_learning_seeds(raw: pd.DataFrame) -> pd.DataFrame:
    identifiers = set(CELL_KEYS + [
        "policy", "path_id", "reclosure_path_id", "path_content_sha256",
        "training_seed", "clearance_status",
    ])
    numeric = [
        column for column in raw.columns
        if column not in identifiers
        and pd.api.types.is_numeric_dtype(raw[column])
        and not pd.api.types.is_bool_dtype(raw[column])
    ]
    booleans = [column for column in raw.columns if pd.api.types.is_bool_dtype(raw[column])]
    rows: list[dict[str, Any]] = []
    for keys, group in raw.groupby(CELL_KEYS + ["policy", "path_id"], sort=True):
        *cell_values, policy, path_id = keys
        expected = 3 if policy in LEARNING_POLICIES else 1
        observed = int(group["training_seed"].nunique(dropna=True)) if expected == 3 else len(group)
        if len(group) != expected or observed != expected:
            raise RuntimeError(f"{policy}/{path_id}/{cell_values} must have {expected} seed rows")
        record = dict(zip(CELL_KEYS, cell_values))
        record.update({
            "policy": policy,
            "path_id": path_id,
            "path_content_sha256": group["path_content_sha256"].iloc[0],
            "training_seed_count": expected,
            "seed_aggregation_applied_before_path_inference": expected == 3,
            "inference_unit": "physical_path",
        })
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
        record["clearance_status"] = "right_censored" if censored.all() else "mixed_across_training_seeds" if censored.any() else "cleared"
        record["clearance_weeks_observed"] = np.nan if censored.any() else float(group["clearance_weeks_observed"].mean())
        rows.append(record)
    return pd.DataFrame(rows)


def precision_requirements(path_level: pd.DataFrame, config: Mapping[str, Any], policies: Sequence[str]) -> tuple[pd.DataFrame, int]:
    design = config["path_design"]
    minimum = int(design["minimum_common_physical_paths"])
    cap = int(design["maximum_physical_paths"])
    target = float(design["target_halfwidth"])
    confidence = float(design["confidence_level"])
    coverage = config["layered_policy_coverage"]
    anchor_coordinates = [
        coverage["reference_cell"], coverage["mild_corner"], coverage["severe_corner"]
    ]
    anchor_ids = {
        cell_id_from_coordinates(item["open_interval_weeks"], item["reclosure_intensity"], item["reclosure_duration_weeks"])
        for item in anchor_coordinates
    }
    corners = path_level.loc[path_level["cell_id"].isin(anchor_ids)]
    rows = []
    for cell_id, cell in corners.groupby("cell_id"):
        if cell["path_id"].nunique() != minimum:
            raise RuntimeError(f"Precision corner {cell_id} lacks {minimum} paths")
        coordinates = {key: cell[key].iloc[0] for key in CELL_KEYS[1:]}
        for reference in ("Passive", "Reactive"):
            ref = cell.loc[cell["policy"] == reference, ["path_id", "total_operational_objective"]].rename(columns={"total_operational_objective": "reference_loss"})
            for policy in policies:
                if policy == reference:
                    continue
                paired = cell.loc[cell["policy"] == policy, ["path_id", "total_operational_objective"]].merge(ref, on="path_id", validate="one_to_one")
                differences = (paired["total_operational_objective"] - paired["reference_loss"]).to_numpy(float)
                sd = float(differences.std(ddof=1))
                critical = float(student_t.ppf(0.5 + confidence / 2.0, len(differences) - 1))
                raw_required = 2 if sd <= 1e-14 else int(math.ceil((critical * sd / target) ** 2))
                rows.append({
                    "cell_id": cell_id, **coordinates, "reference_policy": reference,
                    "policy": policy, "variance_estimation_paths": len(differences),
                    "paired_difference_standard_deviation": sd, "confidence_level": confidence,
                    "t_critical": critical, "target_halfwidth": target,
                    "raw_formula_required_paths": raw_required,
                    "minimum_common_paths": minimum, "maximum_path_cap": cap,
                    "required_paths": max(minimum, raw_required),
                    "required_within_cap": max(minimum, raw_required) <= cap,
                })
    frame = pd.DataFrame(rows)
    selected = int(min(max(int(frame["required_paths"].max()), minimum), cap))
    return frame, selected


def cell_id_from_coordinates(open_weeks: int, intensity: float, duration_weeks: int) -> str:
    severity = f"{float(intensity):.2f}".replace(".", "p")
    return f"open_{int(open_weeks):02d}__intensity_{severity}__duration_{int(duration_weeks):02d}"


def update_precision(requirements: pd.DataFrame, path_level: pd.DataFrame, executed: int) -> pd.DataFrame:
    output = requirements.copy()
    achieved = []
    for row in output.itertuples(index=False):
        cell = path_level.loc[path_level["cell_id"] == row.cell_id]
        ref = cell.loc[cell["policy"] == row.reference_policy, ["path_id", "total_operational_objective"]].rename(columns={"total_operational_objective": "reference_loss"})
        paired = cell.loc[cell["policy"] == row.policy, ["path_id", "total_operational_objective"]].merge(ref, on="path_id", validate="one_to_one")
        values = (paired["total_operational_objective"] - paired["reference_loss"]).to_numpy(float)
        ci = interval(values, float(row.confidence_level))
        achieved.append((ci["upper"] - ci["lower"]) / 2.0)
    output["executed_paths"] = executed
    output["achieved_halfwidth"] = achieved
    output["precision_target_met"] = output["achieved_halfwidth"] <= output["target_halfwidth"]
    return output


def paired_effects(path_level: pd.DataFrame, policies: Sequence[str], confidence: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric, column in PAIR_METRICS.items():
        for cell_id, cell in path_level.groupby("cell_id"):
            coordinates = {key: cell[key].iloc[0] for key in CELL_KEYS[1:]}
            cell_policies = [policy for policy in policies if policy in set(cell["policy"])]
            family_label = "five_policy_anchor" if len(cell_policies) == 5 else "three_policy_axial_corner"
            for reference in ("Passive", "Reactive"):
                family = []
                ref = cell.loc[cell["policy"] == reference, ["path_id", column]].rename(columns={column: "reference_value"})
                for policy in cell_policies:
                    if policy == reference:
                        continue
                    paired = cell.loc[cell["policy"] == policy, ["path_id", column]].merge(ref, on="path_id", validate="one_to_one")
                    values = (paired[column] - paired["reference_value"]).to_numpy(float)
                    standard = interval(values, confidence)
                    family.append({
                        "metric": metric, "source_column": column, "cell_id": cell_id,
                        **coordinates, "policy": policy, "reference_policy": reference,
                        "physical_paths": standard["n"], "mean_paired_difference": standard["mean"],
                        "standard_error": standard["se"], "paired_lower": standard["lower"],
                        "paired_upper": standard["upper"], "unadjusted_p_value": one_sample_p(values),
                        "_values": values,
                    })
                adjusted = holm_adjust([item["unadjusted_p_value"] if np.isfinite(item["unadjusted_p_value"]) else 1.0 for item in family])
                for item, adjusted_p in zip(family, adjusted):
                    simultaneous = interval(item.pop("_values"), confidence, len(family))
                    item["simultaneous_lower"] = simultaneous["lower"]
                    item["simultaneous_upper"] = simultaneous["upper"]
                    item["holm_adjusted_p_value"] = float(adjusted_p)
                    item["multiplicity_family_size"] = len(family)
                    item["comparison_family"] = family_label
                    item["evaluated_policy_set"] = "|".join(cell_policies)
                    item["seed_aggregated_within_path_first"] = True
                    item["inference_unit"] = "physical_path"
                    rows.append(item)
    return pd.DataFrame(rows)


def confidence_sets_and_regret(path_level: pd.DataFrame, policies: Sequence[str], confidence: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_rows, regret_rows = [], []
    for cell_id, cell in path_level.groupby("cell_id"):
        coordinates = {key: cell[key].iloc[0] for key in CELL_KEYS[1:]}
        pivot = cell.pivot(index="path_id", columns="policy", values="total_operational_objective")
        cell_policies = [policy for policy in policies if policy in set(pivot.columns)]
        expected_count = 5 if cell_id in set(path_level.loc[path_level["policy"].isin(["Projected stochastic MPC", "Model-guided constrained SAC"]), "cell_id"]) else 3
        if len(cell_policies) != expected_count:
            raise RuntimeError(f"Policy coverage in {cell_id} is {len(cell_policies)}, expected {expected_count}")
        family_size = len(cell_policies) * (len(cell_policies) - 1) // 2
        family_label = "five_policy_anchor" if len(cell_policies) == 5 else "three_policy_axial_corner"
        means = pivot[cell_policies].mean()
        leader = min(cell_policies, key=lambda item: (means[item], item))
        intervals = {}
        for policy in cell_policies:
            values = (pivot[policy] - pivot[leader]).to_numpy(float)
            ci = interval(values, confidence, family_size)
            intervals[policy] = ci
            set_rows.append({
                "cell_id": cell_id, **coordinates, "policy": policy, "sample_leader": leader,
                "mean_total_operational_loss": float(means[policy]),
                "mean_difference_from_leader": ci["mean"], "simultaneous_lower": ci["lower"],
                "simultaneous_upper": ci["upper"],
                "in_simultaneous_confidence_set": bool(policy == leader or ci["lower"] <= 0.0),
                "resolved_better_than_all_competitors": False,
                "all_pair_family_size": family_size,
                "comparison_family": family_label,
                "evaluated_policy_count": len(cell_policies),
                "evaluated_policy_set": "|".join(cell_policies),
            })
        unique = all(intervals[policy]["lower"] > 0.0 for policy in cell_policies if policy != leader)
        for item in set_rows[-len(cell_policies):]:
            if item["policy"] == leader:
                item["resolved_better_than_all_competitors"] = unique
        path_min = pivot[cell_policies].min(axis=1)
        for policy in cell_policies:
            ci = interval((pivot[policy] - path_min).to_numpy(float), confidence, len(cell_policies))
            regret_rows.append({
                "cell_id": cell_id, **coordinates, "policy": policy, "physical_paths": ci["n"],
                "mean_path_paired_regret": ci["mean"], "simultaneous_lower": ci["lower"],
                "simultaneous_upper": ci["upper"],
                "regret_reference": f"minimum loss among the {len(cell_policies)} preregistered policies on the same matched physical path and cell",
                "comparison_family": family_label,
                "evaluated_policy_count": len(cell_policies),
                "evaluated_policy_set": "|".join(cell_policies),
            })
    return pd.DataFrame(set_rows), pd.DataFrame(regret_rows)


def summaries(path_level: pd.DataFrame, clearance_cap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "total_operational_objective", "loss_queue", "loss_waiting", "loss_exit",
        "loss_overflow", "loss_route_resource", "loss_action", "terminal_correction",
        "direct_sue_exit", "duration_attrition", "waiting_model_unit_weeks",
        "committed_delivery", "adaptive_delivery", "ending_outstanding_mass",
    ]
    mechanism = path_level.groupby(CELL_KEYS + ["policy"], as_index=False)[metrics].mean(numeric_only=True)
    clearance_rows = []
    for keys, group in path_level.groupby(CELL_KEYS + ["policy"], sort=True):
        *cell_values, policy = keys
        censored = group["any_seed_right_censored"].astype(bool)
        clearance_rows.append({
            **dict(zip(CELL_KEYS, cell_values)), "policy": policy, "physical_paths": len(group),
            "clearance_probability": float(group["seed_clearance_probability"].mean()),
            "restricted_mean_clearance_time": float(group["restricted_clearance_time_contribution"].mean()),
            "restriction_weeks": clearance_cap,
            "mean_terminal_outstanding_mass": float(group["ending_outstanding_mass"].mean()),
            "right_censored_paths": int(censored.sum()),
            "right_censored_seed_trajectories": int(group["censored_seed_trajectories"].sum()),
            "censored_recorded_as_observed_clearance": bool(group.loc[censored, "clearance_weeks_observed"].notna().any()),
        })
    return mechanism, pd.DataFrame(clearance_rows)
