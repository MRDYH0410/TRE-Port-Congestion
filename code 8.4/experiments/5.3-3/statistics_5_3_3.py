"""Path-level aggregation and matched inference for Experiment 5.3.3."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats


LEARNING_POLICIES = {"Behaviour cloning", "Model-guided constrained SAC"}
CELL_KEYS = ["cell_id", "gateway_count", "architecture", "eligibility"]


def _interval(values: np.ndarray, confidence: float, family: int = 1) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values)) if n else np.nan
    if n < 2:
        return {"mean": mean, "standard_error": np.nan, "halfwidth": np.nan, "lower": np.nan, "upper": np.nan}
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    alpha = (1.0 - confidence) / max(family, 1)
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    halfwidth = critical * se
    return {"mean": mean, "standard_error": se, "halfwidth": halfwidth, "lower": mean - halfwidth, "upper": mean + halfwidth}


def aggregate_learning_seeds(replications: pd.DataFrame) -> pd.DataFrame:
    identifiers = CELL_KEYS + ["policy", "path_id"]
    excluded = set(identifiers + ["training_seed"])
    numeric = [column for column in replications.select_dtypes(include=[np.number, "bool"]).columns if column not in excluded]
    rows = []
    for keys, group in replications.groupby(identifiers, sort=True):
        policy = str(keys[4])
        expected = 3 if policy in LEARNING_POLICIES else 1
        observed = int(group["training_seed"].nunique(dropna=True)) if policy in LEARNING_POLICIES else len(group)
        if observed != expected:
            raise RuntimeError(f"{keys} has {observed} runs; expected {expected}")
        row = dict(zip(identifiers, keys))
        row["training_seed_count"] = expected
        for column in numeric:
            row[column] = float(group[column].astype(float).mean())
        for column in ("path_content_sha256", "released_information_path_sha256", "clearance_status"):
            if column in group:
                values = group[column].astype(str).unique()
                row[column] = values[0] if len(values) == 1 else "MIXED"
        row["right_censored"] = bool(group["right_censored"].astype(bool).any())
        rows.append(row)
    return pd.DataFrame(rows)


def policy_summary(path_level: pd.DataFrame, confidence: float) -> pd.DataFrame:
    rows = []
    for keys, group in path_level.groupby(CELL_KEYS + ["policy"], sort=True):
        interval = _interval(group["total_operational_objective"].to_numpy(), confidence)
        rows.append(
            {
                **dict(zip(CELL_KEYS + ["policy"], keys)),
                "physical_paths": group["path_id"].nunique(),
                "mean_total_operational_loss": interval["mean"],
                "loss_lower": interval["lower"],
                "loss_upper": interval["upper"],
                "mean_waiting_exposure": group["waiting_exposure"].mean(),
                "mean_delivery": group["delivery"].mean(),
                "clearance_probability": 1.0 - group["right_censored"].astype(float).mean(),
                "restricted_mean_clearance_time": group["restricted_clearance_time_contribution"].mean(),
                "mean_terminal_outstanding": group["terminal_outstanding"].mean(),
                "mean_corridor_overload_exposure": group["corridor_overload_exposure"].mean(),
                "mean_resource_week_overload": group["resource_week_overload"].mean(),
                "mean_overloaded_gateway_incidence": group["overloaded_gateway_incidence"].mean(),
                "mean_decision_time_seconds": group["mean_decision_time_seconds"].mean(),
                "maximum_decision_time_seconds": group["maximum_decision_time_seconds"].max(),
            }
        )
    return pd.DataFrame(rows)


def policy_regret(path_level: pd.DataFrame, confidence: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    regret_rows = []
    confidence_rows = []
    for cell_id, cell in path_level.groupby("cell_id", sort=True):
        pivot = cell.pivot(index="path_id", columns="policy", values="total_operational_objective")
        policies = sorted(pivot.columns)
        family = max(len(policies) - 1, 1)
        means = pivot.mean()
        leader = min(policies, key=lambda policy: (means[policy], policy))
        path_min = pivot.min(axis=1)
        metadata = cell.iloc[0][CELL_KEYS].to_dict()
        for policy in policies:
            regret = _interval((pivot[policy] - path_min).to_numpy(), confidence, family)
            difference = _interval((pivot[policy] - pivot[leader]).to_numpy(), confidence, family)
            regret_rows.append({**metadata, "policy": policy, **{f"regret_{k}": v for k, v in regret.items()}})
            confidence_rows.append(
                {
                    **metadata,
                    "sample_leader": leader,
                    "policy": policy,
                    "mean_difference_from_leader": difference["mean"],
                    "simultaneous_lower": difference["lower"],
                    "simultaneous_upper": difference["upper"],
                    "in_simultaneous_confidence_set": bool(policy == leader or difference["lower"] <= 0.0),
                    "policy_family_size": len(policies),
                }
            )
    return pd.DataFrame(regret_rows), pd.DataFrame(confidence_rows)


def _paired_cell_effect(
    path_level: pd.DataFrame,
    left_id: str,
    right_id: str,
    policy: str,
    confidence: float,
    family: int,
) -> dict[str, Any] | None:
    left = path_level.loc[(path_level["cell_id"] == left_id) & (path_level["policy"] == policy), ["path_id", "total_operational_objective"]].rename(columns={"total_operational_objective": "left"})
    right = path_level.loc[(path_level["cell_id"] == right_id) & (path_level["policy"] == policy), ["path_id", "total_operational_objective"]].rename(columns={"total_operational_objective": "right"})
    joined = left.merge(right, on="path_id", validate="one_to_one")
    if joined.empty:
        return None
    interval = _interval((joined["left"] - joined["right"]).to_numpy(), confidence, family)
    return {"left_cell": left_id, "right_cell": right_id, "policy": policy, "physical_paths": len(joined), **interval}


def component_values(path_level: pd.DataFrame, confidence: float) -> pd.DataFrame:
    structural = ["Passive", "Reactive", "Behaviour cloning"]
    declarations: list[tuple[str, str, str, int, str]] = []
    for n in (4, 5, 7, 9):
        for eligibility in ("emergency_only", "precontracted"):
            suffix = eligibility
            declarations += [
                ("choice_value", "n03_reference", f"n{n:02d}_capacity_neutral_{suffix}", n, eligibility),
                ("port_capacity_value", f"n{n:02d}_capacity_neutral_{suffix}", f"n{n:02d}_port_only_{suffix}", n, eligibility),
                ("end_to_end_capacity_value", f"n{n:02d}_port_only_{suffix}", f"n{n:02d}_end_to_end_{suffix}", n, eligibility),
            ]
        for architecture in ("capacity_neutral", "port_only", "end_to_end"):
            declarations.append(("precontracting_value", f"n{n:02d}_{architecture}_emergency_only", f"n{n:02d}_{architecture}_precontracted", n, architecture))
    family = len(declarations) * len(structural)
    rows = []
    for effect, left, right, n, context in declarations:
        for policy in structural:
            row = _paired_cell_effect(path_level, left, right, policy, confidence, family)
            if row is not None:
                row.update({"component": effect, "gateway_count": n, "context": context, "positive_means": f"lower loss in {right}"})
                rows.append(row)
    return pd.DataFrame(rows)


def precision_audit(components: pd.DataFrame, target: float) -> pd.DataFrame:
    focus = components.loc[(components["gateway_count"] == 9) & (components["policy"].isin(["Passive", "Reactive"]))].copy()
    focus["target_halfwidth"] = float(target)
    focus["precision_target_met"] = focus["halfwidth"] <= float(target)
    return focus
