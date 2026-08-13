"""Path-level paired inference for Experiment 5.3.4."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


LEARNING_POLICIES = {"Behaviour cloning", "Model-guided constrained SAC"}
CELL_KEYS = [
    "cell_id",
    "cell_type",
    "family",
    "display_factor",
    "display_level",
    "full_policy_anchor",
]


def _interval(values: np.ndarray, confidence: float, family: int = 1) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values)) if n else np.nan
    if n < 2:
        return {
            "mean": mean,
            "standard_error": np.nan,
            "halfwidth": np.nan,
            "lower": np.nan,
            "upper": np.nan,
        }
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    alpha = (1.0 - confidence) / max(int(family), 1)
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    halfwidth = critical * se
    return {
        "mean": mean,
        "standard_error": se,
        "halfwidth": halfwidth,
        "lower": mean - halfwidth,
        "upper": mean + halfwidth,
    }


def _one_sample_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    sd = float(np.std(values, ddof=1))
    mean = float(np.mean(values))
    if np.isclose(sd, 0.0):
        return 1.0 if np.isclose(mean, 0.0) else 0.0
    return float(stats.ttest_1samp(values, 0.0).pvalue)


def _holm(values: Sequence[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    output = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return output.tolist()
    ordered = valid[np.argsort(p[valid], kind="mergesort")]
    running = 0.0
    m = len(ordered)
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p[index]))
        output[index] = running
    return output.tolist()


def aggregate_learning_seeds(replications: pd.DataFrame) -> pd.DataFrame:
    identifiers = CELL_KEYS + ["policy", "path_id"]
    excluded = set(identifiers + ["training_seed"])
    numeric = [
        column
        for column in replications.select_dtypes(include=[np.number, "bool"]).columns
        if column not in excluded
    ]
    rows = []
    for keys, group in replications.groupby(identifiers, sort=True, dropna=False):
        policy = str(keys[len(CELL_KEYS)])
        expected = 3 if policy in LEARNING_POLICIES else 1
        observed = (
            int(group["training_seed"].nunique(dropna=True))
            if policy in LEARNING_POLICIES
            else len(group)
        )
        if observed != expected:
            raise RuntimeError(f"{keys} has {observed} runs; expected {expected}")
        row = dict(zip(identifiers, keys))
        row["training_seed_count"] = expected
        for column in numeric:
            row[column] = float(group[column].astype(float).mean())
        for column in (
            "path_content_sha256",
            "released_information_path_sha256",
            "clearance_status",
        ):
            if column in group:
                values = group[column].astype(str).unique()
                row[column] = values[0] if len(values) == 1 else "MIXED"
        row["right_censored"] = bool(group["right_censored"].astype(bool).any())
        rows.append(row)
    return pd.DataFrame(rows)


def policy_summary(path_level: pd.DataFrame, confidence: float) -> pd.DataFrame:
    rows = []
    for keys, group in path_level.groupby(CELL_KEYS + ["policy"], sort=True, dropna=False):
        interval = _interval(group["total_operational_objective"].to_numpy(), confidence)
        rows.append(
            {
                **dict(zip(CELL_KEYS + ["policy"], keys)),
                "physical_paths": group["path_id"].nunique(),
                "mean_total_operational_loss": interval["mean"],
                "loss_lower": interval["lower"],
                "loss_upper": interval["upper"],
                "mean_waiting_exposure": group["waiting_exposure"].mean(),
                "mean_direct_sue_exit": group["direct_sue_exit"].mean(),
                "mean_duration_attrition": group["duration_attrition"].mean(),
                "mean_delivery": group["delivery"].mean(),
                "clearance_probability": 1.0 - group["right_censored"].astype(float).mean(),
                "restricted_mean_clearance_time": group["restricted_clearance_time_contribution"].mean(),
                "mean_terminal_outstanding": group["terminal_outstanding"].mean(),
                "mean_corridor_overload_exposure": group["corridor_overload_exposure"].mean(),
                "mean_port_stage_overload_exposure": group["port_stage_overload_exposure"].mean(),
                "mean_resource_week_overload": group["resource_week_overload"].mean(),
                "mean_decision_time_seconds": group["mean_decision_time_seconds"].mean(),
                "maximum_decision_time_seconds": group["maximum_decision_time_seconds"].max(),
            }
        )
    return pd.DataFrame(rows)


def policy_regret(path_level: pd.DataFrame, confidence: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    regret_rows, confidence_rows = [], []
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
            regret_rows.append(
                {**metadata, "policy": policy, **{f"regret_{key}": value for key, value in regret.items()}}
            )
            confidence_rows.append(
                {
                    **metadata,
                    "sample_leader": leader,
                    "policy": policy,
                    "mean_difference_from_leader": difference["mean"],
                    "simultaneous_lower": difference["lower"],
                    "simultaneous_upper": difference["upper"],
                    "in_simultaneous_confidence_set": bool(
                        policy == leader or difference["lower"] <= 0.0
                    ),
                    "policy_family_size": len(policies),
                }
            )
    return pd.DataFrame(regret_rows), pd.DataFrame(confidence_rows)


def paired_cell_effects(
    path_level: pd.DataFrame,
    cells: Sequence[Any],
    confidence: float,
) -> pd.DataFrame:
    reference = path_level.loc[
        path_level["cell_id"] == "reference",
        ["path_id", "policy", "total_operational_objective"],
    ].rename(columns={"total_operational_objective": "reference_loss"})
    rows = []
    for cell in cells:
        if cell.cell_id == "reference":
            continue
        current = path_level.loc[
            path_level["cell_id"] == cell.cell_id,
            ["path_id", "policy", "total_operational_objective"],
        ]
        joined = current.merge(reference, on=["path_id", "policy"], validate="one_to_one")
        for policy, group in joined.groupby("policy", sort=True):
            differences = (
                group["total_operational_objective"] - group["reference_loss"]
            ).to_numpy(dtype=float)
            # The simultaneous family is defined within the cell's evaluated policy family.
            family_size = int(joined["policy"].nunique())
            interval = _interval(differences, confidence, family_size)
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "cell_type": cell.cell_type,
                    "family": cell.family,
                    "display_factor": cell.display_factor,
                    "display_level": cell.display_level,
                    "policy": policy,
                    "physical_paths": len(group),
                    "effect_definition": "cell loss minus common-reference loss",
                    **interval,
                    "unadjusted_p": _one_sample_p(differences),
                }
            )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = np.nan
    for _, index in result.groupby(["cell_type", "display_factor"], sort=True).groups.items():
        result.loc[index, "holm_adjusted_p"] = _holm(result.loc[index, "unadjusted_p"])
    return result


def clearance_endpoint_diagnostic(
    reference_path_level: pd.DataFrame,
    diagnostics: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows = []
    for item in diagnostics:
        tolerance = float(item["level"])
        for policy, group in reference_path_level.groupby("policy", sort=True):
            outstanding = group["terminal_outstanding"].to_numpy(dtype=float)
            rows.append(
                {
                    "factor": "clearance_tolerance",
                    "tolerance": tolerance,
                    "reference_tolerance": float(item["reference"]),
                    "policy": policy,
                    "physical_paths": len(group),
                    "endpoint_classified_clear_probability": float(np.mean(outstanding <= tolerance)),
                    "mean_endpoint_outstanding": float(np.mean(outstanding)),
                    "changes_actions_transition_or_loss": False,
                    "interpretation": "classification of the same terminal states; not a reoptimised clearance trajectory",
                }
            )
    return pd.DataFrame(rows)
