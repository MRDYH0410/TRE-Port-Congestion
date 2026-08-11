"""Process-isolated production replay and compact provenance aggregation for 5.3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from features import LinearActor
from mechanism import run_mechanism_replication
from model import build_model
from policies import (
    ActorPolicy,
    MPCPolicy,
    ModelGuidedPolicy,
    PassivePolicy,
    ReactivePolicy,
)


@dataclass
class CommitmentArtifacts:
    replication: dict[str, Any]
    weekly: list[dict[str, Any]]
    contract: dict[str, Any]


_MODEL: Any | None = None
_POLICIES: list[Any] = []
_CHI = float("nan")


def _actor(spec: Mapping[str, Any]) -> LinearActor:
    return LinearActor(
        np.asarray(spec["weights"], dtype=float),
        np.asarray(spec["log_standard_deviation"], dtype=float),
    )


def initialise_worker(
    model_config: Mapping[str, Any],
    policy_specs: Sequence[Mapping[str, Any]],
    chi: float,
) -> None:
    """Build a single immutable production model and policy set per worker."""

    global _MODEL, _POLICIES, _CHI
    _MODEL = build_model(model_config)
    _CHI = float(chi)
    if not np.isclose(float(_MODEL.config["committed_fraction_reference"]), _CHI):
        raise RuntimeError("Worker commitment fraction differs from the model configuration")
    policies: list[Any] = []
    for spec in policy_specs:
        kind = str(spec["kind"])
        if kind == "passive":
            policies.append(PassivePolicy(_MODEL))
        elif kind == "reactive":
            policies.append(ReactivePolicy(_MODEL))
        elif kind == "mpc":
            policies.append(MPCPolicy(_MODEL))
        elif kind == "actor":
            policies.append(
                ActorPolicy(
                    str(spec["name"]),
                    _MODEL,
                    _actor(spec),
                    int(spec["training_seed"]),
                )
            )
        elif kind == "model_guided":
            policies.append(
                ModelGuidedPolicy(
                    model=_MODEL,
                    bc_actor=LinearActor(
                        np.asarray(spec["bc_weights"], dtype=float),
                        np.asarray(spec["bc_log_standard_deviation"], dtype=float),
                    ),
                    sac_actor=LinearActor(
                        np.asarray(spec["sac_weights"], dtype=float),
                        np.asarray(spec["sac_log_standard_deviation"], dtype=float),
                    ),
                    training_seed=int(spec["training_seed"]),
                )
            )
        else:
            raise ValueError(f"Unknown 5.3.1 policy kind: {kind}")
    _POLICIES = policies


def _first_nonempty(series: pd.Series, default: Any = "") -> Any:
    values = series.dropna()
    if values.empty:
        return default
    text = values.astype(str)
    nonempty = values.loc[text.str.len() > 0]
    return nonempty.iloc[0] if not nonempty.empty else values.iloc[0]


def _action_summary(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame(columns=["scope", "period_offset"])
    rows: list[dict[str, Any]] = []
    for (scope, period_offset), group in actions.groupby(
        ["scope", "period_offset"], sort=False
    ):
        record: dict[str, Any] = {
            "scope": scope,
            "period_offset": int(period_offset),
            "decision_week": _first_nonempty(group["decision_week"]),
            "decision_source": _first_nonempty(group["decision_source"]),
            "information_vector_sha256": _first_nonempty(
                group["information_vector_sha256"]
            ),
            "observation_sha256": _first_nonempty(group["observation_sha256"]),
            "controller_input_sha256": _first_nonempty(
                group["controller_input_sha256"]
            ),
            "budget_before": float(group["budget_before"].iloc[0]),
            "budget_after": float(group["budget_after"].iloc[0]),
            "period_action_cost_logged": float(group["period_action_cost"].iloc[0]),
            "projection_feasibility_violation": float(
                group["projection_feasibility_violation"].max()
            ),
            "decision_time_seconds": float(group["decision_time_seconds"].max()),
            "active_projection_constraints": _first_nonempty(
                group["active_projection_constraints"]
            ),
        }
        for block, block_group in group.groupby("action_block", sort=False):
            safe = str(block).replace(" ", "_")
            record[f"requested_{safe}"] = float(
                block_group["original_requested_model_units"].sum()
            )
            record[f"implemented_{safe}"] = float(
                block_group["implemented_model_units"].sum()
            )
            record[f"implemented_ratio_{safe}"] = float(
                block_group["implemented_reference_ratio"].mean()
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _behavior_summary(behavior: pd.DataFrame) -> pd.DataFrame:
    if behavior.empty:
        return pd.DataFrame(columns=["scope", "period_offset"])
    rows: list[dict[str, Any]] = []
    for (scope, period_offset), group in behavior.groupby(
        ["scope", "period_offset"], sort=False
    ):
        new = group.loc[group["source_type"] == "new"]
        rows.append(
            {
                "scope": scope,
                "period_offset": int(period_offset),
                "new_decision_eligible_mass": float(new["decision_mass"].sum()),
                "released_waiting_decision_mass": float(
                    group.loc[
                        group["source_type"] == "released_waiting_vintage",
                        "decision_mass",
                    ].sum()
                ),
                "choose_waiting_mass": float(group["choose_waiting"].sum()),
                "sue_exit_mass_behavior": float(group["choose_sue_exit"].sum()),
                "duration_attrition_mass_behavior": float(
                    group["duration_attrition"].sum()
                ),
                "released_waiting_mass": float(group["released_waiting_mass"].sum()),
                "maximum_source_simplex_residual": float(
                    group["source_simplex_residual"].abs().max()
                ),
                "maximum_waiting_identity_residual": float(
                    group["waiting_identity_residual"].abs().max()
                ),
                "sue_residual": float(group["sue_residual"].max()),
                "sue_iterations": int(group["sue_iterations"].max()),
            }
        )
    return pd.DataFrame(rows)


def _physical_summary(physical: pd.DataFrame) -> pd.DataFrame:
    if physical.empty:
        return pd.DataFrame(columns=["scope", "period_offset"])
    rows: list[dict[str, Any]] = []
    for (scope, period_offset), group in physical.groupby(
        ["scope", "period_offset"], sort=False
    ):
        record: dict[str, Any] = {
            "scope": scope,
            "period_offset": int(period_offset),
        }
        for provenance in ("committed", "adaptive"):
            pgroup = group.loc[group["provenance"] == provenance]
            record[f"{provenance}_delivery"] = float(pgroup["delivered_cargo"].sum())
            record[f"{provenance}_physical_outstanding_after"] = float(
                pgroup["queue_next_state"].sum()
            )
            if "new_dispatch" in pgroup:
                record[f"{provenance}_new_dispatch"] = float(
                    pgroup["new_dispatch"].fillna(0.0).sum()
                )
            else:
                record[f"{provenance}_new_dispatch"] = 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def summarise_mechanism_artifact(artifact: Any, chi: float) -> CommitmentArtifacts:
    """Reduce the accepted detailed ledger to one row per production week."""

    losses = pd.DataFrame(artifact.losses)
    actions = _action_summary(pd.DataFrame(artifact.actions))
    behavior = _behavior_summary(pd.DataFrame(artifact.behavior))
    physical = _physical_summary(pd.DataFrame(artifact.physical))
    weekly = losses.merge(actions, on=["scope", "period_offset"], how="left")
    weekly = weekly.merge(behavior, on=["scope", "period_offset"], how="left")
    weekly = weekly.merge(physical, on=["scope", "period_offset"], how="left")
    weekly["chi"] = float(chi)
    weekly["policy"] = weekly.pop("base_policy")
    numeric_fill = [
        "new_decision_eligible_mass",
        "released_waiting_decision_mass",
        "choose_waiting_mass",
        "sue_exit_mass_behavior",
        "duration_attrition_mass_behavior",
        "released_waiting_mass",
        "committed_delivery",
        "adaptive_delivery",
        "committed_physical_outstanding_after",
        "adaptive_physical_outstanding_after",
        "committed_new_dispatch",
        "adaptive_new_dispatch",
    ]
    for column in numeric_fill:
        if column not in weekly:
            weekly[column] = 0.0
        weekly[column] = weekly[column].fillna(0.0)
    weekly["new_committed_mass"] = weekly["committed_new_dispatch"]
    weekly["new_blocked_mass"] = (
        weekly["new_committed_mass"] + weekly["new_decision_eligible_mass"]
    )
    weekly["committed_split_identity_residual"] = (
        weekly["new_committed_mass"] - float(chi) * weekly["new_blocked_mass"]
    )
    weekly["adaptive_split_identity_residual"] = (
        weekly["new_decision_eligible_mass"]
        - (1.0 - float(chi)) * weekly["new_blocked_mass"]
    )
    weekly["commitment_applied_to_new_cohort_only"] = True

    replication = dict(artifact.replication)
    replication["policy"] = replication.pop("base_policy")
    replication["chi"] = float(chi)
    replication["initial_committed_outstanding"] = 0.0
    replication["total_new_blocked_mass"] = float(weekly["new_blocked_mass"].sum())
    replication["total_new_committed_mass"] = float(
        weekly["new_committed_mass"].sum()
    )
    replication["total_new_decision_eligible_mass"] = float(
        weekly["new_decision_eligible_mass"].sum()
    )
    replication["committed_delivery"] = float(weekly["committed_delivery"].sum())
    replication["adaptive_delivery"] = float(weekly["adaptive_delivery"].sum())
    physical_weekly = weekly.loc[weekly["scope"] != "terminal"]
    if physical_weekly.empty:
        terminal_committed = terminal_adaptive = 0.0
    else:
        last = physical_weekly.iloc[-1]
        terminal_committed = float(last["committed_physical_outstanding_after"])
        terminal_adaptive = float(last["adaptive_physical_outstanding_after"])
    replication["terminal_committed_outstanding"] = terminal_committed
    replication["terminal_adaptive_physical_outstanding"] = terminal_adaptive
    committed_available = (
        replication["initial_committed_outstanding"]
        + replication["total_new_committed_mass"]
    )
    replication["committed_delivery_share"] = (
        replication["committed_delivery"] / committed_available
        if committed_available > 0.0
        else np.nan
    )
    replication["terminal_committed_outstanding_share"] = (
        terminal_committed / committed_available if committed_available > 0.0 else np.nan
    )
    replication["committed_conservation_residual"] = (
        committed_available
        - replication["committed_delivery"]
        - terminal_committed
    )
    replication["maximum_committed_split_residual"] = float(
        weekly["committed_split_identity_residual"].abs().max()
    )
    replication["maximum_adaptive_split_residual"] = float(
        weekly["adaptive_split_identity_residual"].abs().max()
    )
    replication["waiting_exposure"] = float(
        replication.get("waiting_model_unit_weeks", 0.0)
    )

    contract = dict(artifact.contract)
    contract["policy"] = contract.pop("base_policy")
    contract["chi"] = float(chi)
    tolerance = 1e-6
    contract["commitment_split_identity_passed"] = (
        replication["maximum_committed_split_residual"] <= tolerance
        and replication["maximum_adaptive_split_residual"] <= tolerance
    )
    contract["committed_provenance_conservation_passed"] = (
        abs(replication["committed_conservation_residual"]) <= tolerance
    )
    contract["existing_state_relabelled_by_chi"] = False
    contract["initial_committed_outstanding"] = 0.0
    return CommitmentArtifacts(
        replication=replication,
        weekly=weekly.to_dict(orient="records"),
        contract=contract,
    )


def evaluate_task(task: tuple[Any, int]) -> CommitmentArtifacts:
    if _MODEL is None or not _POLICIES:
        raise RuntimeError("5.3.1 worker was not initialised")
    path, policy_index = task
    artifact = run_mechanism_replication(
        model=_MODEL,
        base_policy=_POLICIES[int(policy_index)],
        path=path,
        restriction="full_action",
        no_release_pacing_baseline=1.0,
        store_detail=True,
    )
    return summarise_mechanism_artifact(artifact, _CHI)
