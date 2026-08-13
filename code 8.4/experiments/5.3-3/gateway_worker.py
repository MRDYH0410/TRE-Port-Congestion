"""Process-isolated gateway-cell replay for Experiment 5.3.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from features import LinearActor
from mechanism import run_mechanism_replication
from model import build_model
from policies import ActorPolicy, MPCPolicy, ModelGuidedPolicy, PassivePolicy, ReactivePolicy


@dataclass
class GatewayArtifacts:
    replication: dict[str, Any]
    resources: list[dict[str, Any]]
    contract: dict[str, Any]


_MODEL: Any | None = None
_POLICIES: list[Any] = []
_CELL: dict[str, Any] = {}


def _actor(spec: Mapping[str, Any]) -> LinearActor:
    return LinearActor(
        np.asarray(spec["weights"], dtype=float),
        np.asarray(spec["log_standard_deviation"], dtype=float),
    )


def initialise_worker(
    model_config: Mapping[str, Any],
    policy_specs: Sequence[Mapping[str, Any]],
    cell: Mapping[str, Any],
) -> None:
    global _MODEL, _POLICIES, _CELL
    _MODEL = build_model(model_config)
    _CELL = dict(cell)
    expected = int(_MODEL.config["network_design"]["expected_action_dimension"])
    if len(_MODEL.layout.keys) != expected:
        raise RuntimeError("Dynamic action dimension differs from the registered formula")
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
            policies.append(ActorPolicy(str(spec["name"]), _MODEL, _actor(spec), int(spec["training_seed"])))
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
            raise ValueError(kind)
    _POLICIES = policies


def summarise(artifact: Any, cell: Mapping[str, Any]) -> GatewayArtifacts:
    capacity = pd.DataFrame(artifact.capacity)
    replication = dict(artifact.replication)
    replication["policy"] = replication.pop("base_policy")
    for key, value in cell.items():
        replication[key] = value
    if capacity.empty:
        corridor_overload = 0.0
        resource_weeks = 0
        overloaded_gateways = 0
        resource_rows: list[dict[str, Any]] = []
    else:
        capacity = capacity.loc[capacity["scope"] != "terminal"].copy()
        corridor = capacity.loc[capacity["stage"] == "corridor"]
        gateway = capacity.loc[capacity["stage"].isin(["berth", "yard", "gate"])]
        corridor_overload = float(corridor["resource_overload_mass"].sum())
        resource_weeks = int((capacity["resource_overload_mass"] > 1e-9).sum())
        overloaded_gateways = int(
            gateway.loc[gateway["resource_overload_mass"] > 1e-9, "location"].nunique()
        )
        capacity["policy"] = replication["policy"]
        for key, value in cell.items():
            capacity[key] = value
        resource_rows = capacity.to_dict(orient="records")
    n = int(cell["gateway_count"])
    replication.update(
        {
            "corridor_overload_exposure": corridor_overload,
            "resource_week_overload": resource_weeks,
            "overloaded_gateway_count": overloaded_gateways,
            "overloaded_gateway_incidence": overloaded_gateways / n,
            "waiting_exposure": float(replication.get("waiting_model_unit_weeks", 0.0)),
            "delivery": float(replication.get("delivered_landbridge", 0.0)),
            "terminal_outstanding": float(replication.get("ending_outstanding_mass", 0.0)),
        }
    )
    contract = dict(artifact.contract)
    contract["policy"] = contract.pop("base_policy")
    for key, value in cell.items():
        contract[key] = value
    contract["dynamic_action_dimension_passed"] = (
        int(replication["action_dimension"]) == 10 * n + 4
    )
    contract["semi_synthetic_names_are_generic"] = all(
        str(value).startswith("SemiSynthetic_Gateway_")
        for value in _MODEL.config["network_design"]["semi_synthetic_gateway_ids"]
    )
    return GatewayArtifacts(replication, resource_rows, contract)


def evaluate_task(task: tuple[Any, int]) -> GatewayArtifacts:
    if _MODEL is None or not _POLICIES:
        raise RuntimeError("5.3.3 worker was not initialised")
    path, policy_index = task
    artifact = run_mechanism_replication(
        model=_MODEL,
        base_policy=_POLICIES[int(policy_index)],
        path=path,
        restriction="full_action",
        no_release_pacing_baseline=1.0,
        store_detail=True,
    )
    return summarise(artifact, _CELL)
