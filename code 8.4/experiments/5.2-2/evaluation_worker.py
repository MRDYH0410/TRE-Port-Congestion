"""Process-isolated evaluation worker for unchanged 5.2.2 production logic."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from features import LinearActor
from model import build_model
from policies import ActorPolicy, MPCPolicy, ModelGuidedPolicy, PassivePolicy, ReactivePolicy
from simulator import ReplicationArtifacts, run_replication


_MODEL: Any | None = None
_POLICIES: list[Any] = []


def initialise_worker(
    config: Mapping[str, Any], policy_specs: Sequence[Mapping[str, Any]]
) -> None:
    """Build one read-only production model and policy set per worker process."""

    global _MODEL, _POLICIES
    _MODEL = build_model(config)
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
            actor = LinearActor(
                np.asarray(spec["weights"], dtype=float),
                np.asarray(spec["log_standard_deviation"], dtype=float),
            )
            policies.append(
                ActorPolicy(str(spec["name"]), _MODEL, actor, int(spec["training_seed"]))
            )
        elif kind == "model_guided":
            bc = LinearActor(
                np.asarray(spec["bc_weights"], dtype=float),
                np.asarray(spec["bc_log_standard_deviation"], dtype=float),
            )
            sac = LinearActor(
                np.asarray(spec["sac_weights"], dtype=float),
                np.asarray(spec["sac_log_standard_deviation"], dtype=float),
            )
            policies.append(
                ModelGuidedPolicy(
                    model=_MODEL,
                    bc_actor=bc,
                    sac_actor=sac,
                    training_seed=int(spec["training_seed"]),
                )
            )
        else:
            raise ValueError(f"Unknown policy worker specification: {kind}")
    _POLICIES = policies


def evaluate_task(task: tuple[Any, int]) -> ReplicationArtifacts:
    if _MODEL is None or not _POLICIES:
        raise RuntimeError("5.2.2 evaluation worker was not initialised")
    path, policy_index = task
    return run_replication(
        model=_MODEL,
        policy=_POLICIES[int(policy_index)],
        path=path,
    )
