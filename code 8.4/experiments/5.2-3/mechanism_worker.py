"""Process-isolated compact replay worker for Experiment 5.2.3."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from features import LinearActor
from mechanism import MechanismArtifacts, run_mechanism_replication
from model import build_model
from policies import ActorPolicy, ModelGuidedPolicy, PassivePolicy, ReactivePolicy


_MODEL: Any | None = None
_POLICIES: list[Any] = []
_NO_PACING_BASELINE = 1.0


def initialise_worker(
    benchmark_config: Mapping[str, Any],
    policy_specs: Sequence[Mapping[str, Any]],
    no_pacing_baseline: float,
) -> None:
    """Build one immutable model and frozen policy set per worker process."""

    global _MODEL, _POLICIES, _NO_PACING_BASELINE
    _MODEL = build_model(benchmark_config)
    _NO_PACING_BASELINE = float(no_pacing_baseline)
    policies: list[Any] = []
    for spec in policy_specs:
        kind = str(spec["kind"])
        if kind == "passive":
            policies.append(PassivePolicy(_MODEL))
        elif kind == "reactive":
            policies.append(ReactivePolicy(_MODEL))
        elif kind == "actor":
            actor = LinearActor(
                np.asarray(spec["weights"], dtype=float),
                np.asarray(spec["log_standard_deviation"], dtype=float),
            )
            policies.append(
                ActorPolicy(
                    str(spec["name"]),
                    _MODEL,
                    actor,
                    int(spec["training_seed"]),
                )
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
            raise ValueError(f"Unknown 5.2.3 policy worker specification: {kind}")
    _POLICIES = policies


def evaluate_task(task: tuple[Any, int, str]) -> MechanismArtifacts:
    if _MODEL is None or not _POLICIES:
        raise RuntimeError("5.2.3 worker was not initialised")
    path, policy_index, restriction = task
    return run_mechanism_replication(
        model=_MODEL,
        base_policy=_POLICIES[int(policy_index)],
        path=path,
        restriction=str(restriction),
        no_release_pacing_baseline=_NO_PACING_BASELINE,
        store_detail=False,
    )
