"""Production-policy nonanticipativity probes for Experiment 5.2.2."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from model import BenchmarkModel
from paths import PhysicalPath
from preparation import prepare_period
from simulator import BenchmarkPolicy


def policy_nonanticipativity_checks(
    *,
    model: BenchmarkModel,
    policies: Sequence[BenchmarkPolicy],
    path: PhysicalPath,
) -> list[dict[str, Any]]:
    """Perturb same-week and later physical outcomes while preserving history."""

    tolerance = float(model.config["numerics"]["mass_tolerance"])
    baseline_frame = path.frame.copy(deep=True)
    future_frame = path.frame.copy(deep=True)
    if len(future_frame) > 1:
        future_frame.loc[future_frame.index[1:], "serviceability"] = 1.0 - np.asarray(
            future_frame.loc[future_frame.index[1:], "serviceability"], dtype=float
        )
        future_frame.loc[future_frame.index[1:], "normal_model_units"] *= 1.5
    current_frame = path.frame.copy(deep=True)
    current_frame.loc[current_frame.index[0], "serviceability"] = 1.0 - float(
        current_frame.iloc[0]["serviceability"]
    )
    current_frame.loc[current_frame.index[0], "normal_model_units"] *= 1.5
    variants = {
        "future_payload_after_t": replace(
            path,
            path_id=f"{path.path_id}_future_probe",
            frame=future_frame,
            path_hash="nonanticipativity-future-probe",
        ),
        "same_week_unrealised_outcome": replace(
            path,
            path_id=f"{path.path_id}_same_week_probe",
            frame=current_frame,
            path_hash="nonanticipativity-same-week-probe",
        ),
    }
    baseline_row = baseline_frame.iloc[0].to_dict()
    rows: list[dict[str, Any]] = []
    for policy in policies:
        initial = model.initial_state(baseline_row)
        baseline = prepare_period(model=model, state=initial, row=baseline_row)
        baseline_decision = policy.decide(
            state=baseline.state,
            row=baseline_row,
            path=path,
            offset=0,
            bundle=baseline.scenarios,
        )
        baseline_vector = baseline_decision.raw_action.vector(model.layout.keys)
        for probe_name, variant in variants.items():
            variant_row = variant.frame.iloc[0].to_dict()
            prepared = prepare_period(
                model=model,
                state=model.initial_state(variant_row),
                row=variant_row,
            )
            decision = policy.decide(
                state=prepared.state,
                row=variant_row,
                path=variant,
                offset=0,
                bundle=prepared.scenarios,
            )
            vector = decision.raw_action.vector(model.layout.keys)
            maximum_difference = float(np.max(np.abs(vector - baseline_vector)))
            rows.append(
                {
                    "policy": policy.name,
                    "training_seed": policy.training_seed,
                    "probe": probe_name,
                    "maximum_raw_action_difference": maximum_difference,
                    "information_hash_equal": (
                        prepared.information_vector_hash
                        == baseline.information_vector_hash
                    ),
                    "observation_hash_equal": (
                        prepared.observation_hash == baseline.observation_hash
                    ),
                    "tolerance": tolerance,
                    "passed": bool(
                        maximum_difference <= tolerance
                        and prepared.information_vector_hash
                        == baseline.information_vector_hash
                        and prepared.observation_hash == baseline.observation_hash
                    ),
                }
            )
    return rows

