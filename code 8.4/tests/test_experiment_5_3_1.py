from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
EXP = CODE_ROOT / "experiments" / "5.3-1"
BENCHMARK = CODE_ROOT / "experiments" / "5.2-2"
MECHANISM = CODE_ROOT / "experiments" / "5.2-3"
for entry in (EXP, BENCHMARK, MECHANISM, CODE_ROOT / "src"):
    sys.path.insert(0, str(entry))

from statistics_5_3_1 import aggregate_learning_seeds, endpoint_precision


def test_grid_is_preregistered_nine_points() -> None:
    config = json.loads((EXP / "config_5_3_1.json").read_text(encoding="utf-8"))
    assert config["commitment_grid"] == [
        0.0,
        0.125,
        0.25,
        0.375,
        0.5,
        0.625,
        0.75,
        0.875,
        1.0,
    ]
    assert config["path_design"]["precision_endpoints"] == [0.0, 1.0]


def _replication(policy: str, seed: float, path: str, loss: float) -> dict:
    return {
        "chi": 0.5,
        "policy": policy,
        "path_id": path,
        "path_content_sha256": f"hash-{path}",
        "released_information_path_sha256": f"info-{path}",
        "training_seed": seed,
        "information_source": "released_5_2_1",
        "projector_id": "shared",
        "kernel_id": "shared",
        "clearance_status": "cleared",
        "right_censored": False,
        "clearance_weeks_observed": 2.0,
        "restricted_clearance_time_contribution": 2.0,
        "total_operational_objective": loss,
        "all_step_acceptance_passed": True,
    }


def test_learning_seeds_are_averaged_within_path() -> None:
    rows = [_replication("Passive", np.nan, "p0", 10.0)]
    rows += [
        _replication("Behaviour cloning", float(seed), "p0", value)
        for seed, value in enumerate((8.0, 9.0, 10.0))
    ]
    result = aggregate_learning_seeds(pd.DataFrame(rows))
    bc = result.loc[result["policy"] == "Behaviour cloning"].iloc[0]
    assert bc["training_seed_count"] == 3
    assert bc["seed_aggregation_applied_before_path_inference"]
    assert np.isclose(bc["total_operational_objective"], 9.0)


def test_endpoint_precision_never_selects_below_88() -> None:
    policies = [
        "Passive",
        "Reactive",
        "Projected stochastic MPC",
        "Behaviour cloning",
        "Model-guided constrained SAC",
    ]
    rows = []
    for chi in (0.0, 1.0):
        for path in range(88):
            for index, policy in enumerate(policies):
                rows.append(
                    {
                        "chi": chi,
                        "path_id": f"p{path:03d}",
                        "policy": policy,
                        "total_operational_objective": float(path + index),
                    }
                )
    config = {
        "path_design": {
            "minimum_common_physical_paths": 88,
            "maximum_physical_paths": 196,
            "target_halfwidth": 2255.637825,
            "confidence_level": 0.95,
            "precision_endpoints": [0.0, 1.0],
        }
    }
    requirements, selected = endpoint_precision(
        pd.DataFrame(rows), config=config, policies=policies
    )
    assert selected == 88
    assert (requirements["required_paths"] >= 88).all()
    assert len(requirements) == 16


def test_runner_never_imports_legacy_5_3_results() -> None:
    source = (EXP / "run_5_3_1.py").read_text(encoding="utf-8").lower()
    assert "8.3" not in source
    assert "old 5.3" not in source
    assert "output/5.3" not in source
