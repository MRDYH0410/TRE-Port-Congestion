from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = CODE_ROOT / "experiments" / "5.3-4"
for entry in (
    EXPERIMENT_DIR,
    CODE_ROOT / "experiments" / "5.2-2",
    CODE_ROOT / "experiments" / "5.2-3",
    CODE_ROOT / "src",
):
    sys.path.insert(0, str(entry))

from model import build_model  # noqa: E402
from paths import PhysicalPath, _canonical_path_hash  # noqa: E402
from robustness_5_3_4 import (  # noqa: E402
    build_cells,
    cell_registry,
    dimension_changed_cell,
    full_policy_anchor,
    model_config,
    transform_paths,
)


def _configs():
    experiment = json.loads((EXPERIMENT_DIR / "config_5_3_4.json").read_text(encoding="utf-8"))
    base = json.loads((CODE_ROOT / experiment["base_model_config"]).read_text(encoding="utf-8"))
    return experiment, base


def test_cell_and_policy_coverage_is_exact_and_commitment_is_not_repeated():
    experiment, _ = _configs()
    cells, diagnostics = build_cells(experiment)
    assert len(cells) == 31
    assert len({cell.cell_id for cell in cells}) == 31
    assert sum(full_policy_anchor(cell) for cell in cells) == 4
    assert sum(dimension_changed_cell(cell) for cell in cells) == 6
    assert len(diagnostics) == 3
    assert all("commit" not in cell.cell_id.lower() for cell in cells)
    registry = cell_registry(cells, experiment)
    assert len(registry) == 155
    assert registry["policy_evaluated"].sum() == 101
    expected_seed_rows = registry.loc[registry["policy_evaluated"]].assign(
        seeds=lambda frame: frame["policy"].isin(
            ["Behaviour cloning", "Model-guided constrained SAC"]
        ).map({True: 3, False: 1})
    )["seeds"].sum()
    assert expected_seed_rows == 161


def test_every_simulated_cell_retains_chapter_3_and_4_dimensions_and_domains():
    experiment, base = _configs()
    cells, _ = build_cells(experiment)
    for cell in cells:
        config = model_config(base, experiment, cell)
        model = build_model(config)
        expected_action = 94 if cell.network_stress != "reference" else 34
        assert len(model.layout.keys) == expected_action
        assert len(model.resources) == (28 if cell.network_stress != "reference" else 10)
        assert np.isclose(config["committed_fraction_reference"], 0.5)
        assert 0.0 <= float(config["information"]["gamma_I"]) <= 1.0
        assert float(config["behavior"]["hazard_power"]) > 0.0
        assert float(config["behavior"]["exit_failure_cost_per_unit"]) > 0.0


def test_parameter_operators_change_only_the_declared_contract():
    experiment, base = _configs()
    cells, _ = build_cells(experiment)
    lookup = {cell.cell_id: cell for cell in cells}
    route = model_config(base, experiment, lookup["route_sensitivity__2"])
    assert route["behavior"]["logit_theta"] == 2.0
    assert route["routes"] == base["routes"]
    lag = model_config(base, experiment, lookup["maritime_lag__0p5"])
    assert [item["maritime_lag_weeks"] for item in lag["routes"]] == [1, 2, 2]
    budget = model_config(base, experiment, lookup["authority_budget__1"])
    assert budget["action"]["period_budget_fraction"] == 1.0
    assert budget["action"]["cumulative_budget_fraction"] == 1.0
    sigma = model_config(base, experiment, lookup["waiting_error_scale__2"])
    assert sigma["information"]["waiting_error_scale_robustness_multiplier"] == 2.0
    route_cost = model_config(base, experiment, lookup["route_resource_cost__0p5"])
    assert route_cost["route_resource_cost"]["robustness_multiplier"] == 0.5
    credibility = model_config(base, experiment, lookup["interaction__lead16__low_credibility"])
    assert credibility["information"]["gamma_I"] == 0.5
    assert credibility["action"]["readiness_lead_weeks"] == 16


def test_network_exposure_changes_only_path_demand_and_hash():
    experiment, _ = _configs()
    cells, _ = build_cells(experiment)
    cell = next(item for item in cells if item.cell_id == "network_exposure__2")
    frame = pd.DataFrame(
        {
            "week": pd.date_range("2026-01-05", periods=2, freq="W-MON"),
            "normal_model_units": [10.0, 12.0],
            "serviceability": [0.5, 0.75],
            "filtered_high_risk_probability": [0.2, 0.3],
            "lead_time_high_risk_probability": [0.4, 0.5],
            "release_date": pd.date_range("2026-01-05", periods=2, freq="W-MON"),
            "source_observation_month": pd.to_datetime(["2025-12-01", "2025-12-01"]),
            "timing_valid": [True, True],
            "information_source": ["frozen", "frozen"],
            "residual_date": pd.to_datetime(["2025-01-06", "2025-01-13"]),
        }
    )
    path = PhysicalPath(
        path_id="test",
        split="test",
        frame=frame,
        path_hash=_canonical_path_hash(frame),
        construction="frozen",
        residual_start="2025-01-06",
        residual_end="2025-01-13",
        onset_week=0,
        active_duration_weeks=2,
        severity_floor=0.5,
        has_reclosure=False,
    )
    transformed = transform_paths([path], cell, experiment)[0]
    assert np.allclose(transformed.frame["normal_model_units"], [20.0, 24.0])
    for column in frame.columns.difference(["normal_model_units"]):
        assert transformed.frame[column].equals(frame[column])
    assert transformed.path_id == path.path_id
    assert transformed.path_hash != path.path_hash
