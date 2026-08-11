from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
EXP = CODE_ROOT / "experiments" / "5.3-2"
BENCHMARK = CODE_ROOT / "experiments" / "5.2-2"
MECHANISM = CODE_ROOT / "experiments" / "5.2-3"
for entry in (EXP, BENCHMARK, MECHANISM, CODE_ROOT / "src"):
    sys.path.insert(0, str(entry))

from paths import PhysicalPath
from reclosure_worker import GridCell, _formal_sue_source_masses, build_cell_path
from tre84.keys import SourceKey
from run_5_3_2 import _certificate_cells, _cells_by_spec, _policy_cells
from statistics_5_3_2 import aggregate_learning_seeds, confidence_sets_and_regret, precision_requirements


def _config() -> dict:
    return json.loads((EXP / "config_5_3_2.json").read_text(encoding="utf-8"))


def test_layered_coverage_is_exactly_150_certificate_16_policy_and_3_anchor_cells() -> None:
    config = _config()
    certificate = _certificate_cells(config)
    policy, anchors = _policy_cells(config)
    assert len(certificate) == len({cell.cell_id for cell in certificate}) == 150
    assert len(policy) == len({cell.cell_id for cell in policy}) == 16
    assert len(anchors) == len({cell.cell_id for cell in anchors}) == 3
    assert set(anchors).issubset(policy)
    assert GridCell(4, .85, 8) in anchors
    assert GridCell(12, .40, 2) in anchors
    assert GridCell(1, .95, 32) in anchors


def test_policy_task_coverage_is_92_seed_cell_evaluations_per_path() -> None:
    config = _config()
    policy, anchors = _policy_cells(config)
    specs = [
        {"policy": "Passive"}, {"policy": "Reactive"},
        {"policy": "Projected stochastic MPC"},
        {"policy": "Behaviour cloning"}, {"policy": "Model-guided constrained SAC"},
        {"policy": "Behaviour cloning"}, {"policy": "Model-guided constrained SAC"},
        {"policy": "Behaviour cloning"}, {"policy": "Model-guided constrained SAC"},
    ]
    coverage = _cells_by_spec(specs, policy, anchors)
    assert [len(cells) for cells in coverage] == [16, 16, 3, 16, 3, 16, 3, 16, 3]
    assert sum(map(len, coverage)) == 92


def test_event_aligned_constructor_changes_only_the_registered_extension() -> None:
    config = _config()
    base = pd.DataFrame({
        "week": pd.date_range("2026-02-23", periods=21, freq="W-MON"),
        "normal_model_units": np.linspace(8.0, 10.0, 21),
        "serviceability": np.linspace(0.8, 1.0, 21),
        "filtered_high_risk_probability": np.linspace(0.1, 0.2, 21),
        "lead_time_high_risk_probability": np.linspace(0.2, 0.3, 21),
        "release_date": pd.Timestamp("2026-02-01"),
        "source_observation_month": pd.Timestamp("2026-01-01"),
        "timing_valid": True,
    })
    physical = PhysicalPath(
        path_id="unit_path", split="test", frame=base, path_hash="unit-hash",
        construction="unit test", residual_start=0, residual_end=20,
        onset_week=0, active_duration_weeks=21, severity_floor=0.8, has_reclosure=False,
    )
    cell = GridCell(open_weeks=4, intensity=0.7, duration_weeks=8)
    constructed = build_cell_path(physical, cell)
    recovery = config["event_aligned_constructor"]["post_reclosure_recovery_weeks"]
    assert len(constructed.frame) == 21 + 4 + 8 + recovery
    pd.testing.assert_frame_equal(constructed.frame.iloc[:21][base.columns].reset_index(drop=True), base.reset_index(drop=True), check_dtype=False)
    extension = constructed.frame.iloc[21:]
    assert extension["week"].is_monotonic_increasing and extension["timing_valid"].all()
    assert np.allclose(extension.iloc[:4]["serviceability"], 1.0)
    assert np.allclose(extension.iloc[4:12]["serviceability"], 0.3)
    assert np.isclose(extension.iloc[-1]["serviceability"], 1.0)


def _row(policy: str, path: str, seed: float, loss: float, cell: GridCell | None = None) -> dict:
    cell = cell or GridCell(4, .85, 8)
    return {
        "cell_id": cell.cell_id, "open_interval_weeks": cell.open_weeks,
        "reclosure_intensity": cell.intensity, "reclosure_duration_weeks": cell.duration_weeks,
        "policy": policy, "path_id": path, "reclosure_path_id": f"{path}-cell",
        "path_content_sha256": f"hash-{path}", "training_seed": seed,
        "clearance_status": "cleared", "right_censored": False,
        "clearance_weeks_observed": 2.0, "restricted_clearance_time_contribution": 2.0,
        "total_operational_objective": loss, "all_step_acceptance_passed": True,
    }


def test_learning_seeds_are_aggregated_inside_physical_paths() -> None:
    raw = [_row("Passive", "p0", np.nan, 12.0)]
    raw += [_row("Behaviour cloning", "p0", float(seed), loss) for seed, loss in enumerate((8.0, 9.0, 10.0))]
    learned = aggregate_learning_seeds(pd.DataFrame(raw)).loc[lambda frame: frame["policy"] == "Behaviour cloning"].iloc[0]
    assert learned["training_seed_count"] == 3
    assert learned["seed_aggregation_applied_before_path_inference"]
    assert learned["inference_unit"] == "physical_path"
    assert np.isclose(learned["total_operational_objective"], 9.0)


def test_precision_design_uses_three_full_policy_anchors() -> None:
    config = _config()
    policies = config["main_policies"]
    _, anchors = _policy_cells(config)
    rows = []
    for cell in anchors:
        for path in range(88):
            for policy_index, policy in enumerate(policies):
                rows.append({
                    "cell_id": cell.cell_id, "open_interval_weeks": cell.open_weeks,
                    "reclosure_intensity": cell.intensity, "reclosure_duration_weeks": cell.duration_weeks,
                    "path_id": f"p{path:03d}", "policy": policy,
                    "total_operational_objective": float(path + policy_index),
                })
    requirements, selected = precision_requirements(pd.DataFrame(rows), config, policies)
    assert requirements["cell_id"].nunique() == 3
    assert len(requirements) == 3 * 8
    assert selected == 88
    assert (requirements["required_paths"] >= 88).all()


def test_confidence_sets_use_three_policies_outside_anchors_and_five_at_anchor() -> None:
    policies = _config()["main_policies"]
    rows = []
    for cell, cell_policies in ((GridCell(2, .85, 8), ["Passive", "Reactive", "Behaviour cloning"]), (GridCell(4, .85, 8), policies)):
        for path in range(10):
            for index, policy in enumerate(cell_policies):
                rows.append({
                    "cell_id": cell.cell_id, "open_interval_weeks": cell.open_weeks,
                    "reclosure_intensity": cell.intensity, "reclosure_duration_weeks": cell.duration_weeks,
                    "path_id": f"p{path:02d}", "policy": policy,
                    "total_operational_objective": float(path + index),
                })
    confidence, regret = confidence_sets_and_regret(pd.DataFrame(rows), policies, .95)
    assert set(confidence.groupby("cell_id")["evaluated_policy_count"].first()) == {3, 5}
    assert set(regret.groupby("cell_id")["evaluated_policy_count"].first()) == {3, 5}


def test_runner_does_not_read_old_sensitivity_outputs_or_retrain() -> None:
    source = (EXP / "run_5_3_2.py").read_text(encoding="utf-8").lower()
    config = _config()
    assert "8.3" not in source
    assert "output/5.3.2" not in source
    assert "frozen accepted 5.2.2 checkpoints" in config["policy_rule"].lower()
    assert "no grid-cell retraining" in config["policy_rule"].lower()


def test_period_acceptance_uses_formal_sue_inputs_not_post_solver_flow_sums() -> None:
    tiny = float(np.nextafter(0.0, 1.0))
    result = SimpleNamespace(
        transition=SimpleNamespace(
            demand_split=SimpleNamespace(decision_eligible={"cargo": 3.0})
        ),
        equilibrium=SimpleNamespace(
            releases={"cargo": np.asarray([0.0, tiny, 2.0])}
        ),
    )
    masses = _formal_sue_source_masses(result)
    assert masses == {
        SourceKey("cargo", None): 3.0,
        SourceKey("cargo", 1): tiny,
        SourceKey("cargo", 2): 2.0,
    }
