from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
for entry in (
    CODE_ROOT / "experiments" / "5.3-3",
    CODE_ROOT / "experiments" / "5.2-2",
    CODE_ROOT / "src",
):
    sys.path.insert(0, str(entry))

from model import build_model  # noqa: E402
from network_5_3_3 import build_cell_config, declared_cells  # noqa: E402


def _configs():
    base = json.loads((CODE_ROOT / "experiments/5.2-2/config_5_2_2.json").read_text())
    experiment = json.loads((CODE_ROOT / "experiments/5.3-3/config_5_3_3.json").read_text())
    return base, experiment


def test_declares_exactly_25_unique_cells_and_dynamic_action_dimensions():
    base, experiment = _configs()
    cells = declared_cells()
    assert len(cells) == len({cell.cell_id for cell in cells}) == 25
    for cell in cells:
        model = build_model(build_cell_config(base, experiment, cell))
        assert len(model.layout.keys) == 10 * cell.gateway_count + 4


def test_capacity_architectures_and_commitment_eligibility_are_closed():
    base, experiment = _configs()
    cells = {cell.cell_id: cell for cell in declared_cells()}
    baseline = build_model(build_cell_config(base, experiment, cells["n03_reference"]))
    baseline_total = sum(baseline.gateway_scales.values())
    for n in (4, 5, 7, 9):
        neutral = build_model(build_cell_config(base, experiment, cells[f"n{n:02d}_capacity_neutral_emergency_only"]))
        port = build_model(build_cell_config(base, experiment, cells[f"n{n:02d}_port_only_emergency_only"]))
        end = build_model(build_cell_config(base, experiment, cells[f"n{n:02d}_end_to_end_emergency_only"]))
        corridor = next(resource for resource in neutral.resources if resource.stage.value == "corridor")
        assert np.isclose(sum(neutral.gateway_scales.values()), baseline_total)
        assert np.isclose(neutral.base_capacity[corridor], baseline_total)
        assert sum(port.gateway_scales.values()) > baseline_total
        assert np.isclose(port.base_capacity[corridor], baseline_total)
        assert end.base_capacity[corridor] > baseline_total
        assert np.isclose(end.base_capacity[corridor], sum(end.gateway_scales.values()))
        assert all(share == 0.0 for tag, share in neutral.committed_shares.items() if tag.route.startswith("SemiSynthetic_"))
        pre = build_model(build_cell_config(base, experiment, cells[f"n{n:02d}_end_to_end_precontracted"]))
        assert sum(share for tag, share in pre.committed_shares.items() if tag.route.startswith("SemiSynthetic_")) > 0.0
        assert np.isclose(sum(pre.committed_shares.values()), 1.0)
        assert np.isclose(sum(pre.reference_loading_shares.values()), 1.0)


def test_semi_synthetic_nodes_use_median_templates_and_generic_names():
    base, experiment = _configs()
    cell = next(cell for cell in declared_cells() if cell.cell_id == "n09_end_to_end_precontracted")
    config = build_cell_config(base, experiment, cell)
    design = config["network_design"]
    assert len(design["semi_synthetic_gateway_ids"]) == 6
    assert all(name.startswith("SemiSynthetic_Gateway_") for name in design["semi_synthetic_gateway_ids"])
    assert np.isclose(design["template_scale_model_units"], 30.796000000000003)
    assert design["template_maritime_lag_weeks"] == 3
    observed_sigma = list(json.loads((CODE_ROOT / "experiments/5.2-2/config_5_2_2.json").read_text())["information"]["waiting_error_scale_weeks_by_route"].values())
    assert np.isclose(design["template_waiting_error_scale_weeks"], np.median(observed_sigma))


def test_three_gateway_reference_preserves_the_accepted_benchmark_network():
    base, experiment = _configs()
    accepted = build_model(base)
    reference = build_model(build_cell_config(base, experiment, declared_cells()[0]))
    assert accepted.gateway_scales == reference.gateway_scales
    assert accepted.committed_shares == reference.committed_shares
    assert accepted.reference_loading_shares == reference.reference_loading_shares
    assert accepted.base_capacity == reference.base_capacity
    assert accepted.action_upper.shape == reference.action_upper.shape == (34,)
    assert np.allclose(accepted.action_upper, reference.action_upper)
