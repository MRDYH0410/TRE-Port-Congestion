from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = CODE_ROOT / "experiments" / "5.2-5"
OUTPUT = CODE_ROOT / "output" / "5.2.5_computational_methodological_acceptance"


def test_configuration_declares_noncompensatory_contracts() -> None:
    config = json.loads((EXPERIMENT / "config_5_2_5.json").read_text(encoding="utf-8"))
    required = {
        "M0_UPSTREAM_LOCKS",
        "M1_RELEASED_INFORMATION", "M3_FEASIBLE_PROJECTION", "M4_RCMSA_EQUILIBRIUM",
        "M5_TAGGED_TRANSITION", "M6_COMPLETE_LOSS", "M7_NONANTICIPATIVITY",
        "M8_NESTED_MPC", "M9_BC_SAC_SELECTOR", "M10_TRAVEL_LAG",
        "M11_CAPACITY_TIMING", "M12_REPRODUCIBILITY",
        "M13_SAC_LATENT_GAUSSIAN", "M14_SAC_ACTOR_MEAN_UPDATE",
        "M15_SAC_LOG_STD_UPDATE", "M16_SAC_ENTROPY_ACTOR_TERM",
        "M17_SAC_ENTROPY_TEMPERATURE", "M18_SAC_TWIN_REWARD_CRITICS",
        "M19_SAC_CONSTRAINT_CRITIC", "M20_SAC_CONSTRAINT_DUAL",
        "M21_SAC_PROJECTION_GRADIENT", "M22_SAC_FINITE_DIFFERENCE",
        "M23_VALIDATION_CHECKPOINT", "M24_CHECKPOINT_REPLAY",
        "M25_UNAVAILABLE_ROUTE_HOLD", "M26_CLEARANCE_TERMINAL",
    }
    assert required.issubset(config["critical_contracts"])
    assert config["upstream_artifact_locks"]["5.2.4"]["acceptance_5_2_4.json"] == "8FA1E997FCA3C3D32238F1E63C7580F5CE9176F0E173B235A6526E32566EC87A"


def test_runner_does_not_hardcode_acceptance_pass() -> None:
    source = (EXPERIMENT / "run_5_2_5.py").read_text(encoding="utf-8")
    assert '"OVERALL_ACCEPTANCE": "PASS"' not in source
    assert "acceptance_summary(registry" in source


def test_completed_outputs_have_explicit_statuses() -> None:
    if not (OUTPUT / "method_contract_registry.csv").exists():
        return
    registry = pd.read_csv(OUTPUT / "method_contract_registry.csv")
    assert set(registry["status"]).issubset({"PASS", "FAIL", "BLOCKED", "NOT_TESTED"})
    assert registry["critical"].any()
    assert (~registry["critical"]).any()  # missing diagnostics remain explicit, noncritical NOT_TESTED rows
    assert not registry.loc[registry["critical"], "status"].eq("NOT_TESTED").any()
    summary = json.loads((OUTPUT / "acceptance_summary.json").read_text(encoding="utf-8"))
    assert summary["OVERALL_ACCEPTANCE"] in {"PASS", "FAIL", "BLOCKED", "NOT_TESTED"}


def test_output_figures_are_png_and_pdf() -> None:
    figures = EXPERIMENT / "figures"
    if not figures.exists():
        return
    for stem in (
        "figure_5_2_5a_numerical_convergence",
        "figure_5_2_5b_learning_selector",
        "figure_5_2_5c_contract_runtime",
    ):
        assert (figures / f"{stem}.png").exists()
        assert (figures / f"{stem}.pdf").exists()
