"""Blocking contracts for experiment 5.2.3 mechanism evidence."""

from __future__ import annotations

import json
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_522 = CODE_ROOT / "experiments" / "5.2-2"
EXPERIMENT_523 = CODE_ROOT / "experiments" / "5.2-3"
OUTPUT = CODE_ROOT / "output" / "5.2.3_action_and_congestion_mechanisms"
sys.path.insert(0, str(EXPERIMENT_523))
sys.path.insert(0, str(EXPERIMENT_522))
sys.path.insert(0, str(CODE_ROOT / "src"))

from mechanism import apply_restriction  # noqa: E402
from model import build_model  # noqa: E402
from tre84.actions import Action, Block  # noqa: E402


class Experiment523Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_522 = json.loads(
            (EXPERIMENT_522 / "config_5_2_2.json").read_text(encoding="utf-8")
        )
        cls.model = build_model(cls.config_522)
        cls.raw_vector = 0.4 * cls.model.action_upper
        cls.raw_action = Action.from_vector(cls.model.layout.keys, cls.raw_vector)

    def restricted_vector(self, restriction: str) -> np.ndarray:
        action = apply_restriction(
            model=self.model,
            raw_action=self.raw_action,
            restriction=restriction,
            no_release_pacing_baseline=1.0,
        )
        return action.vector(self.model.layout.keys)

    def test_restrictions_preserve_unrestricted_action_rights(self) -> None:
        blocks = [key.block for key in self.model.layout.keys]
        readiness = np.asarray([block == Block.READINESS_ORDER for block in blocks])
        exercise = np.asarray([block == Block.READINESS_EXERCISE for block in blocks])
        direct = np.asarray([block == Block.DIRECT_ORDER for block in blocks])
        release = np.asarray([block == Block.RELEASE for block in blocks])
        disclosure = np.asarray([block == Block.DISCLOSURE for block in blocks])

        no_readiness = self.restricted_vector("no_readiness")
        self.assertTrue(np.allclose(no_readiness[readiness | exercise], 0.0))
        self.assertTrue(np.allclose(no_readiness[direct], self.raw_vector[direct]))

        no_direct = self.restricted_vector("no_direct_capacity")
        self.assertTrue(np.allclose(no_direct[direct], 0.0))
        self.assertTrue(np.allclose(no_direct[readiness | exercise], self.raw_vector[readiness | exercise]))

        no_pacing = self.restricted_vector("no_release_pacing_authority")
        self.assertTrue(np.allclose(no_pacing[release], self.model.action_upper[release]))

        no_disclosure = self.restricted_vector("no_disclosure")
        self.assertTrue(np.allclose(no_disclosure[disclosure], 0.0))
        self.assertTrue(np.allclose(no_disclosure[~disclosure], self.raw_vector[~disclosure]))

    def test_completed_run_passes_all_blocking_checks(self) -> None:
        acceptance = json.loads((OUTPUT / "acceptance_5_2_3.json").read_text(encoding="utf-8"))
        self.assertEqual(acceptance["status"], "complete")
        self.assertEqual(acceptance["blocking_failures"], [])
        self.assertTrue(all(acceptance["blocking_checks"].values()))

    def test_locked_5_2_2_inputs_match_authorised_hashes(self) -> None:
        config = json.loads(
            (EXPERIMENT_523 / "config_5_2_3.json").read_text(encoding="utf-8")
        )
        upstream = CODE_ROOT / config["input_5_2_2"]
        for name, expected in config["input_locks"].items():
            actual = hashlib.sha256((upstream / name).read_bytes()).hexdigest()
            self.assertEqual(actual, expected)

    def test_full_action_exactly_reproduces_frozen_5_2_2(self) -> None:
        reproduction = pd.read_csv(OUTPUT / "full_action_reproduction.csv")
        self.assertGreater(len(reproduction), 0)
        self.assertTrue(reproduction["passed"].astype(bool).all())
        self.assertLessEqual(
            reproduction["absolute_difference"].max(),
            reproduction["tolerance"].min(),
        )
        self.assertEqual(
            set(reproduction["policy"]),
            {"Passive", "Reactive", "Model-guided constrained SAC"},
        )
        self.assertTrue(reproduction.groupby("policy")["path_id"].nunique().eq(88).all())

    def test_medoid_is_external_and_restrictions_use_physical_paths(self) -> None:
        medoid = pd.read_csv(OUTPUT / "path_medoid_selection.csv")
        self.assertEqual(int(medoid["selected_physical_path_medoid"].astype(bool).sum()), 1)
        self.assertFalse(medoid["selection_uses_policy_outcomes"].astype(bool).any())

        effects = pd.read_csv(OUTPUT / "restricted_action_paired_effects.csv")
        expected_outcomes = {
            "total_loss",
            "waiting_exposure",
            "sue_exit",
            "attrition_exit",
            "overload",
            "route_resource_loss",
            "action_loss",
            "clearance_probability",
            "restricted_mean_clearance_time",
            "final_outstanding",
        }
        self.assertEqual(set(effects["outcome"]), expected_outcomes)
        self.assertTrue((effects["physical_paths"] == 88).all())
        self.assertTrue(effects["multiplicity_family"].str.contains("four restrictions").all())

    def test_proposed_selector_and_all_action_channels_are_audited(self) -> None:
        activation = pd.read_csv(OUTPUT / "proposed_policy_activation_audit.csv")
        values = dict(zip(activation["metric"], activation["value"]))
        self.assertEqual(values["BC_proposal_selected_count"] + values["SAC_proposal_selected_count"], 5544)
        self.assertEqual(values["fallback_count"], 0)
        channel_rows = activation.loc[activation["metric"] == "activation_count"]
        self.assertEqual(
            set(channel_rows["module"]),
            {
                "readiness order",
                "direct order",
                "readiness exercise",
                "release",
                "disclosure",
            },
        )
        self.assertTrue((channel_rows["value"] > 0).all())

    def test_three_distinct_figure_families_are_manifested(self) -> None:
        chart_map = pd.read_csv(OUTPUT / "chart_map_5_2_3.csv")
        self.assertEqual(len(chart_map), 3)
        self.assertEqual(chart_map["chart_type"].nunique(), 3)
        for filename in (
            "figure_5_2_3a_action_congestion_trajectories.png",
            "figure_5_2_3b_tagged_route_stage_heatmap.png",
            "figure_5_2_3c_restricted_action_forest.png",
        ):
            self.assertTrue((EXPERIMENT_523 / "figures" / filename).exists())


if __name__ == "__main__":
    unittest.main()
