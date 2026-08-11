"""Blocking information-timing and capacity-right contracts for Experiment 5.2.4."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
EXP_522 = CODE_ROOT / "experiments" / "5.2-2"
EXP_524 = CODE_ROOT / "experiments" / "5.2-4"
OUTPUT = CODE_ROOT / "output" / "5.2.4_released_risk_information_capacity_preparation"
for path in (EXP_524, EXP_522, CODE_ROOT / "src"):
    sys.path.insert(0, str(path))

from controller_factory import CapacityRightsProjector  # noqa: E402
from information_design import InformationProvider, load_hmm_inputs, monthly_transition_count  # noqa: E402
from model import build_model  # noqa: E402
from paths import PhysicalPath, build_test_paths, load_frozen_5_2_1_inputs, sha256_file  # noqa: E402
from run_5_2_4 import _load_locked_medoid, _validate_test_path_reconstruction, _verify_upstream_locks  # noqa: E402
from tre84.actions import Action, Block  # noqa: E402


class Experiment524Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((EXP_522 / "config_5_2_2.json").read_text(encoding="utf-8"))
        cls.model = build_model(cls.config)
        cls.hmm = load_hmm_inputs(CODE_ROOT / "output" / "5.2.1_data_event_information_validity")

    def test_monthly_horizon_is_calendar_based_not_weekly(self) -> None:
        self.assertEqual(monthly_transition_count("2025-12-01", "2026-02-23"), 2)
        self.assertEqual(monthly_transition_count("2025-12-01", "2026-04-20"), 4)

    def test_no_new_information_is_stationary_and_nonzero(self) -> None:
        provider = InformationProvider(hmm=self.hmm, readiness_lead_weeks=8)
        frame = pd.DataFrame(
            {
                "week": pd.date_range("2026-02-23", periods=2, freq="W-MON"),
                "normal_model_units": [1.0, 1.0],
                "serviceability": [1.0, 1.0],
                "filtered_high_risk_probability": [0.0, 1.0],
                "lead_time_high_risk_probability": [0.2, 0.8],
                "release_date": pd.to_datetime(["2026-01-31", "2026-02-28"]),
                "source_observation_month": pd.to_datetime(["2025-12-01", "2026-01-01"]),
                "timing_valid": [True, True],
                "information_source": ["test", "test"],
                "residual_date": pd.NaT,
            }
        )
        path = PhysicalPath("p", "test", frame, "hash", "test", "", "", 2, 0, 1.0, False)
        transformed = provider.apply_training_regime(path, "I0")
        baseline = float(self.hmm.stationary[-1])
        self.assertGreater(baseline, 0.0)
        self.assertTrue(np.allclose(transformed.frame["filtered_high_risk_probability"], baseline))
        self.assertTrue(np.allclose(transformed.frame["lead_time_high_risk_probability"], baseline))

    def test_capacity_rights_preserve_noncapacity_actions(self) -> None:
        raw = Action.from_vector(self.model.layout.keys, 0.4 * self.model.action_upper)
        blocks = [key.block for key in self.model.layout.keys]
        for rights, readiness_allowed, direct_allowed in (
            ("R", True, False),
            ("D", False, True),
            ("NONE", False, False),
        ):
            wrapper = CapacityRightsProjector(inner=self.model.projector, model=self.model, capacity_rights=rights)
            restricted = wrapper.restrict(raw).vector(self.model.layout.keys)
            readiness = np.asarray(
                [block in {Block.READINESS_ORDER, Block.READINESS_EXERCISE} for block in blocks]
            )
            direct = np.asarray([block == Block.DIRECT_ORDER for block in blocks])
            other = ~(readiness | direct)
            self.assertTrue(np.allclose(restricted[other], raw.vector(self.model.layout.keys)[other]))
            self.assertEqual(bool(np.any(restricted[readiness] > 0)), readiness_allowed)
            self.assertEqual(bool(np.any(restricted[direct] > 0)), direct_allowed)

    def test_capacity_rights_projector_chains_the_formal_local_jacobian(self) -> None:
        frozen = load_frozen_5_2_1_inputs(self.config)
        first = build_test_paths(config=self.config, frozen=frozen, count=1)[0].frame.iloc[0].to_dict()
        state = self.model.initial_state(first)
        wrapper = CapacityRightsProjector(inner=self.model.projector, model=self.model, capacity_rights="D")
        raw_vector = 0.2 * np.asarray(self.model.action_upper, dtype=float)
        raw = Action.from_vector(self.model.layout.keys, raw_vector)
        projection = wrapper.project(raw, state)
        jacobian = wrapper.local_jacobian(raw, state, projection=projection)
        readiness_columns = [
            index
            for index, key in enumerate(self.model.layout.keys)
            if key.block in {Block.READINESS_ORDER, Block.READINESS_EXERCISE}
        ]
        self.assertTrue(np.allclose(jacobian[:, readiness_columns], 0.0))
        retained_index = next(
            index for index, key in enumerate(self.model.layout.keys) if key.block == Block.RELEASE
        )
        step = 1e-6 * float(self.model.action_upper[retained_index])
        plus = raw_vector.copy()
        minus = raw_vector.copy()
        plus[retained_index] += step
        minus[retained_index] -= step
        plus_vector = wrapper.project(Action.from_vector(self.model.layout.keys, plus), state).action.vector(self.model.layout.keys)
        minus_vector = wrapper.project(Action.from_vector(self.model.layout.keys, minus), state).action.vector(self.model.layout.keys)
        finite_difference = (plus_vector - minus_vector) / (2.0 * step)
        self.assertTrue(np.allclose(finite_difference, jacobian[:, retained_index], atol=2e-4, rtol=2e-4))

    def test_compressed_action_trace_exposes_rights_contract_totals(self) -> None:
        source = (EXP_524 / "evaluation_5_2_4.py").read_text(encoding="utf-8")
        self.assertNotIn('row["action_block"]', source)
        self.assertIn('row["implemented_readiness_order"]', source)
        self.assertIn('row["implemented_readiness_exercise"]', source)
        self.assertIn('row["implemented_direct_order"]', source)

    def test_seven_upstream_locks_and_accepted_medoid(self) -> None:
        config_524 = json.loads((EXP_524 / "config_5_2_4.json").read_text(encoding="utf-8"))
        input_521 = CODE_ROOT / config_524["input_5_2_1"]
        input_522 = CODE_ROOT / config_524["input_5_2_2"]
        input_523 = CODE_ROOT / config_524["input_5_2_3"]
        actual = _verify_upstream_locks(
            config=config_524,
            input_5_2_1=input_521,
            input_5_2_2=input_522,
            input_5_2_3=input_523,
        )
        self.assertEqual(actual, {key: value.lower() for key, value in config_524["upstream_locks"].items()})
        frozen = load_frozen_5_2_1_inputs(self.config)
        paths = build_test_paths(config=self.config, frozen=frozen, count=88)
        _validate_test_path_reconstruction(paths, input_522)
        medoid_id, audit = _load_locked_medoid(input_523, paths)
        self.assertEqual(medoid_id, "test_017_ebd333a3cdba")
        self.assertEqual(sha256_file(input_523 / "path_medoid_selection.csv"), config_524["upstream_locks"]["path_medoid_selection_5_2_3"])
        self.assertEqual(int(audit["selected_physical_path_medoid"].astype(bool).sum()), 1)

    def test_completed_output_contracts_when_present(self) -> None:
        if not OUTPUT.exists():
            self.skipTest("full 5.2.4 output has not been generated")
        acceptance = json.loads((OUTPUT / "acceptance_5_2_4.json").read_text(encoding="utf-8"))
        if "all_seven_upstream_hash_locks_match" not in acceptance.get("blocking_checks", {}):
            self.skipTest("stale pre-repair 5.2.4 output is not admissible evidence")
        self.assertEqual(acceptance["status"], "complete")
        self.assertEqual(acceptance["blocking_failures"], [])
        self.assertTrue(all(acceptance["blocking_checks"].values()))
        effects = pd.read_csv(OUTPUT / "paired_effects.csv")
        self.assertEqual(
            set(effects["evidence_layer"]),
            {
                "reoptimized_information_value",
                "fixed_policy_information_responsiveness",
                "reoptimized_capacity_rights",
            },
        )
        fixed = pd.read_csv(OUTPUT / "fixed_policy_information_diagnostic.csv")
        self.assertEqual(set(fixed["controller_id"]), {"IL_RD"})
        self.assertNotIn("ORACLE", set(fixed["evaluation_information_regime"]))
        precision = pd.read_csv(OUTPUT / "pilot_precision.csv")
        self.assertEqual(len(precision), 40)
        self.assertTrue(precision["precision_target_met"].all())
        primary = pd.read_csv(OUTPUT / "primary_information_value_path_results.csv")
        self.assertEqual(primary["base_path_id"].nunique(), 88)


if __name__ == "__main__":
    unittest.main()
