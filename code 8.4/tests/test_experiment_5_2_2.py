"""Blocking unit and integration contracts for experiment 5.2.2."""

from __future__ import annotations

import json
import copy
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = CODE_ROOT / "experiments" / "5.2-2"
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(CODE_ROOT / "src"))

from model import build_model, route_resource_cost_register  # noqa: E402
from paths import (  # noqa: E402
    build_test_paths,
    build_training_validation_paths,
    load_frozen_5_2_1_inputs,
)
from training import sac_actor_gradient_check, train_sac  # noqa: E402
from policies import PassivePolicy  # noqa: E402
from reporting import parameter_registry, scientific_parameter_traceability  # noqa: E402
from simulator import build_realization, run_replication  # noqa: E402


class Experiment522Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((EXPERIMENT / "config_5_2_2.json").read_text(encoding="utf-8"))
        cls.frozen = load_frozen_5_2_1_inputs(cls.config)
        cls.model = build_model(cls.config)

    def test_frozen_5_2_1_interface_has_no_future_information_or_manual_ramp(self) -> None:
        frame = self.frozen.historical
        self.assertEqual(len(frame), 21)
        self.assertTrue(frame["timing_valid"].all())
        self.assertTrue((frame["release_date"] <= frame["week"]).all())
        self.assertTrue(frame["risk_information_source"].str.contains("released", case=False).all())
        self.assertFalse(frame["risk_information_source"].str.contains("ramp", case=False).any())
        self.assertEqual(
            self.frozen.interface_hash,
            "407b06106c86f7173d399d8a66283f48ac6746b146c869c14cf77bad3ba3a976",
        )

    def test_training_validation_test_paths_are_hash_disjoint(self) -> None:
        train, validation = build_training_validation_paths(
            config=self.config,
            residuals=self.frozen.residuals,
            reference_normal_model_units=sum(self.model.gateway_scales.values()),
        )
        test = build_test_paths(config=self.config, frozen=self.frozen, count=2)
        train_hashes = {path.path_hash for path in train}
        validation_hashes = {path.path_hash for path in validation}
        test_hashes = {path.path_hash for path in test}
        self.assertFalse(train_hashes & validation_hashes)
        self.assertFalse(train_hashes & test_hashes)
        self.assertFalse(validation_hashes & test_hashes)
        self.assertEqual(len(test_hashes), len(test))

    def test_extended_test_paths_are_unique_without_replacing_pilot_prefix(self) -> None:
        pilot = build_test_paths(config=self.config, frozen=self.frozen, count=4)
        extended = build_test_paths(config=self.config, frozen=self.frozen, count=50)
        self.assertEqual(
            [path.path_hash for path in extended[:4]],
            [path.path_hash for path in pilot],
        )
        self.assertEqual(len({path.path_hash for path in extended}), 50)

    def test_projector_exposes_piecewise_subgradient_without_changing_projection(self) -> None:
        train, _ = build_training_validation_paths(
            config=self.config,
            residuals=self.frozen.residuals,
            reference_normal_model_units=sum(self.model.gateway_scales.values()),
        )
        first = train[0].frame.iloc[0].to_dict()
        state = self.model.initial_state(first)
        normalised = np.full(len(self.model.layout.keys), 0.05)
        raw = self.model.action_from_normalised(normalised)
        projected = self.model.projector.project(raw, state)
        jacobian = self.model.projector.local_jacobian(raw, state, projection=projected)
        free = np.flatnonzero(np.diag(jacobian) > 0.5)
        self.assertGreater(len(free), 0)
        coordinate = int(free[0])
        epsilon = 1e-6
        vector = raw.vector(self.model.layout.keys)
        perturbed = vector.copy()
        perturbed[coordinate] += epsilon
        from tre84.actions import Action

        shifted = self.model.projector.project(
            Action.from_vector(self.model.layout.keys, perturbed), state
        ).action.vector(self.model.layout.keys)
        finite_difference = (
            shifted - projected.action.vector(self.model.layout.keys)
        ) / epsilon
        np.testing.assert_allclose(
            finite_difference,
            jacobian[:, coordinate],
            atol=2e-5,
            rtol=2e-5,
        )

    def test_sac_updates_latent_variance_entropy_temperature_and_all_critics(self) -> None:
        config = copy.deepcopy(self.config)
        config["training"]["minimum_episodes"] = 3
        config["training"]["maximum_episodes"] = 3
        config["training"]["validation_interval_episodes"] = 3
        model = build_model(config)
        train, validation = build_training_validation_paths(
            config=config,
            residuals=self.frozen.residuals,
            reference_normal_model_units=sum(model.gateway_scales.values()),
        )
        result = train_sac(
            model=model,
            training_paths=train[:1],
            validation_paths=validation[:1],
            seed_index=0,
            constrained=True,
        )
        initial_log_std = float(config["state_and_actor"]["initial_log_action_standard_deviation"])
        self.assertFalse(np.allclose(result.actor.log_standard_deviation, initial_log_std))
        self.assertNotAlmostEqual(
            result.entropy_temperature,
            float(config["training"]["sac_entropy_temperature"]),
        )
        curve = pd.DataFrame(result.training_curve)
        for column in (
            "critic_loss_q1",
            "critic_loss_q2",
            "constraint_critic_loss",
            "latent_policy_entropy",
            "entropy_temperature",
            "mean_log_standard_deviation",
        ):
            self.assertTrue(np.isfinite(curve[column]).all(), column)
        self.assertTrue(curve["update_method"].str.contains("auto_entropy").all())
        for column in (
            "period_update_count",
            "reward_critic_q1_update_count",
            "reward_critic_q2_update_count",
            "constraint_critic_update_count",
            "actor_update_count",
            "entropy_temperature_update_count",
            "constraint_dual_update_count",
        ):
            self.assertTrue(curve[column].eq(config["event_weeks"]).all(), column)

    def test_sac_actor_gradient_matches_independent_central_difference(self) -> None:
        rows = pd.DataFrame(sac_actor_gradient_check(self.model))
        self.assertEqual(len(rows), 44)
        self.assertTrue(rows["passed"].astype(bool).all())
        self.assertLess(
            rows["relative_error"].max(),
            self.config["numerics"]["sac_gradient_check_relative_tolerance"],
        )

    def test_common_action_authority_and_route_cost_closure(self) -> None:
        self.assertEqual(len(self.model.controlled_resources), 10)
        self.assertEqual(len(self.model.layout.keys), 34)
        self.assertEqual(len(self.model.layout.readiness_order), 10)
        self.assertEqual(len(self.model.layout.direct_order), 10)
        self.assertEqual(len(self.model.layout.readiness_exercise), 10)
        self.assertEqual(len(self.model.layout.release), 1)
        self.assertEqual(len(self.model.layout.disclosure), 3)
        self.assertFalse(any("exit" in name for name in self.model.layout.names))
        self.assertEqual(self.config["forbidden_main_policy"], "Equal allocation")
        self.assertNotIn("Equal allocation", self.config["main_policies"])
        costs = route_resource_cost_register(self.config)
        self.assertFalse(costs["total_incremental_resource_cost"].isna().any())
        self.assertTrue((costs["total_incremental_resource_cost"] > 0).any())

    def test_chapter_3_late_exit_and_disclosure_contracts(self) -> None:
        self.assertEqual(self.config["behavior"]["late_exit_cost_per_vintage"], 0.0)
        _, validation = build_training_validation_paths(
            config=self.config,
            residuals=self.frozen.residuals,
            reference_normal_model_units=sum(self.model.gateway_scales.values()),
        )
        rows = [row._asdict() for row in validation[0].frame.itertuples(index=False)]
        state = self.model.initial_state(rows[0])
        action_vector = np.zeros(len(self.model.layout.keys), dtype=float)
        action_vector[-len(self.model.layout.disclosure) :] = 1.0
        action = self.model.action_from_normalised(action_vector)
        from preparation import prepare_period

        prepared = prepare_period(model=self.model, state=state, row=rows[2])
        problem = self.model.kernel.behavior_factory(
            prepared.state,
            action,
            build_realization(model=self.model, state=prepared.state, row=rows[2]),
        )
        self.assertTrue(all(value == 0.0 for value in problem.parameters.late_exit_cost.values()))
        registered = self.config["information"]["waiting_error_scale_weeks_by_route"]
        for (_, route), scale in problem.disclosure.error_scale.items():
            self.assertAlmostEqual(scale, registered[route])
        self.assertTrue(
            any(
                not np.isclose(problem.disclosure.public_signal[key], problem.disclosure.reference_forecast[key])
                for key in problem.disclosure.public_signal
            )
        )

    def test_waiting_error_calibration_and_scientific_parameters_are_traceable(self) -> None:
        calibration = pd.read_csv(EXPERIMENT / "waiting_forecast_error_calibration.csv")
        self.assertTrue((calibration["validation_errors"] == 40).all())
        self.assertFalse(calibration["uses_historical_test_event"].astype(bool).any())
        registered = self.config["information"]["waiting_error_scale_weeks_by_route"]
        for row in calibration.itertuples(index=False):
            self.assertAlmostEqual(float(row.sigma_W_rmse_weeks), float(registered[row.route]))
        registry = parameter_registry(self.config)
        self.assertTrue(registry["parameter"].eq("information.gamma_I").any())
        self.assertEqual(
            registry["parameter"].str.startswith("information.calibration_sigma_W.").sum(),
            3,
        )
        trace = scientific_parameter_traceability(self.config)
        self.assertGreaterEqual(len(trace), 15)
        self.assertTrue(trace["traceable"].astype(bool).all())

    def test_passive_keeps_sue_and_formal_transition_active(self) -> None:
        train, _ = build_training_validation_paths(
            config=self.config,
            residuals=self.frozen.residuals,
            reference_normal_model_units=sum(self.model.gateway_scales.values()),
        )
        artifact = run_replication(
            model=self.model,
            policy=PassivePolicy(self.model),
            path=train[0],
        )
        checks = artifact.contract_checks
        self.assertTrue(all(value for key, value in checks.items() if isinstance(value, (bool, np.bool_))))
        self.assertLessEqual(
            checks["maximum_transition_residual"],
            self.config["numerics"]["mass_tolerance"],
        )
        actions = artifact.actions
        direct_columns = [
            f"implemented_{name}"
            for name in self.model.layout.names
            if name.startswith("direct_order__")
        ]
        self.assertEqual(len(direct_columns), 10)
        self.assertTrue(all(row[column] == 0.0 for row in actions for column in direct_columns))
        self.assertGreater(artifact.replication["total_operational_objective"], 0.0)

    def test_completed_output_is_manifested_and_accepted(self) -> None:
        if os.environ.get("TRE84_VERIFY_FORMAL_OUTPUT") != "1":
            self.skipTest("Formal output is checked only after the authorised from-scratch run")
        output = CODE_ROOT / self.config["output_directory"]
        acceptance_path = output / "acceptance_5_2_2.json"
        manifest_path = output / "run_manifest.json"
        self.assertTrue(acceptance_path.exists(), "Run experiment 5.2.2 before final acceptance tests")
        self.assertTrue(manifest_path.exists())
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        self.assertEqual(acceptance["status"], "complete")
        self.assertEqual(acceptance["blocking_failures"], [])
        for required_check in (
            "unidentified_late_exit_cost_is_zero",
            "disclosure_error_scale_matches_frozen_calibration",
            "disclosure_gamma_is_registered_and_bounded",
            "public_signal_and_reference_loading_are_registered",
            "all_scientific_parameter_sections_are_frozen",
            "all_identified_scientific_parameter_families_are_traceable",
        ):
            self.assertTrue(acceptance["blocking_checks"][required_check])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = {item["path"] for item in manifest["outputs"]}
        for required in (
            "benchmark_replications.csv",
            "paired_policy_effects.csv",
            "acceptance_5_2_2.json",
            "figures/figure_5_2_2a_policy_performance.png",
            "figures/figure_5_2_2b_loss_decomposition.png",
            "figures/figure_5_2_2c_clearance_censoring.png",
        ):
            self.assertIn(required, names)


if __name__ == "__main__":
    unittest.main()
