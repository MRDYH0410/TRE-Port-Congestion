# 5.2.2 formula-to-code map

| Requirement | Production implementation and output evidence |
|---|---|
| Unique hash-locked 5.2.1 interface | `paths.load_frozen_5_2_1_inputs`; `test_path_manifest.csv`; `run_manifest.json` |
| Beginning-of-week released information and event-aligned scenarios | `preparation.prepare_period`; `benchmark_period_paths.csv`; `policy_nonanticipativity_checks.csv` |
| Physical realization only after action | `preparation.realize_period`; `simulator.run_replication` |
| Complete timestamped scenario state | `preparation.PreparedPeriod`; `ModelState.scenario_ids`, readiness/operational weights and information timestamps |
| Reference commitment split and permanent tags | `model.build_realization`; `tre84.transition.construct_demand_split`; `trajectory_contract_checks.csv` |
| Common five-block authority and convex projection | `model.build_model`; `tre84.actions.ActionProjector`; `requested_and_implemented_actions.csv` |
| Route resource cost decomposition | `model.route_resource_cost_register`; `tre84.loss.OperationalLoss.compute`; `route_resource_cost_register.csv` |
| Passive and reactive comparators | `policies.PassivePolicy`; `policies.ReactivePolicy`; shared `BenchmarkSimulator` |
| Formal projected stochastic MPC | `policies.MPCPolicy`; `tre84.control.ProjectedStochasticMPC`; `solver_diagnostics.csv` |
| BC teacher and feasible action fit | `training.generate_teacher_data`; `training.train_bc`; `teacher_actions.csv`; checkpoint files |
| PPO from-scratch formal-reward updates | `training.train_ppo`; `training_curves.csv`; checkpoint files |
| Reparameterised SAC latent sample and common projection | `training.train_sac`; latent Gaussian/logistic draw followed by `ActionProjector.project` |
| Twin reward critics | `training.train_sac`; `critic_loss_q1`, `critic_loss_q2` and per-period update counts in `training_curves.csv` |
| Constraint critic and dual | constrained branch of `training.train_sac`; constraint-critic loss and dual update counts in `training_curves.csv` |
| Sampled actor, entropy and temperature updates | `training.train_sac`; sampled projected actor gradient, log-standard-deviation, entropy-temperature and actor update columns |
| Independent SAC gradient verification | `training.sac_actor_gradient_check`; `sac_actor_gradient_check.csv` |
| BC--Constrained-SAC two-proposal selector | `policies.ModelGuidedPolicy`; `tre84.control.TwoProposalSelector`; `proposal_selection_log.csv` |
| Formal operational objective and clearance | `simulator.run_replication`; `tre84.metrics.compute_trajectory_statistics`; `benchmark_replications.csv` |
| Pilot precision path count | `statistics.select_path_count`; `pilot_precision.csv`; `selected_path_count.csv` |
| Within-path seed aggregation | `statistics.aggregate_learning_seeds`; `path_level_seed_aggregated.csv` |
| Paired effects and Holm correction | `statistics.paired_policy_effects`; `tre84.inference.holm_adjust`; `paired_policy_effects.csv` |
| Right-censored clearance and RMST | `statistics.clearance_summary`; `clearance_summary.csv` |
| Three result figures and manifest | `reporting.create_figures`; `reporting.write_run_manifest`; `figures/` |
| Blocking acceptance | `reporting.acceptance_payload`; `acceptance_5_2_2.json`; `ACCEPTANCE_REPORT.md` |
| Unidentified late consequence is zero | `config_5_2_2.json: behavior.late_exit_cost_per_vintage`; `scientific_parameter_traceability.csv` |
| Held-out waiting-error credibility scale | `waiting_forecast_error_calibration.csv`; `model.CommonDisclosureBehaviorFactory` |
| Physical feedback and capacity pipelines | `config_5_2_2.json`; `model.build_model`; shared `tre84.transition.TaggedTransition` |

Passing this experiment supports only a conditional common-authority comparison. It does not establish global optimality, an empirical committed share, causal port effects, HMM information value, gateway-expansion value or renewed-closure boundaries.
