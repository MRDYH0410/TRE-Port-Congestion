# Formula-to-production mapping

| Contract | Formula or claim | Production implementation | 5.2.5 evidence |
|---|---|---|---|
| M0 | authorized upstream identity | `extended_audits.verify_upstream_locks` | `upstream_lock_audit.csv` |
| M1 | released `alpha_nu(t) P^h` enters observation | `experiments/5.2-2/features.py::state_features`, 5.2.4 release provider | `cross_module_trace.csv`, `release_nonanticipativity.csv` |
| M2 | matched exogenous paths and separated splits | 5.2.2 path manifests and replication engine | `cross_module_trace.csv`, `method_contract_registry.csv` |
| M3 | weighted convex feasible projection | `tre84.actions.ActionProjector.project` | `projection_feasibility.csv`, `projection_kkt_trace.csv` |
| M4 | residual-controlled lower-level SUE with true starts and final-trial certification | `tre84.behavior.RCMSASolver.solve` | `rcmsa_iteration_trace.csv`, `rcmsa_comparison_summary.csv`, `rcmsa_start_certification.csv` |
| M5 | route-tagged pipeline and four stages | `tre84.transition.TaggedTransition.step` | `tagged_mass_balance.csv`, `travel_lag_acceptance.csv` |
| M6 | complete operational and terminal loss | `tre84.loss`, 5.2.2 production runner | `loss_reconciliation.csv`, `clearance_terminal_acceptance.csv` |
| M7 | beginning-of-week nonanticipativity | 5.2.2 runner order and frozen policy interfaces | `policy_nonanticipativity.csv`, `release_nonanticipativity.csv` |
| M8 | scenario-weighted nested MPC | `tre84.control.ProjectedStochasticMPC` | `mpc_candidate_rollouts.csv`, `mpc_objective_recalculation.csv` |
| M9 | formal BC--SAC two-proposal selector | `tre84.control.TwoProposalSelector` | `selector_decision_trace.csv`, `selector_regret.csv` |
| M10 | route-lag convolution | `TaggedTransition._inject_and_advance_pipeline` | `travel_lag_acceptance.csv` |
| M11 | readiness/direct timing | `tre84.capacity.CapacityDynamics.transition` | `capacity_pipeline_acceptance.csv` |
| M12 | matched deterministic reproduction | frozen output aggregation and 5.2.4 anchor replay | `reproducibility_audit.csv` |
| M13 | reparameterised latent Gaussian SAC sampling | `features.LinearActor.sample_latent_normalised` | `sac_update_recalculation.csv` |
| M14 | SAC actor mean update | `training.train_sac` | `sac_checkpoint_replay.csv` |
| M15 | SAC actor log-standard-deviation update | `training.train_sac` | `sac_checkpoint_replay.csv`, `sac_training_trace.csv` |
| M16 | entropy term enters actor loss | `training._sac_sample_objective_and_gradient` | `sac_update_recalculation.csv` |
| M17 | adaptive entropy-temperature update | `training.train_sac` | `sac_update_recalculation.csv`, `sac_training_trace.csv` |
| M18 | twin reward critics | `training.train_sac` | `sac_episode_replay_summary.csv` |
| M19 | constraint critic | `training.train_sac` | `sac_update_recalculation.csv` |
| M20 | nonnegative constraint-dual update | `training.train_sac` | `sac_update_recalculation.csv` |
| M21 | formal projection local Jacobian in actor chain | `tre84.actions.ActionProjector.local_jacobian`, `training._normalised_projection_jacobian` | `sac_projection_jacobian.csv` |
| M22 | finite-difference actor gradient | `training.sac_actor_gradient_check` | `sac_actor_gradient_recalculation.csv` |
| M23 | validation-only checkpoint selection | `training.train_bc`, `training.train_sac` | `sac_checkpoint_replay.csv` |
| M24 | frozen checkpoint hash and action replay | `features.LinearActor.raw_action` | `sac_checkpoint_replay.csv` |
| M25 | unavailable route holds mass and provides zero service | `tre84.transition.TaggedTransition.step` | `unavailable_route_acceptance.csv` |
| M26 | right censoring and one terminal correction | `tre84.clearance.ClearanceRunner`, 5.2.2 runner | `clearance_terminal_acceptance.csv` |
| M27 | `Sel_t` distance on complete master choice support | `tre84.behavior.RCMSASolver._distance_to_previous` | `chapter4_contract_reinforcement.csv` |
| M28 | disclosure reference forecast uses `a_t^{-I}` | `tre84.factory.disclosure_reference_action`, `StandardBehaviorProblemFactory.__call__` | `chapter4_contract_reinforcement.csv` |
| M29 | every waiting vintage ages once and is never reset | `tre84.transition.TaggedTransition.step`, `tre84.acceptance._physical_certificate` | `chapter4_contract_reinforcement.csv` |
| M30 | all MPC/selector module certificates and mechanical selection logs are retained | `tre84.control.ProjectedStochasticMPC`, `tre84.control.TwoProposalSelector` | `chapter4_contract_reinforcement.csv` |
| M31 | contract-only repairs preserve accepted scientific quantities | `extended_audits.core_repair_numerical_equivalence_audit` | `core_repair_numerical_equivalence.csv` |

The registry CSV is authoritative because it also records criticality, tolerances, status, and failure reasons. SAC actor loss, temperature updates, critic targets, the projection Jacobian and the gradient check are independently reconstructed from accepted production traces rather than inferred from checkpoint readability.
