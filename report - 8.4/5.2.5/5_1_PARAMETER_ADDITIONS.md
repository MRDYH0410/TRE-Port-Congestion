# 5.1 Parameter and Diagnostic Additions from Experiment 5.2.5

**DATA: NO CHANGE.** Experiment 5.2.5 introduces no new empirical data, event window, HMM estimate, or upstream policy result.

**NUMERICAL PARAMETERS: NO ADDITION.** The refresh adds acceptance certificates and one machine-precision equivalence check; it does not change an economic, physical, statistical-design, training, or solver parameter.

The following registry is code generated for later 5.1 updating. Values are not inserted into the manuscript automatically.

| Parameter | Value | Basis category | Basis |
|---|---:|---|---|
| rcmsa_stopping_tolerance | 1e-06 | chapter_method | Chapter 4 and 5.1 registered equilibrium tolerance |
| rcmsa_maximum_iterations | 500 | chapter_method | Chapter 4 and 5.1 computational cap |
| rcmsa_step_size_rule | min(1,kappa/(n+1)), kappa in {1,2,4}; choose least residual | chapter_method | Chapter 4 RC-MSA definition |
| conventional_msa_comparator | 1/(n+1) | algorithmic_comparator | classical MSA comparator on the identical fixed-point problem |
| projection_solver | SciPy SLSQP weighted Euclidean convex projection | solver | Chapter 4 projection plus solver implementation |
| projection_tolerance | 1e-08 | chapter_method | 5.1 registered solver tolerance |
| projection_max_iterations | 300 | chapter_method | 5.1 registered solver cap |
| mpc_horizon_weeks | 8 | chapter_method | 5.1 readiness-aligned control horizon |
| mpc_scenario_count | 3 | designed_experiment | 5.2.2 frozen structural scenario bundle |
| mpc_candidate_construction | zero + five block endpoints + balanced midpoint | designed_experiment | registered endpoint-midpoint finite lattice |
| mpc_terminal_loss | shared TerminalMassCorrection.compute | chapter_method | Chapter 3 terminal outstanding correction |
| selector_horizon_and_scenarios | H=8; scenarios=3 | chapter_method | same formal MPC evaluator for BC and SAC |
| bc_architecture | complete-state linear sigmoid actor with 34 output actions | designed_experiment | 5.2.2 frozen state_and_actor contract and checkpoint shape |
| bc_loss | projected-action mean squared error on formal MPC teacher plus ridge | chapter_method | 5.2.2 training.py::train_bc |
| bc_batch_size | full frozen teacher set per epoch | algorithm_implementation | train_bc uses all teacher rows after a deterministic permutation |
| bc_ridge | 0.001 | designed_experiment | 5.2.2 frozen training configuration |
| sac_architecture | linear sigmoid actor; twin ridge critics with projection dimension 32 and interaction head 4 | designed_experiment | 5.2.2 training.py::train_sac |
| sac_discount | 0.98 | designed_experiment | 5.2.2 frozen training configuration |
| sac_initial_entropy_temperature | 0.05 | chapter_method | Initial condition only; adaptively updated by the Chapter 4/5.1 temperature loss |
| sac_entropy_temperature_learning_rate | 0.02 | chapter_method | 5.1 registered adaptive entropy-temperature update |
| sac_target_entropy_rule | negative_action_dimension | chapter_method | 5.1 target entropy equals negative full action dimension |
| sac_constraint_dual_step | 0.1 | designed_experiment | 5.2.2 frozen constrained-SAC setting |
| sac_gradient_check_step | 6.055454452393343e-06 | machine_precision | cube root of IEEE-754 double machine epsilon |
| sac_gradient_check_relative_tolerance | 0.0001 | chapter_method | 5.1 numerical gradient acceptance tolerance |
| sac_update_replay_scope | first complete 21-period production episode for Vanilla and Constrained SAC | acceptance_design | deterministic replay of the accepted production training path and optimizer |
| checkpoint_selection_rule | validation every 3 episodes; fractional improvement 0.001; patience 2 | designed_experiment | 5.2.2 validation-only checkpoint rule |
| training_seeds | 3 | designed_experiment | 5.2.2 frozen deterministic seed construction |
| training_minimum_and_cap | minimum=9; maximum=12 | designed_experiment | 5.2.2 preregistered stopping rule |
| numerical_precision_target | all registered residual tolerances; no averaged acceptance score | acceptance_design | noncompensatory contract rule |
| paired_path_precision_target | 5.2.2 selected_path_count.csv; 88 physical paths and target half-width 2255.637825 | statistical_design | 5.2.2 preregistered statistical precision |
| mass_conservation_tolerance | 1e-06 | chapter_method | 5.1 production numerical tolerance |
| loss_reconciliation_tolerance | 1e-06 | chapter_method | 5.1 production numerical tolerance |
| clearance_tolerance | 1e-06 | chapter_method | 5.1 clearance rule |
| runtime_environment | Python 3.12.13; Windows-11-10.0.26200-SP0 | runtime_observation | captured at run time |
| hardware_configuration | AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD | runtime_observation | captured at run time |
| runtime_timeout_rule | No new timeout introduced; frozen upstream caps apply | chapter_method | preserves production settings |
| 5.2.5_controlled_rcmsa_cases | 3 | acceptance_design | boundary audit count, not an economic parameter |
| rcmsa_master_choice_distance_audit | complete current-plus-previous master support; unavailable current routes have zero share | acceptance_contract | Chapter 4 deterministic Sel_t contract |
| disclosure_reference_action_audit | reference forecast receives a_t^{-I} at the StandardBehaviorProblemFactory boundary | acceptance_contract | Chapter 4 frozen disclosure baseline contract |
| waiting_vintage_no_reset_audit | every source vintage balance plus independent age-zero renewed-waiting identity | acceptance_contract | Chapter 4 final behavioral/physical acceptance contract |
| mpc_selector_module_certificate_audit | raw/projected action, projection, RC-MSA, tagged transition, loss, terminal, failure and selection records | acceptance_contract | Chapter 4 integrated algorithm return contract |
| core_repair_equivalence_tolerance | 1e-12 | machine_precision | accepted pre-repair deterministic scientific-output comparison; runtime excluded |

New diagnostic scope: the RC-MSA selector distance now has a complete-master support certificate; disclosure-reference construction has an interface-level a_t^{-I} certificate; waiting transition has a per-vintage no-reset certificate; MPC and the two-proposal selector persist complete nested module certificates and mechanical selection logs; accepted scientific quantities are compared against the pre-repair baseline at 1e-12. These are metrics/contracts, not new model parameters.

SAC critic, entropy, adaptive-temperature and dual traces remain audited directly; they are not parameter additions. Remaining noncritical instrumentation gaps are persisted solver KKT multipliers, BC action-coordinate validation errors, historical training wall time, and peak memory. No convenient default is introduced for any gap.
