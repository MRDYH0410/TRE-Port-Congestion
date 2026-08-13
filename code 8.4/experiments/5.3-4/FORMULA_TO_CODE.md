# Formula-to-code contract for Experiment 5.3.4

| Contract | Mathematical or experimental object | Production implementation | Formal output |
|---|---|---|---|
| F534-01 | Reference-centred structural transformation | `robustness_5_3_4.model_config` | `parameter_registry_5_3_4.csv` |
| F534-02 | Waiting attrition profile `(j/Jbar)^p` | `model.build_model` with the cell exponent | `parameter_registry_5_3_4.csv` |
| F534-03 | Network exposure `m_Q varrho^(Q,ref)` | `robustness_5_3_4.transform_paths` | `test_path_cell_manifest.csv` |
| F534-04 | Nonzero route-resource cost multiplier | `model.route_resource_cost_register` | `parameter_registry_5_3_4.csv` |
| F534-05 | RMSE waiting-scale multiplier | `model.build_model` validation contract | `parameter_registry_5_3_4.csv` |
| F534-06 | Severe reclosure path `(1,0.95,32)` | `robustness_5_3_4._severe_reclosure_path` | `test_path_cell_manifest.csv` |
| F534-07 | Nine-gateway structural constructor | `network_5_3_3.build_cell_config` | `test_path_cell_manifest.csv` |
| F534-08 | Matched BC and full constrained-SAC training | `run_5_3_4._train_matched_bundle` | `matched_*_curves.csv`, derivative audits |
| F534-09 | Common projector, RC-MSA, transition, and complete loss | `robustness_worker.evaluate_task` | `trajectory_contract_checks.csv` |
| F534-10 | Learning seeds averaged within physical path | `statistics_5_3_4.aggregate_learning_seeds` | `path_level_seed_aggregated.csv` |
| F534-11 | Cell-minus-reference paired effect | `statistics_5_3_4.paired_cell_effects` | `paired_parameter_effects.csv` |
| F534-12 | Policy regret and simultaneous confidence set | `statistics_5_3_4.policy_regret` | `policy_regret.csv`, `policy_confidence_set.csv` |
| F534-13 | Clearance reclassification only | `statistics_5_3_4.clearance_endpoint_diagnostic` | `clearance_tolerance_diagnostic.csv` |

All multipliers are designed structural stresses, not estimated confidence intervals. The clearance diagnostic cannot change actions, transitions, loss, or terminal state. Missing or dimension-incompatible policy-cell pairs are recorded rather than imputed.
