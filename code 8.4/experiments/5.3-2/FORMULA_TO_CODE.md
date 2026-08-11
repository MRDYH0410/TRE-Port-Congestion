# 5.3.2 Formula-to-Code Mapping

| Contract | Production implementation | Formal output |
|---|---|---|
| Reclosure serviceability, `a_reclose=1-intensity` | `reclosure_worker.GridCell.serviceability` | `reclosure_policy_path_manifest.csv` |
| One event-aligned open/reclosure/recovery constructor | `reclosure_worker.build_cell_path` | `reclosure_policy_path_manifest.csv` |
| Same information → action → projector → RC-MSA → tagged transition → loss chain | `reclosure_worker._advance` | `trajectory_contract_checks.csv` |
| Frozen deployed policy coverage | `run_5_3_2._cells_by_spec` | `cell_policy_coverage_registry.csv` |
| Seed aggregation within physical path | `statistics_5_3_2.aggregate_learning_seeds` | `path_level_seed_aggregated.csv` |
| Paired effects and Holm adjustment | `statistics_5_3_2.paired_effects` | `paired_effects.csv` |
| Family-specific simultaneous confidence sets and regret | `statistics_5_3_2.confidence_sets_and_regret` | `policy_confidence_set.csv`, `policy_regret.csv` |
| Anchor precision path-count rule | `statistics_5_3_2.precision_requirements` | `anchor_precision_requirements.csv` |
| Route-lag-aware optimistic absorption at reclosure onset | `absorption_5_3_2.absorption_certificate`; `tre84.diagnostics.AbsorptionBoundaryLP` | `absorption_certificate_path_results.csv` |
| Sufficient violation `Q^C(H)>C_abs(H)` | first cumulative violating horizon in `absorption_certificate` | `absorption_certificate_summary.csv` |
| Clearance cap as right censoring | `reclosure_worker._finalise`; `tre84.clearance.ClearanceRunner` | `clearance_and_censoring.csv` |
| Historical RC-MSA shares remain on the master-choice simplex, including subnormal old vintages | `tre84.behavior.RCMSASolver._shares`; `tre84.acceptance._behavior_certificate` | `single_path_gate.json`, `trajectory_contract_checks.csv` |
