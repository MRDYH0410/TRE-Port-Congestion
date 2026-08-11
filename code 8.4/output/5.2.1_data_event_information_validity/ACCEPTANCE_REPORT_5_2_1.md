# 5.2.1 Acceptance Report

## Overall status: COMPLETE

This report concerns data, counterfactual predictive validity, released geopolitical-risk information, and formula-derived event inputs only. It contains no policy training or comparison.

## Blocking acceptance checks

| Check | Result | Observed | Expected |
|---|---:|---|---|
| used_input_fields_complete | PASS | 0 | 0 |
| frozen_input_hashes_and_grain | PASS | {'pass': 5} | {'pass': 5} |
| event_window_21_complete_monday_weeks | PASS | 21 | 21 |
| counterfactual_event_leakage_absent | PASS | 2026-02-16 | 2026-02-16 |
| all_feasible_weekly_origins_used | PASS | 216 | 216 |
| counterfactual_selection_rule_prefrozen | PASS | cumulative_path_WAPE | cumulative_path_WAPE at 21 weeks with declared tie breaks |
| residual_library_event_free_unique_one_step | PASS | {'rows': 216, 'latest': '2026-02-16', 'duplicates': 0} | unique one-step residuals ending no later than 2026-02-16 |
| hmm_training_and_heldout_split | PASS | {'training': 456, 'held_out': 18} | {'training': 456, 'held_out': 18} |
| hmm_em_converged | PASS | {'iterations': 8, 'last_change': 1.0862977433134802e-06} | absolute log-likelihood change <= 1e-05 |
| hmm_has_no_closure_or_event_label | PASS | ['gpr_threat', 'gpr_act', 'gpr_threat_diff', 'gpr_act_diff', 'gpr_threat_diff_volatility_24m', 'gpr_act_diff_volatility_24m', 'gpr_threat_jump', 'gpr_act_jump'] | threat/act levels, differences, volatilities, and declared jumps only |
| hmm_feature_missingness | PASS | 0 | 0 |
| heldout_parameters_never_refit | PASS | {False: 162} | {False: 162} |
| release_date_not_after_decision | PASS | 0 | 0 |
| monthly_transition_matrix_not_applied_weekly | PASS | {'maximum_monthly_transitions': 4, 'weekly_applications': 0} | calendar-month difference with zero weekly applications |
| serviceability_closed_unit_interval | PASS | [0.0, 0.7396289684964209] | [0, 1] |
| blocked_flow_nonnegative | PASS | 849609.9232757334 | >= 0 |
| blocked_flow_identity | PASS | 0.0 | <= 1e-09 |
| model_unit_conversion | PASS | 0.0 | <= 1e-10 with U_Q=1000.0 |
| no_artificial_risk_ramp | PASS | {'risk_sources': ['released_hmm_filter'], 'event_sources': ['formula_derived_5.2.1']} | released_hmm_filter plus formula_derived_5.2.1 only |
| committed_share_absent | PASS | False | False |
| required_parameters_registered | PASS | [] | [] |
| figures_reproducible_300dpi_and_mapped | PASS | {'figure_5_2_1a_counterfactual_predictive_validity.png': {'exists': True, 'dpi': [300, 300], 'source_tables': ['figure_5_2_1a_data.csv', 'counterfactual_rolling_origin_summary.csv', 'counterfactual_one_step_residual_acf.csv', 'counterfactual_model_selection.csv']}, 'figure_5_2_1b_released_hmm_validity.png': {'exists': True, 'dpi': [300, 300], 'source_tables': ['figure_5_2_1b_data.csv', 'heldout_density_summary.csv', 'released_hmm_filter.csv']}, 'figure_5_2_1c_event_input_release_clock.png': {'exists': True, 'dpi': [300, 300], 'source_tables': ['figure_5_2_1c_data.csv', 'historical_information_event_path.csv', 'release_clock.csv']}} | three code-generated figures at 300 dpi with declared source tables |
| independent_recalculation_matches_formal_csv | PASS | {'passed': 10, 'total': 10} | {'passed': 10, 'total': 10} |
| critical_frozen_inputs_and_downstream_outputs_reproduced | PASS | {'byte_identical': 9, 'compared': 9} | {'byte_identical': 9, 'compared': 9} |
| all_required_pre_manifest_outputs_created | PASS | ['baseline_reproduction_audit.csv', 'counterfactual_model_selection.csv', 'counterfactual_model_specifications.csv', 'counterfactual_one_step_residual_acf.csv', 'counterfactual_residual_library.csv', 'counterfactual_rolling_origin_predictions.csv', 'counterfactual_rolling_origin_summary.csv', 'data_audit.csv', 'experiment_input_register.csv', 'figure_5_2_1a_counterfactual_predictive_validity.pdf', 'figure_5_2_1a_counterfactual_predictive_validity.png', 'figure_5_2_1a_data.csv', 'figure_5_2_1b_data.csv', 'figure_5_2_1b_released_hmm_validity.pdf', 'figure_5_2_1b_released_hmm_validity.png', 'figure_5_2_1c_data.csv', 'figure_5_2_1c_event_input_release_clock.pdf', 'figure_5_2_1c_event_input_release_clock.png', 'heldout_density_scores.csv', 'heldout_density_summary.csv', 'historical_event_replay.csv', 'historical_information_event_path.csv', 'hmm_parameter_manifest.csv', 'independent_recalculation_checks.csv', 'parameter_registry_5_2_1.csv', 'release_clock.csv', 'released_hmm_filter.csv'] | ['baseline_reproduction_audit.csv', 'counterfactual_model_selection.csv', 'counterfactual_model_specifications.csv', 'counterfactual_one_step_residual_acf.csv', 'counterfactual_residual_library.csv', 'counterfactual_rolling_origin_predictions.csv', 'counterfactual_rolling_origin_summary.csv', 'data_audit.csv', 'experiment_input_register.csv', 'figure_5_2_1a_counterfactual_predictive_validity.pdf', 'figure_5_2_1a_counterfactual_predictive_validity.png', 'figure_5_2_1a_data.csv', 'figure_5_2_1b_data.csv', 'figure_5_2_1b_released_hmm_validity.pdf', 'figure_5_2_1b_released_hmm_validity.png', 'figure_5_2_1c_data.csv', 'figure_5_2_1c_event_input_release_clock.pdf', 'figure_5_2_1c_event_input_release_clock.png', 'heldout_density_scores.csv', 'heldout_density_summary.csv', 'historical_event_replay.csv', 'historical_information_event_path.csv', 'hmm_parameter_manifest.csv', 'independent_recalculation_checks.csv', 'parameter_registry_5_2_1.csv', 'release_clock.csv', 'released_hmm_filter.csv'] |
| input_scope_excludes_policy_and_old_chapter5_results | PASS | [] | [] |
| downstream_interface_hash_and_contract | PASS | valid | hash-verified 21-week released-information path |
| run_manifest_created_with_figure_sources | PASS | ['figure_5_2_1a_counterfactual_predictive_validity.pdf', 'figure_5_2_1a_counterfactual_predictive_validity.png', 'figure_5_2_1b_released_hmm_validity.pdf', 'figure_5_2_1b_released_hmm_validity.png', 'figure_5_2_1c_event_input_release_clock.pdf', 'figure_5_2_1c_event_input_release_clock.png'] | ['figure_5_2_1a_counterfactual_predictive_validity.pdf', 'figure_5_2_1a_counterfactual_predictive_validity.png', 'figure_5_2_1b_released_hmm_validity.pdf', 'figure_5_2_1b_released_hmm_validity.png', 'figure_5_2_1c_event_input_release_clock.pdf', 'figure_5_2_1c_event_input_release_clock.png'] |

## Counterfactual evidence

The frozen selection rule selected **Harmonic ridge** using 21-week cumulative path WAPE, the predeclared relative tie tolerance, absolute bias, and unique one-step residual dependence in that order.

| Model | 21-week WAPE | Bias | One-step dependence | Selected |
|---|---:|---:|---:|---:|
| Harmonic ridge | 0.1363 | 184003.31 | 1.124 | yes |
| Seasonal naive | 0.1673 | 10849.30 | 1.047 | no |
| Damped local trend | 0.1754 | 32838.75 | 0.752 | no |

## HMM held-out evidence

Mean log predictive density is reported without suppressing horizons where a benchmark performs better.

| Horizon | HMM transition | Unconditional | Persistence |
|---:|---:|---:|---:|
| 1 | -4.6635 | -4.7919 | -7.4397 |
| 2 | -4.7905 | -4.7919 | -10.2311 |
| 3 | -4.7965 | -4.7919 | -10.6170 |

## Negative, weak, or uncertain findings

- Counterfactual: the selected model was not the lowest-WAPE model at these shorter horizons: none. This does not overturn the frozen 21-week rule.
- Counterfactual bias: the selected 21-week path has raw Bias 184003.31 metric tonnes and normalised Bias 5.72%. The positive bias can increase estimated event shortfall and must remain visible in interpretation.
- Residual dependence: selected-model one-step ACF exceeded the approximate 95% bounds at lags [1, 9]. Any remaining dependence must be retained in downstream resampling design.
- HMM comparison: benchmarks with higher held-out LPD were ['unconditional at 3 month(s)']. A weaker HMM horizon is evidence, not a code failure.
- HMM sharpness: 21 of 21 event-week current beliefs lie at or beyond 0.01/0.99, while the eight-week lead forecast ranges from 0.115 to 0.127. This sharp regime separation is an uncertainty and must not be presented as calibrated closure probability.
- Event reference: the formula-derived 2026-07-13 serviceability is 0.162634; the prior manuscript target is 0.164000. Any difference is a counterfactual-reconstruction result and was not manually overwritten.
- Interpretation: filtered state probabilities and lead-time forecasts describe geopolitical-risk regimes. They are not closure probabilities, closure labels, or closure dates.
- Provenance caveat: exact external retrieval dates were recorded for PortWatch. Other frozen public files have hash-verified project freeze dates but no independently recorded retrieval date in the source manifest.

## Evidence boundary and next gate

If every blocking check above passes, the frozen historical information/event interface is admissible for 5.2.2. This acceptance does not establish policy superiority, readiness value, a historical committed share, or any structural boundary.
