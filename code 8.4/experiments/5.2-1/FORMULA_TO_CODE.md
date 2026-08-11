# 5.2.1 formula-to-code map

| Experimental contract | Implementation | Output evidence |
|---|---|---|
| Monday-based complete weeks and frozen source audit | `experiments.data.prepare.build_shared_data`; `reporting.build_data_audit` | `data_audit.csv`, `experiment_input_register.csv` |
| Harmonic ridge with within-prefix order/ridge selection | `counterfactual.fit_harmonic_ridge` | rolling predictions and model specifications |
| Damped local trend with estimated damping | `counterfactual.fit_damped_local_trend` | per-origin `fitted_parameters` in rolling predictions |
| Calendar seasonal naive | `counterfactual.fit_seasonal_naive` | model specification and rolling predictions |
| Cumulative path WAPE, MAE, and Bias | `counterfactual.summarise_rolling_predictions` | `counterfactual_rolling_origin_summary.csv` |
| Unique ordered one-step residual ACF | `counterfactual.one_step_residual_acf` | `counterfactual_one_step_residual_acf.csv` |
| Frozen 21-week selection and tie breaks | `counterfactual.select_counterfactual_model` | `counterfactual_model_selection.csv` |
| Event-free residual library | selected one-step slice in `run_5_2_1.main` | `counterfactual_residual_library.csv` |
| Two-state diagonal-Gaussian HMM | `hmm_validity.fit_frozen_hmm` using `tre84.information.GaussianHMM` | `hmm_parameter_manifest.csv` |
| Held-out LPD against unconditional and persistence | `hmm_validity.heldout_density_scores` | held-out score and summary tables |
| Released current belief \(\alpha_{\nu(t)}\) | `hmm_validity.filter_feature_history`; `build_release_clock` | released filter and release clock |
| \(\alpha_{\nu(t)}P_Z^{h(t,\Lambda^R)}\) | `calendar_month_transitions`; `build_release_clock` | monthly-transition and lead-time columns |
| \(\widehat a_t^H=\min(1,A_t^{obs}/\widehat A_t^0)\) | `event_input.construct_historical_event` | `historical_event_replay.csv` |
| \(\widehat q_t^B=\max(\widehat A_t^0-A_t^{obs},0)\) | `event_input.construct_historical_event` | blocked value and identity residual columns |
| \(q_t^{B,model}=\varrho^{Q,ref}\widehat q_t^B/U_Q\) | `event_input.construct_historical_event` | model-unit and conversion-residual columns |
| Unique downstream path with no artificial ramp | `interface.load_historical_path` | hash-frozen interface and `run_manifest.json` |
| Blocking acceptance | `_build_checks` in `run_5_2_1.py` | acceptance JSON and report |
| Independent WAPE, MAE, Bias, calendar, release, and unit recomputation | `validation_5_2_1.independent_recalculation` | `independent_recalculation_checks.csv` |
| Staging and atomic publication | `run_5_2_1.main` | formal output directory is replaced only after all blocking checks pass |
| Figure source layer | `validation_5_2_1.figure_data_tables` | three `figure_5_2_1*_data.csv` files and matched PNG/PDF files |

The runner never creates a closure label, policy action, policy loss, committed
share, gateway expansion, or reclosure-boundary input.
