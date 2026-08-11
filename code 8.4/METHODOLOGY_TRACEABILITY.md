# Chapter 3 and Chapter 4 traceability contract

This document is the acceptance map for the 8.4 code rewrite. Equation labels
refer to the current `TRE_Port_Congestion_8.4.tex`. The map covers model and
algorithm support only; it does not claim that a Chapter 5 calibration or result
has been reproduced.

## Chapter 3 model to code

| Paper contract | Code implementation | Verification |
|---|---|---|
| `eq:blocked-flow`, `eq:commitment-split` | `transition.construct_demand_split`; `factory.StandardBehaviorProblemFactory` | Transition test recomputes the new blocked, committed, and decision-eligible masses. |
| `eq:committed-route-share` | `ExogenousRealization.committed_route_share`; committed `Tag` and `Provenance.COMMITTED`; permanent master routes in `Network` | Shares must sum to one for each positive committed cohort. Closing a choice route never renormalizes an existing tag. |
| `eq:waiting-vintages`, `eq:waiting-release`, `eq:oldest-first` | `ModelState.waiting`; `behavior.oldest_first`; `build_decision_masses` | Unit tests verify oldest-first allocation and the transition verifies release equals the action's `rho`. |
| `eq:decision-masses`, `eq:source-conservation` | `SourceKey`; `DecisionMasses`; every RC-MSA source slice contains route, wait, and exit | Each positive and zero-mass source retains its declared simplex mass; zero-mass sources skip cost/softmax evaluation. |
| `eq:waiting-hazard`, `eq:waiting-transition`, `eq:endogenous-exit`, `eq:adaptive-conservation` | `BehaviorCostParameters.hazard`; `TaggedTransition` waiting update; `EquilibriumResult.direct_exit`; duration attrition audit | Terminal hazard is required to equal one. Direct SUE exit and later attrition are reported separately and the adaptive identity is audited by class. |
| `eq:formal-action`, `eq:phase-actions`, `eq:budget`, `eq:safe-action-set` | `ActionKey`, five `Block` values, `ActionDomain`, `ActionProjector` | Projection enforces phase, stock, component, period-budget, and remaining-budget bounds. Exit is not an action. |
| `eq:route-dispatch`, `eq:route-arrival` | RC-MSA route aggregation; normalized route lag kernels; `PipelineLot` with remaining lag and provenance | Pipeline mass is audited. Zero-lag arrivals may enter current berth service; unavailable physical tags are held. |
| `eq:tagged-conservation`, `eq:landbridge-balance`, `eq:aggregate-tagged-states` | Four tag dictionaries in `ModelState`; event-ordered balances in `TaggedTransition` | Tag-level balance residuals, nonnegativity, and state completeness must pass. Gate release is not counted as delivery. |
| `eq:work-conserving`, `eq:proportional-service` | Stage/location aggregation and `_proportional_service` | Berth can serve current external arrivals. Internal releases cannot receive a second service in the same period. |
| `eq:shared-corridor` | `Network.shared_corridors`; one corridor queue and service capacity shared by all route tags | The transition test uses two gateways competing for the same corridor. No extra border queue is created. |
| `prop:controllability`, `eq:shapeable-bound` | `diagnostics.check_adaptive_shaping_bound` | The half-L1 comparison is conditional on the same source masses and fixed release. |
| `eq:absorption-capacity`, `eq:committed-arrivals-window`, `prop:absorption`, `eq:unavoidable-condition` | `diagnostics.AbsorptionBoundaryLP` | The time-expanded LP is optimistic, non-work-conserving, capacity-envelope based, and kept separate from policy execution. Gateway-specific outputs are not additive. |
| `eq:anticipated-wait` | `queue_projection.ContinuousQueueProjection` | The provider receives candidate route dispatch so marginal candidate workload cannot be cached away. Scenario sampling remains outside the SUE solve. |
| `eq:disclosed-wait`, `eq:disclosure-credibility` | `FrozenDisclosure`; `DisclosureForecast`; predetermined reference loading | Signal, reference, error scale, and intensity are frozen before RC-MSA. The coordinator changes the weight, not the sign of the signal-prior difference. |
| `eq:generalized-cost` | `BehaviorProblem.costs` route cost | Private resource use, exogenous market cost, and perceived waiting enter behavior; perceived terms are not copied into operational loss. |
| `eq:projected-waiting`, `eq:waiting-resource-cost`, `eq:waiting-cost`, `eq:new-source-age`, `eq:old-source-age`, `eq:attrition-consequence` | `BehaviorProblem.projected_waiting_out` and source-age cost construction | Released vintages retain their age. Waiting internalizes the same-period attrition consequence and survival continuation value. |
| `eq:exit-cost`, `eq:choice-set`, `eq:logit-loading`, `eq:sue-fixed-point` | Source-specific exit cost, route/`WAIT`/`EXIT` choices, stable Logit, `RCMSASolver` | No fixed exit share exists. Residual is normalized by total decision mass and KL is a secondary diagnostic. |
| `eq:equilibrium-selector`, `prop:sue-existence` | strict zero-load free-flow start; final-trial certification; normalized-share multistart deduplication; preceding-share distance; exact start provenance; lexicographic tie rule | Every selected trial, including the final allowed update, can become the retained certified iterate. Nonconvergence is returned explicitly and cannot advance the physical state. The code does not claim general uniqueness. |
| `eq:effective-capacities`, `eq:feedback-chain` | `CapacityDynamics`; `ServiceParameters` feedback functions | Yard occupancy can reduce berth capacity and lagged corridor pressure can reduce gate capacity. No gate-to-yard link is added. |
| `eq:corridor-release`, `eq:lagged-corridor-weight` | gate-service history by gateway/corridor; lag-window weights with declared fallback share | Current gate releases update the next state only; current capacity uses strictly previous history. |
| `eq:congestion-indicators`, `eq:cascade-times`, `def:cascade` | `ThresholdSnapshot`; `ex_post_cascade_statistics` | Times are computed only after realized and matched no-disruption paths are complete. Missing crossings remain censored. |
| `eq:local-jacobian` | `local_selected_branch_jacobian` | Requires supplied fixed-branch derivatives and reports the inverse condition and spectral radius; it is not an event threshold. |
| `eq:risk-belief`, `eq:lead-time-risk` | `GaussianHMM`; `ReleasedRiskInference`; official `ReleaseRecord` calendar | Belief changes only when data are officially released and the transition matrix is powered by monthly transitions, never weeks. |
| `eq:risk-scenario-kernel`, `eq:scenario-weights` | `EventPath`, `ScenarioBundle`, `TimestampedOperationalContext`, `RevealedEventHistory`, `CommonScenarioConstructor` | One physical support has readiness and operational weights. Operational evidence is timestamped and future payload is exposed only as a strictly past prefix to continuation decisions. The ablation replaces readiness weights only. |
| `eq:phase`, `eq:control-state` | `ModelState.phase`; full state dataclasses | Phase depends only on whether disruption has ever been announced. Queue thresholds and ex-post cascade times never define the action phase. |
| `eq:readiness-pipeline`, `eq:readiness-transition` | readiness order buckets, mature stock, consuming exercise, decay, capacity yield, and an independent order/stock identity audit | An option order is not current capacity; maturity is not exercise; exercised stock cannot be reused. |
| `eq:direct-capacity-pipeline`, `eq:spot-capacity`, `eq:direct-capacity-stock` | independent direct order buckets, phase-dependent lead, current spot stock, decay, and an independent order/stock identity audit | A policy without readiness still has its declared direct-procurement right. |
| `eq:action-cost` | per-coordinate convex piecewise-linear cost curves | Release can have zero cash cost while remaining physically bounded. |
| `eq:unified-model`, `eq:unified-objective` | `orchestration.FiveModuleExecutor`; `engine.ModelKernel`; projected MPC and learning runners | The same behavior solver and tagged transition are called by teacher, training, selection, execution, and clearance. |
| `eq:stage-loss`, `eq:loss-components` | `OperationalLoss`, `LossBreakdown` | Queues, waiting vintages, both exit channels, real route increments, overflow, and actions are separate components counted once. |
| `eq:terminal-mass` | `ClearanceRunner`; `TerminalMassCorrection` | Clearance keeps using Modules 4 and 5. At the cap, all remaining waiting, pipeline, and tagged mass is charged and marked right-censored. |

## Chapter 4 algorithm modules

| Chapter 4 module | Input to output implementation |
|---|---|
| Module 1 released risk inference | `ReleaseRecord` plus frozen HMM to filtered belief, lead-time forecast, release timestamps, and HMM diagnostics. |
| Module 2 event-aligned scenarios | Frozen common paths plus timestamped released/operational context to dual weights, active weights, future serviceability, reclosure probability, and seed manifest. MPC/SAC continuation callbacks receive only the already revealed path prefix. |
| Module 3 feasible control | Complete state plus scenario bundle to projected MPC/BC/SAC proposals and one hard-feasible five-block action. |
| Module 4 vintage endogenous response | State, feasible action, realization, and preceding shares to oldest-first release, selected route-wait-exit SUE, residual, KL, dispersion, iterations, and status. |
| Module 5 augmented transition | Complete state, feasible action, selected equilibrium, and realization to complete next state, delivered mass, exit channels, loss breakdown, and conservation audit. |
| Integrated training/execution | `learning.py`, `control.py`, `engine.py`, and `orchestration.py` retain one nested implementation boundary and explicit checkpoint/backend contracts. |

## Acceptance layers

`acceptance.evaluate_acceptance` is the period-level certificate and keeps five
claims separate. `acceptance.evaluate_trajectory_acceptance` is the final
trajectory certificate; it additionally requires an ordered decision trace,
the explicit clearance tail, and a reported objective that reconstructs as
decision loss plus clearance loss plus exactly one terminal correction.

1. Information timing: every feature and weight timestamp is no later than the decision.
2. Action feasibility: all five blocks pass phase, stock, component, and budget constraints.
3. Behavioral closure: oldest-first release, source conservation, convergence status, selected residual, and exact multi-start provenance pass.
4. Physical closure: route-availability holding, pipeline, tag, waiting, both capacity pipelines, budget, history, nonnegativity, and state checks pass.
5. Objective closure: all declared operational loss components are finite, nonnegative, and exposed separately.

No one layer substitutes for another. In particular, convergence of RC-MSA does
not validate the tagged transition, and passing the test suite does not validate
an empirical calibration or a Chapter 5 result.

## Shared Section 5.1 experiment contracts

| Paper contract | Code implementation | Verification |
|---|---|---|
| Frozen PortWatch retrieval and empirical dimensions | `experiments/data/raw/portwatch`; `manifests/datasets.json`; `quality.audit_frame` | SHA-256, row/field counts, grain, missing cells, duplicate keys, and date ranges are blocking checks. |
| Monday-based aggregation and zero retention | `construction.build_portwatch_weekly` | The build verifies 395/393 chokepoint weeks, 1,580 port-weeks, the 393-week common window, 21 event weeks, and all four zero counts. |
| Official monthly GPR data | `construction.build_gpr_monthly`; `build_gpr_continuous_features` | The build verifies 498 consecutive complete months and 474 continuous-feature months. The jump threshold is intentionally deferred until an experiment declares it. |
| `eq:gateway-activity-scale` | `construction.build_gateway_reference_scales` | Uses complete pre-event weeks, import plus export container proxy, 0.90 quantile, and `U_Q=1,000`; no positive floor. |
| `eq:network-exposure-reference` | `construction.network_exposure_reference` | Reconstructs 0.0263 as an activity-scale alignment, not observed diversion. |
| `eq:committed-allocation-reference` | `construction.committed_itinerary_shares` | Reconstructs 0.5952/0.1071/0.2976 from official annual nameplates; no weekly service claim. |
| Public event coordinates | raw event advisory dates plus manifest `derived_reference_targets` | Duration is reconstructed as 23 days/3.29 weeks; 0.164/0.836 is labelled a manuscript estimate for 5.2.1 regeneration; final duration remains right-censored. |
| `eq:experimental-model-loss` | `metrics.compute_trajectory_statistics` | Decision loss, clearance loss, and terminal correction are reconciled and counted once. |
| `eq:experimental-pressure`, `eq:experimental-outcomes` | `metrics.stage_pressures`; `compute_trajectory_statistics` | Aggregates permanent tags to the four stages and reports Peak, Overload, Waiting, the two exit channels, and landbridge discharge. |
| `eq:experimental-outstanding-mass` and censored clearance | `ModelState.cargo_mass`; `TrajectoryStatistics` | Ending waiting, maritime pipeline, and all tagged queues are counted. A capped run never receives the cap as an observed clearance time. |

These are shared input and calculation contracts only. They do not constitute a
Section 5.2/5.3 experiment, trained policy, counterfactual validation, or
empirical result.

## Section 5.2.1 validity experiment contract

| Experiment requirement | Implementation and blocking evidence |
|---|---|
| Frozen inputs and used-field audit | `experiments/5.2-1/reporting.build_data_audit` verifies shared-manifest hashes, dimensions, dates, duplicates, complete weeks, and missingness only in fields actually used by each model. Unused GPR wide-table nulls are reported separately. |
| Event-free counterfactual validation | `counterfactual.run_rolling_origins` uses every eligible Monday origin through 2026-02-16 and compares harmonic ridge, estimated damped local trend, and calendar seasonal naive at 1/2/4/8/12/21 weeks. Hyperparameters are re-estimated within each training prefix. |
| Frozen selection and residual library | `counterfactual.select_counterfactual_model` applies 21-week cumulative WAPE, absolute-bias, and unique one-step-residual dependence in that order. `build_residual_library` excludes every event-period residual. |
| Fixed chronological HMM evaluation | `hmm_validity.fit_frozen_hmm` fits the declared two-state model through 2024-12 using deterministic initialisations and training-only scaling. `score_heldout_density` retains 2025-01 through 2026-06 and reports HMM, unconditional, and persistence LPD without result-dependent retuning. |
| Released information and monthly lead alignment | `hmm_validity.build_release_clock` selects the latest observation whose declared release date is no later than the Monday decision and powers the monthly transition matrix by calendar-month distance to readiness maturity. |
| Historical event formulas and units | `event_input.construct_historical_event` implements serviceability, nonnegative estimated shortfall, and `0.0263 * blocked_proxy / 1000`; no `chi` or committed-share split enters this experiment. |
| Unique downstream interface | `interface.load_historical_path` verifies the manifest hash, 21-week calendar, release timing, source, and formula lineage of `historical_information_event_path.csv`; hash tampering and artificial ramps fail closed. |
| Reproducible evidence and figures | `run_5_2_1.py` writes the registered CSV/JSON evidence, acceptance report, and three 300-dpi figures. `run_manifest.json` binds every published result to its SHA-256 source artifact. |

The detailed equation-to-function map is retained in
`experiments/5.2-1/FORMULA_TO_CODE.md`. Passing this experiment establishes the
validity of the frozen historical information input, not policy superiority or
the economic value of readiness.

## Section 5.3.1 Commitment Boundary code contract

| Experiment requirement | Implementation and blocking evidence |
|---|---|
| Only `chi` changes new blocked cohorts | `commitment_boundary.audits.extract_replication` recomputes committed and decision-eligible mass from every decision transition. Frozen design, checkpoint, path, model, and clearance hashes must remain constant within each policy/seed/path trace. |
| Base grid and one pilot refinement round | `BASE_CHI_GRID` is exactly 0, 0.25, 0.50, 0.75, 1. Pilot sample-best or simultaneous-confidence-set changes trigger one adjacent midpoint; recursion is prohibited and pilot paths are excluded from final inference. |
| Endpoint contracts | Every cargo class and period records split error, committed dispatch, new endogenous route/wait/exit flow, committed-tag presence, and transition audit. Chi zero and one are blocking endpoints. |
| Full mass identity | Every replication checks blocked mass against Delivered, direct SUE exit, duration attrition, and final waiting/pipeline/tagged queues. Seed-averaged flow decompositions must also close before plotting. |
| Formal experimental loss | Decision `C_Q`, `C_W`, `C_E`, `C_over`, `C_route_res`, and `C_A` plus complete `ell_clear` must reconstruct `J_op`; clearance operational loss and terminal correction remain separately visible. |
| Clearance and censoring | A censored trace must execute all `T_clear` periods. Observed clearance time remains missing, while a separately named restricted-time contribution supports RMST. |
| Frozen policy comparison | Every policy artifact is hash checked. Learned seeds are averaged inside policy/path/chi before physical paths become the independent inference units. Training chi and support are recorded, and extrapolation cells are flagged. |
| Precision and multiplicity | Shared `tre84.inference.PrecisionRule` converts pilot paired variance and target halfwidth to required paths. Final Passive comparisons use one Holm family and Bonferroni simultaneous Student intervals; policy confidence sets use the global all-pair family. |
| Resolved policy switch | A policy is resolved only when its simultaneous paired-loss interval is below every competitor. Sample mean alone cannot resolve a switch. |
| Result publication | `CommitmentBoundaryExperiment.run` writes to a temporary staging directory and atomically publishes the required CSV, PNG, registry, and manifest only after all checks pass. Existing result folders are never overwritten. |

The current repository contains the experiment code and contract tests, not a
5.3.1 paper-result run. A final run remains correctly blocked until its Section
5.2 policy backend and newly rebuilt checkpoints are available.
