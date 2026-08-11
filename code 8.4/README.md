# TRE Port Congestion 8.4 code framework

This directory is a clean-room implementation of the formal model in Chapter 3
and the nested algorithms in Chapter 4 of `TRE_Port_Congestion_8.4.tex`. No
earlier project code was read or copied. Chapter 5 experiments are added only
as isolated, declared modules. The shared Section 5.1 data/provenance layer and
the independent Section 5.2.1 validity experiment are described below; neither
trains nor compares a policy.

## What is implemented

The package follows the paper's five-module execution order.

1. `information.py` implements diagonal-Gaussian HMM estimation, released
   filtering, the official release calendar, monthly lead-time forecasting,
   likelihood/information-criterion diagnostics, and rolling-origin scoring.
2. `scenarios.py` keeps one event-aligned path support and assigns separate
   readiness and operational weights. The HMM ablation can replace readiness
   weights only, leaving the support, operational information, paths, and seeds
   unchanged. Operational inputs require a release-timestamp ledger, while MPC
   and training continuations receive only the strictly revealed event prefix.
3. `actions.py`, `control.py`, and `learning.py` implement the formal five-block
   action, hard phase/stock/component/budget projection, projected stochastic
   MPC, behavior-cloning contracts, the constrained-SAC actor loss, replay
   transitions, per-period released-information preparation, and the frozen
   BC/SAC two-proposal selector.
4. `behavior.py`, `queue_projection.py`, and `factory.py` implement oldest-first
   release, source/vintage decision masses, verifiable disclosure, candidate-
   dependent queue projection, route-wait-exit Logit loading, multi-start
   residual-controlled MSA, strict zero-load free-flow initialization, final-
   trial certification, residual/KL certificates, and deterministic selection
   with exact start provenance.
5. `capacity.py`, `transition.py`, and `loss.py` implement the two capacity
   pipelines, committed/adaptive maritime provenance, permanent route tags,
   berth-yard-gate-corridor conservation, shared-corridor service, strictly
   lagged feedback, duration attrition, budget/resource updates, operational
   loss, physical route-holding, and full transition audits for both capacity
   pipelines, budget, and histories.

`engine.py` is the shared Module 4/5 kernel. MPC, teacher generation, SAC,
execution, replay, and clearance must call this same kernel. `orchestration.py`
joins Modules 1 through 5. `clearance.py` continues the full behavioral and
physical model after the decision horizon and marks capped trajectories as
right-censored. `diagnostics.py` contains the adaptive-shaping check,
time-expanded optimistic absorption LP, fixed-branch local Jacobian, and ex-post
matched cascade statistics.

`acceptance.evaluate_trajectory_acceptance` is the final model-level gate. It
cannot pass on a single transition: it requires the complete decision trace,
explicit frozen-recovery clearance, and exact one-time reconciliation of the
terminal correction.

The exact paper-to-code mapping is in
[`METHODOLOGY_TRACEABILITY.md`](METHODOLOGY_TRACEABILITY.md).

## Shared Section 5.1 layer

`experiments/data/` now supplies the common data layer required before adding
individual Section 5.2 or 5.3 experiments. It contains only `raw/`,
`processed/`, and `manifests/` data products plus the clean-room construction
code. Run:

```powershell
python -m experiments.data.prepare
```

The command verifies frozen hashes and 48 data contracts, then regenerates the
Monday-week PortWatch panels, the official monthly GPR panel and continuous
features, gateway activity scales, network exposure, nameplate allocation
shares, and the event reference marker. Raw zero activity is retained. Jebel
Ali remains a comparator and never enters the three-gateway reference set.
Event severity remains an estimated value that the first event-validity
experiment must regenerate; renewed-closure duration remains right-censored.

`tre84.metrics.compute_trajectory_statistics` is the shared output calculator
for every later experiment. It reconstructs formal operational loss, stage
pressures, Peak, Overload, Waiting, direct SUE exit, duration attrition, total
exit, landbridge Delivered, terminal outstanding mass, clearance status, and
transition-audit acceptance. By default physical metrics span the decision and
clearance transitions. A censored run reports the realised follow-up length but
leaves observed clearance time empty.

## Environment and verification

Python 3.11 or newer is required. The deterministic core depends only on NumPy
and SciPy. A neural implementation may use the optional `learning` dependency;
the present framework deliberately leaves network size and training
hyperparameters to the later declared experiment rather than inventing them in
the model layer.

```powershell
python -m pip install -e ".[test]"
python -m pytest tests/test_acceptance.py tests/test_actions_and_behavior.py tests/test_capacity_pipelines.py tests/test_diagnostics.py tests/test_inference.py tests/test_information_and_scenarios.py tests/test_metrics.py tests/test_transition_and_clearance.py
```

The core test suite checks released information timing, the continuation
information firewall, common-support dual weighting, the hard action projector,
oldest-first release, the joint route-wait-exit simplex, RC-MSA final-trial and
start-provenance certificates, distinct readiness/direct pipelines,
unavailable-route holding, shared-corridor tagged conservation, the four-stage
service clock, terminal-mass censoring and exact objective reconciliation, the
structural LP, the local Jacobian, and completed matched-path cascade timing.

The separate experiment tests and runners are outside this core command. No
Section 5.2 or 5.3 policy result is regenerated by the core verification.

## Independent Section 5.2.1 experiment

`experiments/5.2-1/` validates the frozen data, pre-event no-disruption
counterfactual, two-state released-information HMM, release clock, and
formula-derived historical event path. It reads only `experiments/data/` and
does not train or compare policies. Run the complete experiment with:

```powershell
python experiments/5.2-1/run_5_2_1.py
```

The command regenerates the shared frozen data, writes all registered tables,
JSON acceptance evidence, three 300-dpi figures, and the single downstream
interface to `output/5.2.1_data_event_information_validity/`. It exits nonzero
if any blocking timing, leakage, unit, identity, provenance, or manifest check
fails. Later historical-path experiments must call
`experiments/5.2-1/interface.py`; direct reconstruction of a risk ramp or a
second event path is outside the experiment contract.

## Independent Section 5.3.1 experiment

`experiments/5.3-1/` contains the complete commitment-boundary experiment
orchestrator, fail-closed backend contract, endpoint and conservation audits,
pilot-only one-round refinement, shared precision-rule call, path-level seed
aggregation, paired and multiplicity-controlled inference, parameter registry,
and reproducible figure generation.

The example configuration is intentionally non-runnable. Paper-result execution
remains blocked until the final Section 5.2 benchmark backend, rebuilt policy
artifacts, HMM information paths, declared horizons and tolerances, and pilot
halfwidth design exist. Unit tests exercise the full output chain with a clearly
labelled synthetic contract backend in a temporary directory; those artifacts
are not empirical or paper results.

## Configuration boundary

There are no hidden baseline values. All of the following must be supplied by a
later declared calibration or experiment:

- network routes, master tags, shared corridors, and maritime lag kernels;
- demand/serviceability paths and committed itinerary shares;
- waiting hazards, costs, continuation values, and Logit sensitivities;
- disclosure reference loading, forecast-error scales, and credibility bounds;
- physical capacities, thresholds, feedback laws, capacity technology, and
  action cost curves;
- HMM initialization/training data and scenario log-weight functions;
- MPC lattice, terminal value, recovery schedule, neural architecture, and
  training hyperparameters.

This separation prevents experimental assumptions from silently changing the
Chapter 3 model. A numerical run is admissible only when information,
feasibility, equilibrium, physical, and objective acceptance checks all pass.
