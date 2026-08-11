# 5.2.5 Computational and Methodological Acceptance Report

Overall outcome: **PASS**.

## Independent acceptance layers

- ENGINEERING_ACCEPTANCE: **PASS**
- NUMERICAL_ACCEPTANCE: **PASS**
- METHODOLOGY_CONTRACT_ACCEPTANCE: **PASS**
- EXPERIMENTAL_EVIDENCE_ACCEPTANCE: **PASS**
- OVERALL_ACCEPTANCE: **PASS**

## Interpretation

All 32 critical production-method contracts were executed; 32 passed. The paired design independently recovers 88 physical paths, with three learning seeds aggregated within path. Its maximum half-width is 2,237.542 against the registered 2,255.638 target.

The complete SAC gate evaluates 12 critical learning contracts: latent Gaussian sampling, actor mean and log-standard-deviation updates, entropy in the actor objective, adaptive temperature, twin reward critics, the constraint critic and dual, the projection-gradient chain, finite differences, validation-only selection, and checkpoint replay. These are reconstructed from persisted production traces rather than inferred from checkpoint readability.

The RC-MSA comparison is a numerical fixed-point diagnostic, not a policy comparison. MPC exactness is established only over the preregistered finite candidate lattice. Selector regret is relative only to the frozen BC and SAC candidates. Runtime results are a computational profile because no external operational response deadline is registered.

## Chapter 4 contract reinforcement

The refresh adds five noncompensatory checks: complete-master RC-MSA history distance, interface-level $a_t^{-I}$, independent per-vintage no-reset identities, complete MPC/selector module certificates, and pre/post-repair numerical equivalence. 5/5 passed.

These are audit-contract reinforcements. They do not change the frozen network, costs, paths, checkpoints, policy definitions, training settings, or the accepted 5.2.1--5.2.4 results.

## Numerical and selector findings

RC-MSA converged in 3/3 controlled production problems, with median 7 iterations and maximum terminal residual 7.828e-07. Conventional MSA converged in 0/3 within the common 500-iteration cap; this negative comparator result is retained and is not an acceptance failure for RC-MSA.

The BC-SAC selector was mechanically consistent in every recorded decision, but its path-level ex-post regret relative to the better frozen candidate averaged 510.457 and ranged from 157.354 to 872.348. This is negative candidate-set evidence and does not contradict mechanical selector acceptance.

## Missing or weaker evidence

- SLSQP primal feasibility and an independent projection objective/Jacobian check are available; solver dual and complementarity multipliers were not persisted, so that noncritical KKT subdiagnostic remains `NOT_TESTED`.
- BC action-coordinate validation errors were not persisted; aggregate training and validation imitation loss remain available.
- Historical training wall time and peak memory were not persisted, and no retrospective values are fabricated.
- The three-scenario MPC prefix analysis is diagnostic; the formal statistical evidence gate remains the 88-path paired precision calculation.

## Noncritical NOT_TESTED diagnostics

- D1_PROJECTION_DUAL_KKT: SLSQP multipliers were not persisted by the production ProjectionResult.
- D2_BC_ACTION_DIMENSION_ERRORS: 5.2.2 persisted aggregate imitation loss but not action-coordinate validation errors or projected teacher gap.
- D4_TRAINING_MEMORY_AND_TIME: Training wall time by algorithm and peak memory were not persisted; no retrospective value is fabricated.

## Upstream reproduction

The audit contains 1,408 path/checkpoint replay rows across 5.2.2--5.2.4. Maximum accepted-anchor difference: 5.821e-11.

## Evidence boundary

This experiment verifies that the named methods are connected to the production chain and reports numerical acceptance. It does not prove global optimality, universal policy superiority, causal validity for real ports, or real-time deployability.
