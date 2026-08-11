# 5.2.2 Acceptance Report

Status: **complete**.

## Result boundary

Conditional policy comparison for the declared reference network, chi=0.5 design point, cost proxies, common authority and frozen information only.

## Main numerical result

The sample-lowest mean loss policy is **Reactive**. This is not labelled universally optimal; the simultaneous confidence-set field determines whether it is statistically resolved.

## Paired effects versus Passive

- Reactive: mean difference -186461; simultaneous 95% interval [-189562, -183359]; Holm p=8.98028e-110.
- Projected stochastic MPC: mean difference -177279; simultaneous 95% interval [-180128, -174429]; Holm p=5.33726e-111.
- Behaviour cloning: mean difference -171048; simultaneous 95% interval [-174096, -168000]; Holm p=2.90127e-107.
- PPO: mean difference -111649; simultaneous 95% interval [-113981, -109316]; Holm p=1.66516e-101.
- Vanilla SAC: mean difference -52.4152; simultaneous 95% interval [-56.4939, -48.3364]; Holm p=3.78023e-53.
- Constrained SAC: mean difference -31.016; simultaneous 95% interval [-35.3915, -26.6406]; Holm p=1.47724e-33.
- Model-guided constrained SAC: mean difference -170538; simultaneous 95% interval [-173581, -167494]; Holm p=2.90127e-107.

## Clearance and censoring

- Behaviour cloning: clearance probability 1.000; restricted mean 79.000 weeks; censored paths 0; mean final outstanding 5.74154e-07.
- Constrained SAC: clearance probability 1.000; restricted mean 78.000 weeks; censored paths 0; mean final outstanding 7.26541e-07.
- Model-guided constrained SAC: clearance probability 1.000; restricted mean 79.000 weeks; censored paths 0; mean final outstanding 5.74566e-07.
- PPO: clearance probability 1.000; restricted mean 78.996 weeks; censored paths 0; mean final outstanding 5.59729e-07.
- Passive: clearance probability 1.000; restricted mean 78.000 weeks; censored paths 0; mean final outstanding 7.26558e-07.
- Projected stochastic MPC: clearance probability 1.000; restricted mean 79.000 weeks; censored paths 0; mean final outstanding 5.63375e-07.
- Reactive: clearance probability 1.000; restricted mean 79.000 weeks; censored paths 0; mean final outstanding 5.61256e-07.
- Vanilla SAC: clearance probability 1.000; restricted mean 78.000 weeks; censored paths 0; mean final outstanding 7.26533e-07.

## Negative, weak, uncertain, or incomplete evidence

- Best learning policy Behaviour cloning exceeded transparent benchmark Reactive by 15412.468480 mean loss units.
- Model-guided constrained SAC exceeded Behaviour cloning by 510.457073 mean loss units.
- Vanilla SAC reduced mean paired loss versus Passive by only 52.415172 units despite satisfying the SAC implementation contract.
- Constrained SAC reduced mean paired loss versus Passive by only 31.016034 units despite satisfying the SAC implementation contract.

## Blocking failures

- None.

## Interpretation limits

These outputs do not estimate the historical committed share, causal port effects, universal Hormuz performance, or global optimality. Designed route-resource proxies require the registered 5.3.4 sensitivity analysis.
