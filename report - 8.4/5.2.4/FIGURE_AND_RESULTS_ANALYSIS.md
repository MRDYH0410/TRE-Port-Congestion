# 5.2.4 Released Risk Information and Capacity Preparation

Acceptance status: **complete**.

## Reoptimized information value

| Scenario | Comparison | Mean effect | Simultaneous 95% interval | Holm p |
|---|---|---:|---:|---:|
| GH | IF vs I0 | -86.407 | [-97.139, -75.675] | 6.198e-39 |
| GH | IL vs I0 | 2,644.626 | [2,558.785, 2,730.467] | 7.647e-87 |
| GH | ORACLE vs I0 | 134.187 | [104.233, 164.140] | 5.651e-22 |
| GH | IL vs IF | 2,731.033 | [2,651.208, 2,810.857] | 7.743e-89 |
| GH | oracle gap | -2,510.439 | [-2,597.473, -2,423.406] | 1.109e-82 |
| GT | IF vs I0 | -86.407 | [-97.139, -75.675] | 6.198e-39 |
| GT | IL vs I0 | 2,644.636 | [2,558.795, 2,730.476] | 7.647e-87 |
| GT | ORACLE vs I0 | 134.187 | [104.233, 164.140] | 5.651e-22 |
| GT | IL vs IF | 2,731.042 | [2,651.217, 2,810.867] | 7.743e-89 |
| GT | oracle gap | -2,510.449 | [-2,597.482, -2,423.416] | 1.109e-82 |
| GL | IF vs I0 | -86.407 | [-97.139, -75.675] | 6.198e-39 |
| GL | IL vs I0 | 2,644.625 | [2,558.784, 2,730.466] | 7.647e-87 |
| GL | ORACLE vs I0 | 134.187 | [104.233, 164.140] | 5.651e-22 |
| GL | IL vs IF | 2,731.032 | [2,651.207, 2,810.856] | 7.743e-89 |
| GL | oracle gap | -2,510.439 | [-2,597.472, -2,423.405] | 1.109e-82 |
| GFW | IF vs I0 | 0.082 | [0.082, 0.082] | 0 |
| GFW | IL vs I0 | -7.072 | [-7.072, -7.072] | 0 |
| GFW | ORACLE vs I0 | -12.915 | [-12.915, -12.915] | 0 |
| GFW | IL vs IF | -7.154 | [-7.154, -7.154] | 0 |
| GFW | oracle gap | -5.843 | [-5.843, -5.843] | 0 |

## Fixed-checkpoint information responsiveness

These rows use the identical IL_RD checkpoint bundle and are not information value estimates.

| Scenario | Comparison | Mean effect | Simultaneous 95% interval | Holm p |
|---|---|---:|---:|---:|
| GH | IF vs I0 | 242.870 | [227.988, 257.752] | 7.492e-62 |
| GH | IL vs I0 | 192.778 | [179.077, 206.479] | 8.302e-57 |
| GT | IF vs I0 | 242.870 | [227.988, 257.752] | 7.492e-62 |
| GT | IL vs I0 | 192.788 | [179.087, 206.489] | 8.302e-57 |
| GL | IF vs I0 | 242.870 | [227.988, 257.752] | 7.492e-62 |
| GL | IL vs I0 | 192.777 | [179.076, 206.478] | 8.302e-57 |
| GFW | IF vs I0 | 2.036 | [2.036, 2.036] | 0 |
| GFW | IL vs I0 | 0.754 | [0.754, 0.754] | 0 |

## Reoptimized capacity rights

| Scenario | Comparison | Mean effect | Simultaneous 95% interval | Holm p |
|---|---|---:|---:|---:|
| GH | V_R_given_D | 4,596.391 | [4,501.585, 4,691.198] | 5.556e-104 |
| GH | V_D_given_R | 3,195.783 | [3,105.803, 3,285.763] | 1.811e-92 |
| GH | S_RD | 3,030.880 | [2,938.156, 3,123.605] | 1.168e-89 |
| GT | V_R_given_D | 4,596.390 | [4,501.583, 4,691.196] | 5.556e-104 |
| GT | V_D_given_R | 3,195.783 | [3,105.803, 3,285.763] | 1.811e-92 |
| GT | S_RD | 3,030.879 | [2,938.155, 3,123.604] | 1.168e-89 |
| GL | V_R_given_D | 4,596.391 | [4,501.585, 4,691.198] | 5.556e-104 |
| GL | V_D_given_R | 3,195.782 | [3,105.802, 3,285.762] | 1.811e-92 |
| GL | S_RD | 3,030.880 | [2,938.155, 3,123.605] | 1.168e-89 |
| GFW | V_R_given_D | -9.385 | [-9.385, -9.385] | 0 |
| GFW | V_D_given_R | -20.817 | [-20.817, -20.817] | 0 |
| GFW | S_RD | -6.168 | [-6.168, -6.168] | 0 |

## False-warning and precision evidence

Mean false-warning costs by regime: {'IF': -0.082, 'IL': 7.072, 'ORACLE': 12.915}.
All 40 preregistered contrasts met the paired half-width target.
Right-censored aggregate cells report 0 censored physical paths; the simulation cap is never recorded as an observed clearance week.

## Result interpretation

- Under historical release, the reoptimized current-filtered controller relative to I0 increases loss by 86.4; the adjusted interval excludes zero.
- Under historical release, the reoptimized lead-aligned controller relative to I0 lowers loss by 2,644.6; the adjusted interval excludes zero. Its incremental paired effect versus current filtering is 2,731.0, with adjusted interval [2,651.2, 2,810.9].
- Moving the target release packet between GH, GT, and GL changes the IF effect by up to 0.000 and the IL effect by up to 0.010. These are designed timing stresses; identical or zero-variance outcomes are retained rather than converted into a historical timing claim.
- Under historical release, the controller trained with ORACLE information relative to I0 lowers loss by 134.2; the adjusted interval excludes zero. The achieved oracle gap, defined as J_IL - J_ORACLE, is -2,510.4 with adjusted interval [-2,597.5, -2,423.4]. Because it is negative, the trained ORACLE controller did not attain the theoretical perfect-information performance bound; this is retained as an optimization/training limitation rather than relabelled as an upper-bound result.
- Under GH, V_R|D=4,596.4, V_D|R=3,195.8, and S_RD=3,030.9. The positive combination statistic indicates complementarity in this matched design.
- False-warning effects are retained with their signs: {'IF': -0.082, 'IL': 7.072, 'ORACLE': 12.915}. Positive values mean information increased loss without a physical disruption; negative values mean it reduced loss in that designed no-disruption comparison.
- Some GFW contrasts have zero across-path variance and hence zero-width intervals. Their deterministic matched-path differences are descriptive for this design and their nominal p-values should not be read as conventional sampling evidence.

## Interpretation boundaries

- GH is a historical release replay; GT, GL, and GFW are designed timing stresses.
- ORACLE information is unattainable and cannot be interpreted as an implementable policy. Its information set is theoretically dominant, but the achieved learned-controller result is not an empirical upper bound when the recorded oracle gap is negative.
- Fixed-checkpoint substitutions show signal dependence or out-of-distribution response, not information value.
- Capacity-right effects are conditional simulation results under the reference network and reoptimized controllers.
- Negative values, false-warning costs, right censoring, and any unmet precision targets are retained.
