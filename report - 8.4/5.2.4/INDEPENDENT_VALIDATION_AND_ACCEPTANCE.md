# Independent validation and acceptance for Experiment 5.2.4

## Overall assessment

Execution acceptance is **complete**. The implementable information and capacity-right comparisons are **ready to share with mandatory caveats**. The trained ORACLE controller did not attain the theoretical perfect-information performance bound, so its realized loss must not be presented as an empirical upper bound.

## Data and inference grain

- The primary reoptimized-information layer contains 1,408 path rows: 4 information regimes, 4 warning scenarios, and 88 physical paths.
- The fixed-checkpoint responsiveness layer contains 1,056 path rows: 3 evaluation inputs, 4 warning scenarios, and 88 physical paths.
- The reoptimized capacity-right layer contains 1,408 path rows: 4 right sets, 4 warning scenarios, and 88 physical paths.
- The 11,616 seed-level replications consistently contain three learning seeds per controller-path cell. Seeds are averaged within physical paths before inference.
- The action, capacity, period-loss, and release-information traces each contain 336,864 weekly rows, equal to 11,616 replications multiplied by 29 decision weeks.

## Independent calculation checks

- All 40 paired effects were independently reconstructed from the three path-level tables. The maximum absolute discrepancy was (3.95\times10^{-12}) for means, (5.23\times10^{-13}) for standard errors, and (4.04\times10^{-12}) for interval endpoints.
- All 40 preregistered half-width targets passed. The maximum achieved half-width was 64.0297 against the frozen target of 2,255.6378.
- Maximum loss-reconciliation error was (2.91\times10^{-11}).
- Maximum 5.2.2 historical-anchor reproduction error was (1.46\times10^{-11}), below the (10^{-6}) tolerance.
- Maximum transition residual was (9.99991\times10^{-7}), within the registered (10^{-6}) tolerance.
- No aggregate cell was right censored; the clearance cap was never recorded as an observed clearance week.

## Contract and provenance checks

- All seven registered upstream SHA256 locks matched.
- All 17 manifest inputs and 114 original run artifacts matched their recorded sizes and SHA256 values before this independent report was added.
- All 11,616 trajectory-contract rows passed action projection, capacity-right enforcement, release timing, monthly HMM propagation, nonnegative stock, common-kernel acceptance, loss closure, censoring, and controller-information trace checks.
- The accepted 5.2.3 medoid `test_017_ebd333a3cdba` is used only for the timing figure; all formal inference uses 88 physical paths.
- Targeted Chapter 3/4 and Experiments 5.2.1-5.2.4 regression tests returned 59 passed and 1 authorized 5.2.2 output-check skip. Experiment 5.2.4 itself returned 7 of 7 passed.

## Mandatory evidence caveats

1. Under GH, the reoptimized IF controller increases loss by 86.4 relative to I0; the adjusted interval excludes zero.
2. Under GH, IL reduces loss by 2,644.6 relative to I0 and by 2,731.0 relative to IF.
3. GH, GT, and GL differ by at most 0.010 in the reported information effects. The designed release-date changes therefore do not provide meaningful timing-value evidence in this run.
4. The achieved oracle gap (J_{IL}-J_{ORACLE}) is -2,510.4 under GH, with adjusted interval [-2,597.5, -2,423.4]. Perfect information is theoretically dominant, but the trained ORACLE controller underperforms IL; this is an optimization/training limitation, not an implementable recommendation or an attained upper bound.
5. Ten GFW contrasts have zero or numerically negligible across-path variance. Their zero-width intervals and nominal zero p-values are deterministic matched-design descriptions, not conventional sampling evidence.
6. Under GH, (V_{R\mid D}=4,596.4), (V_{D\mid R}=3,195.8), and (S_{RD}=3,030.9), indicating complementarity in this reference design. Under GFW all three signs reverse, preserving the false-warning cost boundary.

## Figure validation

- All three PNG files are 300 dpi and have matching vector PDF versions.
- Figure 5.2.4a distinguishes reoptimized information value from fixed-checkpoint responsiveness and labels ORACLE as an unattainable information input rather than an attained upper bound.
- Figure 5.2.4b uses the frozen physical-path medoid, keeps current and lead-aligned geopolitical risk-state beliefs distinct, and uses no dual axes.
- Figure 5.2.4c retains the common absolute loss scale and adds an explicitly separate-scale GFW inset so false-warning preparation costs remain visible without distorting the disrupted cases.

## Scope conclusion

The experiment supports conditional comparisons of implementable released-risk information and reoptimized capacity rights under the frozen reference network. It does not establish historical timing value for GT or GL, causal capacity effects outside the designed comparisons, a deployable ORACLE policy, or an attained perfect-information upper bound.
