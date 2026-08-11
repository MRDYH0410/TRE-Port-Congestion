# 5.2.1 Results and Figure Analysis

## Acceptance status: COMPLETE

This rerun validates the frozen data, event-free counterfactual, released geopolitical-risk state information, and formula-derived historical event input. It does not train, rank, or evaluate any policy.

## Reproduction result

- Scientific-output comparison with the pre-rerun baseline: 11/19 pre-existing scientific CSV/PNG artifacts are byte-identical.
- This is not a claim of full byte-for-byte reproduction when the count above is below the comparison total. Any such difference is retained in baseline_reproduction_audit.csv; the frozen residual library, HMM/release outputs, and unique downstream interface must still reproduce exactly.
- Independent arithmetic and contract checks: 10/10 passed.
- The unique downstream interface is fail-closed and hash-frozen in run_manifest.json.

## Figure 5.2.1a: Counterfactual predictive validity

The frozen rule selects `harmonic_ridge`. Its 21-week cumulative-path WAPE is 0.1363, MAE is 438,033.0 proxy tonnes, and Bias is 184,003.3 proxy tonnes. All three candidates use 216 feasible weekly one-step origins; shorter horizons remain diagnostics and do not alter selection.

## Figure 5.2.1b: Released HMM validity

The two-state HMM is a geopolitical-risk regime model, not a Hormuz closure classifier. Held-out mean log predictive densities are reported below without hiding weaker horizons:

| Horizon (months) | HMM transition | Unconditional | Persistence |
|---:|---:|---:|---:|
| 1 | -4.6635 | -4.7919 | -7.4397 |
| 2 | -4.7905 | -4.7919 | -10.2311 |
| 3 | -4.7965 | -4.7919 | -10.6170 |

The monthly transition matrix is powered by calendar-month distance from the released source month to readiness maturity; it is never applied once per week.

## Figure 5.2.1c: Event input and release clock

The interface contains 21 complete Monday weeks. On 2026-07-13, formula-derived serviceability is 0.162634, estimated blocked activity is 2,992,145.9 AIS-proxy tonnes, and converted blocked mass is 78.696 model units.

Blocked activity is an estimated positive shortfall, not observed diversion. The HMM belief and lead-aligned forecast are geopolitical-risk state probabilities, not closure probabilities or closure-date forecasts.

## Evidence boundary

The output is admissible only as the unique frozen input to later experiments. It provides no evidence about policy superiority, readiness economic value, historical committed share, gateway expansion, or reclosure boundaries.
