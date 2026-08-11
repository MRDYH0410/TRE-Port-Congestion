# Experiment 5.2.1: Data, Event, and Released Information Validity

This experiment validates the frozen data, the event-free no-disruption
counterfactual, the two-state geopolitical-risk HMM, official-information
timing, and the formula-derived 21-week historical event input. It does not
train, select, or compare any policy and contains no committed-share, gateway-
expansion, or reclosure-boundary analysis.

Run once from the `code 8.4` root:

```powershell
python experiments/5.2-1/run_5_2_1.py
```

The command builds in a staging directory and atomically replaces the formal
output only after every blocking check, the independent recalculation, the
fail-closed interface test, and manifest hash verification pass. It also
publishes 300 dpi PNG and vector PDF figures plus their figure-data CSV files.

The command reads only `experiments/data`, writes all tables, manifests,
figures, the frozen downstream interface, and the acceptance report to
`output/5.2.1_data_event_information_validity`, and exits nonzero if a blocking
acceptance check fails.

All empirical choices are frozen in `config_5_2_1.json`. In particular, the
counterfactual selection rule is fixed before seeing the event window, HMM
parameters are frozen at the 2024 calendar-year boundary, publication uses a
conservative full-month lag, and the eight-week readiness maturity is only a
designed timing anchor. It is not an economic calibration.

Later historical experiments must call `interface.load_historical_path` and
read `historical_information_event_path.csv`. They must not reconstruct a
second risk path or insert an artificial pre-event probability ramp.
