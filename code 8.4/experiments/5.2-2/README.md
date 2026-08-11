# Experiment 5.2.2: Common Authority Policy Benchmark

This directory is an isolated Chapter 5 experiment. It reads the hash-verified
5.2.1 historical information path and pre-event residual library, then compares
policies under one network, information set, action domain, projector, budget,
loss, physical-path support, RC-MSA solver, tagged transition, and clearance
rule. It never reads an older checkpoint or any previous Chapter 5 result.

Run from the `code 8.4` root:

```powershell
python experiments/5.2-2/run_5_2_2.py
```

Before the full run, the route-wise disclosure error-scale evidence can be
reproduced with `python experiments/5.2-2/calibrate_waiting_error_scale.py`.
The full command verifies that these held-out validation RMSE values exactly
match the frozen configuration before any teacher action or checkpoint is
generated.

Tables and machine-readable acceptance evidence are written to
`output/5.2.2_common_authority_benchmark`. The three 300-dpi figures are also
published to this directory's `figures/` folder. The command exits nonzero when
any blocking preflight, trajectory, authority, objective, statistical, hash, or
figure check fails.

The comparison is conditional on the declared designed reference network and
`chi=0.5`. The route cost is a non-zero physical resource index based only on
declared excess maritime lag; it is not a monetary estimate and is explicitly
reserved for sensitivity analysis in 5.3.4.

The control horizon and readiness lead are both eight weeks because the unique
5.2.1 interface publishes an eight-week maturity-aligned forecast. Residual
uncertainty uses contiguous 21-week blocks so the lag-1 and lag-9 dependence
reported in 5.2.1 is not erased by independent weekly resampling.
