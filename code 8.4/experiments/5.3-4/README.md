# Experiment 5.3.4 Parameter Robustness

This experiment evaluates conditional deployment robustness under 13 behavioural, physical, economic, control, and information factors, four preregistered combined stresses, and a clearance-classification audit. Commitment is fixed at the registered midpoint because its full structural domain is owned by the commitment-sensitivity experiment.

The 31 simulated cells are one common reference, two nonreference levels for each of 13 simulated factors, and four combined stresses. Clearance tolerance is diagnostic only. Dimension-compatible one-factor cells use frozen accepted policies. Maritime-lag and readiness-lead cells use Passive, Reactive, and online MPC because the state representation changes and checkpoint padding is prohibited. The long-lag/severe-reclosure anchor receives matched teacher generation, BC training, full constrained-SAC training, gradient checks, and projection-Jacobian checks. The nine-gateway cell uses the frozen size-specific checkpoint from the completed gateway experiment.

The convex-hazard/low-exit combined anchor deliberately retains the requested `p=2` level. Since `p=2` is the historical reference, it is algebraically identical to the low-exit main-effect cell; an explicit identity audit prevents it from being interpreted as a non-additive interaction.

Run from the code root:

```powershell
.\.venv\Scripts\python.exe experiments\5.3-4\run_5_3_4.py --phase all
```

The time gate uses validation paths and matched-training time only. It selects at most 20 formal physical paths without consulting test outcomes and enforces the eight-hour wall-clock budget. Cache reuse requires a matching configuration, source, upstream, checkpoint, and cell-path signature.
