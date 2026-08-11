# Experiment 5.3.2 — Reclosure Sensitivity

This implementation uses the frozen **layered coverage** design:

- all 150 structural cells: policy-independent corrected Prop. 2 certificate;
- 16 preregistered axial/corner cells: Passive, Reactive and frozen BC;
- reference, mild and severe anchors: add formal stochastic MPC and frozen MG constrained SAC.

No reclosure-cell retraining is permitted. Three learning seeds are averaged within each matched physical path before inference. Non-evaluated policy-cell combinations are `NOT_EVALUATED_BY_DESIGN`, never zero-filled.

From the code root:

```powershell
.\.venv\Scripts\python.exe experiments\5.3-2\run_5_3_2.py --phase all
```

The single-path and eight-path gates can be run separately:

```powershell
.\.venv\Scripts\python.exe experiments\5.3-2\run_5_3_2.py --phase gate1
.\.venv\Scripts\python.exe experiments\5.3-2\run_5_3_2.py --phase gate8
.\.venv\Scripts\python.exe experiments\5.3-2\run_5_3_2.py --phase formal
```

Progress is resumable at the physical-path/policy-seed task level. Monitor without starting a second simulation:

```powershell
.\.venv\Scripts\python.exe experiments\5.3-2\run_5_3_2.py --phase status
Get-Content experiments\5.3-2\run_layered.stdout.log -Wait
```

Formal outputs are published atomically to `output/5.3.2_reclosure_sensitivity`; the three requested 300 dpi PNG figures are also written to `experiments/5.3-2/figures`.

The certificate is anchored at reclosure onset, retains already dispatched committed maritime mass and route lags, removes future base/adaptive arrivals and grants optimistic capacity. A violation is sufficient for unavoidable threshold crossing; a nonviolation proves neither feasibility nor policy existence.
