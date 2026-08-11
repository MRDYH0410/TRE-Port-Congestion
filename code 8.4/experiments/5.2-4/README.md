# Experiment 5.2.4 — Released Risk Information and Capacity Preparation

Run from the `code 8.4` root:

```powershell
.venv\Scripts\python.exe experiments\5.2-4\run_5_2_4.py
```

The runner fails closed unless the registered 5.2.1 interface and the accepted
5.2.2/5.2.3 artifacts match their frozen SHA256 values. It uses all 88 accepted
physical paths, averages the three learning seeds within each path, and does not
read a previous 5.2.4 output or modify the manuscript.

The experiment keeps three estimands separate:

1. Reoptimized information value for I0, IF, IL, and the unattainable ORACLE.
2. Fixed IL-checkpoint information responsiveness under I0, IF, and IL.
3. Reoptimized capacity rights RD, R, D, and NONE under IL.

GH is a historical release replay. GT, GL, and GFW are designed timing stresses.
ORACLE is an unattainable upper bound. HMM output is interpreted only as a
geopolitical-risk state belief. The accepted 21-week 5.2.2 event-window replay is
retained as a separate numerical anchor because the designed timing stresses add
the preparation window mechanically implied by the readiness lead.

IL-RD reuses only the hash-locked accepted 5.2.2 BC and constrained-SAC
checkpoints. I0-RD, IF-RD, ORACLE-RD, IL-R, IL-D, and IL-NONE are trained from
scratch and selected on independent validation paths. Every controller uses the
same production `ModelKernel`, action projector, RC-MSA response, tagged
transition, capacity pipelines, operational loss, and clearance rule. Capacity
rights are restricted after each raw proposal and before the common projector.
