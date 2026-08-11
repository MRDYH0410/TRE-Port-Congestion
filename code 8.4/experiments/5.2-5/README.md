# 5.2.5 Computational and Methodological Acceptance

This experiment audits the production chain used by Experiments 5.2.1--5.2.4. It does not rank policies and does not create a replacement environment.

Run once from `code 8.4`:

```powershell
python experiments/5.2-5/run_5_2_5.py
```

The command first fail-closes on the authorized 5.2.1--5.2.4 SHA256 locks. It then independently recalculates production-output contracts, runs controlled cases through the shared `tre84` projection, RC-MSA, MPC, tagged-transition and capacity modules, verifies the complete-master RC-MSA history distance, interface-level `a_t^{-I}`, per-vintage no-reset identities, and complete MPC/selector module logs, reconstructs the complete SAC update from accepted production traces, independently checks actor gradients, tests pre/post-repair numerical equivalence, recalculates 88-path paired precision, writes all CSV/JSON evidence, and generates the three PNG/PDF figures. A missing diagnostic is recorded as `NOT_TESTED`; any critical `FAIL` or `NOT_TESTED` makes methodology acceptance fail under the noncompensatory rule.

No manuscript file is edited by this experiment.
