# Experiment 5.2.3: Action and Congestion Mechanisms

Run from the `code 8.4` root:

```powershell
python experiments/5.2-3/run_5_2_3.py
```

The experiment reads only the accepted 5.2.2 outputs and their current-model
checkpoints. It does not train a policy. The main policy comparison is rebuilt
from all held-out physical paths. Fine-grained source, vintage, provenance and
stage traces are generated for the externally selected physical-path medoid.
The restricted-action diagnostic uses all held-out paths and the automatically
selected 5.2.2 benchmark leader.

`full_action` is required to reproduce the corresponding 5.2.2 path result.
The other restrictions are applied to the frozen policy's raw action before the
shared projector. They are mechanism diagnostics, not reoptimised action-right
values or causal marginal effects.
