# 5.1 Parameter and Metric Additions for Experiment 5.3.1

## Data

NO CHANGE. Experiment 5.3.1 reads the accepted 5.2.1 interface and the accepted 5.2 production path builder. It introduces no new observed data and does not reinterpret χ as an empirical estimate.

## Parameters to register

- Structural commitment grid: `[0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]`.
- Grid role: endpoints and quartiles plus four preregistered eighth-grid resolution points.
- Application rule: χ splits only each period's newly blocked cohort through `qC=χqB` and `qD=(1-χ)qB`.
- Common path rule: 88 minimum, endpoint variance recalculation, common expansion up to 196.
- Precision target: 2255.6378250000002 loss-index units, inherited unchanged from 5.2.2.
- Policy set: Passive, Reactive, projected stochastic MPC, BC, and MG constrained SAC.
- Retraining rule: teacher data, BC, and constrained SAC are regenerated at each χ; test paths never select checkpoints.

## Metrics to add

- Total new committed mass `QC`, committed landbridge delivery `DC`, terminal committed outstanding `OC`, committed delivery share, and terminal committed outstanding share.
- Waiting exposure, SUE exit, duration attrition, committed delivery, and adaptive delivery by policy and χ.
- Matched effects versus both Passive and Reactive, path-paired policy regret, and simultaneous confidence-set membership.
- Clearance probability, restricted mean clearance time, terminal outstanding mass, and right-censoring count.

No arbitrary absorption score is introduced.
