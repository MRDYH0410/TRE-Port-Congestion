# 5.2.3 formula-to-code map

| Mechanism or identity | Implementation |
|---|---|
| Raw action, restriction, common projection | `mechanism.py::apply_restriction`, `run_mechanism_replication` |
| Waiting release and oldest-first allocation | `tre84.behavior::oldest_first`; logged by `mechanism.py::_behavior_rows` |
| Source route-wait-exit simplex | `tre84.behavior::RCMSASolver`; audited by `mechanism.py::_behavior_rows` for the medoid and `_behavior_residuals` for compact all-path checks |
| Waiting vintage transition and attrition | `tre84.transition::TaggedTransition.step`; audited by `waiting_identity_residual` |
| Committed/adaptive maritime provenance | `tre84.state::PipelineLot`; logged by `mechanism.py::_physical_rows` |
| Four-stage route-tagged transition | `tre84.transition::TaggedTransition.step`; decomposed by the proportional-service provenance shadow ledger |
| Capacity pipelines and physical feedback | `tre84.capacity::CapacityDynamics.transition`; logged by `mechanism.py::_capacity_rows` |
| Period loss and terminal correction | `tre84.loss::OperationalLoss`, `TerminalMassCorrection`; logged by `mechanism.py::_loss_row` |
| Learning-seed-first aggregation | `statistics_5_2_3.py::aggregate_full_policy_mechanisms`, `run_5_2_3.py::_aggregate_restricted_seeds` |
| Matched policy and restricted-action differences | `statistics_5_2_3.py::restricted_action_effects` |
| Physical path medoid | `statistics_5_2_3.py::select_physical_path_medoid` |
| Full-action no-difference reproduction | `run_5_2_3.py::_reproduction_checks` |
