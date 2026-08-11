# Formula-to-code map: Experiment 5.3.1

| Formula or contract | Production implementation | Experiment evidence |
|---|---|---|
| \(q^C_{kt}=\chi q^B_{kt}\), \(q^D_{kt}=(1-\chi)q^B_{kt}\) | `src/tre84/transition.py::construct_demand_split` | `weekly_commitment_trajectories.csv`; `trajectory_contract_checks.csv` |
| \(\chi\) applies only to new blocked cargo | `experiments/5.2-2/preparation.py::build_realization` passes the current configured fraction into the production transition; existing state is not rewritten | `parameter_registry_5_3_1.csv`; `trajectory_contract_checks.csv` |
| Committed/adaptive tagged flow through maritime, berth, yard, gate, and landbridge | `src/tre84/transition.py::TaggedTransition`; accepted provenance ledger in `experiments/5.2-3/mechanism.py::_physical_rows` | `weekly_commitment_trajectories.csv`; `path_level_results.csv` |
| \(Q^C=Q^{C,0}+\sum q^C\), \(D^C=\sum s^{M,C}\), \(O^C=Q^C-D^C\) | `commitment_worker.py::summarise_mechanism_artifact` | committed inflow, delivery, terminal outstanding and conservation columns |
| Same formal policy/model kernel per grid cell | `model.py::build_model`, `policies.py`, `mechanism.py::run_mechanism_replication` | `checkpoint_manifest.csv`; `trajectory_contract_checks.csv` |
| Learning seeds averaged within physical path | `statistics_5_3_1.py::aggregate_learning_seeds` | `path_level_seed_aggregated.csv` |
| Paired effects versus Passive and Reactive | `statistics_5_3_1.py::paired_effects` | `paired_effects.csv` |
| Required paths from endpoint paired variance | `statistics_5_3_1.py::endpoint_precision` | `endpoint_precision_requirements.csv`; `selected_path_count.csv` |
| Right-censored clearance | production `tre84.clearance.ClearanceRunner`; aggregation in `statistics_5_3_1.py::clearance_summary` | `clearance_and_censoring.csv` |
