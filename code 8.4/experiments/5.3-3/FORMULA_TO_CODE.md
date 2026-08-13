# Formula-to-code map for Experiment 5.3.3

| Contract | Formula or rule | Code | Output certificate |
|---|---|---|---|
| Network size | n in {3,4,5,7,9} | `network_5_3_3.declared_cells` | `declared_network_cells.csv` |
| Capacity neutral | fixed total stage and common-corridor capacity | `network_5_3_3.build_cell_config` | `network_resource_registry.csv` |
| Port only | median-template port stages; fixed common corridor | `network_5_3_3.build_cell_config` | `network_component_values.csv` |
| End to end | port stages and common corridor both expand | `network_5_3_3.build_cell_config` | `network_component_values.csv` |
| End-to-end reference capacity | min of berth, yard, gate, and proportional corridor share | `network_5_3_3._e2e_capacity` | `network_resource_registry.csv` |
| Emergency eligibility | new committed shares equal zero | `network_5_3_3.build_cell_config` | `network_resource_registry.csv` |
| Precontracted eligibility | capacity-proportional new share; official relative observed shares retained | `network_5_3_3.build_cell_config` | `network_resource_registry.csv` |
| Dynamic action | dim a(n)=3(3n+1)+1+n=10n+4 | `model.build_model`; `gateway_worker.initialise_worker` | `trajectory_contract_checks.csv` |
| Size-specific learning | fresh teacher, BC, full constrained SAC, selector validation | `run_5_3_3._train_size` | `checkpoint_manifest.csv` |
| SAC contract | full actor gradient finite difference | `training.sac_actor_gradient_check` | `sac_actor_gradient_check.csv` |
| Projection contract | formal local Jacobian versus central difference | `run_5_3_3._projection_jacobian_audit` | `projection_jacobian_check.csv` |
| Matched component values | choice, port, end-to-end, and precontracting loss differences | `statistics_5_3_3.component_values` | `network_component_values.csv` |
| Runtime gate | validation timing only; test outcomes do not select N | `run_5_3_3.run` | `computational_gate.csv` |

The observed three-gateway cell is a numerical reproduction anchor. Expanded nodes are semi-synthetic and may support structural mechanism claims only.
