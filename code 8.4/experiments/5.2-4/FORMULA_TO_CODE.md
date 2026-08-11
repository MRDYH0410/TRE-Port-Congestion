# Formula-to-code map

| Construct | Code |
|---|---|
| Latest released packet \(\nu(t)\) | `information_design.ReleaseTimingScenarioBuilder._packet_schedule` and `_latest_packet` |
| Monthly lead forecast \(\alpha_{\nu(t)}P^h\) | `information_design.monthly_transition_count` and `ReleaseTimingScenarioBuilder.build` |
| No-new-information comparator \(\bar\alpha_{train}P^h\) | `information_design._stationary_distribution` and `InformationProvider.apply` |
| Current-filter comparator | `InformationProvider.apply(..., "IF")` |
| Perfect-information upper bound | `InformationProvider.apply(..., "ORACLE")` |
| \(g^R\) and \(g^D\) | `information_design.ReleaseTimingScenarioBuilder.build` and `evaluation_5_2_4.run_information_replication` |
| Rights restriction before projection | `controller_factory.CapacityRightsProjector` |
| Rights-aware projection Jacobian | `controller_factory.CapacityRightsProjector.local_jacobian` composed with the common projector Jacobian |
| Common physical transition and loss | `evaluation_5_2_4.run_information_replication` calling the 5.2.2 `ModelKernel` |
| Beginning-of-week information firewall | `evaluation_5_2_4.run_information_replication` calling the 5.2.2 `prepare_period` before policy evaluation |
| IL-RD accepted anchor reproduction | `run_5_2_4._build_anchor_specs` and `reporting_5_2_4.acceptance_payload` |
| \(V_{I,g,\omega}\), \(V^{L\mid F}\), oracle gap | `statistics_5_2_4.information_effects` |
| \(V_{R\mid D}\), \(V_{D\mid R}\), \(S_{RD}\) | `statistics_5_2_4.capacity_effects` |
| False-warning cost | `statistics_5_2_4.false_warning_costs` |
| Seed-first path inference | `statistics_5_2_4.aggregate_learning_seeds` |
| Paired intervals and multiplicity | `statistics_5_2_4._effect_row` |
| Blocking validation | `reporting_5_2_4.acceptance_payload` |
