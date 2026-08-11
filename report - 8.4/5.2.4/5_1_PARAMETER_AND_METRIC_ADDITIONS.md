# 5.1 additions required by Experiment 5.2.4

These entries are generated from the accepted 5.2.1/5.2.2 interfaces and the frozen 5.2.4 registry. No convenient default was introduced.

| Parameter | Value | Evidence | Basis |
|---|---|---|---|
| `HMM.transition_matrix` | [[0.9103344690884552, 0.08966553091154487], [0.6803176991315283, 0.31968230086847177]] | training-sample estimate | estimated on 456 months through 2024-12-31 |
| `HMM.training_stationary_distribution` | [0.8835487223448633, 0.11645127765513666] | derived estimate | stationary row distribution satisfying alpha_bar P = alpha_bar |
| `release_mapping.nu_t` | latest packet with release_date <= decision week | historical information interface | nonanticipative carry-forward at Monday decisions |
| `lead_horizon.h` | calendar month ordinal(maturity)-ordinal(source month) | model definition | monthly matrix is never multiplied once per week |
| `information_regime.I0` | {"label": "No newly released risk information", "controller_input": "training stationary distribution propagated at the native monthly frequency", "evidence_class": "implementable comparator"} | implementable comparator | training stationary distribution propagated at the native monthly frequency |
| `information_regime.IF` | {"label": "Current filtered risk only", "controller_input": "latest legally released filtered high-risk belief repeated at the lead coordinate", "evidence_class": "implementable current-state comparator"} | implementable current-state comparator | latest legally released filtered high-risk belief repeated at the lead coordinate |
| `information_regime.IL` | {"label": "Released lead-aligned forecast", "controller_input": "latest legally released alpha multiplied by P to the calendar-month distance from source month to readiness maturity", "evidence_class": "primary implementable regime"} | primary implementable regime | latest legally released alpha multiplied by P to the calendar-month distance from source month to readiness maturity |
| `information_regime.ORACLE` | {"label": "Perfect information oracle", "controller_input": "realized current disruption state and realized readiness-maturity disruption state", "evidence_class": "unattainable theoretical upper bound"} | unattainable theoretical upper bound | realized current disruption state and realized readiness-maturity disruption state |
| `warning_scenario.GH` | {"label": "Historical release", "construction": "canonical 5.2.1 public release dates and historical event physical path", "evidence_class": "historical information replay"} | historical information replay | canonical 5.2.1 public release dates and historical event physical path |
| `warning_scenario.GT` | {"label": "Minimally sufficient true release", "construction": "move the last event-pre information packet to event onset minus readiness lead so g_R equals zero", "evidence_class": "designed information timing stress"} | designed information timing stress | move the last event-pre information packet to event onset minus readiness lead so g_R equals zero |
| `warning_scenario.GL` | {"label": "Insufficient lead", "construction": "move the same packet to the first Monday after the g_R equals zero boundary", "evidence_class": "designed information timing stress"} | designed information timing stress | move the same packet to the first Monday after the g_R equals zero boundary |
| `warning_scenario.GFW` | {"label": "False warning", "construction": "use GT release timing with the matched normal-demand path and serviceability fixed at one", "evidence_class": "designed information timing stress"} | designed information timing stress | use GT release timing with the matched normal-demand path and serviceability fixed at one |
| `capacity_rights.RD` | {"label": "Readiness and direct procurement", "readiness_allowed": true, "direct_allowed": true} | reoptimized action-right design | restriction after raw proposal and before common projector |
| `capacity_rights.R` | {"label": "Readiness only", "readiness_allowed": true, "direct_allowed": false} | reoptimized action-right design | restriction after raw proposal and before common projector |
| `capacity_rights.D` | {"label": "Direct procurement only", "readiness_allowed": false, "direct_allowed": true} | reoptimized action-right design | restriction after raw proposal and before common projector |
| `capacity_rights.NONE` | {"label": "Neither readiness nor direct procurement", "readiness_allowed": false, "direct_allowed": false} | reoptimized action-right design | restriction after raw proposal and before common projector |
| `5.2.4.statistics.confidence_level` | 0.95 | preregistered experiment design | frozen before test evaluation |
| `5.2.4.statistics.bootstrap_resamples` | 5000 | preregistered experiment design | frozen before test evaluation |
| `5.2.4.statistics.bootstrap_seed_namespace` | 5.2.4-paired-bootstrap | preregistered experiment design | frozen before test evaluation |
| `5.2.4.statistics.multiplicity_adjustment` | Holm adjusted paired p-values plus within-family simultaneous t intervals | preregistered experiment design | frozen before test evaluation |
| `5.2.4.statistics.pilot_paths` | 4 | preregistered experiment design | frozen before test evaluation |
| `5.2.4.statistics.executed_paths_source` | 5.2.2 selected_path_count.csv | preregistered experiment design | frozen before test evaluation |
| `5.2.4.statistics.inference_unit` | physical path after learning-seed averaging | preregistered experiment design | frozen before test evaluation |
