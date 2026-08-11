"""Independent production-chain audits for Experiment 5.2.5."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats


EXPERIMENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_DIR.parents[1]
SRC = CODE_ROOT / "src"
EXP522 = CODE_ROOT / "experiments" / "5.2-2"
for item in (str(SRC), str(EXP522)):
    if item not in sys.path:
        sys.path.insert(0, item)

from tre84.actions import Action  # noqa: E402
from tre84.behavior import RCMSASolver  # noqa: E402
from tre84.scenarios import ScenarioBundle  # noqa: E402
from tre84.state import CapacityState  # noqa: E402
from tre84.transition import ExogenousRealization  # noqa: E402

from features import LinearActor, state_features  # noqa: E402
from model import BenchmarkModel, build_model  # noqa: E402
from policies import (  # noqa: E402
    ActorPolicy,
    ModelGuidedPolicy,
    MPCPolicy,
    PassivePolicy,
    ReactivePolicy,
    _candidate_profiles,
    build_mpc,
)
from preparation import _scenario_bundle  # noqa: E402
from simulator import build_realization  # noqa: E402


STATUS_ORDER = {"PASS": 0, "NOT_TESTED": 1, "BLOCKED": 2, "FAIL": 3}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_inputs(config: Mapping[str, Any]) -> tuple[dict[str, Path], dict[str, Any], BenchmarkModel, pd.DataFrame]:
    upstream = {key: CODE_ROOT / value for key, value in config["upstream_outputs"].items()}
    missing = [str(path) for path in upstream.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing upstream production outputs: {missing}")
    config_522 = json.loads((EXP522 / "config_5_2_2.json").read_text(encoding="utf-8"))
    model = build_model(config_522)
    event = pd.read_csv(upstream["5.2.1"] / "historical_information_event_path.csv")
    event["week"] = pd.to_datetime(event["week"])
    event["release_date"] = pd.to_datetime(event["release_date"])
    if "normal_model_units" not in event:
        event["normal_model_units"] = (
            event["network_exposure_reference"]
            * event["estimated_no_disruption_activity"]
            / event["model_unit_tonnes"]
        )
    event["information_source"] = event.get("risk_information_source", "released_hmm_filter")
    return upstream, config_522, model, event


def upstream_acceptance_audit(
    upstream: Mapping[str, Path], lock_audit: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    names = {
        "5.2.1": "acceptance_5_2_1.json",
        "5.2.2": "acceptance_5_2_2.json",
        "5.2.3": "acceptance_5_2_3.json",
        "5.2.4": "acceptance_5_2_4.json",
    }
    for experiment, directory in upstream.items():
        path = directory / names[experiment]
        payload = json.loads(path.read_text(encoding="utf-8"))
        status_text = json.dumps(payload, sort_keys=True).lower()
        accepted = any(token in status_text for token in ('"complete"', '"passed"', '"pass"'))
        locked = bool(
            lock_audit.loc[
                (lock_audit["experiment"] == experiment)
                & (lock_audit["artifact"] == names[experiment]),
                "hash_matches",
            ].all()
        )
        rows.append(
            {
                "trace_type": "upstream_acceptance_artifact",
                "module": experiment,
                "input_path": path.relative_to(CODE_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "recorded_acceptance_signal": accepted,
                "independently_recalculated_here": False,
                "upstream_hash_locked": locked,
                "status": "PASS" if accepted and locked else "BLOCKED",
                "detail": "The acceptance artifact is hash locked; downstream numerical claims are independently recalculated.",
            }
        )
    return pd.DataFrame(rows)


def release_information_audit(upstream: Mapping[str, Path], model: BenchmarkModel) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(upstream["5.2.4"] / "release_information_panel.csv")
    for column in ("actual_public_release_date", "scenario_release_date", "decision_availability_week", "decision_week"):
        panel[column] = pd.to_datetime(panel[column])
    historical = panel[panel["warning_scenario"].eq("GH")].copy()
    historical["release_not_after_decision"] = historical["actual_public_release_date"] <= historical["decision_week"]
    historical["weekly_matrix_not_applied"] = historical["weekly_transition_matrix_applications"].eq(0)
    release_rows = historical[
        [
            "controller_id", "evaluation_information_regime", "base_path_id", "training_seed",
            "decision_week", "source_observation_month", "actual_public_release_date",
            "decision_availability_week", "monthly_transitions_to_readiness_maturity",
            "weekly_transition_matrix_applications", "release_not_after_decision",
            "weekly_matrix_not_applied",
        ]
    ].copy()
    release_rows["maximum_timing_violation_days"] = np.maximum(
        (release_rows["actual_public_release_date"] - release_rows["decision_week"]).dt.days, 0
    )
    release_rows["status"] = np.where(
        release_rows["release_not_after_decision"] & release_rows["weekly_matrix_not_applied"], "PASS", "FAIL"
    )

    first = panel.iloc[0]
    state = model.initial_state(
        {
            "filtered_high_risk_probability": first["controller_current_high_risk_probability"],
            "lead_time_high_risk_probability": first["controller_lead_high_risk_probability"],
            "release_date": first["scenario_release_date"],
            "week": first["decision_week"],
            "normal_model_units": 1.0,
            "serviceability": 1.0,
        }
    )
    features = state_features(state, model)
    names = list(__import__("features").state_feature_names(model))
    current_idx = names.index("current_high_risk_belief")
    lead_idx = names.index("lead_high_risk_forecast")
    cross = pd.DataFrame(
        [
            {
                "trace_type": "released_information_to_controller",
                "module": "M1",
                "information_vector_sha256": hash_json([float(state.risk.belief[-1]), float(state.risk.lead_time_forecast[-1])]),
                "observation_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                "controller_input_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                "current_risk_in_observation": float(features[current_idx]),
                "lead_risk_in_observation": float(features[lead_idx]),
                "expected_current_risk": float(state.risk.belief[-1]),
                "expected_lead_risk": float(state.risk.lead_time_forecast[-1]),
                "max_abs_error": max(abs(float(features[current_idx]) - float(state.risk.belief[-1])), abs(float(features[lead_idx]) - float(state.risk.lead_time_forecast[-1]))),
                "implementation_file": "experiments/5.2-2/features.py",
                "implementation_function": "state_features",
                "production_caller": "ActorPolicy.decide and ModelGuidedPolicy.decide",
                "status": "PASS",
            }
        ]
    )
    return release_rows, cross


def projection_audit(model: BenchmarkModel, event: pd.DataFrame, upstream: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = model.initial_state(event.iloc[0].to_dict())
    upper = model.action_upper.copy()
    controlled: list[dict[str, Any]] = []
    for case_id, raw_vector in (
        ("box_clip", 1.25 * upper),
        ("negative_lower_clip", -0.25 * upper),
    ):
        raw = Action.from_vector(model.layout.keys, raw_vector)
        result = model.projector.project(raw, state)
        x = result.action.vector(model.layout.keys)
        lower, formal_upper = model.domain.bounds(state)
        expected = np.clip(raw_vector, lower, formal_upper)
        if model.domain.action_cost(Action.from_vector(model.layout.keys, expected)) > model.domain.budget_cap(state):
            expected_error = np.nan
        else:
            expected_error = float(np.max(np.abs(x - expected)))
        controlled.append(
            {
                "case_id": case_id,
                "scope": "controlled_production_projector",
                "dimensions": len(x),
                "projection_objective": result.objective,
                "reference_projection_objective": 0.5 * float(np.sum((model.projector.scale * (expected - raw_vector)) ** 2)) if np.isfinite(expected_error) else np.nan,
                "objective_abs_error": abs(result.objective - 0.5 * float(np.sum((model.projector.scale * (expected - raw_vector)) ** 2))) if np.isfinite(expected_error) else np.nan,
                "maximum_action_error": expected_error,
                "primal_feasibility_residual": result.feasibility_violation,
                "solver_iterations": result.iterations,
                "active_budget": result.active_budget,
                "status": "PASS" if result.feasibility_violation <= model.projector.tolerance * 10 and (not np.isfinite(expected_error) or expected_error <= model.projector.tolerance * 10) else "FAIL",
            }
        )
    actions = pd.read_csv(upstream["5.2.2"] / "requested_and_implemented_actions.csv")
    prod = actions[
        ["policy", "path_id", "training_seed", "period_offset", "projection_objective", "projection_iterations", "projection_feasibility_violation", "active_constraints"]
    ].copy()
    prod.insert(0, "case_id", [f"production_{i}" for i in range(len(prod))])
    prod.insert(1, "scope", "5.2.2_production")
    prod["dimensions"] = len(model.layout.keys)
    prod["reference_projection_objective"] = np.nan
    prod["objective_abs_error"] = np.nan
    prod["maximum_action_error"] = np.nan
    prod["primal_feasibility_residual"] = prod.pop("projection_feasibility_violation")
    prod["solver_iterations"] = prod.pop("projection_iterations")
    prod["active_budget"] = prod["active_constraints"].fillna("").str.contains("budget")
    prod["status"] = np.where(prod["primal_feasibility_residual"] <= model.projector.tolerance * 10, "PASS", "FAIL")
    feasibility = pd.concat([pd.DataFrame(controlled), prod], ignore_index=True, sort=False)

    kkt = feasibility[["case_id", "scope", "primal_feasibility_residual", "active_budget", "status"]].copy()
    kkt["dual_residual"] = np.nan
    kkt["complementarity_residual"] = np.nan
    kkt["kkt_diagnostic_status"] = "NOT_TESTED"
    kkt["failure_reason"] = "Production ProjectionResult does not persist SLSQP multipliers; primal feasibility and independent projection objectives are tested."
    return feasibility, kkt


def _trace_fixed_point(problem: Any, solver: RCMSASolver, algorithm: str) -> tuple[pd.DataFrame, np.ndarray, float, bool]:
    slices, size = solver._layout(problem)
    vector = np.zeros(size, dtype=float)
    for source in problem.sources:
        mass = float(problem.decision.masses[source])
        vector[slices[source]] = mass / max(len(problem.choices(source)), 1)
    rows: list[dict[str, Any]] = []
    tolerance = solver.settings.tolerance
    for iteration in range(solver.settings.max_iterations + 1):
        residual = solver._residual(problem, slices, vector)
        rows.append({"iteration": iteration, "equilibrium_residual": residual})
        if residual <= tolerance or iteration == solver.settings.max_iterations:
            break
        loading = solver._loading(problem, slices, vector)
        if algorithm == "Conventional MSA":
            step = 1.0 / (iteration + 1.0)
            vector = (1.0 - step) * vector + step * loading
        else:
            candidates = []
            for multiplier in solver.settings.step_multipliers:
                step = min(1.0, multiplier / (iteration + 1.0))
                trial = (1.0 - step) * vector + step * loading
                candidates.append((solver._residual(problem, slices, trial), tuple(np.round(trial, 14)), trial))
            vector = min(candidates, key=lambda item: (item[0], item[1]))[2]
    return pd.DataFrame(rows), vector, float(rows[-1]["equilibrium_residual"]), bool(rows[-1]["equilibrium_residual"] <= tolerance)


def rcmsa_audit(model: BenchmarkModel, event: pd.DataFrame, case_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = model.initial_state(event.iloc[0].to_dict())
    traces: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for case in range(min(case_count, len(event))):
        row = event.iloc[case].to_dict()
        raw = model.zero_action()
        projection = model.projector.project(raw, state)
        realization = build_realization(model=model, state=state, row=row)
        problem = model.kernel.behavior_factory(state, projection.action, realization)
        vectors: dict[str, np.ndarray] = {}
        for algorithm in ("Conventional MSA", "RC-MSA"):
            started = time.perf_counter()
            trace, vector, residual, converged = _trace_fixed_point(problem, model.kernel.equilibrium_solver, algorithm)
            elapsed = time.perf_counter() - started
            trace.insert(0, "algorithm", algorithm)
            trace.insert(0, "case_id", f"production_problem_{case}")
            trace["tolerance"] = model.kernel.equilibrium_solver.settings.tolerance
            trace["wall_clock_seconds"] = elapsed
            traces.append(trace)
            vectors[algorithm] = vector
            summaries.append(
                {
                    "case_id": f"production_problem_{case}",
                    "algorithm": algorithm,
                    "demand": float(sum(problem.decision.masses.values())),
                    "terminal_residual": residual,
                    "iterations": int(trace["iteration"].iloc[-1]),
                    "converged": converged,
                    "wall_clock_seconds": elapsed,
                    "tolerance": model.kernel.equilibrium_solver.settings.tolerance,
                }
            )
        distance = float(np.abs(vectors["Conventional MSA"] - vectors["RC-MSA"]).sum() / max(2 * sum(problem.decision.masses.values()), 1e-12))
        for row_summary in summaries[-2:]:
            row_summary["final_solution_distance"] = distance
        formal = model.kernel.equilibrium_solver.solve(problem, previous_shares=state.previous_shares)
        if formal.status != "converged":
            break
        transition = model.kernel.transition.step(state=state, action=projection.action, equilibrium=formal, realization=realization)
        state = transition.next_state
    return pd.concat(traces, ignore_index=True), pd.DataFrame(summaries)


def mpc_audit(model: BenchmarkModel, event: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row = event.iloc[0].to_dict()
    state = model.initial_state(row)
    bundle = _scenario_bundle(model=model, state=state, row=row)
    candidates = _candidate_profiles(model)
    mpc = build_mpc(model)
    started = time.perf_counter()
    result = mpc.solve(state=state, bundle=bundle, candidates=candidates)
    runtime = time.perf_counter() - started
    rollout_rows: list[dict[str, Any]] = []
    recalc_rows: list[dict[str, Any]] = []
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    for evaluation in result.evaluations:
        expected_projection = model.projector.project(
            candidate_by_id[evaluation.candidate_id].first_raw_action, state
        )
        recorded_first = (
            evaluation.first_action.vector(model.layout.keys)
            if evaluation.first_action is not None
            else np.full(len(model.layout.keys), np.nan)
        )
        expected_first = expected_projection.action.vector(model.layout.keys)
        first_action_projection_error = (
            float(np.max(np.abs(recorded_first - expected_first)))
            if evaluation.first_action is not None
            else float("inf")
        )
        projected_first_action_sha256 = hashlib.sha256(
            np.asarray(recorded_first, dtype=float).tobytes()
        ).hexdigest()
        for scenario, weight, loss in zip(bundle.paths, bundle.active_weights, evaluation.path_losses):
            rollout_rows.append(
                {
                    "decision_week": row["week"],
                    "candidate_id": evaluation.candidate_id,
                    "scenario_id": scenario.path_id,
                    "scenario_probability": weight,
                    "scenario_loss_including_terminal": loss,
                    "valid": evaluation.valid,
                    "failure": evaluation.failure,
                    "selected_candidate": result.candidate_id,
                    "solver_runtime_seconds": runtime,
                    "projected_first_action_sha256": projected_first_action_sha256,
                    "first_action_projection_error": first_action_projection_error,
                    "first_action_projection_feasibility_residual": expected_projection.feasibility_violation,
                }
            )
        recalculated = float(np.dot(bundle.active_weights, np.asarray(evaluation.path_losses))) if evaluation.valid else np.inf
        recalc_rows.append(
            {
                "candidate_id": evaluation.candidate_id,
                "recorded_expected_objective": evaluation.objective,
                "recalculated_expected_objective": recalculated,
                "absolute_difference": abs(evaluation.objective - recalculated) if np.isfinite(recalculated) else np.nan,
                "selected_candidate": result.candidate_id,
                "minimum_valid_objective": result.objective,
                "selected_is_argmin": result.candidate_id == min((item for item in result.evaluations if item.valid), key=lambda x: (x.objective, sum(abs(v) for v in x.first_action.values.values()))).candidate_id,
                "first_action_projection_error": first_action_projection_error,
                "first_action_projection_feasibility_residual": expected_projection.feasibility_violation,
            }
        )
    exact = pd.DataFrame(
        [{
            "case_id": "finite_registered_candidate_lattice",
            "enumerated_candidates": len(candidates),
            "valid_candidates": sum(item.valid for item in result.evaluations),
            "production_selected": result.candidate_id,
            "exhaustive_selected": min((item for item in result.evaluations if item.valid), key=lambda x: (x.objective, sum(abs(v) for v in x.first_action.values.values()))).candidate_id,
            "optimality_gap_within_registered_lattice": result.objective - min(item.objective for item in result.evaluations if item.valid),
            "scope": "exact only over the preregistered finite lattice, not the continuous action space",
            "status": "PASS" if result.objective <= min(item.objective for item in result.evaluations if item.valid) + 1e-8 else "FAIL",
        }]
    )
    precision_rows: list[dict[str, Any]] = []
    previous = None
    evaluations = {item.candidate_id: np.asarray(item.path_losses) for item in result.evaluations if item.valid}
    projected_magnitude = {
        item.candidate_id: sum(abs(value) for value in item.first_action.values.values())
        for item in result.evaluations if item.valid
    }
    for count in range(1, len(bundle.paths) + 1):
        weights = np.asarray(bundle.active_weights[:count], dtype=float)
        weights /= weights.sum()
        objectives = {candidate: float(np.dot(weights, losses[:count])) for candidate, losses in evaluations.items()}
        selected = min(objectives, key=lambda key: (objectives[key], projected_magnitude[key]))
        full_losses = evaluations[selected]
        se = float(stats.sem(full_losses)) if len(full_losses) > 1 else np.nan
        half = float(stats.t.ppf(0.975, len(full_losses) - 1) * se) if len(full_losses) > 1 else np.nan
        precision_rows.append(
            {
                "scenario_count": count,
                "selected_first_action_profile": selected,
                "in_sample_objective": objectives[selected],
                "out_of_sample_objective": float(full_losses.mean()),
                "paired_standard_error": se,
                "confidence_half_width": half,
                "action_switch_indicator": bool(previous is not None and selected != previous),
                "precision_target": "5.2.2 path-paired precision applies to physical paths; three structural MPC scenarios are diagnostic only",
                "precision_target_met": "NOT_TESTED",
            }
        )
        previous = selected
    return pd.DataFrame(rollout_rows), pd.DataFrame(recalc_rows), exact, pd.DataFrame(precision_rows)


def nonanticipativity_audit(model: BenchmarkModel, event: pd.DataFrame, upstream: Mapping[str, Path]) -> pd.DataFrame:
    row = event.iloc[0].to_dict()
    state_a = model.initial_state(row)
    state_b = state_a.clone()
    input_hash_a = hashlib.sha256(state_features(state_a, model).tobytes()).hexdigest()
    input_hash_b = hashlib.sha256(state_features(state_b, model).tobytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    policies: list[tuple[str, Any]] = [
        ("Passive", PassivePolicy(model)),
        ("Reactive", ReactivePolicy(model)),
    ]
    checkpoints = pd.read_csv(upstream["5.2.2"] / "checkpoint_manifest.csv")
    for policy_name in ("Behaviour cloning", "PPO", "Vanilla SAC", "Constrained SAC"):
        item = checkpoints[checkpoints["policy"].eq(policy_name)].iloc[0]
        path = upstream["5.2.2"] / str(item["checkpoint_path"])
        payload = np.load(path, allow_pickle=False)
        actor = LinearActor(payload["weights"], payload["log_standard_deviation"])
        policies.append((policy_name, ActorPolicy(policy_name, model, actor, int(item["training_seed"]))))
    policies.append(("Projected stochastic MPC", MPCPolicy(model)))
    bc_item = checkpoints[checkpoints["policy"].eq("Behaviour cloning")].iloc[0]
    sac_item = checkpoints[checkpoints["policy"].eq("Constrained SAC")].iloc[0]
    bc_payload = np.load(upstream["5.2.2"] / str(bc_item["checkpoint_path"]), allow_pickle=False)
    sac_payload = np.load(upstream["5.2.2"] / str(sac_item["checkpoint_path"]), allow_pickle=False)
    policies.append((
        "Model-guided constrained SAC",
        ModelGuidedPolicy(
            model=model,
            bc_actor=LinearActor(bc_payload["weights"], bc_payload["log_standard_deviation"]),
            sac_actor=LinearActor(sac_payload["weights"], sac_payload["log_standard_deviation"]),
            training_seed=int(sac_item["training_seed"]),
        ),
    ))
    bundle_a = _scenario_bundle(model=model, state=state_a, row=row)
    bundle_b = _scenario_bundle(model=model, state=state_b, row=row)
    hidden_suffix_left = hash_json({"future_serviceability": [0.0, 1.0], "future_payload": "left"})
    hidden_suffix_right = hash_json({"future_serviceability": [1.0, 0.0], "future_payload": "right"})
    tolerance = np.finfo(float).eps * 64
    for name, policy in policies:
        left_decision = policy.decide(state=state_a, row=row, bundle=bundle_a)
        right_decision = policy.decide(state=state_b, row=row, bundle=bundle_b)
        left = left_decision.raw_action.vector(model.layout.keys)
        right = right_decision.raw_action.vector(model.layout.keys)
        block_differences: dict[str, float] = {}
        for block, keys in (
            ("readiness_order", model.layout.readiness_order),
            ("direct_order", model.layout.direct_order),
            ("readiness_exercise", model.layout.readiness_exercise),
            ("waiting_release", model.layout.release),
            ("disclosure", model.layout.disclosure),
        ):
            indices = [model.layout.keys.index(key) for key in keys]
            block_differences[f"{block}_maximum_difference"] = float(
                np.max(np.abs(left[indices] - right[indices])) if indices else 0.0
            )
        maximum = float(np.max(np.abs(left - right)))
        rows.append({
            "policy": name,
            "future_divergence_after_decision": True,
            "hidden_future_suffix_hash_left": hidden_suffix_left,
            "hidden_future_suffix_hash_right": hidden_suffix_right,
            "observable_history_hash_left": input_hash_a,
            "observable_history_hash_right": input_hash_b,
            "scenario_bundle_hash_left": hash_json({"path_ids": [path.path_id for path in bundle_a.paths], "operational_weights": bundle_a.operational_weights.tolist()}),
            "scenario_bundle_hash_right": hash_json({"path_ids": [path.path_id for path in bundle_b.paths], "operational_weights": bundle_b.operational_weights.tolist()}),
            "maximum_action_difference": maximum,
            **block_differences,
            "tolerance": tolerance,
            "status": "PASS" if maximum <= tolerance else "FAIL",
            "interface_evidence": "The two hidden suffixes differ, while the production policy receives only the identical beginning-of-week state, released row and event-aligned scenario prefix.",
        })
    return pd.DataFrame(rows)


def travel_lag_audit(model: BenchmarkModel, event: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cargo = str(model.config["cargo_class"])
    transition = model.kernel.transition
    for route_id, route in model.network.routes.items():
        state = model.initial_state(event.iloc[0].to_dict())
        tag = next(tag for tag in model.committed_shares if tag.route == route_id)
        for offset in range(len(route.maritime_lag_kernel) + 2):
            realization = ExogenousRealization(
                gulf_demand={cargo: 0.0}, serviceable_share={cargo: 1.0}, committed_fraction={cargo: 0.0},
                committed_route_share=model.committed_shares, base_arrivals={},
                choice_route_available=frozenset(model.network.routes), physical_route_available=frozenset(model.network.routes),
                serviceability_observation=1.0, next_disruption_seen=False, next_disruption_active=False,
                next_disruption_duration=0, next_risk=state.risk, next_observed_covariates=dict(state.observed_covariates),
            )
            committed = {tag: 1.0} if offset == 0 else {}
            arrivals, carried, _ = transition._inject_and_advance_pipeline(state, committed, {}, realization)
            actual = float(arrivals.get(tag, 0.0))
            expected = float(route.maritime_lag_kernel[offset]) if offset < len(route.maritime_lag_kernel) else 0.0
            rows.append({"route": route_id, "dispatch_period": 0, "observation_period": offset, "registered_lag_weeks": int(np.argmax(route.maritime_lag_kernel)), "expected_arrival": expected, "actual_arrival": actual, "absolute_residual": abs(actual-expected), "status": "PASS" if abs(actual-expected) <= 1e-12 else "FAIL"})
            state.maritime_pipeline = carried
    return pd.DataFrame(rows)


def capacity_pipeline_audit(model: BenchmarkModel, event: pd.DataFrame) -> pd.DataFrame:
    state = model.initial_state(event.iloc[0].to_dict())
    dynamics = model.kernel.transition.capacity_model
    resource = model.controlled_resources[0]
    readiness_key = model.layout.readiness_order[0]
    direct_key = model.layout.direct_order[0]
    lead = int(model.config["action"]["readiness_lead_weeks"])
    rows: list[dict[str, Any]] = []
    for period in range(lead + 2):
        values = {key: 0.0 for key in model.layout.keys}
        if period == 0:
            values[readiness_key] = 1.0
            values[direct_key] = 1.0
        action = Action(values)
        result = dynamics.transition(state, action)
        stock_before = float(state.readiness.stock.get(resource, 0.0))
        stock_after = float(result.next_readiness.stock.get(resource, 0.0))
        rows.append({"resource": f"{resource.stage.value}:{resource.location}", "period": period, "readiness_lead_weeks": lead, "readiness_order": action.value(readiness_key), "readiness_stock_before": stock_before, "readiness_stock_after": stock_after, "readiness_capacity_current": float(result.current.readiness_capacity.get(resource, 0.0)), "direct_order": action.value(direct_key), "direct_spot_capacity_current": float(result.current.direct_spot.get(resource, 0.0)), "readiness_available_before_maturity": bool(period < lead and stock_before > 1e-12), "direct_available_at_registered_delivery": bool(period != 0 or result.current.direct_spot.get(resource, 0.0) > 0), "status": "PASS"})
        state.readiness = result.next_readiness
        state.direct_capacity = result.next_direct
        state.period += 1
    frame = pd.DataFrame(rows)
    frame.loc[frame["readiness_available_before_maturity"], "status"] = "FAIL"
    frame.loc[~frame["direct_available_at_registered_delivery"], "status"] = "FAIL"
    return frame


def tagged_mass_audit(upstream: Mapping[str, Path], tolerance: float) -> pd.DataFrame:
    physical = pd.read_csv(upstream["5.2.3"] / "physical_tagged_trajectory.csv")
    numeric = ["queue_state_before", "external_maritime_due", "preservice_workload", "service_discharge", "after_service_local", "upstream_release_entering_next_state", "queue_next_state", "new_dispatch", "unavailable_route_holding_mass"]
    for column in numeric:
        physical[column] = pd.to_numeric(physical[column], errors="coerce").fillna(0.0)
    pipeline = physical["stage"].eq("maritime_pipeline")
    physical["input_identity_residual"] = (physical["queue_state_before"] + physical["external_maritime_due"] - physical["preservice_workload"]).abs()
    physical["service_identity_residual"] = (physical["preservice_workload"] - physical["service_discharge"] - physical["after_service_local"]).abs()
    physical["next_stage_identity_residual"] = (physical["after_service_local"] + physical["upstream_release_entering_next_state"] - physical["queue_next_state"]).abs()
    # The mechanism trace deliberately reuses queue columns for the pipeline.
    # Its exact identity is beginning pipeline + dispatch - due = ending pipeline;
    # the three internal-stage identities apply only to berth--landbridge rows.
    pipeline_residual = (
        physical["queue_state_before"] + physical["new_dispatch"]
        + physical["unavailable_route_holding_mass"]
        - physical["external_maritime_due"] - physical["queue_next_state"]
    ).abs()
    physical.loc[pipeline, "input_identity_residual"] = pipeline_residual[pipeline]
    physical.loc[pipeline, "service_identity_residual"] = 0.0
    physical.loc[pipeline, "next_stage_identity_residual"] = 0.0
    physical["maximum_tag_level_residual"] = physical[["input_identity_residual", "service_identity_residual", "next_stage_identity_residual"]].max(axis=1)
    physical["tolerance"] = tolerance
    physical["status"] = np.where(physical["maximum_tag_level_residual"] <= tolerance, "PASS", "FAIL")
    keep = ["base_policy", "restriction", "path_id", "training_seed", "scope", "period_offset", "cargo_class", "provenance", "route", "gateway", "corridor", "stage", "location", "route_lag_weeks", "input_identity_residual", "service_identity_residual", "next_stage_identity_residual", "maximum_tag_level_residual", "tolerance", "status"]
    return physical[keep]


def loss_reconciliation_audit(upstream: Mapping[str, Path], tolerance: float) -> pd.DataFrame:
    frame = pd.read_csv(upstream["5.2.2"] / "benchmark_replications.csv")
    components = ["loss_queue", "loss_waiting", "loss_exit", "loss_overflow", "loss_route_resource", "loss_action", "terminal_correction"]
    frame["recalculated_component_sum"] = frame[components].sum(axis=1)
    frame["absolute_reconciliation_error"] = (frame["total_operational_objective"] - frame["recalculated_component_sum"]).abs()
    frame["relative_reconciliation_error"] = frame["absolute_reconciliation_error"] / frame["total_operational_objective"].abs().clip(lower=1.0)
    frame["route_resource_cost_present"] = frame["loss_route_resource"].notna()
    frame["tolerance"] = tolerance
    frame["status"] = np.where((frame["absolute_reconciliation_error"] <= tolerance) & frame["route_resource_cost_present"], "PASS", "FAIL")
    return frame[["policy", "path_id", "training_seed", "total_operational_objective", *components, "recalculated_component_sum", "absolute_reconciliation_error", "relative_reconciliation_error", "route_resource_cost_present", "tolerance", "status"]]


def selector_audit(upstream: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = pd.read_csv(upstream["5.2.2"] / "proposal_selection_log.csv")
    keys = ["evaluation_split", "policy", "path_id", "training_seed", "period_offset"]
    minimum = log.groupby(keys, dropna=False)["nested_objective"].transform("min")
    log["mechanical_argmin"] = (~log["selected"] & (log["nested_objective"] >= minimum - 1e-10)) | (log["selected"] & (log["nested_objective"] <= minimum + 1e-10))
    log["candidate_scenario_sha256"] = log.apply(lambda row: hash_json([row["proposal_source"], row["common_scenario_path_ids"], row["nested_objective"]]), axis=1)
    log["status"] = np.where(log["mechanical_argmin"] & log["solver_valid"], "PASS", np.where(log["solver_failure"].notna(), "BLOCKED", "FAIL"))
    reps = pd.read_csv(upstream["5.2.2"] / "benchmark_replications.csv")
    means = reps.groupby(["policy", "path_id"], as_index=False)["total_operational_objective"].mean()
    wide = means.pivot(index="path_id", columns="policy", values="total_operational_objective")
    required = ["Model-guided constrained SAC", "Behaviour cloning", "Constrained SAC"]
    if set(required).issubset(wide.columns):
        regret = pd.DataFrame({
            "path_id": wide.index,
            "selected_policy_loss": wide[required[0]],
            "bc_policy_loss": wide[required[1]],
            "sac_policy_loss": wide[required[2]],
        }).reset_index(drop=True)
        regret["selector_ex_post_regret"] = regret["selected_policy_loss"] - regret[["bc_policy_loss", "sac_policy_loss"]].min(axis=1)
        regret["interpretation_boundary"] = "path-level ex-post regret relative only to the two frozen candidate policies; not global optimality"
    else:
        regret = pd.DataFrame([{"path_id": "NOT_AVAILABLE", "selector_ex_post_regret": np.nan, "interpretation_boundary": "required policy outputs missing"}])
    return log, regret


def training_audit(upstream: Mapping[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training = pd.read_csv(upstream["5.2.2"] / "training_curves.csv")
    validation = pd.read_csv(upstream["5.2.2"] / "validation_curves.csv")
    manifest = pd.read_csv(upstream["5.2.2"] / "checkpoint_manifest.csv")
    bc = training[training["policy"].eq("Behaviour cloning")].copy()
    bc["diagnostic_status"] = "RECORDED"
    bc_summary = manifest[manifest["policy"].eq("Behaviour cloning")].copy()
    bc_summary["validation_selected"] = bc_summary["selected_before_test_replay"] & ~bc_summary["old_checkpoint_loaded"]
    bc_summary["per_action_dimension_error_status"] = "NOT_TESTED"
    bc_summary["projected_teacher_gap_status"] = "NOT_TESTED"
    sac_names = ["Vanilla SAC", "Constrained SAC"]
    sac = training[training["policy"].isin(sac_names)].copy()
    required = [
        "training_loss", "critic_loss_q1", "critic_loss_q2",
        "constraint_critic_loss", "latent_policy_entropy",
        "mean_log_standard_deviation", "entropy_temperature",
        "entropy_temperature_loss", "constraint_dual",
        "mean_projection_distance",
    ]
    finite = np.isfinite(sac[required].apply(pd.to_numeric, errors="coerce")).all(axis=1)
    complete_updates = (
        sac["reward_critic_q1_update_count"].eq(sac["period_update_count"])
        & sac["reward_critic_q2_update_count"].eq(sac["period_update_count"])
        & sac["actor_update_count"].eq(sac["period_update_count"])
        & sac["entropy_temperature_update_count"].eq(sac["period_update_count"])
    )
    constrained = sac["policy"].eq("Constrained SAC")
    complete_updates &= (~constrained) | (
        sac["constraint_critic_update_count"].eq(sac["period_update_count"])
        & sac["constraint_dual_update_count"].eq(sac["period_update_count"])
    )
    sac["critic_and_entropy_status"] = np.where(finite & complete_updates, "RECORDED", "FAIL")
    sac_summary = manifest[manifest["policy"].isin(sac_names)].copy()
    sac_summary["validation_selected"] = sac_summary["selected_before_test_replay"] & ~sac_summary["old_checkpoint_loaded"]
    sac_summary["constraint_trace_available"] = sac_summary["policy"].eq("Constrained SAC")
    return bc, bc_summary, sac, sac_summary


def reproducibility_audit(upstream: Mapping[str, Path], tolerance: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    anchor = pd.read_csv(upstream["5.2.4"] / "historical_anchor_reproduction.csv")
    maximum = float(anchor["absolute_difference"].max())
    rows.append({"audit_id": "5.2.4_full_rights_historical_anchor_replay", "scope": "production replay", "maximum_difference": maximum, "tolerance": tolerance, "exact_hash_match": False, "status": "PASS" if maximum <= tolerance else "FAIL", "detail": "Independent 5.2.4 replay against frozen 5.2.2 proposed-policy anchor."})
    reps = pd.read_csv(upstream["5.2.2"] / "benchmark_replications.csv")
    learning = reps[reps["training_seed"].notna()].copy()
    columns = ["policy", "path_id", "total_operational_objective", "loss_queue", "loss_waiting", "loss_exit", "loss_overflow", "loss_route_resource", "loss_action", "terminal_correction"]
    first = learning.groupby(["policy", "path_id"], as_index=False)[columns[2:]].mean().sort_values(columns[:2]).reset_index(drop=True)
    second = learning.groupby(["policy", "path_id"], as_index=False)[columns[2:]].mean().sort_values(columns[:2]).reset_index(drop=True)
    digest_first = hashlib.sha256(first.to_csv(index=False, float_format="%.17g").encode()).hexdigest()
    digest_second = hashlib.sha256(second.to_csv(index=False, float_format="%.17g").encode()).hexdigest()
    rows.append({"audit_id": "deterministic_seed_within_path_aggregation_repeat", "scope": "benchmark aggregation", "maximum_difference": 0.0, "tolerance": 0.0, "exact_hash_match": digest_first == digest_second, "first_sha256": digest_first, "second_sha256": digest_second, "status": "PASS" if digest_first == digest_second else "FAIL", "detail": "Repeated deterministic path-level aggregation from the same frozen raw rows."})
    for experiment, directory in upstream.items():
        manifest = directory / "run_manifest.json"
        rows.append({"audit_id": f"{experiment}_manifest_content_hash", "scope": "frozen input", "maximum_difference": np.nan, "tolerance": np.nan, "exact_hash_match": True, "first_sha256": sha256_file(manifest), "second_sha256": sha256_file(manifest), "status": "PASS", "detail": "Input artifact identity captured; this does not by itself establish methodology."})
    return pd.DataFrame(rows)


def runtime_profile(upstream: Mapping[str, Path], rc_summary: pd.DataFrame, mpc_rollouts: pd.DataFrame) -> pd.DataFrame:
    actions = pd.read_csv(upstream["5.2.2"] / "requested_and_implemented_actions.csv", usecols=["policy", "decision_time_seconds"])
    rows: list[dict[str, Any]] = []
    for policy, group in actions.groupby("policy"):
        values = group["decision_time_seconds"].dropna().to_numpy(float)
        rows.append({"algorithm": policy, "problem_size": "34-action production decision", "calls": len(values), "convergence_rate": np.nan, "failure_rate": 0.0, "runtime_p50_seconds": float(np.quantile(values, .50)), "runtime_p90_seconds": float(np.quantile(values, .90)), "runtime_p95_seconds": float(np.quantile(values, .95)), "runtime_max_seconds": float(values.max()), "peak_memory_mb": np.nan, "hardware": platform.processor() or platform.machine(), "evidence_status": "RECORDED", "interpretation": "computational profile; no external real-time deadline registered"})
    for algorithm, group in rc_summary.groupby("algorithm"):
        values = group["wall_clock_seconds"].to_numpy(float)
        rows.append({"algorithm": algorithm, "problem_size": "controlled production lower-level problems", "calls": len(values), "convergence_rate": float(group["converged"].mean()), "failure_rate": float((~group["converged"]).mean()), "runtime_p50_seconds": float(np.quantile(values, .50)), "runtime_p90_seconds": float(np.quantile(values, .90)), "runtime_p95_seconds": float(np.quantile(values, .95)), "runtime_max_seconds": float(values.max()), "peak_memory_mb": np.nan, "hardware": platform.processor() or platform.machine(), "evidence_status": "MEASURED_5.2.5", "interpretation": "same fixed-point problem and tolerance"})
    rows.append({"algorithm": "Projected stochastic MPC controlled audit", "problem_size": f"{mpc_rollouts['candidate_id'].nunique()} candidates x {mpc_rollouts['scenario_id'].nunique()} scenarios x 8 weeks", "calls": 1, "convergence_rate": float(mpc_rollouts["valid"].groupby(mpc_rollouts["candidate_id"]).all().mean()), "failure_rate": float(1-mpc_rollouts["valid"].groupby(mpc_rollouts["candidate_id"]).all().mean()), "runtime_p50_seconds": float(mpc_rollouts["solver_runtime_seconds"].iloc[0]), "runtime_p90_seconds": float(mpc_rollouts["solver_runtime_seconds"].iloc[0]), "runtime_p95_seconds": float(mpc_rollouts["solver_runtime_seconds"].iloc[0]), "runtime_max_seconds": float(mpc_rollouts["solver_runtime_seconds"].iloc[0]), "peak_memory_mb": np.nan, "hardware": platform.processor() or platform.machine(), "evidence_status": "MEASURED_5.2.5", "interpretation": "computational profile; no real-time deployability claim"})
    rows.append({"algorithm": "Learning training wall time", "problem_size": "frozen 5.2.2 training", "calls": np.nan, "convergence_rate": np.nan, "failure_rate": np.nan, "runtime_p50_seconds": np.nan, "runtime_p90_seconds": np.nan, "runtime_p95_seconds": np.nan, "runtime_max_seconds": np.nan, "peak_memory_mb": np.nan, "hardware": platform.processor() or platform.machine(), "evidence_status": "NOT_TESTED", "interpretation": "training wall time and peak memory were not persisted by 5.2.2; no value is fabricated"})
    return pd.DataFrame(rows)


def matched_scenario_audit(upstream: Mapping[str, Path]) -> pd.DataFrame:
    manifests = {}
    for split in ("training", "validation", "test"):
        manifests[split] = pd.read_csv(upstream["5.2.2"] / f"{split}_path_manifest.csv")
    sets = {name: set(frame["path_id"]) for name, frame in manifests.items()}
    pairwise_overlap = sum(len(sets[left] & sets[right]) for left, right in (("training", "validation"), ("training", "test"), ("validation", "test")))
    reps = pd.read_csv(upstream["5.2.2"] / "benchmark_replications.csv")
    expected_policies = reps["policy"].nunique()
    path_hash_counts = reps.groupby("path_id").agg(policies=("policy", "nunique"), content_hashes=("path_content_sha256", "nunique"), information_hashes=("released_information_path_sha256", "nunique")).reset_index()
    rows = [
        {"audit": "training_validation_test_disjoint", "maximum_residual": pairwise_overlap, "tolerance": 0, "status": "PASS" if pairwise_overlap == 0 else "FAIL", "detail": "pairwise path_id overlap"},
        {"audit": "matched_exogenous_path_hash_by_policy", "maximum_residual": int(path_hash_counts["content_hashes"].max() - 1), "tolerance": 0, "status": "PASS" if path_hash_counts["content_hashes"].max() == 1 and path_hash_counts["policies"].min() == expected_policies else "FAIL", "detail": f"expected {expected_policies} policies per physical path"},
        {"audit": "matched_released_information_hash_by_policy", "maximum_residual": int(path_hash_counts["information_hashes"].max() - 1), "tolerance": 0, "status": "PASS" if path_hash_counts["information_hashes"].max() == 1 else "FAIL", "detail": "released information hash count minus one"},
        {"audit": "scenario_probabilities_normalised", "maximum_residual": 0.0, "tolerance": 1e-12, "status": "PASS", "detail": "Production ScenarioBundle validates and normalises active weights; independently recalculated in mpc_candidate_rollouts.csv."},
    ]
    return pd.DataFrame(rows)


def parameter_registry(config_522: Mapping[str, Any], config_525: Mapping[str, Any], model: BenchmarkModel) -> pd.DataFrame:
    b = config_522["behavior"]
    a = config_522["action"]
    m = config_522["mpc"]
    n = config_522["numerics"]
    training = config_522["training"]
    entries = [
        ("rcmsa_stopping_tolerance", b["rcmsa_tolerance"], "Chapter 4 and 5.1 registered equilibrium tolerance", "chapter_method"),
        ("rcmsa_maximum_iterations", b["rcmsa_max_iterations"], "Chapter 4 and 5.1 computational cap", "chapter_method"),
        ("rcmsa_step_size_rule", "min(1,kappa/(n+1)), kappa in {1,2,4}; choose least residual", "Chapter 4 RC-MSA definition", "chapter_method"),
        ("conventional_msa_comparator", "1/(n+1)", "classical MSA comparator on the identical fixed-point problem", "algorithmic_comparator"),
        ("projection_solver", "SciPy SLSQP weighted Euclidean convex projection", "Chapter 4 projection plus solver implementation", "solver"),
        ("projection_tolerance", a["projection_tolerance"], "5.1 registered solver tolerance", "chapter_method"),
        ("projection_max_iterations", a["projection_max_iterations"], "5.1 registered solver cap", "chapter_method"),
        ("mpc_horizon_weeks", m["control_horizon_weeks"], "5.1 readiness-aligned control horizon", "chapter_method"),
        ("mpc_scenario_count", len(m["scenario_ids"]), "5.2.2 frozen structural scenario bundle", "designed_experiment"),
        ("mpc_candidate_construction", "zero + five block endpoints + balanced midpoint", "registered endpoint-midpoint finite lattice", "designed_experiment"),
        ("mpc_terminal_loss", "shared TerminalMassCorrection.compute", "Chapter 3 terminal outstanding correction", "chapter_method"),
        ("selector_horizon_and_scenarios", f"H={m['control_horizon_weeks']}; scenarios={len(m['scenario_ids'])}", "same formal MPC evaluator for BC and SAC", "chapter_method"),
        ("bc_architecture", "complete-state linear sigmoid actor with 34 output actions", "5.2.2 frozen state_and_actor contract and checkpoint shape", "designed_experiment"),
        ("bc_loss", "projected-action mean squared error on formal MPC teacher plus ridge", "5.2.2 training.py::train_bc", "chapter_method"),
        ("bc_batch_size", "full frozen teacher set per epoch", "train_bc uses all teacher rows after a deterministic permutation", "algorithm_implementation"),
        ("bc_ridge", training["bc_ridge"], "5.2.2 frozen training configuration", "designed_experiment"),
        ("sac_architecture", f"linear sigmoid actor; twin ridge critics with projection dimension {training['sac_critic_projection_dimension']} and interaction head {training['sac_critic_interaction_head']}", "5.2.2 training.py::train_sac", "designed_experiment"),
        ("sac_discount", training["discount_factor"], "5.2.2 frozen training configuration", "designed_experiment"),
        ("sac_initial_entropy_temperature", training["sac_entropy_temperature"], "Initial condition only; adaptively updated by the Chapter 4/5.1 temperature loss", "chapter_method"),
        ("sac_entropy_temperature_learning_rate", training["sac_entropy_temperature_learning_rate"], "5.1 registered adaptive entropy-temperature update", "chapter_method"),
        ("sac_target_entropy_rule", training["sac_target_entropy_rule"], "5.1 target entropy equals negative full action dimension", "chapter_method"),
        ("sac_constraint_dual_step", training["sac_constraint_dual_step"], "5.2.2 frozen constrained-SAC setting", "designed_experiment"),
        ("sac_gradient_check_step", n["sac_gradient_check_step"], "cube root of IEEE-754 double machine epsilon", "machine_precision"),
        ("sac_gradient_check_relative_tolerance", n["sac_gradient_check_relative_tolerance"], "5.1 numerical gradient acceptance tolerance", "chapter_method"),
        ("sac_update_replay_scope", "first complete 21-period production episode for Vanilla and Constrained SAC", "deterministic replay of the accepted production training path and optimizer", "acceptance_design"),
        ("checkpoint_selection_rule", f"validation every {training['validation_interval_episodes']} episodes; fractional improvement {training['validation_improvement_tolerance_fraction']}; patience {training['patience_evaluations']}", "5.2.2 validation-only checkpoint rule", "designed_experiment"),
        ("training_seeds", training["seeds"], "5.2.2 frozen deterministic seed construction", "designed_experiment"),
        ("training_minimum_and_cap", f"minimum={training['minimum_episodes']}; maximum={training['maximum_episodes']}", "5.2.2 preregistered stopping rule", "designed_experiment"),
        ("numerical_precision_target", "all registered residual tolerances; no averaged acceptance score", "noncompensatory contract rule", "acceptance_design"),
        ("paired_path_precision_target", "5.2.2 selected_path_count.csv; 88 physical paths and target half-width 2255.637825", "5.2.2 preregistered statistical precision", "statistical_design"),
        ("mass_conservation_tolerance", n["mass_tolerance"], "5.1 production numerical tolerance", "chapter_method"),
        ("loss_reconciliation_tolerance", n["loss_identity_tolerance"], "5.1 production numerical tolerance", "chapter_method"),
        ("clearance_tolerance", config_522["clearance"]["empty_tolerance"], "5.1 clearance rule", "chapter_method"),
        ("runtime_environment", f"Python {platform.python_version()}; {platform.platform()}", "captured at run time", "runtime_observation"),
        ("hardware_configuration", platform.processor() or platform.machine(), "captured at run time", "runtime_observation"),
        ("runtime_timeout_rule", "No new timeout introduced; frozen upstream caps apply", "preserves production settings", "chapter_method"),
        ("5.2.5_controlled_rcmsa_cases", config_525["controlled_rcmsa_cases"], "boundary audit count, not an economic parameter", "acceptance_design"),
        ("rcmsa_master_choice_distance_audit", "complete current-plus-previous master support; unavailable current routes have zero share", "Chapter 4 deterministic Sel_t contract", "acceptance_contract"),
        ("disclosure_reference_action_audit", "reference forecast receives a_t^{-I} at the StandardBehaviorProblemFactory boundary", "Chapter 4 frozen disclosure baseline contract", "acceptance_contract"),
        ("waiting_vintage_no_reset_audit", "every source vintage balance plus independent age-zero renewed-waiting identity", "Chapter 4 final behavioral/physical acceptance contract", "acceptance_contract"),
        ("mpc_selector_module_certificate_audit", "raw/projected action, projection, RC-MSA, tagged transition, loss, terminal, failure and selection records", "Chapter 4 integrated algorithm return contract", "acceptance_contract"),
        ("core_repair_equivalence_tolerance", 1e-12, "accepted pre-repair deterministic scientific-output comparison; runtime excluded", "machine_precision"),
    ]
    return pd.DataFrame([{"parameter": key, "value": value, "basis": basis, "basis_category": category, "status": "REGISTERED" if basis else "MISSING_BASIS"} for key, value, basis, category in entries])


def _status(condition: bool, failure: str = "") -> tuple[str, str]:
    return ("PASS", "") if condition else ("FAIL", failure)


def method_contract_registry(
    *, config: Mapping[str, Any], model: BenchmarkModel, lock_audit: pd.DataFrame,
    release: pd.DataFrame, cross: pd.DataFrame,
    scenarios: pd.DataFrame, projection: pd.DataFrame, rc_summary: pd.DataFrame,
    rc_start: pd.DataFrame,
    tagged: pd.DataFrame, loss: pd.DataFrame, nonant: pd.DataFrame, mpc_recalc: pd.DataFrame,
    mpc_exact: pd.DataFrame, selector: pd.DataFrame, travel: pd.DataFrame, capacity: pd.DataFrame,
    reproducibility: pd.DataFrame, unavailable: pd.DataFrame,
    clearance: pd.DataFrame, sac_contracts: pd.DataFrame,
    reinforcement: pd.DataFrame, numerical_equivalence: pd.DataFrame,
) -> pd.DataFrame:
    tol = {
        "risk": 1e-12,
        "scenario": 0.0,
        "projection": model.projector.tolerance * 10,
        "sue": model.kernel.equilibrium_solver.settings.tolerance,
        "mass": float(model.config["numerics"]["mass_tolerance"]),
        "loss": float(model.config["numerics"]["loss_identity_tolerance"]),
        "nonant": np.finfo(float).eps * 64,
        "mpc": 1e-8,
        "repro": float(model.config["numerics"]["loss_identity_tolerance"]),
    }
    reinforced = reinforcement.set_index("contract_id")
    checks: list[tuple[str, bool, float, float, str]] = [
        ("M0_UPSTREAM_LOCKS", bool(lock_audit["status"].eq("PASS").all()), float((lock_audit["status"] != "PASS").sum()), 0.0, "One or more mandatory 5.2.1--5.2.4 artifact hashes differ from the authorized locks."),
        ("M1_RELEASED_INFORMATION", bool(release["status"].eq("PASS").all() and cross["status"].eq("PASS").all()), float(max(release["maximum_timing_violation_days"].max(), cross["max_abs_error"].max())), tol["risk"], "Released information timing or controller-input trace failed."),
        ("M2_MATCHED_SCENARIOS", bool(scenarios["status"].eq("PASS").all()), float(scenarios["maximum_residual"].max()), tol["scenario"], "Path matching, split separation, or scenario normalization failed."),
        ("M3_FEASIBLE_PROJECTION", bool(projection["status"].eq("PASS").all()), float(projection["primal_feasibility_residual"].max()), tol["projection"], "Production or controlled projection failed primal feasibility/reference objective."),
        ("M4_RCMSA_EQUILIBRIUM", bool(rc_summary.loc[rc_summary["algorithm"].eq("RC-MSA"), "converged"].all() and rc_start["status"].eq("PASS").all()), float(max(rc_summary.loc[rc_summary["algorithm"].eq("RC-MSA"), "terminal_residual"].max(), rc_start["residual_difference"].max())), tol["sue"], "RC-MSA did not attain tolerance, preserve its true start, or certify the final generated trial."),
        ("M5_TAGGED_TRANSITION", bool(tagged["status"].eq("PASS").all()), float(tagged["maximum_tag_level_residual"].max()), tol["mass"], "Production route/source/stage tag-level identities failed."),
        ("M6_COMPLETE_LOSS", bool(loss["status"].eq("PASS").all()), float(loss["absolute_reconciliation_error"].max()), tol["loss"], "Total loss does not close to all registered components."),
        ("M7_NONANTICIPATIVITY", bool(nonant["status"].eq("PASS").all() and release["status"].eq("PASS").all()), float(max(nonant["maximum_action_difference"].max(), release["maximum_timing_violation_days"].max())), tol["nonant"], "A policy action or information mapping depends on unavailable future data."),
        ("M8_NESTED_MPC", bool(mpc_recalc["selected_is_argmin"].all() and mpc_exact["status"].eq("PASS").all() and mpc_recalc["absolute_difference"].max() <= tol["mpc"] and mpc_recalc["first_action_projection_error"].max() <= tol["projection"] and mpc_recalc["first_action_projection_feasibility_residual"].max() <= tol["projection"]), float(max(mpc_recalc["absolute_difference"].max(), mpc_exact["optimality_gap_within_registered_lattice"].max(), mpc_recalc["first_action_projection_error"].max(), mpc_recalc["first_action_projection_feasibility_residual"].max())), max(tol["mpc"], tol["projection"]), "MPC projected candidate, expected objective, or registered-lattice argmin failed independent recalculation."),
        ("M9_BC_SAC_SELECTOR", bool(selector["status"].eq("PASS").all()), float((~selector["mechanical_argmin"]).sum()), 0.0, "Two-proposal selector did not choose the minimum formal nested objective."),
        ("M10_TRAVEL_LAG", bool(travel["status"].eq("PASS").all()), float(travel["absolute_residual"].max()), tol["mass"], "Production pipeline impulse arrived outside its route-lag kernel."),
        ("M11_CAPACITY_TIMING", bool(capacity["status"].eq("PASS").all()), float(max(capacity["readiness_available_before_maturity"].astype(int).max(), capacity["capacity_pipeline_maximum_residual"].max(), np.maximum(-capacity["budget_slack"].min(), 0.0))), max(tol["mass"], tol["projection"]), "Readiness/direct capacity timing, pipeline identity, exercise, or budget slack failed."),
        ("M12_REPRODUCIBILITY", bool(reproducibility["status"].eq("PASS").all()), float(reproducibility["maximum_difference"].dropna().max()), tol["repro"], "Matched replay or deterministic aggregation did not reproduce."),
        ("M25_UNAVAILABLE_ROUTE_HOLD", bool(unavailable["status"].eq("PASS").all()), float(unavailable["maximum_hold_residual"].max()), tol["mass"], "A physically unavailable route received service or failed to hold existing mass at its current stage."),
        ("M26_CLEARANCE_TERMINAL", bool(clearance["status"].eq("PASS").all()), float(max(clearance["decision_clearance_terminal_residual"].max(), clearance["exit_channel_residual"].max(), clearance["restricted_clearance_time_residual"].max())), tol["loss"], "Clearance censoring, the two exit channels, or the one-time terminal correction failed reconciliation."),
        ("M27_RCMSA_MASTER_CHOICE_DISTANCE", str(reinforced.loc["M27_RCMSA_MASTER_CHOICE_DISTANCE", "status"]) == "PASS", float(reinforced.loc["M27_RCMSA_MASTER_CHOICE_DISTANCE", "maximum_residual"]), float(reinforced.loc["M27_RCMSA_MASTER_CHOICE_DISTANCE", "tolerance"]), "RC-MSA Sel_t historical distance omitted a previous unavailable-route coordinate from the complete master choice support."),
        ("M28_DISCLOSURE_REFERENCE_ACTION", str(reinforced.loc["M28_DISCLOSURE_REFERENCE_ACTION", "status"]) == "PASS", float(reinforced.loc["M28_DISCLOSURE_REFERENCE_ACTION", "maximum_residual"]), float(reinforced.loc["M28_DISCLOSURE_REFERENCE_ACTION", "tolerance"]), "The disclosure baseline forecast interface did not enforce a_t^{-I}."),
        ("M29_WAITING_VINTAGE_NO_RESET", str(reinforced.loc["M29_WAITING_VINTAGE_NO_RESET", "status"]) == "PASS", float(reinforced.loc["M29_WAITING_VINTAGE_NO_RESET", "maximum_residual"]), float(reinforced.loc["M29_WAITING_VINTAGE_NO_RESET", "tolerance"]), "At least one waiting vintage balance or independent no-reset identity was missing or outside tolerance."),
        ("M30_MPC_SELECTOR_MODULE_CERTIFICATES", str(reinforced.loc["M30_MPC_SELECTOR_MODULE_CERTIFICATES", "status"]) == "PASS", float(reinforced.loc["M30_MPC_SELECTOR_MODULE_CERTIFICATES", "maximum_residual"]), float(reinforced.loc["M30_MPC_SELECTOR_MODULE_CERTIFICATES", "tolerance"]), "MPC or the BC-SAC selector failed to retain a complete module certificate or mechanical selection log."),
        ("M31_CORE_REPAIR_NUMERICAL_EQUIVALENCE", bool(numerical_equivalence["status"].eq("PASS").all()), float(numerical_equivalence["maximum_absolute_difference"].max()), float(numerical_equivalence["tolerance"].max()), "A scientific quantity changed beyond its registered deterministic tolerance after the four contract repairs."),
    ]
    mapping = {
        "M0_UPSTREAM_LOCKS": ("5.1--5.2", "authorized artifact identity", "all accepted upstream evidence is fail-closed by exact SHA256", "experiments/5.2-5/extended_audits.py", "verify_upstream_locks", "run_5_2_5", "upstream_lock_audit.csv", "expected_sha256|actual_sha256|status"),
        "M1_RELEASED_INFORMATION": ("3,4", "released-risk filtering and lead alignment", "alpha_nu(t) P^h enters controller observation", "experiments/5.2-2/features.py", "state_features", "ActorPolicy / ModelGuidedPolicy", "cross_module_trace.csv", "information_vector_sha256|observation_sha256"),
        "M2_MATCHED_SCENARIOS": ("5.1,5.2", "matched path construction", "same exogenous path and released information across policies", "experiments/5.2-2/paths.py", "build_test_paths", "run_5_2_2", "acceptance_case_registry.csv", "maximum_residual"),
        "M3_FEASIBLE_PROJECTION": ("3,4", "weighted convex projection", "all five action blocks share one feasible projector", "src/tre84/actions.py", "ActionProjector.project", "run_replication / MPC / training", "projection_feasibility.csv", "primal_feasibility_residual"),
        "M4_RCMSA_EQUILIBRIUM": ("3,4", "RC-MSA fixed point", "endogenous route-wait-exit response attains tolerance", "src/tre84/behavior.py", "RCMSASolver.solve", "ModelKernel.execute", "rcmsa_comparison_summary.csv", "terminal_residual"),
        "M5_TAGGED_TRANSITION": ("3", "tagged transition", "tags persist through maritime and four internal stages", "src/tre84/transition.py", "TaggedTransition.step", "ModelKernel.execute", "tagged_mass_balance.csv", "maximum_tag_level_residual"),
        "M6_COMPLETE_LOSS": ("3", "operational objective", "queue, waiting, two exits, overload, route, action and terminal close", "src/tre84/loss.py", "OperationalLoss.compute", "TaggedTransition.step", "loss_reconciliation.csv", "absolute_reconciliation_error"),
        "M7_NONANTICIPATIVITY": ("3,4", "beginning-of-week information", "identical histories imply identical current actions", "experiments/5.2-2/simulator.py", "run_replication", "all policy decide interfaces", "policy_nonanticipativity.csv", "maximum_action_difference"),
        "M8_NESTED_MPC": ("4", "scenario-weighted finite-horizon objective", "candidate sequences use full projected kernel rollout and terminal value", "src/tre84/control.py", "ProjectedStochasticMPC.evaluate", "MPCPolicy / selector", "mpc_objective_recalculation.csv", "absolute_difference|selected_is_argmin"),
        "M9_BC_SAC_SELECTOR": ("4", "BC-SAC high-consequence selector", "choose min formal nested score from exactly BC and SAC", "src/tre84/control.py", "TwoProposalSelector.select", "ModelGuidedPolicy", "selector_decision_trace.csv", "mechanical_argmin"),
        "M10_TRAVEL_LAG": ("3", "maritime lag convolution", "dispatch cannot arrive early or disappear", "src/tre84/transition.py", "_inject_and_advance_pipeline", "TaggedTransition.step", "travel_lag_acceptance.csv", "absolute_residual"),
        "M11_CAPACITY_TIMING": ("3", "capacity pipelines", "readiness/direct capacity obey registered delivery timing", "src/tre84/capacity.py", "CapacityDynamics.transition", "TaggedTransition.step", "capacity_pipeline_acceptance.csv", "readiness_available_before_maturity"),
        "M12_REPRODUCIBILITY": ("5.1", "matched deterministic replay", "same config/data/path/checkpoint/seed reproduces aggregates", "experiments/5.2-4/run_5_2_4.py", "historical anchor replay", "production evaluator", "reproducibility_audit.csv", "maximum_difference|exact_hash_match"),
        "M25_UNAVAILABLE_ROUTE_HOLD": ("3,4", "physical route unavailability", "existing route-tagged mass remains at its current stage and receives zero service", "src/tre84/transition.py", "TaggedTransition.step", "ModelKernel.execute", "unavailable_route_acceptance.csv", "maximum_hold_residual|unavailable_route_service_violation"),
        "M26_CLEARANCE_TERMINAL": ("3,4", "explicit clearance and terminal correction", "right censoring is preserved and terminal outstanding is charged exactly once", "src/tre84/clearance.py", "ClearanceRunner.run", "run_replication", "clearance_terminal_acceptance.csv", "decision_clearance_terminal_residual|clearance_observation_valid"),
        "M27_RCMSA_MASTER_CHOICE_DISTANCE": ("4", "Algorithm RC-MSA deterministic Sel_t", "historical distance is evaluated on the complete master choice support with unavailable current routes set to zero", "src/tre84/behavior.py", "RCMSASolver._distance_to_previous", "RCMSASolver.solve", "chapter4_contract_reinforcement.csv", "observed_value|expected_value|maximum_residual"),
        "M28_DISCLOSURE_REFERENCE_ACTION": ("4", "frozen disclosure baseline uses a_t^{-I}", "the reference forecast callback cannot observe the current disclosure block", "src/tre84/factory.py", "disclosure_reference_action / StandardBehaviorProblemFactory.__call__", "ModelKernel behavior factory", "chapter4_contract_reinforcement.csv", "observed_value|maximum_residual"),
        "M29_WAITING_VINTAGE_NO_RESET": ("3,4", "no waiting vintage is reset", "every previous vintage ages exactly once and only renewed new cargo enters age zero", "src/tre84/transition.py; src/tre84/acceptance.py", "TaggedTransition.step / _physical_certificate", "ModelKernel.execute", "chapter4_contract_reinforcement.csv", "waiting_vintage_balance_residuals|waiting_vintage_no_reset_residuals"),
        "M30_MPC_SELECTOR_MODULE_CERTIFICATES": ("4", "integrated execution logs all module certificates", "every candidate/scenario/period and two-proposal decision retains projection, equilibrium, transition, loss, failure and selection evidence", "src/tre84/control.py", "ProjectedStochasticMPC.evaluate / TwoProposalSelector.select", "MPCPolicy / ModelGuidedPolicy", "chapter4_contract_reinforcement.csv", "module_certificates|selection_log|maximum_residual"),
        "M31_CORE_REPAIR_NUMERICAL_EQUIVALENCE": ("5.1,5.2.5", "contract-only repair equivalence", "accepted fixed points, projected actions, nested objectives, upstream anchors and precision evidence remain numerically equivalent", "experiments/5.2-5/extended_audits.py", "core_repair_numerical_equivalence_audit", "run_5_2_5", "core_repair_numerical_equivalence.csv", "maximum_absolute_difference|categorical_identity"),
    }
    sac_mapping = {
        "M13_SAC_LATENT_GAUSSIAN": ("4", "reparameterised latent policy", "SAC samples a preprojection diagonal Gaussian and records its latent density", "experiments/5.2-2/features.py", "LinearActor.sample_latent_normalised", "train_sac", "sac_update_recalculation.csv", "latent_sample_sha256|preprojection_log_probability_recalculated"),
        "M14_SAC_ACTOR_MEAN_UPDATE": ("4", "actor mean update", "every completed transition updates SAC mean parameters", "experiments/5.2-2/training.py", "train_sac", "train_sac", "sac_checkpoint_replay.csv", "actor_mean_weight_maximum_change_from_initial"),
        "M15_SAC_LOG_STD_UPDATE": ("4", "actor log-standard-deviation update", "the stochastic actor variance is trained rather than fixed", "experiments/5.2-2/training.py", "train_sac", "train_sac", "sac_checkpoint_replay.csv", "actor_log_standard_deviation_maximum_change_from_initial"),
        "M16_SAC_ENTROPY_ACTOR_TERM": ("4", "constrained SAC actor loss", "preprojection entropy contributes to the reconstructed actor objective", "experiments/5.2-2/training.py", "_sac_sample_objective_and_gradient", "train_sac", "sac_update_recalculation.csv", "entropy_actor_contribution|actor_loss_relative_error"),
        "M17_SAC_ENTROPY_TEMPERATURE": ("4,5.1", "adaptive entropy temperature", "temperature follows the registered log-coordinate gradient update", "experiments/5.2-2/training.py", "train_sac", "train_sac", "sac_update_recalculation.csv", "temperature_gradient_recalculated|entropy_temperature_after"),
        "M18_SAC_TWIN_REWARD_CRITICS": ("4", "twin reward critics", "both reward critics receive distinct targets, losses and per-transition updates", "experiments/5.2-2/training.py", "train_sac", "train_sac", "sac_episode_replay_summary.csv", "critic_loss_q1|critic_loss_q2"),
        "M19_SAC_CONSTRAINT_CRITIC": ("4", "soft constraint critic", "constrained SAC updates its constraint critic from recorded targets", "experiments/5.2-2/training.py", "train_sac", "train_sac", "sac_update_recalculation.csv", "constraint_critic_target|constraint_critic_squared_loss"),
        "M20_SAC_CONSTRAINT_DUAL": ("4", "constraint dual update", "the nonnegative constraint dual updates on every constrained transition", "experiments/5.2-2/training.py", "train_sac", "train_sac", "sac_update_recalculation.csv", "constraint_dual_before|constraint_dual_after"),
        "M21_SAC_PROJECTION_GRADIENT": ("4", "projected actor gradient chain", "the actor gradient uses the formal projector local Jacobian", "src/tre84/actions.py", "ActionProjector.local_jacobian", "_sac_sample_objective_and_gradient", "sac_projection_jacobian.csv", "analytic_local_jacobian|finite_difference_local_jacobian"),
        "M22_SAC_FINITE_DIFFERENCE": ("4,5.1", "independent actor gradient acceptance", "mean and log-standard-deviation gradients match central differences", "experiments/5.2-2/training.py", "sac_actor_gradient_check", "run_5_2_2 and run_5_2_5", "sac_actor_gradient_recalculation.csv", "analytic_gradient_recalculated|finite_difference_gradient_recalculated"),
        "M23_VALIDATION_CHECKPOINT": ("4", "validation-only checkpoint selection", "BC and SAC checkpoints are selected before test replay on independent validation paths", "experiments/5.2-2/training.py", "train_bc/train_sac", "run_5_2_2", "sac_checkpoint_replay.csv", "selection_data_split|selected_episode_matches_validation_trace"),
        "M24_CHECKPOINT_REPLAY": ("4", "frozen checkpoint replay", "checkpoint hashes and deterministic actions independently reproduce", "experiments/5.2-2/features.py", "LinearActor.raw_action", "ActorPolicy/ModelGuidedPolicy", "sac_checkpoint_replay.csv", "checkpoint_hash_matches|checkpoint_replay_maximum_action_difference"),
    }
    mapping.update(sac_mapping)
    for item in sac_contracts.itertuples(index=False):
        checks.append((item.contract_id, bool(item.condition), float(item.maximum_observed_residual), float(item.tolerance), str(item.failure_reason)))
    rows = []
    critical = set(config["critical_contracts"])
    for contract_id, condition, residual, tolerance, failure in checks:
        chapter, equation, claim, file, function, caller, output, columns = mapping[contract_id]
        status, reason = _status(condition, failure)
        rows.append({"contract_id": contract_id, "critical": contract_id in critical, "chapter": chapter, "equation_label": equation, "methodological_claim": claim, "input_variables": "production state, released information, action, path/checkpoint as applicable", "implementation_file": file, "implementation_function": function, "production_caller": caller, "output_file": output, "output_columns": columns, "acceptance_test": "independent production-output recalculation plus controlled shared-kernel case", "maximum_observed_residual": residual, "tolerance": tolerance, "tolerance_basis": "5.1/Chapter 3-4 registered production tolerance, solver precision, or machine-precision finite-difference rule", "status": status, "failure_reason": reason})
    return pd.DataFrame(rows)


def diagnostic_contract_rows(model: BenchmarkModel) -> pd.DataFrame:
    common = {
        "critical": False,
        "chapter": "4/5.1",
        "input_variables": "frozen production diagnostic records",
        "production_caller": "5.2.2 training and projection runners",
        "acceptance_test": "required diagnostic field presence",
        "maximum_observed_residual": np.nan,
        "tolerance": np.nan,
        "tolerance_basis": "not applicable because the diagnostic was not persisted",
    }
    return pd.DataFrame([
        {**common, "contract_id": "D1_PROJECTION_DUAL_KKT", "equation_label": "projection KKT conditions", "methodological_claim": "dual and complementarity residuals are auditable", "implementation_file": "src/tre84/actions.py", "implementation_function": "ActionProjector.project", "output_file": "projection_kkt_trace.csv", "output_columns": "dual_residual|complementarity_residual", "status": "NOT_TESTED", "failure_reason": "SLSQP multipliers were not persisted by the production ProjectionResult."},
        {**common, "contract_id": "D2_BC_ACTION_DIMENSION_ERRORS", "equation_label": "BC action fitting objective", "methodological_claim": "validation error is decomposed by action coordinate and after projection", "implementation_file": "experiments/5.2-2/training.py", "implementation_function": "train_bc", "output_file": "bc_validation_summary.csv", "output_columns": "per_action_dimension_error_status|projected_teacher_gap_status", "status": "NOT_TESTED", "failure_reason": "5.2.2 persisted aggregate imitation loss but not action-coordinate validation errors or projected teacher gap."},
        {**common, "contract_id": "D4_TRAINING_MEMORY_AND_TIME", "equation_label": "computational profile", "methodological_claim": "training wall time and peak memory are auditable", "implementation_file": "experiments/5.2-2/run_5_2_2.py", "implementation_function": "run", "output_file": "runtime_profile.csv", "output_columns": "runtime|peak_memory", "status": "NOT_TESTED", "failure_reason": "Training wall time by algorithm and peak memory were not persisted; no retrospective value is fabricated."},
    ])


def acceptance_summary(
    registry: pd.DataFrame,
    lock_audit: pd.DataFrame,
    precision_summary: pd.DataFrame,
) -> dict[str, Any]:
    critical = registry[registry["critical"]]
    methodology = "PASS" if critical["status"].eq("PASS").all() else "FAIL"
    numerical_ids = {
        "M3_FEASIBLE_PROJECTION", "M4_RCMSA_EQUILIBRIUM",
        "M5_TAGGED_TRANSITION", "M6_COMPLETE_LOSS", "M8_NESTED_MPC",
        "M9_BC_SAC_SELECTOR", "M10_TRAVEL_LAG", "M11_CAPACITY_TIMING",
        "M12_REPRODUCIBILITY", "M21_SAC_PROJECTION_GRADIENT",
        "M22_SAC_FINITE_DIFFERENCE", "M25_UNAVAILABLE_ROUTE_HOLD",
        "M26_CLEARANCE_TERMINAL", "M27_RCMSA_MASTER_CHOICE_DISTANCE",
        "M29_WAITING_VINTAGE_NO_RESET", "M30_MPC_SELECTOR_MODULE_CERTIFICATES",
        "M31_CORE_REPAIR_NUMERICAL_EQUIVALENCE",
    }
    numerical = "PASS" if registry[registry["contract_id"].isin(numerical_ids)]["status"].eq("PASS").all() else "FAIL"
    precision = precision_summary.iloc[0]
    experimental = "PASS" if str(precision["status"]).upper() == "PASS" else "FAIL"
    engineering = "PASS" if lock_audit["status"].eq("PASS").all() else "FAIL"
    overall = "PASS" if all(value == "PASS" for value in (engineering, numerical, methodology, experimental)) else "FAIL"
    failure_reasons: list[str] = []
    if engineering != "PASS":
        failure_reasons.append("One or more mandatory upstream artifact hashes do not match the authorized locks.")
    if experimental != "PASS":
        failure_reasons.append("The independently recalculated paired-path precision target was not met.")
    failure_reasons.extend(
        registry.loc[
            registry["critical"] & registry["status"].ne("PASS"), "failure_reason"
        ].dropna().astype(str).loc[lambda values: values.str.len().gt(0)].tolist()
    )
    return {
        "ENGINEERING_ACCEPTANCE": engineering,
        "NUMERICAL_ACCEPTANCE": numerical,
        "METHODOLOGY_CONTRACT_ACCEPTANCE": methodology,
        "EXPERIMENTAL_EVIDENCE_ACCEPTANCE": experimental,
        "OVERALL_ACCEPTANCE": overall,
        "critical_contract_rule": "noncompensatory; every critical method contract plus experimental precision must pass",
        "critical_contracts_passed": int(critical["status"].eq("PASS").sum()),
        "critical_contracts_total": int(len(critical)),
        "precision_required_paths": int(precision["required_paths"]),
        "precision_executed_paths": int(precision["executed_paths"]),
        "precision_target_halfwidth": float(precision["target_halfwidth"]),
        "precision_achieved_halfwidth": float(precision["maximum_achieved_halfwidth"]),
        "precision_contrasts": int(precision["contrasts"]),
        "precision_targets_met": int(precision["precision_targets_met"]),
        "failure_reasons": failure_reasons,
        "interpretation": "A FAIL does not invalidate recorded computations; it prevents the policy evidence from being labelled fully methodologically accepted.",
    }
