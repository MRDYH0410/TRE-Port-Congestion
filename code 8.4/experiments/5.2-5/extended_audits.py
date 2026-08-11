"""Extended production-chain audits required by the rebuilt 5.2.5 gate."""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats


EXPERIMENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_DIR.parents[1]
EXP522 = CODE_ROOT / "experiments" / "5.2-2"
SRC = CODE_ROOT / "src"
for item in (str(SRC), str(EXP522)):
    if item not in sys.path:
        sys.path.insert(0, item)

from tre84.actions import Action, Block  # noqa: E402
from tre84.behavior import RCMSASettings, RCMSASolver  # noqa: E402
from tre84.control import TwoProposalSelector  # noqa: E402
from tre84.factory import (  # noqa: E402
    StandardBehaviorProblemFactory,
    disclosure_reference_action,
)
from tre84.keys import SourceKey  # noqa: E402
from tre84.state import CapacityState, Tag  # noqa: E402

from policies import _candidate_profiles, build_mpc  # noqa: E402
from preparation import _scenario_bundle  # noqa: E402
from simulator import build_realization  # noqa: E402


def _load_statistics_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "tre84_exp522_statistics", EXP522 / "statistics.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the accepted 5.2.2 statistics module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "pass", "passed"})


def chapter4_contract_reinforcement_audit(
    model: Any, event: pd.DataFrame
) -> pd.DataFrame:
    """Independently exercise the four strengthened Chapter 4 contracts."""

    rows: list[dict[str, Any]] = []
    tolerance = float(model.config["numerics"]["mass_tolerance"])

    source = SourceKey(str(model.config["cargo_class"]), None)

    class _MasterProblem:
        decision = SimpleNamespace(masses={source: 1.0})
        sources = (source,)

        @staticmethod
        def choices(_source: SourceKey) -> tuple[str, ...]:
            return ("available", "__WAIT__", "__EXIT__")

    solver = RCMSASolver(RCMSASettings(1e-8, 10, 1e-8))
    problem = _MasterProblem()
    slices, _ = solver._layout(problem)
    observed_distance = solver._distance_to_previous(
        problem,
        slices,
        np.asarray([1.0, 0.0, 0.0]),
        {
            source: {
                "available": 0.0,
                "now_unavailable": 1.0,
                "__WAIT__": 0.0,
                "__EXIT__": 0.0,
            }
        },
    )
    rows.append(
        {
            "contract_id": "M27_RCMSA_MASTER_CHOICE_DISTANCE",
            "audit_case": "unavailable_previous_route_is_zero_on_current_master_simplex",
            "observed_value": observed_distance,
            "expected_value": 2.0,
            "maximum_residual": abs(observed_distance - 2.0),
            "tolerance": 1e-12,
            "status": "PASS" if abs(observed_distance - 2.0) <= 1e-12 else "FAIL",
            "detail": "Historical L1 distance includes both the lost unavailable-route share and the gained current-route share.",
        }
    )

    full_action = model.zero_action()
    full_action.values[model.layout.disclosure[0]] = 0.75
    full_action.values[model.layout.release[0]] = 0.25
    reference_action = disclosure_reference_action(full_action)
    interface_bound = "disclosure_reference_action(action)" in inspect.getsource(
        StandardBehaviorProblemFactory.__call__
    )
    disclosure_leak = max(
        (abs(value) for key, value in reference_action.values.items() if key.block is Block.DISCLOSURE),
        default=0.0,
    )
    noninformation_error = max(
        (
            abs(reference_action.value(key) - full_action.value(key))
            for key in full_action.values
            if key.block is not Block.DISCLOSURE
        ),
        default=0.0,
    )
    reference_residual = max(disclosure_leak, noninformation_error, 0.0 if interface_bound else 1.0)
    rows.append(
        {
            "contract_id": "M28_DISCLOSURE_REFERENCE_ACTION",
            "audit_case": "factory_boundary_enforces_a_t_minus_I",
            "observed_value": disclosure_leak,
            "expected_value": 0.0,
            "maximum_residual": reference_residual,
            "tolerance": 0.0,
            "status": "PASS" if reference_residual == 0.0 else "FAIL",
            "detail": "The disclosure baseline callback receives no disclosure coordinate while every non-information action coordinate is preserved.",
        }
    )

    row = event.iloc[0].to_dict()
    state = model.initial_state(row)
    projection = model.projector.project(model.zero_action(), state)
    realization = build_realization(model=model, state=state, row=row)
    kernel_result = model.kernel.execute(
        state=state,
        action=projection.action,
        realization=realization,
        projection=projection,
    )
    audit = kernel_result.transition.audit
    balance_values = [
        abs(value) for _, _, value in audit.waiting_vintage_balance_residuals
    ]
    no_reset_values = [
        abs(value) for _, value in audit.waiting_vintage_no_reset_residuals
    ]
    vintage_coverage = (
        audit.waiting_vintage_certificate_complete
        and len(balance_values) == audit.waiting_vintage_expected_balance_count
        and len(no_reset_values) == audit.waiting_vintage_expected_no_reset_count
    )
    vintage_residual = max(
        max(balance_values, default=0.0),
        max(no_reset_values, default=0.0),
        0.0 if vintage_coverage else 1.0,
    )
    rows.append(
        {
            "contract_id": "M29_WAITING_VINTAGE_NO_RESET",
            "audit_case": "production_transition_each_vintage_and_age_zero_identity",
            "observed_value": vintage_residual,
            "expected_value": 0.0,
            "maximum_residual": vintage_residual,
            "tolerance": tolerance,
            "status": "PASS" if vintage_residual <= tolerance else "FAIL",
            "detail": f"Certified {len(balance_values)} vintage balances and {len(no_reset_values)} independent no-reset identities.",
        }
    )

    state = model.initial_state(row)
    bundle = _scenario_bundle(model=model, state=state, row=row)
    candidates = _candidate_profiles(model)
    mpc = build_mpc(model)
    mpc_result = mpc.solve(state=state, bundle=bundle, candidates=candidates)
    expected_ids = tuple(candidate.candidate_id for candidate in candidates)
    selection_log_ok = (
        mpc_result.selection_log.considered_candidate_ids == expected_ids
        and mpc_result.selection_log.selected_candidate_id == mpc_result.candidate_id
    )
    certificate_residual = 0.0
    for evaluation in mpc_result.evaluations:
        if not evaluation.valid:
            continue
        expected_count = len(bundle.paths) * mpc.lookahead
        if len(evaluation.module_certificates) != expected_count:
            certificate_residual = max(certificate_residual, 1.0)
            continue
        by_scenario = {
            scenario_id: [
                certificate
                for certificate in evaluation.module_certificates
                if certificate.scenario_id == scenario_id
            ]
            for scenario_id in evaluation.scenario_ids
        }
        for scenario_index, scenario_id in enumerate(evaluation.scenario_ids):
            certificates = sorted(by_scenario[scenario_id], key=lambda item: item.offset)
            reconstructed = sum(item.period_loss.total for item in certificates) + evaluation.terminal_losses[scenario_index]
            certificate_residual = max(
                certificate_residual,
                abs(reconstructed - evaluation.path_losses[scenario_index]),
                max(
                    (
                        max(
                            item.projection.feasibility_violation,
                            item.transition_audit.unavailable_route_service_violation,
                            item.equilibrium.residual - model.kernel.equilibrium_solver.settings.tolerance,
                        )
                        for item in certificates
                    ),
                    default=0.0,
                ),
            )
    selector = TwoProposalSelector(
        mpc_evaluator=mpc,
        fallback_raw_action=lambda current_state: model.zero_action(),
        continuation=candidates[0].continuation,
    )
    proposal = selector.select(
        state=state,
        bundle=bundle,
        bc_raw_action=candidates[0].first_raw_action,
        sac_raw_action=candidates[1].first_raw_action,
    )
    proposal_log_ok = (
        proposal.selection_log.considered_candidate_ids == ("BC", "SAC")
        and proposal.selection_log.selected_candidate_id == proposal.source
        and set(proposal.evaluations) >= {"BC", "SAC"}
    )
    certificate_residual = max(
        certificate_residual,
        0.0 if selection_log_ok and proposal_log_ok else 1.0,
    )
    rows.append(
        {
            "contract_id": "M30_MPC_SELECTOR_MODULE_CERTIFICATES",
            "audit_case": "production_nested_rollout_and_two_proposal_selection_log",
            "observed_value": certificate_residual,
            "expected_value": 0.0,
            "maximum_residual": certificate_residual,
            "tolerance": max(tolerance, 1e-8),
            "status": "PASS" if certificate_residual <= max(tolerance, 1e-8) else "FAIL",
            "detail": "Every valid candidate/scenario/period retains raw and projected actions, projection, RC-MSA, transition, loss, terminal, failure and mechanical selection evidence.",
        }
    )
    return pd.DataFrame(rows)


def core_repair_numerical_equivalence_audit(
    *,
    rc_summary: pd.DataFrame,
    mpc_recalculation: pd.DataFrame,
    precision_summary: pd.DataFrame,
    reproducibility: pd.DataFrame,
    baseline_path: Path,
) -> pd.DataFrame:
    """Compare refreshed scientific quantities with the accepted pre-repair baseline."""

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    tolerance = float(baseline["tolerance"])
    rows: list[dict[str, Any]] = []

    current_rc = {
        (str(row.case_id), str(row.algorithm)): row
        for row in rc_summary.itertuples(index=False)
    }
    maximum = 0.0
    exact = True
    for case_id, algorithm, demand, terminal, iterations, converged, registered_tol, distance in baseline["rcmsa"]:
        row = current_rc.get((case_id, algorithm))
        if row is None:
            exact = False
            maximum = float("inf")
            continue
        maximum = max(
            maximum,
            abs(float(row.demand) - float(demand)),
            abs(float(row.terminal_residual) - float(terminal)),
            abs(float(row.tolerance) - float(registered_tol)),
            abs(float(row.final_solution_distance) - float(distance)),
        )
        exact = exact and int(row.iterations) == int(iterations) and bool(row.converged) is bool(converged)
    rows.append(
        {
            "comparison": "RC-MSA selected fixed-point diagnostics",
            "maximum_absolute_difference": maximum,
            "categorical_identity": exact,
            "tolerance": tolerance,
            "status": "PASS" if exact and maximum <= tolerance else "FAIL",
        }
    )

    current_mpc = {
        str(row.candidate_id): row
        for row in mpc_recalculation.itertuples(index=False)
    }
    maximum = 0.0
    exact = True
    for candidate_id, objective, selected in baseline["mpc"]:
        row = current_mpc.get(candidate_id)
        if row is None:
            exact = False
            maximum = float("inf")
            continue
        maximum = max(
            maximum,
            abs(float(row.recorded_expected_objective) - float(objective)),
            abs(float(row.recalculated_expected_objective) - float(objective)),
            abs(float(row.first_action_projection_error)),
            abs(float(row.first_action_projection_feasibility_residual)),
        )
        exact = exact and str(row.selected_candidate) == str(selected) and bool(row.selected_is_argmin)
    rows.append(
        {
            "comparison": "MPC candidate objectives, projection and selected action",
            "maximum_absolute_difference": maximum,
            "categorical_identity": exact,
            "tolerance": tolerance,
            "status": "PASS" if exact and maximum <= tolerance else "FAIL",
        }
    )

    precision = precision_summary.iloc[0]
    baseline_precision = baseline["precision"]
    maximum = max(
        abs(float(precision["target_halfwidth"]) - float(baseline_precision["target_halfwidth"])),
        abs(float(precision["maximum_achieved_halfwidth"]) - float(baseline_precision["maximum_achieved_halfwidth"])),
    )
    exact = all(
        int(precision[key]) == int(baseline_precision[key])
        for key in (
            "required_paths",
            "executed_paths",
            "contrasts",
            "precision_targets_met",
        )
    ) and bool(precision["precision_target_met"]) is bool(baseline_precision["precision_target_met"])
    rows.append(
        {
            "comparison": "88-path statistical precision evidence",
            "maximum_absolute_difference": maximum,
            "categorical_identity": exact,
            "tolerance": tolerance,
            "status": "PASS" if exact and maximum <= tolerance else "FAIL",
        }
    )

    replay_maximum = float(reproducibility["maximum_difference"].max())
    replay_tolerance = float(reproducibility["tolerance"].max())
    rows.append(
        {
            "comparison": "5.2.2--5.2.4 accepted path, checkpoint, loss and clearance anchors",
            "maximum_absolute_difference": replay_maximum,
            "categorical_identity": bool(reproducibility["status"].eq("PASS").all()),
            "tolerance": replay_tolerance,
            "status": "PASS" if reproducibility["status"].eq("PASS").all() and replay_maximum <= replay_tolerance else "FAIL",
        }
    )
    return pd.DataFrame(rows)


def verify_upstream_locks(
    config: Mapping[str, Any], upstream: Mapping[str, Path], sha256_file: Any
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for experiment, artifacts in config["upstream_artifact_locks"].items():
        for name, expected in artifacts.items():
            path = upstream[experiment] / name
            actual = sha256_file(path) if path.is_file() else "MISSING"
            matched = actual.upper() == str(expected).upper()
            rows.append(
                {
                    "experiment": experiment,
                    "artifact": name,
                    "relative_path": path.relative_to(CODE_ROOT).as_posix(),
                    "expected_sha256": str(expected).upper(),
                    "actual_sha256": actual.upper(),
                    "hash_matches": matched,
                    "status": "PASS" if matched else "BLOCKED",
                    "failure_reason": "" if matched else "Required upstream artifact hash mismatch.",
                }
            )
    return pd.DataFrame(rows)


def experimental_precision_audit(
    *, upstream: Mapping[str, Path], config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    statistics_module = _load_statistics_module()
    replications = pd.read_csv(upstream["5.2.2"] / "benchmark_replications.csv")
    path_level = statistics_module.aggregate_learning_seeds(
        replications,
        learning_policies=config["learning_policies"],
    )
    test_manifest = pd.read_csv(upstream["5.2.2"] / "test_path_manifest.csv")
    selected = pd.read_csv(upstream["5.2.2"] / "selected_path_count.csv").iloc[0]
    target = float(selected["target_halfwidth"])
    pilot_count = int(selected["pilot_paths"])
    pilot_ids = test_manifest["path_id"].iloc[:pilot_count].tolist()
    passive = path_level.loc[
        path_level["policy"].eq("Passive"),
        ["path_id", "total_operational_objective"],
    ].rename(columns={"total_operational_objective": "passive_loss"})
    passive_pilot = passive[passive["path_id"].isin(pilot_ids)]
    confidence = float(config["paths"]["confidence_level"])
    pilot_critical = float(stats.t.ppf((1.0 + confidence) / 2.0, pilot_count - 1))
    final_critical = float(stats.t.ppf((1.0 + confidence) / 2.0, len(passive) - 1))
    minimum = int(config["paths"]["minimum_final_count"])
    maximum = int(config["paths"]["maximum_final_count"])
    rows: list[dict[str, Any]] = []
    for policy in config["main_policies"]:
        if policy == "Passive":
            continue
        values = path_level.loc[
            path_level["policy"].eq(policy),
            ["path_id", "total_operational_objective"],
        ].merge(passive, on="path_id", validate="one_to_one")
        differences = (
            values["total_operational_objective"] - values["passive_loss"]
        ).to_numpy(float)
        halfwidth = float(
            final_critical * np.std(differences, ddof=1) / np.sqrt(len(differences))
        )
        pilot_values = path_level.loc[
            path_level["policy"].eq(policy)
            & path_level["path_id"].isin(pilot_ids),
            ["path_id", "total_operational_objective"],
        ].merge(passive_pilot, on="path_id", validate="one_to_one")
        pilot_differences = (
            pilot_values["total_operational_objective"]
            - pilot_values["passive_loss"]
        ).to_numpy(float)
        pilot_sd = float(np.std(pilot_differences, ddof=1))
        raw_required = (
            2
            if pilot_sd == 0.0
            else int(math.ceil((pilot_critical * pilot_sd / target) ** 2))
        )
        required = max(minimum, raw_required)
        learning = policy in set(config["learning_policies"])
        seed_counts = path_level.loc[
            path_level["policy"].eq(policy), "training_seed_count"
        ]
        seed_aggregation_valid = bool(
            seed_counts.eq(3 if learning else 1).all()
            and path_level.loc[
                path_level["policy"].eq(policy), "inference_unit"
            ].eq("physical_path").all()
        )
        status = (
            len(differences) == int(selected["executed_paths"])
            and len(differences) >= required
            and halfwidth <= target
            and seed_aggregation_valid
        )
        rows.append(
            {
                "policy": policy,
                "physical_paths": len(differences),
                "learning_policy": learning,
                "learning_seeds_aggregated_within_path_first": seed_aggregation_valid,
                "pilot_paths": pilot_count,
                "pilot_paired_standard_deviation": pilot_sd,
                "pilot_t_critical": pilot_critical,
                "raw_formula_required_paths": raw_required,
                "required_paths": required,
                "maximum_path_cap": maximum,
                "target_halfwidth": target,
                "achieved_halfwidth": halfwidth,
                "precision_target_met": status,
                "status": "PASS" if status else "FAIL",
            }
        )
    frame = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "required_paths": int(frame["required_paths"].max()),
                "executed_paths": int(frame["physical_paths"].min()),
                "target_halfwidth": target,
                "maximum_achieved_halfwidth": float(frame["achieved_halfwidth"].max()),
                "contrasts": len(frame),
                "precision_targets_met": int(frame["precision_target_met"].sum()),
                "precision_target_met": bool(frame["precision_target_met"].all()),
                "status": "PASS" if frame["precision_target_met"].all() else "FAIL",
            }
        ]
    )
    return frame, summary


def _row_numeric_difference(left: pd.Series, right: pd.Series, columns: list[str]) -> float:
    differences: list[float] = []
    for column in columns:
        lvalue, rvalue = left[column], right[column]
        if pd.isna(lvalue) and pd.isna(rvalue):
            differences.append(0.0)
        else:
            differences.append(abs(float(lvalue) - float(rvalue)))
    return max(differences, default=0.0)


def upstream_anchor_replay_audit(
    *, upstream: Mapping[str, Path], config: Mapping[str, Any], tolerance: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    statistics_module = _load_statistics_module()
    replications = pd.read_csv(upstream["5.2.2"] / "benchmark_replications.csv")
    recalculated = statistics_module.aggregate_learning_seeds(
        replications,
        learning_policies=config["learning_policies"],
    )
    accepted = pd.read_csv(upstream["5.2.2"] / "path_level_seed_aggregated.csv")
    merged = accepted.merge(
        recalculated,
        on=["policy", "path_id"],
        suffixes=("_accepted", "_recalculated"),
        validate="one_to_one",
    )
    common_numeric = [
        column
        for column in recalculated.columns
        if column not in {"policy", "path_id"}
        and f"{column}_accepted" in merged
        and f"{column}_recalculated" in merged
        and pd.api.types.is_numeric_dtype(recalculated[column])
        and not pd.api.types.is_bool_dtype(recalculated[column])
    ]
    for item in merged.itertuples(index=False):
        differences = []
        for column in common_numeric:
            left = getattr(item, f"{column}_accepted")
            right = getattr(item, f"{column}_recalculated")
            differences.append(
                0.0
                if pd.isna(left) and pd.isna(right)
                else abs(float(left) - float(right))
            )
        maximum = max(differences, default=0.0)
        rows.append(
            {
                "experiment": "5.2.2",
                "audit_id": "raw_replications_to_path_seed_aggregation",
                "policy": item.policy,
                "path_id": item.path_id,
                "training_seed": np.nan,
                "maximum_difference": maximum,
                "tolerance": tolerance,
                "status": "PASS" if maximum <= tolerance else "FAIL",
                "detail": "Three learning seeds are averaged within physical path before comparison with the accepted path-level file.",
            }
        )

    replay_523 = pd.read_csv(
        upstream["5.2.3"] / "full_action_reproduction_summary.csv"
    )
    for item in replay_523.itertuples(index=False):
        maximum = float(item.maximum_absolute_difference)
        rows.append(
            {
                "experiment": "5.2.3",
                "audit_id": "full_action_path_replay",
                "policy": item.policy,
                "path_id": item.path_id,
                "training_seed": item.training_seed,
                "maximum_difference": maximum,
                "tolerance": tolerance,
                "status": "PASS" if bool(item.all_metrics_passed) and maximum <= tolerance else "FAIL",
                "detail": "Frozen full-action mechanism replay against accepted 5.2.2 trajectory.",
            }
        )

    replay_524 = pd.read_csv(
        upstream["5.2.4"] / "historical_anchor_reproduction.csv"
    )
    grouped = (
        replay_524.groupby(["base_path_id", "training_seed"], as_index=False)[
            "absolute_difference"
        ]
        .max()
        .rename(columns={"absolute_difference": "maximum_difference"})
    )
    for item in grouped.itertuples(index=False):
        maximum = float(item.maximum_difference)
        rows.append(
            {
                "experiment": "5.2.4",
                "audit_id": "lead_aligned_full_rights_historical_anchor",
                "policy": "I_L / RD anchor",
                "path_id": item.base_path_id,
                "training_seed": item.training_seed,
                "maximum_difference": maximum,
                "tolerance": tolerance,
                "status": "PASS" if maximum <= tolerance else "FAIL",
                "detail": "Accepted 5.2.4 historical anchor replay against accepted 5.2.2 MG controller.",
            }
        )
    return pd.DataFrame(rows)


def rcmsa_start_certification_audit(
    model: Any, event: pd.DataFrame, case_count: int
) -> pd.DataFrame:
    solver = model.kernel.equilibrium_solver
    rows: list[dict[str, Any]] = []

    def certify(problem: Any, initial: np.ndarray) -> tuple[float, tuple[float, ...]]:
        slices, _ = solver._layout(problem)
        current = initial.copy()
        best = current.copy()
        best_residual = solver._residual(problem, slices, current)
        selected_steps: list[float] = []
        for iteration in range(solver.settings.max_iterations):
            residual = solver._residual(problem, slices, current)
            if residual < best_residual:
                best, best_residual = current.copy(), residual
            if residual <= solver.settings.tolerance:
                best, best_residual = current.copy(), residual
                break
            loading = solver._loading(problem, slices, current)
            candidates = []
            for multiplier in solver.settings.step_multipliers:
                step = min(1.0, multiplier / (iteration + 1.0))
                trial = (1.0 - step) * current + step * loading
                candidates.append(
                    (solver._residual(problem, slices, trial), multiplier, trial)
                )
            selected_residual, selected_multiplier, current = min(
                candidates,
                key=lambda item: (item[0], solver._lexicographic_key(item[2])),
            )
            selected_steps.append(float(selected_multiplier))
            if selected_residual < best_residual:
                best, best_residual = current.copy(), float(selected_residual)
            if selected_residual <= solver.settings.tolerance:
                break
        return float(solver._residual(problem, slices, best)), tuple(selected_steps)

    for case_index in range(min(case_count, len(event))):
        state = model.initial_state(event.iloc[case_index].to_dict())
        row = event.iloc[case_index].to_dict()
        projection = model.projector.project(model.zero_action(), state)
        realization = build_realization(model=model, state=state, row=row)
        problem = model.kernel.behavior_factory(state, projection.action, realization)
        slices, size = solver._layout(problem)
        zero = np.zeros(size, dtype=float)
        previous = solver._previous_start(problem, slices, state.previous_shares)
        free_flow = solver._loading(problem, slices, zero)
        dispersed = np.zeros(size, dtype=float)
        for source in problem.sources:
            dispersed[slices[source]] = float(problem.decision.masses[source]) / len(
                problem.choices(source)
            )
        expected_starts = {
            "previous": previous,
            "free_flow": free_flow,
            "dispersed": dispersed,
        }
        result = solver.solve(problem, previous_shares=state.previous_shares)
        records = {record.name: record for record in result.starts}
        for start_name, initial in expected_starts.items():
            certified_residual, certified_steps = certify(problem, initial)
            record = records.get(start_name)
            recorded_residual = float(record.residual) if record is not None else np.inf
            residual_difference = abs(recorded_residual - certified_residual)
            selected_provenance_valid = (
                result.selected_start in records
                and abs(
                    float(records[result.selected_start].residual)
                    - float(result.residual)
                )
                <= 1e-12
            )
            status = (
                record is not None
                and residual_difference <= 1e-12
                and tuple(record.selected_step_multipliers) == certified_steps
                and selected_provenance_valid
                and set(records) == set(expected_starts)
            )
            rows.append(
                {
                    "case_id": f"production_problem_{case_index}",
                    "start_name": start_name,
                    "start_source_formula": (
                        "Logit loading at zero adaptive flow"
                        if start_name == "free_flow"
                        else "previous normalized shares"
                        if start_name == "previous"
                        else "uniform dispersed source simplex"
                    ),
                    "zero_loading_vector_norm": float(np.linalg.norm(zero)),
                    "initial_vector_norm": float(np.linalg.norm(initial)),
                    "recorded_residual": recorded_residual,
                    "independently_certified_residual": certified_residual,
                    "residual_difference": residual_difference,
                    "recorded_selected_step_multipliers": "|".join(
                        str(value)
                        for value in (
                            record.selected_step_multipliers if record is not None else ()
                        )
                    ),
                    "certified_selected_step_multipliers": "|".join(
                        str(value) for value in certified_steps
                    ),
                    "final_generated_trial_certified": residual_difference <= 1e-12,
                    "production_selected_start": result.selected_start,
                    "selected_start_provenance_valid": selected_provenance_valid,
                    "tolerance": solver.settings.tolerance,
                    "status": "PASS" if status else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def unavailable_route_hold_audit(model: Any, event: pd.DataFrame) -> pd.DataFrame:
    cargo = str(model.config["cargo_class"])
    rows: list[dict[str, Any]] = []
    for route_id in sorted(model.network.routes):
        state = model.initial_state(event.iloc[0].to_dict())
        tag = Tag(cargo, route_id)
        state.berth[tag] = 1.0
        row = event.iloc[0].to_dict()
        row["normal_model_units"] = 0.0
        row["serviceability"] = 1.0
        realization = build_realization(model=model, state=state, row=row)
        realization = replace(
            realization,
            gulf_demand={cargo: 0.0},
            committed_fraction={cargo: 0.0},
            base_arrivals={},
            choice_route_available=frozenset(model.network.routes),
            physical_route_available=frozenset(
                route for route in model.network.routes if route != route_id
            ),
        )
        projection = model.projector.project(model.zero_action(), state)
        problem = model.kernel.behavior_factory(state, projection.action, realization)
        equilibrium = model.kernel.equilibrium_solver.solve(
            problem, previous_shares=state.previous_shares
        )
        result = model.kernel.transition.step(
            state=state,
            action=projection.action,
            equilibrium=equilibrium,
            realization=realization,
        )
        berth_after = float(result.next_state.berth.get(tag, 0.0))
        yard_after = float(result.next_state.yard.get(tag, 0.0))
        delivered = float(result.delivered.get(tag, 0.0))
        service_violation = float(result.audit.unavailable_route_service_violation)
        maximum = max(abs(berth_after - 1.0), yard_after, delivered, service_violation)
        rows.append(
            {
                "route": route_id,
                "initial_berth_mass": 1.0,
                "ending_berth_mass": berth_after,
                "ending_yard_mass": yard_after,
                "delivered_mass": delivered,
                "unavailable_route_service_violation": service_violation,
                "maximum_hold_residual": maximum,
                "tolerance": float(model.config["numerics"]["mass_tolerance"]),
                "status": (
                    "PASS"
                    if maximum
                    <= float(model.config["numerics"]["mass_tolerance"])
                    else "FAIL"
                ),
            }
        )
    return pd.DataFrame(rows)


def capacity_timing_audit(model: Any, event: pd.DataFrame) -> pd.DataFrame:
    state = model.initial_state(event.iloc[0].to_dict())
    dynamics = model.kernel.transition.capacity_model
    resource = model.controlled_resources[0]
    readiness_key = model.layout.readiness_order[0]
    direct_key = model.layout.direct_order[0]
    exercise_key = model.layout.readiness_exercise[0]
    readiness_lead = int(model.config["action"]["readiness_lead_weeks"])
    direct_lead = int(model.config["action"]["direct_lead_weeks"])
    tolerance = float(model.config["numerics"]["mass_tolerance"])
    rows: list[dict[str, Any]] = []
    implemented_readiness_order = 0.0
    for period in range(readiness_lead + 2):
        values = {key: 0.0 for key in model.layout.keys}
        if period == 0:
            values[readiness_key] = 1.0
            values[direct_key] = 1.0
        if period == readiness_lead:
            values[exercise_key] = 1.0
        raw = Action(values)
        projected = model.projector.project(raw, state)
        action = projected.action
        if period == 0:
            implemented_readiness_order = action.value(readiness_key)
        stock_before = float(state.readiness.stock.get(resource, 0.0))
        result = dynamics.transition(state, action)
        stock_after = float(result.next_readiness.stock.get(resource, 0.0))
        readiness_current = float(
            result.current.readiness_capacity.get(resource, 0.0)
        )
        direct_current = float(result.current.direct_spot.get(resource, 0.0))
        action_cost = float(model.domain.action_cost(action))
        budget_cap = float(model.domain.budget_cap(state))
        budget_slack = budget_cap - action_cost
        readiness_early = bool(
            period < readiness_lead
            and (stock_before > tolerance or readiness_current > tolerance)
        )
        readiness_maturity_valid = bool(
            period != readiness_lead
            or stock_before
            >= implemented_readiness_order
            * float(dynamics.technology.readiness_maturity_yield[resource])
            - tolerance
        )
        readiness_exercise_valid = bool(
            period != readiness_lead
            or (
                readiness_current
                >= action.value(exercise_key)
                * float(dynamics.technology.readiness_capacity_yield[resource])
                - tolerance
                and stock_after <= stock_before + tolerance
            )
        )
        direct_timing_valid = bool(
            (period == direct_lead and direct_current >= 0.0)
            and (period != 0 or direct_current > tolerance)
            if direct_lead == 0 and period == 0
            else (direct_current <= tolerance)
        )
        status = (
            not readiness_early
            and readiness_maturity_valid
            and readiness_exercise_valid
            and direct_timing_valid
            and result.audit.maximum_residual <= tolerance
            and budget_slack >= -tolerance
        )
        rows.append(
            {
                "resource": f"{resource.stage.value}:{resource.location}",
                "period": period,
                "readiness_lead_weeks": readiness_lead,
                "direct_lead_weeks": direct_lead,
                "readiness_order_requested": raw.value(readiness_key),
                "readiness_order_implemented": action.value(readiness_key),
                "readiness_pipeline": float(
                    sum(result.next_readiness.orders.get(resource, {}).values())
                ),
                "readiness_stock_before": stock_before,
                "readiness_stock_after": stock_after,
                "readiness_exercise_implemented": action.value(exercise_key),
                "readiness_capacity_current": readiness_current,
                "direct_order_requested": raw.value(direct_key),
                "direct_order_implemented": action.value(direct_key),
                "direct_pipeline": float(
                    sum(result.next_direct.orders.get(resource, {}).values())
                ),
                "direct_capacity_current": direct_current,
                "readiness_available_before_maturity": readiness_early,
                "readiness_maturity_valid": readiness_maturity_valid,
                "readiness_exercise_valid": readiness_exercise_valid,
                "direct_delivery_timing_valid": direct_timing_valid,
                "capacity_pipeline_maximum_residual": result.audit.maximum_residual,
                "action_cost": action_cost,
                "period_budget_cap": budget_cap,
                "budget_slack": budget_slack,
                "projection_feasibility_residual": projected.feasibility_violation,
                "status": "PASS" if status else "FAIL",
            }
        )
        state.readiness = result.next_readiness
        state.direct_capacity = result.next_direct
        state.budget = max(state.budget - action_cost, 0.0)
        state.period += 1
    return pd.DataFrame(rows)


def clearance_terminal_audit(
    upstream: Mapping[str, Path], config: Mapping[str, Any], tolerance: float
) -> pd.DataFrame:
    frame = pd.read_csv(upstream["5.2.2"] / "benchmark_replications.csv")
    cap = int(config["clearance"]["maximum_weeks"])
    frame["right_censored"] = _as_bool(frame["right_censored"])
    frame["decision_clearance_terminal_residual"] = (
        frame["total_operational_objective"]
        - frame["decision_operational_loss"]
        - frame["clearance_operational_loss"]
        - frame["terminal_correction"]
    ).abs()
    frame["exit_channel_residual"] = (
        frame["loss_exit"]
        - frame["loss_direct_sue_exit"]
        - frame["loss_duration_attrition"]
    ).abs()
    frame["clearance_observation_valid"] = np.where(
        frame["right_censored"],
        frame["clearance_weeks_observed"].isna(),
        frame["clearance_weeks_observed"].notna(),
    )
    expected_restricted = np.where(
        frame["right_censored"], cap, frame["clearance_weeks_observed"]
    )
    frame["restricted_clearance_time_residual"] = (
        frame["restricted_clearance_time_contribution"] - expected_restricted
    ).abs()
    frame["terminal_once_valid"] = frame[
        "decision_clearance_terminal_residual"
    ].le(tolerance)
    frame["status"] = np.where(
        frame["terminal_once_valid"]
        & frame["exit_channel_residual"].le(tolerance)
        & frame["clearance_observation_valid"]
        & frame["restricted_clearance_time_residual"].le(tolerance),
        "PASS",
        "FAIL",
    )
    return frame[
        [
            "policy",
            "path_id",
            "training_seed",
            "right_censored",
            "clearance_weeks_observed",
            "restricted_clearance_time_contribution",
            "ending_outstanding_mass",
            "terminal_correction",
            "decision_clearance_terminal_residual",
            "exit_channel_residual",
            "clearance_observation_valid",
            "restricted_clearance_time_residual",
            "terminal_once_valid",
            "status",
        ]
    ]
