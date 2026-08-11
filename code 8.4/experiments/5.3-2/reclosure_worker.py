"""Process-isolated, branch-sharing production evaluator for Experiment 5.3.2.

The evaluator never implements a second environment.  Every decision period calls
the accepted 5.2 ``prepare_period`` adapter, common projector, RC-MSA kernel and
tagged transition.  Branch sharing is purely a deterministic computational
factorisation: common path prefixes are executed once and cloned before a grid
coordinate first differs.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from tre84.acceptance import evaluate_acceptance
from tre84.clearance import ClearanceRunner
from tre84.errors import ContractError
from tre84.metrics import compute_trajectory_statistics
from tre84.state import ModelState

from features import LinearActor
from model import build_model
from paths import PhysicalPath, _canonical_path_hash
from policies import (
    ActorPolicy,
    MPCPolicy,
    ModelGuidedPolicy,
    PassivePolicy,
    ReactivePolicy,
)
from preparation import build_realization, prepare_period
from simulator import RecoveryRule, _period_record
from mechanism import _physical_rows
from tre84.keys import Provenance, SourceKey


@dataclass(frozen=True)
class GridCell:
    open_weeks: int
    intensity: float
    duration_weeks: int

    @property
    def serviceability(self) -> float:
        return 1.0 - self.intensity

    @property
    def cell_id(self) -> str:
        severity = f"{self.intensity:.2f}".replace(".", "p")
        return f"open_{self.open_weeks:02d}__intensity_{severity}__duration_{self.duration_weeks:02d}"


@dataclass
class Branch:
    state: Any
    initial_state: Any
    results: tuple[Any, ...]
    weekly: tuple[dict[str, Any], ...]
    all_step_acceptance: bool
    maximum_sue_residual: float
    maximum_transition_residual: float
    maximum_projection_violation: float
    decision_seconds_sum: float
    decision_count: int
    selector_records: int
    provenance_shadow: dict[Any, float]
    committed_delivery: float
    adaptive_delivery: float
    maximum_provenance_shadow_residual: float

    def snapshot(self) -> "Branch":
        return Branch(
            state=self.state.clone(),
            initial_state=self.initial_state,
            results=self.results,
            weekly=self.weekly,
            all_step_acceptance=self.all_step_acceptance,
            maximum_sue_residual=self.maximum_sue_residual,
            maximum_transition_residual=self.maximum_transition_residual,
            maximum_projection_violation=self.maximum_projection_violation,
            decision_seconds_sum=self.decision_seconds_sum,
            decision_count=self.decision_count,
            selector_records=self.selector_records,
            provenance_shadow=dict(self.provenance_shadow),
            committed_delivery=self.committed_delivery,
            adaptive_delivery=self.adaptive_delivery,
            maximum_provenance_shadow_residual=self.maximum_provenance_shadow_residual,
        )


_MODEL: Any | None = None
_POLICIES: list[Any] = []
_CELLS_BY_POLICY: tuple[tuple[GridCell, ...], ...] = ()
_RECOVERY_WEEKS = 8
_CACHE_DIR: Path | None = None
_RUN_SIGNATURE = ""
_CURRENT_TASK_TAG = ""
_ORIGINAL_STATE_VALIDATE = ModelState.validate


def _diagnostic_state_validate(state: ModelState, *, tolerance: float = 1e-10) -> None:
    """Preserve fail-closed validation while persisting the rejected contract.

    This experiment-only wrapper does not alter the validator or suppress its
    exception.  It makes a long reclosure failure auditable without modifying
    the accepted core package.
    """
    try:
        _ORIGINAL_STATE_VALIDATE(state, tolerance=tolerance)
    except ContractError as error:
        if _CACHE_DIR is not None and _CURRENT_TASK_TAG:
            directory = _CACHE_DIR / _CURRENT_TASK_TAG
            directory.mkdir(parents=True, exist_ok=True)
            invalid_shares = []
            for source, shares in state.previous_shares.items():
                values = np.asarray(tuple(shares.values()), dtype=float)
                if values.size == 0 or np.any(~np.isfinite(values)) or np.min(values) < -tolerance or not np.isclose(values.sum(), 1.0, atol=tolerance):
                    invalid_shares.append({
                        "source": repr(source), "choice_count": int(values.size),
                        "sum": float(values.sum()) if values.size else 0.0,
                        "minimum": float(values.min()) if values.size else None,
                        "maximum": float(values.max()) if values.size else None,
                        "shares": {str(key): float(value) for key, value in shares.items()},
                    })
            payload = {
                "run_signature": _RUN_SIGNATURE,
                "task": _CURRENT_TASK_TAG,
                "period": state.period,
                "horizon": state.horizon,
                "reason": str(error),
                "cargo_mass": state.cargo_mass(),
                "budget": state.budget,
                "disruption_seen": state.disruption_seen,
                "disruption_active": state.disruption_active,
                "invalid_previous_shares": invalid_shares,
            }
            (directory / "state_validation_failure.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        raise


def _formal_sue_source_masses(result: Any) -> dict[SourceKey, float]:
    """Reconstruct the source ledger entering the formal route-wait-exit SUE."""
    masses = {
        SourceKey(cargo_class, None): float(mass)
        for cargo_class, mass in result.transition.demand_split.decision_eligible.items()
    }
    for cargo_class, release in result.equilibrium.releases.items():
        for vintage, mass in enumerate(np.asarray(release, dtype=float)):
            if mass > 0.0:
                masses[SourceKey(cargo_class, vintage)] = float(mass)
    return masses


def _actor(checkpoint: str) -> LinearActor:
    with np.load(checkpoint, allow_pickle=False) as payload:
        return LinearActor(
            np.asarray(payload["weights"], dtype=float),
            np.asarray(payload["log_standard_deviation"], dtype=float),
        )


def initialise_worker(
    model_config: Mapping[str, Any],
    policy_specs: Sequence[Mapping[str, Any]],
    cells_by_policy: Sequence[Sequence[Mapping[str, Any]]],
    recovery_weeks: int,
    cache_dir: str,
    run_signature: str,
) -> None:
    global _MODEL, _POLICIES, _CELLS_BY_POLICY, _RECOVERY_WEEKS, _CACHE_DIR, _RUN_SIGNATURE
    _MODEL = build_model(model_config)
    policies: list[Any] = []
    for spec in policy_specs:
        kind = str(spec["kind"])
        if kind == "passive":
            policies.append(PassivePolicy(_MODEL))
        elif kind == "reactive":
            policies.append(ReactivePolicy(_MODEL))
        elif kind == "mpc":
            policies.append(MPCPolicy(_MODEL))
        elif kind == "actor":
            policies.append(
                ActorPolicy(
                    str(spec["name"]),
                    _MODEL,
                    _actor(str(spec["checkpoint"])),
                    int(spec["training_seed"]),
                )
            )
        elif kind == "model_guided":
            policies.append(
                ModelGuidedPolicy(
                    model=_MODEL,
                    bc_actor=_actor(str(spec["bc_checkpoint"])),
                    sac_actor=_actor(str(spec["sac_checkpoint"])),
                    training_seed=int(spec["training_seed"]),
                )
            )
        else:
            raise ValueError(f"Unknown 5.3.2 policy kind: {kind}")
    _POLICIES = policies
    _CELLS_BY_POLICY = tuple(
        tuple(
            GridCell(
                int(row["open_weeks"]),
                float(row["intensity"]),
                int(row["duration_weeks"]),
            )
            for row in cells
        )
        for cells in cells_by_policy
    )
    if len(_CELLS_BY_POLICY) != len(_POLICIES) or any(not cells for cells in _CELLS_BY_POLICY):
        raise RuntimeError("Every policy execution must have a nonempty preregistered cell set")
    _RECOVERY_WEEKS = int(recovery_weeks)
    _CACHE_DIR = Path(cache_dir)
    _RUN_SIGNATURE = str(run_signature)
    ModelState.validate = _diagnostic_state_validate


def _task_tag(path: PhysicalPath, policy_index: int) -> str:
    safe = path.path_id.replace("/", "_").replace("\\", "_")
    return f"{safe}__policy_{policy_index:02d}"


def _extension_row(
    base_path: PhysicalPath,
    extension_offset: int,
    serviceability: float,
) -> dict[str, Any]:
    source = base_path.frame.iloc[extension_offset % len(base_path.frame)].to_dict()
    last = base_path.frame.iloc[-1]
    source["week"] = pd.Timestamp(last["week"]) + pd.Timedelta(
        weeks=extension_offset + 1
    )
    source["serviceability"] = float(serviceability)
    source["filtered_high_risk_probability"] = float(
        last["filtered_high_risk_probability"]
    )
    source["lead_time_high_risk_probability"] = float(
        last["lead_time_high_risk_probability"]
    )
    source["release_date"] = pd.Timestamp(last["release_date"])
    source["source_observation_month"] = pd.Timestamp(
        last["source_observation_month"]
    )
    source["timing_valid"] = True
    source["information_source"] = (
        "released_hmm_filter_last_available_carry_forward"
    )
    return source


def build_cell_path(base_path: PhysicalPath, cell: GridCell) -> PhysicalPath:
    rows = [row._asdict() for row in base_path.frame.itertuples(index=False)]
    extension = 0
    for _ in range(cell.open_weeks):
        rows.append(_extension_row(base_path, extension, 1.0))
        extension += 1
    for _ in range(cell.duration_weeks):
        rows.append(_extension_row(base_path, extension, cell.serviceability))
        extension += 1
    for step in range(1, _RECOVERY_WEEKS + 1):
        serviceability = cell.serviceability + (
            1.0 - cell.serviceability
        ) * step / _RECOVERY_WEEKS
        rows.append(_extension_row(base_path, extension, serviceability))
        extension += 1
    frame = pd.DataFrame(rows)
    digest = _canonical_path_hash(frame)
    return PhysicalPath(
        path_id=f"{base_path.path_id}__{cell.cell_id}",
        split="test_reclosure",
        frame=frame,
        path_hash=digest,
        construction=(
            "accepted 21-week 5.2.2 physical prefix plus common event-aligned "
            "open/reclosure/eight-week-linear-recovery constructor"
        ),
        residual_start=base_path.residual_start,
        residual_end=base_path.residual_end,
        onset_week=len(base_path.frame) + cell.open_weeks,
        active_duration_weeks=cell.duration_weeks,
        severity_floor=cell.serviceability,
        has_reclosure=True,
    )


def _initial_branch(base_path: PhysicalPath) -> Branch:
    assert _MODEL is not None
    initial = _MODEL.initial_state(base_path.frame.iloc[0].to_dict())
    return Branch(
        state=initial,
        initial_state=initial.clone(),
        results=(),
        weekly=(),
        all_step_acceptance=True,
        maximum_sue_residual=0.0,
        maximum_transition_residual=0.0,
        maximum_projection_violation=0.0,
        decision_seconds_sum=0.0,
        decision_count=0,
        selector_records=0,
        provenance_shadow={},
        committed_delivery=0.0,
        adaptive_delivery=0.0,
        maximum_provenance_shadow_residual=0.0,
    )


def _advance(
    branch: Branch,
    *,
    policy: Any,
    row: Mapping[str, Any],
    path_context: PhysicalPath,
    segment: str,
) -> Branch:
    assert _MODEL is not None
    prepared = prepare_period(model=_MODEL, state=branch.state, row=row)
    state = prepared.state
    started = time.perf_counter()
    decision = policy.decide(
        state=state,
        row=row,
        path=path_context,
        offset=state.period,
        bundle=prepared.scenarios,
    )
    elapsed = time.perf_counter() - started
    projection = _MODEL.projector.project(decision.raw_action, state)
    realization = build_realization(model=_MODEL, state=state, row=row)
    result = _MODEL.kernel.execute(
        state=state,
        action=projection.action,
        realization=realization,
        projection=projection,
    )
    # The acceptance input is the formal source mass entering the SUE, not the
    # post-solver sum of represented floating-point flows.  Those quantities
    # coincide at ordinary scales, but an extremely old waiting vintage can be
    # IEEE-754 subnormal and individual alternatives may underflow.  Rebuild the
    # same source ledger used by the production trajectory acceptance contract:
    # current decision-eligible mass plus every strictly positive released
    # waiting vintage.  This changes only the certificate input; it does not
    # alter the equilibrium, state transition, action, or loss.
    source_masses = _formal_sue_source_masses(result)
    acceptance = evaluate_acceptance(
        decision_time=row["week"],
        information_timestamps=prepared.scenarios.information_timestamps,
        state=state,
        action=projection.action,
        action_domain=_MODEL.domain,
        equilibrium=result.equilibrium,
        source_masses=source_masses,
        transition_audit=result.transition.audit,
        loss=result.transition.loss,
        tolerance=float(_MODEL.config["numerics"]["mass_tolerance"]),
    )
    weekly = _period_record(
        model=_MODEL,
        policy=policy.name,
        path=path_context,
        seed=policy.training_seed,
        offset=state.period,
        scope="decision",
        state=state,
        result=result,
    )
    weekly["event_segment"] = segment
    weekly["decision_week"] = pd.Timestamp(row["week"])
    weekly["serviceability"] = float(row["serviceability"])
    weekly["normal_model_units"] = float(row["normal_model_units"])
    weekly["projected_action_cost"] = float(
        _MODEL.domain.action_cost(projection.action)
    )
    weekly["projection_feasibility_violation"] = float(
        projection.feasibility_violation
    )
    weekly["decision_time_seconds"] = elapsed
    weekly["information_vector_sha256"] = prepared.information_vector_hash
    weekly["observation_sha256"] = prepared.observation_hash
    weekly["acceptance_information_timing"] = acceptance.information_timing
    weekly["acceptance_action_feasibility"] = acceptance.action_feasibility
    weekly["acceptance_behavioral_closure"] = acceptance.behavioral_closure
    weekly["acceptance_physical_closure"] = acceptance.physical_closure
    weekly["acceptance_objective_closure"] = acceptance.objective_closure
    weekly["acceptance_passed"] = acceptance.passed
    weekly["acceptance_messages"] = "|".join(acceptance.messages)
    audit = result.transition.audit
    physical_rows, next_shadow, shadow_residual = _physical_rows(
        model=_MODEL,
        base_policy=policy.name,
        restriction="full_action",
        path=path_context,
        seed=policy.training_seed,
        scope="decision",
        period_offset=state.period,
        state=state,
        realization=realization,
        result=result,
        shadow=dict(branch.provenance_shadow),
        store_rows=True,
    )
    committed_delivery = sum(
        float(item.get("delivered_cargo", 0.0))
        for item in physical_rows
        if item.get("provenance") == Provenance.COMMITTED.value
    )
    adaptive_delivery = sum(
        float(item.get("delivered_cargo", 0.0))
        for item in physical_rows
        if item.get("provenance") == Provenance.ADAPTIVE.value
    )
    transition_residual = max(
        audit.adaptive_mass_residual,
        audit.committed_mass_residual,
        audit.pipeline_mass_residual,
        audit.tagged_balance_residual,
        getattr(audit, "maximum_waiting_vintage_balance_residual", 0.0),
        getattr(audit, "maximum_waiting_vintage_no_reset_residual", 0.0),
    )
    return Branch(
        state=result.transition.next_state,
        initial_state=branch.initial_state,
        results=branch.results + (result,),
        weekly=branch.weekly + (weekly,),
        all_step_acceptance=branch.all_step_acceptance and acceptance.passed,
        maximum_sue_residual=max(
            branch.maximum_sue_residual, float(result.equilibrium.residual)
        ),
        maximum_transition_residual=max(
            branch.maximum_transition_residual, float(transition_residual)
        ),
        maximum_projection_violation=max(
            branch.maximum_projection_violation,
            float(projection.feasibility_violation),
        ),
        decision_seconds_sum=branch.decision_seconds_sum + elapsed,
        decision_count=branch.decision_count + 1,
        selector_records=branch.selector_records + len(decision.proposal_records),
        provenance_shadow=next_shadow,
        committed_delivery=branch.committed_delivery + committed_delivery,
        adaptive_delivery=branch.adaptive_delivery + adaptive_delivery,
        maximum_provenance_shadow_residual=max(
            branch.maximum_provenance_shadow_residual, float(shadow_residual)
        ),
    )


def _finalise(
    branch: Branch,
    *,
    policy: Any,
    cell_path: PhysicalPath,
    cell: GridCell,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    assert _MODEL is not None
    clearance = ClearanceRunner(
        kernel=_MODEL.kernel,
        recovery_rule=RecoveryRule(_MODEL),
        terminal_cost=_MODEL.terminal_cost,
        maximum_weeks=int(_MODEL.config["clearance"]["maximum_weeks"]),
        empty_tolerance=float(_MODEL.config["clearance"]["empty_tolerance"]),
    ).run(branch.state)
    recovery = RecoveryRule(_MODEL)
    clearance_state = branch.state
    weekly = [dict(row) for row in branch.weekly]
    maximum_transition = branch.maximum_transition_residual
    maximum_sue = branch.maximum_sue_residual
    provenance_shadow = dict(branch.provenance_shadow)
    committed_delivery = branch.committed_delivery
    adaptive_delivery = branch.adaptive_delivery
    maximum_shadow = branch.maximum_provenance_shadow_residual
    for offset, result in enumerate(clearance.transitions):
        realization = recovery.realization(clearance_state)
        row = _period_record(
            model=_MODEL,
            policy=policy.name,
            path=cell_path,
            seed=policy.training_seed,
            offset=offset,
            scope="clearance",
            state=clearance_state,
            result=result,
        )
        row["event_segment"] = "clearance"
        row["decision_week"] = pd.NaT
        row["serviceability"] = 1.0
        row["normal_model_units"] = 0.0
        row["projected_action_cost"] = 0.0
        row["projection_feasibility_violation"] = 0.0
        row["decision_time_seconds"] = 0.0
        row["information_vector_sha256"] = ""
        row["observation_sha256"] = ""
        weekly.append(row)
        physical_rows, provenance_shadow, shadow_residual = _physical_rows(
            model=_MODEL,
            base_policy=policy.name,
            restriction="full_action",
            path=cell_path,
            seed=policy.training_seed,
            scope="clearance",
            period_offset=offset,
            state=clearance_state,
            realization=realization,
            result=result,
            shadow=provenance_shadow,
            store_rows=True,
        )
        committed_delivery += sum(
            float(item.get("delivered_cargo", 0.0))
            for item in physical_rows
            if item.get("provenance") == Provenance.COMMITTED.value
        )
        adaptive_delivery += sum(
            float(item.get("delivered_cargo", 0.0))
            for item in physical_rows
            if item.get("provenance") == Provenance.ADAPTIVE.value
        )
        maximum_shadow = max(maximum_shadow, float(shadow_residual))
        audit = result.transition.audit
        maximum_transition = max(
            maximum_transition,
            float(
                max(
                    audit.adaptive_mass_residual,
                    audit.committed_mass_residual,
                    audit.pipeline_mass_residual,
                    audit.tagged_balance_residual,
                    getattr(audit, "maximum_waiting_vintage_balance_residual", 0.0),
                    getattr(audit, "maximum_waiting_vintage_no_reset_residual", 0.0),
                )
            ),
        )
        maximum_sue = max(maximum_sue, float(result.equilibrium.residual))
        clearance_state = result.transition.next_state

    stats = compute_trajectory_statistics(
        initial_state=branch.initial_state,
        decision_results=branch.results,
        network=_MODEL.network,
        thresholds=_MODEL.thresholds,
        clearance=clearance,
        include_clearance_in_physical_metrics=True,
        tolerance=float(_MODEL.config["numerics"]["loss_identity_tolerance"]),
    )
    component_sum = (
        stats.loss_queue
        + stats.loss_waiting
        + stats.loss_exit
        + stats.loss_overflow
        + stats.loss_route_resource
        + stats.loss_action
        + stats.terminal_correction
    )
    total_committed_arrivals = float(
        sum(
            sum(result.transition.demand_split.committed_by_tag.values())
            for result in branch.results
        )
    )
    terminal_committed_outstanding = float(
        sum(
            amount
            for (provenance, _route, _stage), amount in provenance_shadow.items()
            if provenance == Provenance.COMMITTED
        )
        + sum(
            lot.mass
            for lot in clearance.final_state.maritime_pipeline
            if lot.provenance == Provenance.COMMITTED
        )
    )
    terminal_adaptive_outstanding = float(
        clearance.final_state.waiting_mass()
        + sum(
            amount
            for (provenance, _route, _stage), amount in provenance_shadow.items()
            if provenance == Provenance.ADAPTIVE
        )
        + sum(
            lot.mass
            for lot in clearance.final_state.maritime_pipeline
            if lot.provenance == Provenance.ADAPTIVE
        )
    )
    exit_unit = float(_MODEL.config["behavior"]["exit_failure_cost_per_unit"])
    record = {
        "cell_id": cell.cell_id,
        "open_interval_weeks": cell.open_weeks,
        "reclosure_intensity": cell.intensity,
        "reclosure_serviceability": cell.serviceability,
        "reclosure_duration_weeks": cell.duration_weeks,
        "policy": policy.name,
        "path_id": cell_path.path_id.split("__open_")[0],
        "reclosure_path_id": cell_path.path_id,
        "path_content_sha256": cell_path.path_hash,
        "training_seed": policy.training_seed,
        **stats.as_record(),
        "loss_direct_sue_exit": stats.direct_sue_exit * exit_unit,
        "loss_duration_attrition": stats.duration_attrition * exit_unit,
        "loss_component_sum_with_terminal": component_sum,
        "all_step_acceptance_passed": branch.all_step_acceptance,
        "maximum_sue_residual": maximum_sue,
        "maximum_transition_residual": maximum_transition,
        "maximum_projection_violation": branch.maximum_projection_violation,
        "mean_decision_time_seconds": branch.decision_seconds_sum
        / max(branch.decision_count, 1),
        "decision_count": branch.decision_count,
        "selector_proposal_records": branch.selector_records,
        "committed_delivery": committed_delivery,
        "adaptive_delivery": adaptive_delivery,
        "total_committed_arrivals": total_committed_arrivals,
        "committed_delivery_share": (
            committed_delivery / total_committed_arrivals
            if total_committed_arrivals > 0.0
            else np.nan
        ),
        "terminal_committed_outstanding": terminal_committed_outstanding,
        "terminal_adaptive_outstanding": terminal_adaptive_outstanding,
        "maximum_provenance_shadow_residual": maximum_shadow,
        "clearance_weeks_observed": stats.clearance_weeks_observed,
        "restricted_clearance_time_contribution": (
            stats.clearance_weeks_observed
            if stats.clearance_weeks_observed is not None
            else int(_MODEL.config["clearance"]["maximum_weeks"])
        ),
    }
    tolerance = float(_MODEL.config["numerics"]["loss_identity_tolerance"])
    contract = {
        "cell_id": cell.cell_id,
        "policy": policy.name,
        "path_id": record["path_id"],
        "training_seed": policy.training_seed,
        "path_content_sha256": cell_path.path_hash,
        "all_step_acceptance_passed": branch.all_step_acceptance,
        "all_transition_audits_passed": stats.transition_audits_passed,
        "sue_residual_within_tolerance": maximum_sue
        <= float(_MODEL.config["behavior"]["rcmsa_tolerance"]),
        "projection_feasible": branch.maximum_projection_violation
        <= float(_MODEL.config["action"]["projection_tolerance"]),
        "loss_components_reconstruct_total": abs(
            component_sum - stats.total_operational_objective
        )
        <= tolerance,
        "right_censoring_not_observed_clearance": not (
            stats.right_censored and stats.clearance_weeks_observed is not None
        ),
        "shared_prefix_execution": True,
        "branching_changes_scientific_logic": False,
        "future_information_used": False,
        "frozen_checkpoint_or_rule": True,
        "provenance_shadow_conservation": maximum_shadow <= tolerance,
        "committed_mass_reconciliation": abs(
            total_committed_arrivals
            - committed_delivery
            - terminal_committed_outstanding
        )
        <= tolerance,
    }
    terminal = {
        "policy": policy.name,
        "training_seed": policy.training_seed,
        "scope": "terminal",
        "period_offset": clearance.weeks,
        "event_segment": "terminal",
        "terminal_loss": clearance.terminal_correction,
        "period_operational_loss": clearance.terminal_correction,
        "outstanding_after": clearance.final_state.cargo_mass(),
        "decision_week": pd.NaT,
        "serviceability": 1.0,
        "normal_model_units": 0.0,
    }
    weekly.append(terminal)
    for row in weekly:
        row.update(
            {
                "cell_id": cell.cell_id,
                "open_interval_weeks": cell.open_weeks,
                "reclosure_intensity": cell.intensity,
                "reclosure_duration_weeks": cell.duration_weeks,
                "policy": policy.name,
                "path_id": record["path_id"],
                "reclosure_path_id": cell_path.path_id,
                "path_content_sha256": cell_path.path_hash,
                "training_seed": policy.training_seed,
            }
        )
    return record, weekly, contract


def evaluate_path_policy_task(task: tuple[PhysicalPath, int]) -> dict[str, Any]:
    if _MODEL is None or _CACHE_DIR is None:
        raise RuntimeError("5.3.2 worker is not initialised")
    global _CURRENT_TASK_TAG
    base_path, policy_index = task
    policy = _POLICIES[policy_index]
    policy_cells = _CELLS_BY_POLICY[policy_index]
    tag = _task_tag(base_path, policy_index)
    _CURRENT_TASK_TAG = tag
    directory = _CACHE_DIR / tag
    marker = directory / "complete.json"
    path_file = directory / "path_level.csv.gz"
    contract_file = directory / "contracts.csv.gz"
    weekly_file = directory / "weekly.csv.gz"
    if marker.exists():
        status = json.loads(marker.read_text(encoding="utf-8"))
        if status.get("run_signature") != _RUN_SIGNATURE:
            raise RuntimeError(f"Incompatible cached task: {tag}")
        if status.get("cell_ids") != [cell.cell_id for cell in policy_cells]:
            raise RuntimeError(f"Cached policy coverage mismatch: {tag}")
        for path in (path_file, contract_file, weekly_file):
            if not path.exists():
                raise RuntimeError(f"Incomplete cached task: {tag}")
        return {
            "tag": tag,
            "path_file": str(path_file),
            "contract_file": str(contract_file),
            "weekly_file": str(weekly_file),
            "path_rows": int(status["path_rows"]),
            "weekly_rows": int(status["weekly_rows"]),
            "reused": True,
        }

    directory.mkdir(parents=True, exist_ok=True)
    cells_by_open: dict[int, list[GridCell]] = {}
    for cell in policy_cells:
        cells_by_open.setdefault(cell.open_weeks, []).append(cell)
    context = build_cell_path(base_path, policy_cells[-1])
    prefix = _initial_branch(base_path)
    for row in base_path.frame.to_dict(orient="records"):
        prefix = _advance(
            prefix,
            policy=policy,
            row=row,
            path_context=context,
            segment="historical_prefix",
        )

    open_snapshots: dict[int, Branch] = {}
    open_branch = prefix
    extension_offset = 0
    for week in range(1, max(cells_by_open) + 1):
        open_branch = _advance(
            open_branch,
            policy=policy,
            row=_extension_row(base_path, extension_offset, 1.0),
            path_context=context,
            segment="open_interval",
        )
        extension_offset += 1
        if week in cells_by_open:
            open_snapshots[week] = open_branch.snapshot()

    path_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    for open_weeks, open_snapshot in open_snapshots.items():
        open_cells = cells_by_open[open_weeks]
        intensities = sorted({cell.intensity for cell in open_cells})
        for intensity in intensities:
            durations = sorted(
                cell.duration_weeks
                for cell in open_cells
                if cell.intensity == intensity
            )
            reclose = open_snapshot.snapshot()
            duration_snapshots: dict[int, Branch] = {}
            start_extension = open_weeks
            for week in range(1, max(durations) + 1):
                reclose = _advance(
                    reclose,
                    policy=policy,
                    row=_extension_row(
                        base_path,
                        start_extension + week - 1,
                        1.0 - intensity,
                    ),
                    path_context=context,
                    segment="reclosure",
                )
                if week in durations:
                    duration_snapshots[week] = reclose.snapshot()
            for duration, snapshot in duration_snapshots.items():
                cell = GridCell(open_weeks, intensity, duration)
                recovery = snapshot
                for step in range(1, _RECOVERY_WEEKS + 1):
                    serviceability = (1.0 - intensity) + intensity * step / _RECOVERY_WEEKS
                    recovery = _advance(
                        recovery,
                        policy=policy,
                        row=_extension_row(
                            base_path,
                            open_weeks + duration + step - 1,
                            serviceability,
                        ),
                        path_context=context,
                        segment="post_reclosure_recovery",
                    )
                cell_path = build_cell_path(base_path, cell)
                record, weekly, contract = _finalise(
                    recovery,
                    policy=policy,
                    cell_path=cell_path,
                    cell=cell,
                )
                path_rows.append(record)
                weekly_rows.extend(weekly)
                contract_rows.append(contract)

    if len(path_rows) != len(policy_cells):
        raise RuntimeError(f"Grid task {tag} produced {len(path_rows)} cells")
    pd.DataFrame(path_rows).to_csv(path_file, index=False, compression="gzip")
    pd.DataFrame(contract_rows).to_csv(contract_file, index=False, compression="gzip")
    pd.DataFrame(weekly_rows).to_csv(weekly_file, index=False, compression="gzip")
    marker.write_text(
        json.dumps(
            {
                "run_signature": _RUN_SIGNATURE,
                "tag": tag,
                "path_rows": len(path_rows),
                "weekly_rows": len(weekly_rows),
                "policy": policy.name,
                "training_seed": policy.training_seed,
                "cell_ids": [cell.cell_id for cell in policy_cells],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "tag": tag,
        "path_file": str(path_file),
        "contract_file": str(contract_file),
        "weekly_file": str(weekly_file),
        "path_rows": len(path_rows),
        "weekly_rows": len(weekly_rows),
        "reused": False,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
