from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tre84.actions import Action
from tre84.behavior import EquilibriumResult, StartRecord
from tre84.capacity import CurrentCapacity
from tre84.clearance import ClearanceResult
from tre84.engine import KernelResult
from tre84.errors import ContractError
from tre84.keys import Network, ResourceKey, Route, SourceKey, Stage, Tag
from tre84.loss import LossBreakdown
from tre84.state import CapacityState, ModelState, RiskInformation
from tre84.transition import DemandSplit, TransitionAudit, TransitionResult


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "experiments" / "5.3-1"
sys.path.insert(0, str(EXPERIMENT_ROOT))

from commitment_boundary.contracts import (  # noqa: E402
    ArtifactRef,
    ExperimentConfig,
    PathSpec,
    PolicySpec,
    PrecisionConfig,
    PreflightCheck,
    PreflightReport,
    REQUIRED_PREFLIGHT_CHECKS,
    TraceBundle,
    TraceMetadata,
    sha256_file,
)
from commitment_boundary.runner import (  # noqa: E402
    CommitmentBoundaryExperiment,
    _canonical_hash,
    hash_python_source_tree,
)


NETWORK = Network({"r": Route("r", "c", "g", "e", (1.0,))})
TAG = Tag("c", "r")
RESOURCES = {
    ResourceKey(Stage.BERTH, "g"): 100.0,
    ResourceKey(Stage.YARD, "g"): 100.0,
    ResourceKey(Stage.GATE, "g"): 100.0,
    ResourceKey(Stage.CORRIDOR, "e"): 100.0,
}


def _state(period: int, stage: str | None, mass: float) -> ModelState:
    queues = {name: {} for name in ("berth", "yard", "gate", "corridor")}
    if stage is not None and mass > 0:
        queues[stage] = {TAG: mass}
    return ModelState(
        period=period,
        horizon=10,
        risk=RiskInformation(np.array([1.0]), np.array([1.0])),
        disruption_seen=True,
        disruption_active=True,
        disruption_duration=period,
        waiting={"c": np.array([0.0])},
        berth=queues["berth"],
        yard=queues["yard"],
        gate=queues["gate"],
        corridor=queues["corridor"],
        maritime_pipeline=[],
        previous_shares={},
        corridor_history={("g", "e"): (1.0,)},
        serviceability_history=(0.0,),
        readiness=CapacityState(),
        direct_capacity=CapacityState(),
        budget=10.0,
    )


def _audit() -> TransitionAudit:
    return TransitionAudit(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, True, True)


def _capacity() -> CurrentCapacity:
    return CurrentCapacity(RESOURCES, {}, {}, {("g", "e"): 1.0})


def _equilibrium(decision_mass: float) -> EquilibriumResult:
    source = SourceKey("c", None)
    return EquilibriumResult(
        flows={
            source: {
                "r": decision_mass,
                "__WAIT__": 0.0,
                "__EXIT__": 0.0,
            }
        },
        releases={"c": np.array([0.0])},
        route_dispatch={("c", "r"): decision_mass},
        renewed_waiting={source: 0.0},
        direct_exit={source: 0.0},
        normalized_shares={source: {"r": 1.0} if decision_mass > 0 else {}},
        residual=0.0,
        kl_discrepancy=0.0,
        multi_start_dispersion=0.0,
        iterations=1,
        status="converged",
        starts=(StartRecord("synthetic", 0.0, 1, True),),
    )


def _kernel_result(
    *,
    next_state: ModelState,
    equilibrium: EquilibriumResult,
    split: DemandSplit,
    loss: LossBreakdown,
    delivered: float = 0.0,
) -> KernelResult:
    transition = TransitionResult(
        next_state=next_state,
        loss=loss,
        delivered={TAG: delivered} if delivered else {},
        direct_exit=equilibrium.direct_exit,
        duration_attrition={("c", 0): 0.0},
        demand_split=split,
        capacity=_capacity(),
        audit=_audit(),
    )
    return KernelResult(Action({}), None, equilibrium, transition)


class _SyntheticBackend:
    def __init__(self, *, model_hash: str, frozen_hash: str) -> None:
        self.model_hash = model_hash
        self.frozen_hash = frozen_hash
        self._pilot = tuple(
            PathSpec(f"pilot_{index}", f"pilot_hash_{index}", 10.0 + index)
            for index in range(3)
        )
        self._evaluation = tuple(
            PathSpec(f"eval_{index}", f"eval_hash_{index}", 12.0 + index)
            for index in range(3)
        )

    def preflight(self, config):
        return PreflightReport(
            checks=tuple(
                PreflightCheck(check_id, True, "synthetic contract evidence")
                for check_id in sorted(REQUIRED_PREFLIGHT_CHECKS)
            ),
            model_source_hash=self.model_hash,
            frozen_design_hash=self.frozen_hash,
            clearance_rule_hash="clearance_rule_hash",
        )

    def pilot_paths(self):
        return self._pilot

    def evaluation_paths(self):
        return self._evaluation

    def frozen_parameter_snapshot(self):
        return {"network": "synthetic_three_stage_clock", "capacity": "fixed"}

    def run_replication(self, *, policy, seed, path, chi):
        blocked = float(path.payload)
        committed = chi * blocked
        decision = (1.0 - chi) * blocked
        split = DemandSplit(
            blocked={"c": blocked},
            committed={"c": committed},
            decision_eligible={"c": decision},
            committed_by_tag={TAG: committed} if committed else {},
        )
        policy_loss = {
            "passive_equilibrium": 120.0,
            "reactive_coordination": 90.0 + 50.0 * chi,
            "stochastic_mpc": 100.0 + 5.0 * chi,
            "behaviour_cloning": 105.0 - 5.0 * chi,
            "model_guided_constrained_sac": 95.0 + 15.0 * chi,
        }[policy.policy_id]
        seed_adjustment = 0.0
        if seed.endswith("1"):
            seed_adjustment = -0.2
        elif seed.endswith("2"):
            seed_adjustment = 0.2
        initial = _state(0, None, 0.0)
        berth = _state(1, "berth", blocked)
        decision_result = _kernel_result(
            next_state=berth,
            equilibrium=_equilibrium(decision),
            split=split,
            loss=LossBreakdown(policy_loss + seed_adjustment, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        zero_split = DemandSplit(
            blocked={"c": 0.0},
            committed={"c": 0.0},
            decision_eligible={"c": 0.0},
            committed_by_tag={},
        )
        yard = _state(2, "yard", blocked)
        gate = _state(3, "gate", blocked)
        corridor = _state(4, "corridor", blocked)
        censored = policy.policy_id == "reactive_coordination" and chi == 1.0
        outstanding = 1.0 if censored else 0.0
        final = _state(5, "corridor" if outstanding else None, outstanding)
        clearance_results = (
            _kernel_result(next_state=yard, equilibrium=_equilibrium(0.0), split=zero_split, loss=LossBreakdown(0, 0, 0, 0, 0, 0)),
            _kernel_result(next_state=gate, equilibrium=_equilibrium(0.0), split=zero_split, loss=LossBreakdown(0, 0, 0, 0, 0, 0)),
            _kernel_result(next_state=corridor, equilibrium=_equilibrium(0.0), split=zero_split, loss=LossBreakdown(0, 0, 0, 0, 0, 0)),
            _kernel_result(
                next_state=final,
                equilibrium=_equilibrium(0.0),
                split=zero_split,
                loss=LossBreakdown(0, 0, 0, 0, 0, 0),
                delivered=blocked - outstanding,
            ),
        )
        clearance = ClearanceResult(
            final_state=final,
            weeks=4,
            cleared=not censored,
            right_censored=censored,
            operational_loss=0.0,
            terminal_correction=outstanding,
            transitions=clearance_results,
        )
        metadata = TraceMetadata(
            model_source_hash=self.model_hash,
            checkpoint_bundle_hash=policy.checkpoint_bundle_hash(seed),
            exogenous_path_hash=path.exogenous_path_hash,
            frozen_design_hash=self.frozen_hash,
            clearance_rule_hash="clearance_rule_hash",
            decision_times=("2026-02-23",),
            latest_release_times=("2026-01-31",),
            lead_forecast_months=(2,),
            selector_evaluated_bc_and_sac=True,
            selector_used_widehat_j_hc=True,
            actions_after_decision=0,
            untracked_base_arrivals=0.0,
            committed_route_reassignments=0,
            committed_pipeline_provenance_preserved=True,
            itinerary_tags_preserved_through_queues=True,
            new_source_choice_set_has_route_wait_exit=True,
            capacity_controls_active_for_committed_cargo=True,
            chi_zero_congestion_not_forced=True,
            legacy_result_reads=0,
        )
        return TraceBundle(initial, (decision_result,), clearance, NETWORK, RESOURCES, metadata)


def _policies(tmp_path: Path) -> tuple[PolicySpec, ...]:
    definitions = (
        ("passive_equilibrium", "Passive equilibrium", "passive", False, ("deterministic",)),
        ("reactive_coordination", "Reactive coordination", "reactive", False, ("deterministic",)),
        ("stochastic_mpc", "Stochastic MPC", "stochastic_mpc", False, ("deterministic",)),
        ("behaviour_cloning", "Behaviour cloning", "behaviour_cloning", True, ("seed1", "seed2")),
        (
            "model_guided_constrained_sac",
            "Model guided constrained SAC",
            "model_guided_constrained_sac",
            True,
            ("seed1", "seed2"),
        ),
    )
    policies = []
    for policy_id, label, kind, learned, seeds in definitions:
        artifacts = {}
        for seed in seeds:
            path = tmp_path / f"{policy_id}_{seed}.artifact"
            path.write_text(f"new 8.4 artifact {policy_id} {seed}", encoding="utf-8")
            artifacts[seed] = (
                ArtifactRef(f"{policy_id}_{seed}", path, sha256_file(path)),
            )
        policies.append(
            PolicySpec(
                policy_id=policy_id,
                label=label,
                kind=kind,
                learned=learned,
                seeds=seeds,
                artifacts=artifacts,
                training_chi=0.5 if learned else None,
                training_support=(0.25, 0.75) if learned else None,
            )
        )
    return tuple(policies)


def test_commitment_boundary_generates_complete_independent_artifact(tmp_path: Path) -> None:
    source_root = ROOT / "src" / "tre84"
    model_hash = hash_python_source_tree(source_root)
    snapshot = {"network": "synthetic_three_stage_clock", "capacity": "fixed"}
    frozen_hash = _canonical_hash(snapshot)
    config = ExperimentConfig(
        run_id="synthetic_contract_test",
        base_chi_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        policies=_policies(tmp_path),
        precision=PrecisionConfig(100.0, 0.95, 2, 20),
        interval_method="student",
        multiplicity_method="holm_fwer_with_bonferroni_intervals",
        decision_horizon=1,
        clearance_horizon=4,
        clearance_tolerance=1e-8,
        mass_tolerance=1e-8,
        residual_tolerance=1e-8,
        committed_itinerary_csv=(
            ROOT
            / "experiments"
            / "data"
            / "processed"
            / "anchors"
            / "committed_itinerary_reference.csv"
        ),
        backend_factory="unused:factory",
    )
    output = CommitmentBoundaryExperiment(
        config=config,
        backend=_SyntheticBackend(model_hash=model_hash, frozen_hash=frozen_hash),
        source_root=source_root,
    ).run(tmp_path / "run")

    required = {
        "commitment_replications.csv",
        "commitment_path_level.csv",
        "commitment_loss_components.csv",
        "commitment_flow_decomposition.csv",
        "commitment_paired_effects.csv",
        "commitment_policy_confidence_sets.csv",
        "commitment_endpoint_audit.csv",
        "pilot_precision.csv",
        "refinement_registry.csv",
        "commitment_policy_boundary.png",
        "commitment_mechanism_decomposition.png",
        "commitment_endpoint_contract.png",
        "parameter_registry.csv",
        "run_manifest.json",
    }
    assert required == {path.name for path in output.iterdir()}
    replications = pd.read_csv(output / "commitment_replications.csv")
    assert replications["mass_balance_error"].abs().max() <= 1e-8
    assert replications["loss_decomposition_error"].abs().max() <= 1e-8
    censored = replications[replications["right_censored"].astype(bool)]
    assert not censored.empty
    assert censored["clearance_weeks_observed"].isna().all()
    path_level = pd.read_csv(output / "commitment_path_level.csv")
    assert path_level.loc[
        path_level["policy"].isin(
            ["behaviour_cloning", "model_guided_constrained_sac"]
        ),
        "n_seeds_averaged",
    ].eq(2).all()
    endpoint = pd.read_csv(output / "commitment_endpoint_audit.csv")
    assert endpoint["audit_passed"].astype(bool).all()
    refinement = pd.read_csv(output / "refinement_registry.csv")
    assert refinement["recursive_refinement_allowed"].eq(False).all()
    assert set(refinement.loc[refinement["triggered"], "added_chi"]).issubset(
        {0.125, 0.375, 0.625, 0.875}
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["pilot_paths_excluded_from_final"] is True
    assert manifest["legacy_5_3_1_results_read"] is False


def test_commitment_preflight_fails_closed_when_one_contract_is_missing() -> None:
    missing = "no_fixed_exit_share"
    report = PreflightReport(
        checks=tuple(
            PreflightCheck(check_id, True, "evidence")
            for check_id in sorted(REQUIRED_PREFLIGHT_CHECKS - {missing})
        ),
        model_source_hash="model",
        frozen_design_hash="design",
        clearance_rule_hash="clearance",
    )
    with pytest.raises(ContractError, match="preflight failed"):
        report.require_passed()


def test_example_configuration_cannot_generate_placeholder_results() -> None:
    with pytest.raises(ContractError):
        ExperimentConfig.from_json(EXPERIMENT_ROOT / "experiment_config.example.json")
