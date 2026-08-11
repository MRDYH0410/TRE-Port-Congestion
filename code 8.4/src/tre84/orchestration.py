"""Five-module beginning-of-period preparation and execution graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .actions import Action
from .engine import KernelResult, ModelKernel
from .errors import ContractError
from .information import ReleaseRecord, ReleasedRiskInference, ReleasedRiskResult
from .scenarios import (
    CommonScenarioConstructor,
    ScenarioBundle,
    TimestampedOperationalContext,
)
from .state import ModelState, RiskInformation
from .transition import ExogenousRealization


@dataclass(frozen=True)
class InformationContext:
    decision_time: Any
    readiness_maturity_time: Any
    release_records: tuple[ReleaseRecord, ...]
    operational_context: TimestampedOperationalContext
    monthly_transition_count: Callable[[int | None, Any], int]

    def __post_init__(self) -> None:
        if not isinstance(self.operational_context, TimestampedOperationalContext):
            raise ContractError(
                "InformationContext requires timestamped operational evidence"
            )


@dataclass(frozen=True)
class PreparedPeriod:
    state: ModelState
    released_risk: ReleasedRiskResult
    scenarios: ScenarioBundle


class InformationScenarioModules:
    """Modules 1 and 2, with an explicit information-timestamp boundary."""

    def __init__(
        self,
        *,
        risk_inference: ReleasedRiskInference,
        scenario_constructor: CommonScenarioConstructor,
    ) -> None:
        self.risk_inference = risk_inference
        self.scenario_constructor = scenario_constructor

    def prepare(self, state: ModelState, context: InformationContext) -> PreparedPeriod:
        released = self.risk_inference.infer(
            decision_time=context.decision_time,
            readiness_maturity_time=context.readiness_maturity_time,
            records=context.release_records,
            monthly_transition_count=context.monthly_transition_count,
        )
        if any(timestamp > context.decision_time for timestamp in released.information_timestamps):
            raise ContractError("Unreleased risk information reached the decision state")
        service_timestamps = tuple(
            state.observed_covariates.get("serviceability_timestamps", ())
        )
        service_history = tuple(state.serviceability_history)
        if len(service_timestamps) != len(service_history):
            raise ContractError(
                "Every serviceability-history value needs one release timestamp"
            )
        completed_service = tuple(
            (value, timestamp)
            for value, timestamp in zip(service_history, service_timestamps)
            if timestamp <= context.decision_time
        )
        completed_service_history = tuple(value for value, _ in completed_service)
        completed_service_timestamps = tuple(timestamp for _, timestamp in completed_service)
        scenarios = self.scenario_constructor.build(
            lead_time_risk_forecast=released.lead_time_forecast,
            operational_context=context.operational_context,
            phase=state.phase,
            completed_information_timestamps=(
                *released.information_timestamps,
                *completed_service_timestamps,
            ),
            decision_time=context.decision_time,
        )
        prepared = state.clone()
        prepared.serviceability_history = completed_service_history
        prepared.observed_covariates["serviceability_timestamps"] = (
            completed_service_timestamps
        )
        prepared.risk = RiskInformation(
            belief=released.belief,
            lead_time_forecast=released.lead_time_forecast,
            scenario_ids=tuple(path.path_id for path in scenarios.paths),
            readiness_weights=scenarios.readiness_weights,
            operational_weights=scenarios.operational_weights,
            reclosure_probability=scenarios.reclosure_probability,
            latest_release_period=released.latest_observation_period,
            latest_release_time=released.latest_release_time,
        )
        return PreparedPeriod(prepared, released, scenarios)


class FiveModuleExecutor:
    """Module 3 supplies the action; the shared kernel executes Modules 4 and 5."""

    def __init__(self, modules_1_2: InformationScenarioModules, kernel: ModelKernel) -> None:
        self.modules_1_2 = modules_1_2
        self.kernel = kernel

    def execute_raw(
        self,
        *,
        state: ModelState,
        information: InformationContext,
        raw_action: Action,
        realization: ExogenousRealization,
    ) -> tuple[PreparedPeriod, KernelResult]:
        prepared = self.modules_1_2.prepare(state, information)
        result = self.kernel.execute_raw(
            state=prepared.state,
            raw_action=raw_action,
            realization=realization,
        )
        return prepared, result
