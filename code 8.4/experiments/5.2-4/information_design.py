"""Nonanticipative information regimes and release-timing stress construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from paths import PhysicalPath


@dataclass(frozen=True)
class FrozenHMMInputs:
    transition: np.ndarray
    stationary: np.ndarray
    packet_table: pd.DataFrame
    training_end: str
    training_rows: int


def _canonical_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    payload = frame.loc[:, list(columns)].copy()
    for column in payload.columns:
        if pd.api.types.is_datetime64_any_dtype(payload[column]):
            payload[column] = payload[column].dt.strftime("%Y-%m-%d")
    text = payload.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def monthly_transition_count(source_month: Any, maturity_date: Any) -> int:
    source = pd.Timestamp(source_month).to_period("M")
    maturity = pd.Timestamp(maturity_date).to_period("M")
    count = maturity.ordinal - source.ordinal
    if count < 0:
        raise ValueError("Readiness maturity precedes the released source month")
    return int(count)


def _stationary_distribution(transition: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(np.asarray(transition, dtype=float).T)
    index = int(np.argmin(np.abs(values - 1.0)))
    vector = np.real(vectors[:, index])
    vector = np.maximum(vector, 0.0)
    if vector.sum() <= 0:
        vector = np.abs(np.real(vectors[:, index]))
    return vector / vector.sum()


def load_hmm_inputs(output_5_2_1: Path) -> FrozenHMMInputs:
    manifest = pd.read_csv(output_5_2_1 / "hmm_parameter_manifest.csv")
    transition = np.zeros((2, 2), dtype=float)
    rows = manifest.loc[manifest["category"].eq("transition_matrix")]
    if len(rows) != 4:
        raise RuntimeError("MISSING_BASIS: the frozen 5.2.1 transition matrix is incomplete")
    for row in rows.itertuples(index=False):
        coordinates = str(row.parameter).removeprefix("P[").removesuffix("]").split(",")
        transition[int(coordinates[0]), int(coordinates[1])] = float(row.value)
    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1.0):
        raise RuntimeError("MISSING_BASIS: invalid frozen HMM transition matrix")
    packets = pd.read_csv(
        output_5_2_1 / "released_hmm_filter.csv",
        parse_dates=["observation_month", "assumed_release_date"],
    )
    required = {
        "observation_month",
        "assumed_release_date",
        "filtered_state_0_probability",
        "filtered_state_1_probability",
    }
    if not required.issubset(packets.columns):
        raise RuntimeError("MISSING_BASIS: released HMM packet columns are incomplete")
    design = manifest.set_index("parameter")
    return FrozenHMMInputs(
        transition=transition,
        stationary=_stationary_distribution(transition),
        packet_table=packets.sort_values(["assumed_release_date", "observation_month"]).reset_index(drop=True),
        training_end=str(design.loc["training_end", "value"]),
        training_rows=int(float(design.loc["training_rows", "value"])),
    )


def _latest_packet(packet_table: pd.DataFrame, decision_week: pd.Timestamp) -> pd.Series:
    eligible = packet_table.loc[packet_table["scenario_release_date"] <= decision_week]
    if eligible.empty:
        raise RuntimeError(f"No HMM information packet was available by {decision_week.date()}")
    return eligible.sort_values(["scenario_release_date", "observation_month"]).iloc[-1]


class ReleaseTimingScenarioBuilder:
    """Separate physical paths from historical or designed release timing."""

    def __init__(
        self,
        *,
        hmm: FrozenHMMInputs,
        event_onset: pd.Timestamp,
        readiness_lead_weeks: int,
        reference_normal_model_units: float,
    ) -> None:
        self.hmm = hmm
        self.event_onset = pd.Timestamp(event_onset)
        self.readiness_lead_weeks = int(readiness_lead_weeks)
        self.reference_normal_model_units = float(reference_normal_model_units)
        if self.readiness_lead_weeks < 1:
            raise RuntimeError("MISSING_BASIS: readiness lead must be positive")
        self.sufficient_week = self.event_onset - pd.Timedelta(weeks=self.readiness_lead_weeks)
        self.insufficient_week = self.sufficient_week + pd.Timedelta(weeks=1)
        historical = self.hmm.packet_table.loc[
            self.hmm.packet_table["assumed_release_date"] < self.event_onset
        ]
        if historical.empty:
            raise RuntimeError("MISSING_BASIS: no event-pre HMM packet exists")
        self.target_observation_month = pd.Timestamp(
            historical.sort_values(["assumed_release_date", "observation_month"]).iloc[-1][
                "observation_month"
            ]
        )

    def scenario_registry(self) -> pd.DataFrame:
        target = self.hmm.packet_table.loc[
            self.hmm.packet_table["observation_month"].eq(self.target_observation_month)
        ].iloc[-1]
        rows = []
        for scenario, label, release, physical, evidence in (
            ("GH", "Historical release", pd.Timestamp(target["assumed_release_date"]), "historical event", "historical information replay"),
            ("GT", "Minimally sufficient true release", self.sufficient_week, "historical event", "designed information timing stress"),
            ("GL", "Insufficient lead", self.insufficient_week, "historical event", "designed information timing stress"),
            ("GFW", "False warning", self.sufficient_week, "no disruption", "designed information timing stress"),
        ):
            availability = release + pd.Timedelta(days=(7 - release.weekday()) % 7)
            gap_r = int((self.event_onset - availability).days // 7 - self.readiness_lead_weeks)
            rows.append(
                {
                    "warning_scenario": scenario,
                    "label": label,
                    "target_information_vintage": self.target_observation_month.date(),
                    "historical_release_date": pd.Timestamp(target["assumed_release_date"]).date(),
                    "scenario_release_date": release.date(),
                    "decision_availability_week": availability.date(),
                    "event_onset": self.event_onset.date(),
                    "readiness_lead_weeks": self.readiness_lead_weeks,
                    "g_R_weeks": gap_r,
                    "physical_event": physical,
                    "evidence_class": evidence,
                }
            )
        return pd.DataFrame(rows)

    def _packet_schedule(self, scenario: str) -> pd.DataFrame:
        packets = self.hmm.packet_table.copy()
        packets["scenario_release_date"] = packets["assumed_release_date"]
        target = packets["observation_month"].eq(self.target_observation_month)
        if scenario in {"GT", "GFW"}:
            packets.loc[target, "scenario_release_date"] = self.sufficient_week
        elif scenario == "GL":
            packets.loc[target, "scenario_release_date"] = self.insufficient_week
        elif scenario != "GH":
            raise ValueError(f"Unknown warning scenario {scenario}")
        return packets.sort_values(["scenario_release_date", "observation_month"]).reset_index(drop=True)

    def _physical_frame(self, base_path: PhysicalPath, scenario: str) -> pd.DataFrame:
        event = base_path.frame.copy()
        event["week"] = pd.to_datetime(event["week"])
        event["release_date"] = pd.to_datetime(event["release_date"])
        event["source_observation_month"] = pd.to_datetime(event["source_observation_month"])
        preparation_weeks = pd.date_range(
            self.sufficient_week,
            self.event_onset - pd.Timedelta(weeks=1),
            freq="W-MON",
        )
        preparation = pd.DataFrame(
            {
                "week": preparation_weeks,
                "normal_model_units": self.reference_normal_model_units,
                "serviceability": 1.0,
                "residual_date": pd.NaT,
            }
        )
        keep = ["week", "normal_model_units", "serviceability", "residual_date"]
        physical = pd.concat([preparation[keep], event[keep]], ignore_index=True)
        if scenario == "GFW":
            physical["serviceability"] = 1.0
        physical["preparation_period"] = physical["week"] < self.event_onset
        physical["event_period"] = physical["week"] >= self.event_onset
        return physical

    def build(self, base_path: PhysicalPath, scenario: str) -> PhysicalPath:
        physical = self._physical_frame(base_path, scenario)
        packets = self._packet_schedule(scenario)
        rows: list[dict[str, Any]] = []
        for item in physical.itertuples(index=False):
            week = pd.Timestamp(item.week)
            packet = _latest_packet(packets, week)
            belief = np.asarray(
                [packet["filtered_state_0_probability"], packet["filtered_state_1_probability"]],
                dtype=float,
            )
            belief /= belief.sum()
            maturity = week + pd.Timedelta(weeks=self.readiness_lead_weeks)
            transitions = monthly_transition_count(packet["observation_month"], maturity)
            lead = belief @ np.linalg.matrix_power(self.hmm.transition, transitions)
            row = dict(item._asdict())
            row.update(
                {
                    "filtered_high_risk_probability": float(belief[-1]),
                    "lead_time_high_risk_probability": float(lead[-1]),
                    "release_date": pd.Timestamp(packet["scenario_release_date"]),
                    "actual_public_release_date": pd.Timestamp(packet["assumed_release_date"]),
                    "source_observation_month": pd.Timestamp(packet["observation_month"]),
                    "timing_valid": bool(pd.Timestamp(packet["scenario_release_date"]) <= week),
                    "information_source": "5.2.1_release_clock" if scenario == "GH" else "designed_release_timing_stress",
                    "warning_scenario": scenario,
                    "monthly_transitions_to_readiness_maturity": transitions,
                    "readiness_maturity_date": maturity,
                    "weekly_transition_matrix_applications": 0,
                }
            )
            for horizon in range(5):
                forecast = belief @ np.linalg.matrix_power(self.hmm.transition, horizon)
                row[f"high_risk_forecast_h{horizon}_months"] = float(forecast[-1])
            rows.append(row)
        frame = pd.DataFrame(rows)
        physical_hash = _canonical_hash(
            frame,
            ["week", "normal_model_units", "serviceability", "residual_date"],
        )
        frame["base_path_id"] = base_path.path_id
        frame["base_physical_path_sha256"] = physical_hash
        frame["event_onset"] = self.event_onset
        path_hash = _canonical_hash(frame, sorted(frame.columns))
        active = frame["serviceability"].to_numpy(dtype=float) < 1.0 - 1e-12
        onset = int(np.argmax(active)) if active.any() else len(frame)
        return PhysicalPath(
            path_id=f"{base_path.path_id}__{scenario}",
            split="test",
            frame=frame,
            path_hash=path_hash,
            construction=(
                "5.2.2 matched physical path plus 5.2.1 historical release clock"
                if scenario == "GH"
                else "5.2.2 matched physical path plus designed information timing stress"
            ),
            residual_start=base_path.residual_start,
            residual_end=base_path.residual_end,
            onset_week=onset,
            active_duration_weeks=int(active.sum()),
            severity_floor=float(frame["serviceability"].min()),
            has_reclosure=bool(any((~active[:-1]) & active[1:])) if len(active) > 1 else False,
        )


class InformationProvider:
    """Return only the information legal under the selected regime."""

    def __init__(self, *, hmm: FrozenHMMInputs, readiness_lead_weeks: int) -> None:
        self.hmm = hmm
        self.readiness_lead_weeks = int(readiness_lead_weeks)

    def apply(self, path: PhysicalPath, regime: str) -> PhysicalPath:
        frame = path.frame.copy()
        current_released = frame["filtered_high_risk_probability"].to_numpy(dtype=float)
        lead_released = frame["lead_time_high_risk_probability"].to_numpy(dtype=float)
        if regime == "I0":
            current = np.full(len(frame), float(self.hmm.stationary[-1]))
            lead = np.full(len(frame), float(self.hmm.stationary[-1]))
            source = "training_stationary_unconditional_comparator"
        elif regime == "IF":
            current = current_released
            lead = current_released.copy()
            source = "released_current_state_comparator_repeated_at_lead_coordinate"
        elif regime == "IL":
            current = current_released
            lead = lead_released
            source = "released_lead_aligned_hmm_forecast"
        elif regime == "ORACLE":
            service = frame["serviceability"].to_numpy(dtype=float)
            active = (service < 1.0 - 1e-12).astype(float)
            current = active
            indices = np.minimum(
                np.arange(len(frame)) + self.readiness_lead_weeks,
                len(frame) - 1,
            )
            lead = active[indices]
            source = "unattainable_realized_physical_state_oracle"
        else:
            raise ValueError(f"Unknown information regime {regime}")
        frame["released_filtered_high_risk_probability"] = current_released
        frame["released_lead_high_risk_probability"] = lead_released
        frame["filtered_high_risk_probability"] = np.clip(current, 0.0, 1.0)
        frame["lead_time_high_risk_probability"] = np.clip(lead, 0.0, 1.0)
        frame["information_regime"] = regime
        frame["controller_information_source"] = source
        frame["information_source"] = source
        path_hash = _canonical_hash(frame, sorted(frame.columns))
        return PhysicalPath(
            path_id=f"{path.path_id}__{regime}",
            split=path.split,
            frame=frame,
            path_hash=path_hash,
            construction=f"{path.construction}; information regime {regime}",
            residual_start=path.residual_start,
            residual_end=path.residual_end,
            onset_week=path.onset_week,
            active_duration_weeks=path.active_duration_weeks,
            severity_floor=path.severity_floor,
            has_reclosure=path.has_reclosure,
        )

    def apply_training_regime(self, path: PhysicalPath, regime: str) -> PhysicalPath:
        frame = path.frame.copy()
        frame["actual_public_release_date"] = pd.to_datetime(frame["release_date"])
        frame["monthly_transitions_to_readiness_maturity"] = np.nan
        frame["readiness_maturity_date"] = pd.to_datetime(frame["week"]) + pd.Timedelta(
            weeks=self.readiness_lead_weeks
        )
        frame["weekly_transition_matrix_applications"] = 0
        frame["warning_scenario"] = "TRAINING_SUPPORT"
        frame["base_path_id"] = path.path_id
        frame["base_physical_path_sha256"] = _canonical_hash(
            frame, ["week", "normal_model_units", "serviceability", "residual_date"]
        )
        enriched = PhysicalPath(
            path_id=path.path_id,
            split=path.split,
            frame=frame,
            path_hash=path.path_hash,
            construction=path.construction,
            residual_start=path.residual_start,
            residual_end=path.residual_end,
            onset_week=path.onset_week,
            active_duration_weeks=path.active_duration_weeks,
            severity_floor=path.severity_floor,
            has_reclosure=path.has_reclosure,
        )
        return self.apply(enriched, regime)


def information_regime_registry(hmm: FrozenHMMInputs) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "information_regime": "I0",
                "label": "No newly released risk information",
                "definition": "stationary training-sample comparator at current and lead coordinates",
                "baseline_high_risk_probability": float(hmm.stationary[-1]),
                "oracle": False,
                "evidence_class": "implementable comparator",
            },
            {
                "information_regime": "IF",
                "label": "Current filtered risk only",
                "definition": "latest released current belief repeated at the lead coordinate",
                "baseline_high_risk_probability": np.nan,
                "oracle": False,
                "evidence_class": "estimated current-state comparator",
            },
            {
                "information_regime": "IL",
                "label": "Released lead-aligned forecast",
                "definition": "alpha_nu(t) P^h on the native monthly clock",
                "baseline_high_risk_probability": np.nan,
                "oracle": False,
                "evidence_class": "estimated implementable information",
            },
            {
                "information_regime": "ORACLE",
                "label": "Perfect information oracle",
                "definition": "realized current and readiness-maturity disruption states",
                "baseline_high_risk_probability": np.nan,
                "oracle": True,
                "evidence_class": "unattainable theoretical upper bound",
            },
        ]
    )
