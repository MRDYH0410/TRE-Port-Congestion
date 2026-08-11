from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tre84.information import FrozenStandardizer, GaussianHMM


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "5.2-1"
sys.path.insert(0, str(EXPERIMENT))

from counterfactual import run_rolling_origins
from event_input import construct_historical_event
from hmm_validity import HMMFitResult, build_release_clock, calendar_month_transitions
from interface import REQUIRED_COLUMNS, load_historical_path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rolling_origins_use_every_feasible_week_without_thinning() -> None:
    weeks = pd.date_range("2022-01-03", periods=180, freq="W-MON")
    weekly = pd.DataFrame(
        {
            "week_start": weeks,
            "observed_activity": 100.0 + 10.0 * np.sin(np.arange(180) / 8.0),
        }
    )
    config = {
        "candidate_models": ["seasonal_naive"],
        "minimum_training_weeks": 156,
        "maximum_forecast_horizon_weeks": 21,
        "seasonal_naive_lag_weeks": 52,
    }
    predictions = run_rolling_origins(weekly, config)
    one_step = predictions.loc[predictions["forecast_horizon"].eq(1)]
    assert one_step["origin_date"].nunique() == len(weekly) - 156
    assert not one_step["target_date"].duplicated().any()


def test_event_serviceability_blocked_identity_and_model_units() -> None:
    weeks = pd.date_range("2022-01-03", periods=177, freq="W-MON")
    observed = np.full(len(weeks), 1000.0)
    observed[-21:] = np.linspace(900.0, 500.0, 21)
    weekly = pd.DataFrame({"week_start": weeks, "observed_activity": observed})
    cutoff = weeks[-22]
    event, _ = construct_historical_event(
        weekly,
        selected_model="seasonal_naive",
        counterfactual_config={"seasonal_naive_lag_weeks": 52},
        selection_cutoff=cutoff,
        event_start=weeks[-21],
        event_end=weeks[-1],
        network_exposure_reference=0.0263,
        model_unit_tonnes=1000.0,
    )
    assert len(event) == 21
    assert event["serviceability"].between(0, 1).all()
    assert event["blocked_activity_proxy"].ge(0).all()
    assert np.allclose(event["blocked_identity_residual"], 0.0)
    assert np.allclose(
        event["model_blocked_units"],
        0.0263 * event["blocked_activity_proxy"] / 1000.0,
    )
    assert not any("chi" in column.lower() for column in event.columns)


def test_release_clock_uses_calendar_months_not_weekly_powers() -> None:
    model = GaussianHMM(
        initial=np.array([0.8, 0.2]),
        transition=np.array([[0.9, 0.1], [0.2, 0.8]]),
        means=np.array([[0.0], [1.0]]),
        variances=np.ones((2, 1)),
    )
    fit = HMMFitResult(
        model=model,
        standardizer=FrozenStandardizer(np.array([0.0]), np.array([1.0])),
        feature_names=("x",),
        training_rows=1,
        heldout_rows=1,
        selected_initialisation=0,
        selected_initialisation_name="test",
        likelihood_history=(0.0, 0.0),
        converged=True,
        initialisation_summary=(),
    )
    filtered = pd.DataFrame(
        {
            "observation_month": pd.to_datetime(["2025-12-01", "2026-01-01"]),
            "assumed_release_date": pd.to_datetime(["2026-01-31", "2026-02-28"]),
            "filtered_state_0_probability": [0.8, 0.7],
            "filtered_state_1_probability": [0.2, 0.3],
        }
    )
    clock = build_release_clock(
        [pd.Timestamp("2026-02-23")], filtered, fit, {"readiness_lead_weeks": 8}
    )
    assert clock.loc[0, "source_observation_month"] == pd.Timestamp("2025-12-01")
    expected = calendar_month_transitions(
        pd.Timestamp("2025-12-01"), pd.Timestamp("2026-04-20")
    )
    assert clock.loc[0, "monthly_transitions_to_maturity"] == expected == 4
    assert clock.loc[0, "weekly_transition_matrix_applications"] == 0
    assert bool(clock.loc[0, "timing_valid"])


def test_frozen_interface_detects_tampering(tmp_path: Path) -> None:
    weeks = pd.date_range("2026-02-23", periods=21, freq="W-MON")
    frame = pd.DataFrame({column: [0] * 21 for column in REQUIRED_COLUMNS})
    frame["week"] = weeks
    frame["source_observation_month"] = pd.Timestamp("2025-12-01")
    frame["release_date"] = pd.Timestamp("2026-01-31")
    frame["decision_cutoff"] = weeks
    frame["readiness_maturity_date"] = weeks + pd.Timedelta(weeks=8)
    frame["timing_valid"] = True
    frame["risk_information_source"] = "released_hmm_filter"
    frame["event_path_source"] = "formula_derived_5.2.1"
    frame["contains_committed_share"] = False
    path = tmp_path / "historical_information_event_path.csv"
    frame.to_csv(path, index=False)
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"outputs": [{"path": path.name, "sha256": _hash(path)}]}),
        encoding="utf-8",
    )
    assert len(load_historical_path(tmp_path)) == 21
    frame.loc[0, "risk_information_source"] = "manual_pre_event_ramp"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="does not match"):
        load_historical_path(tmp_path)


def _valid_interface(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    weeks = pd.date_range("2026-02-23", periods=21, freq="W-MON")
    frame = pd.DataFrame({column: [0] * 21 for column in REQUIRED_COLUMNS})
    frame["week"] = weeks
    frame["source_observation_month"] = pd.Timestamp("2025-12-01")
    frame["release_date"] = pd.Timestamp("2026-01-31")
    frame["decision_cutoff"] = weeks
    frame["readiness_maturity_date"] = weeks + pd.Timedelta(weeks=8)
    frame["timing_valid"] = True
    frame["risk_information_source"] = "released_hmm_filter"
    frame["event_path_source"] = "formula_derived_5.2.1"
    frame["contains_committed_share"] = False
    path = tmp_path / "historical_information_event_path.csv"
    frame.to_csv(path, index=False)
    return frame, path


def _write_manifest(tmp_path: Path, path: Path, *, include: bool = True) -> None:
    outputs = [{"path": path.name, "sha256": _hash(path)}] if include else []
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"outputs": outputs}), encoding="utf-8"
    )


def test_frozen_interface_fails_closed_when_manifest_record_is_missing(tmp_path: Path) -> None:
    _, path = _valid_interface(tmp_path)
    _write_manifest(tmp_path, path, include=False)
    with pytest.raises(ValueError, match="does not freeze"):
        load_historical_path(tmp_path)


def test_frozen_interface_fails_closed_on_late_release_even_with_matching_hash(tmp_path: Path) -> None:
    frame, path = _valid_interface(tmp_path)
    frame.loc[0, "release_date"] = frame.loc[0, "week"] + pd.Timedelta(days=1)
    frame.to_csv(path, index=False)
    _write_manifest(tmp_path, path)
    with pytest.raises(ValueError, match="unreleased information"):
        load_historical_path(tmp_path)


def test_frozen_interface_fails_closed_on_manual_ramp_even_with_matching_hash(tmp_path: Path) -> None:
    frame, path = _valid_interface(tmp_path)
    frame["risk_information_source"] = "manual_pre_event_ramp"
    frame.to_csv(path, index=False)
    _write_manifest(tmp_path, path)
    with pytest.raises(ValueError, match="Artificial risk paths"):
        load_historical_path(tmp_path)
