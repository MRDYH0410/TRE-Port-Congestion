from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.data.construction import (
    add_gpr_jump_indicator,
    build_gpr_continuous_features,
    build_portwatch_weekly,
)
from experiments.data.prepare import build_shared_data


def test_frozen_section_51_data_build_passes_all_contracts() -> None:
    report = build_shared_data()
    assert report.passed
    assert len(report.checks) == 48


def test_monday_aggregation_retains_observed_zero_activity() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2026-02-23", periods=7, freq="D"),
            "portname": ["Fujairah"] * 7,
            "import_container": [0] * 7,
            "export_container": [0] * 7,
        }
    )
    weekly = build_portwatch_weekly(
        daily,
        location_columns=("portname",),
        activity_columns=("import_container", "export_container"),
    )
    assert len(weekly) == 1
    assert weekly.loc[0, "week_start"] == pd.Timestamp("2026-02-23")
    assert bool(weekly.loc[0, "is_complete_week"])
    assert weekly.loc[0, "import_container"] + weekly.loc[0, "export_container"] == 0


def test_gpr_base_transform_has_474_rows_and_jump_rule_is_explicit() -> None:
    months = pd.date_range("1985-01-01", periods=498, freq="MS")
    monthly = pd.DataFrame(
        {
            "month": months,
            "gpr": np.linspace(80.0, 120.0, len(months)),
            "gpr_threat": np.sin(np.arange(len(months)) / 8.0) + 100.0,
            "gpr_act": np.cos(np.arange(len(months)) / 9.0) + 100.0,
        }
    )
    features = build_gpr_continuous_features(monthly, volatility_window=24)
    assert len(features) == 474
    assert not any(column.endswith("_jump") for column in features)
    declared = add_gpr_jump_indicator(
        features, volatility_window=24, jump_sigma=2.0
    )
    assert {"gpr_threat_jump", "gpr_act_jump"}.issubset(declared)

