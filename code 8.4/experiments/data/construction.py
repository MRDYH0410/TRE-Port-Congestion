"""Section 5.1 transformations derived only from the current manuscript."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .quality import DataContractError


MODEL_UNIT_TONNES = 1_000.0
EVENT_START_WEEK = pd.Timestamp("2026-02-23")
EVENT_END_WEEK = pd.Timestamp("2026-07-13")
GPR_START_MONTH = pd.Timestamp("1985-01-01")
GPR_END_MONTH = pd.Timestamp("2026-06-01")
GATEWAY_NAME_MAP = {
    "Khor Fakkan": "Khor Fakkan",
    "Fujairah": "Fujairah",
    "Port of Sohar": "Sohar",
    "Jebel Ali": "Jebel Ali",
}
REFERENCE_GATEWAYS = ("Khor Fakkan", "Fujairah", "Sohar")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise DataContractError(f"{name} is missing required columns: {missing}")


def monday_week(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="raise")
    return (dates - pd.to_timedelta(dates.dt.weekday, unit="D")).dt.normalize()


def build_portwatch_weekly(
    daily: pd.DataFrame,
    *,
    location_columns: tuple[str, ...],
    activity_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Sum daily AIS proxies into Monday-based weeks without replacing zeros."""

    _require_columns(
        daily,
        ("date", *location_columns, *activity_columns),
        name="PortWatch daily panel",
    )
    frame = daily.loc[:, ("date", *location_columns, *activity_columns)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["week_start"] = monday_week(frame["date"])
    aggregation: dict[str, tuple[str, str]] = {
        "days_observed": ("date", "nunique"),
    }
    aggregation.update({column: (column, "sum") for column in activity_columns})
    weekly = (
        frame.groupby([*location_columns, "week_start"], sort=True, observed=True)
        .agg(**aggregation)
        .reset_index()
    )
    weekly["is_complete_week"] = weekly["days_observed"].eq(7)
    return weekly


def build_gateway_reference_scales(
    gateway_weekly: pd.DataFrame,
    *,
    event_start_week: pd.Timestamp = EVENT_START_WEEK,
    model_unit_tonnes: float = MODEL_UNIT_TONNES,
) -> pd.DataFrame:
    """Compute Equation (gateway-activity-scale) from complete pre-event weeks."""

    _require_columns(
        gateway_weekly,
        (
            "portname",
            "week_start",
            "is_complete_week",
            "import_container",
            "export_container",
        ),
        name="gateway weekly panel",
    )
    if model_unit_tonnes <= 0:
        raise DataContractError("The model conservation unit must be positive")
    frame = gateway_weekly.copy()
    frame["gateway"] = frame["portname"].map(GATEWAY_NAME_MAP)
    if frame["gateway"].isna().any():
        unknown = sorted(frame.loc[frame["gateway"].isna(), "portname"].unique())
        raise DataContractError(f"Unrecognised PortWatch port names: {unknown}")
    frame["container_activity"] = (
        frame["import_container"] + frame["export_container"]
    )
    pre_event = frame.loc[
        frame["is_complete_week"] & frame["week_start"].lt(event_start_week)
    ]
    scales = (
        pre_event.loc[pre_event["gateway"].isin(REFERENCE_GATEWAYS)]
        .groupby("gateway", sort=False, observed=True)["container_activity"]
        .quantile(0.90)
        .div(model_unit_tonnes)
        .rename("activity_scale_model_units")
        .reindex(REFERENCE_GATEWAYS)
        .reset_index()
    )
    if scales["activity_scale_model_units"].isna().any():
        raise DataContractError("All three reference gateway scales must be identified")
    scales["quantile"] = 0.90
    scales["pre_event_cutoff_exclusive"] = event_start_week
    scales["model_unit_tonnes"] = model_unit_tonnes
    return scales


def network_exposure_reference(
    gateway_scales: pd.DataFrame,
    chokepoint_weekly: pd.DataFrame,
    *,
    event_start_week: pd.Timestamp = EVENT_START_WEEK,
    model_unit_tonnes: float = MODEL_UNIT_TONNES,
) -> pd.DataFrame:
    """Compute the activity-scale alignment in Equation (network-exposure-reference)."""

    _require_columns(
        gateway_scales,
        ("gateway", "activity_scale_model_units"),
        name="gateway scale table",
    )
    _require_columns(
        chokepoint_weekly,
        ("week_start", "is_complete_week", "capacity_container"),
        name="chokepoint weekly panel",
    )
    pre_event = chokepoint_weekly.loc[
        chokepoint_weekly["is_complete_week"]
        & chokepoint_weekly["week_start"].lt(event_start_week),
        "capacity_container",
    ]
    denominator = float(pre_event.quantile(0.90) / model_unit_tonnes)
    numerator = float(gateway_scales["activity_scale_model_units"].sum())
    if denominator <= 0:
        raise DataContractError("The pre-event Hormuz activity scale must be positive")
    return pd.DataFrame(
        {
            "reference_network_exposure": [min(1.0, numerator / denominator)],
            "gateway_scale_sum": [numerator],
            "hormuz_q90_model_units": [denominator],
            "quantile": [0.90],
            "pre_event_cutoff_exclusive": [event_start_week],
            "model_unit_tonnes": [model_unit_tonnes],
        }
    )


def committed_itinerary_shares(capacity: pd.DataFrame) -> pd.DataFrame:
    """Normalise official annual nameplate values only as allocation anchors."""

    _require_columns(
        capacity,
        ("gateway", "annual_container_capacity_teu"),
        name="official gateway capacity anchors",
    )
    selected = capacity.loc[capacity["gateway"].isin(REFERENCE_GATEWAYS)].copy()
    selected = selected.set_index("gateway").reindex(REFERENCE_GATEWAYS).reset_index()
    if len(selected) != len(REFERENCE_GATEWAYS) or selected["annual_container_capacity_teu"].isna().any():
        raise DataContractError("Exactly one official capacity anchor is required per gateway")
    total = float(selected["annual_container_capacity_teu"].sum())
    if total <= 0:
        raise DataContractError("Official nameplate capacity must be positive")
    selected["committed_itinerary_share"] = (
        selected["annual_container_capacity_teu"] / total
    )
    selected["evidence_boundary"] = (
        "relative committed-itinerary allocation only; not weekly usable service"
    )
    return selected


def build_gpr_monthly(workbook: pd.DataFrame) -> pd.DataFrame:
    """Retain the official 1985-01 to 2026-06 GPR/threat/act monthly panel."""

    _require_columns(workbook, ("month", "GPR", "GPRT", "GPRA"), name="GPR workbook")
    frame = workbook.loc[:, ["month", "GPR", "GPRT", "GPRA"]].copy()
    frame["month"] = pd.to_datetime(frame["month"], errors="coerce")
    frame = frame.loc[frame["month"].between(GPR_START_MONTH, GPR_END_MONTH)].copy()
    frame = frame.rename(
        columns={"GPR": "gpr", "GPRT": "gpr_threat", "GPRA": "gpr_act"}
    ).sort_values("month", ignore_index=True)
    return frame


def build_gpr_continuous_features(
    monthly: pd.DataFrame, *, volatility_window: int = 24
) -> pd.DataFrame:
    """Build levels, first differences, and rolling difference volatility.

    The 24-month rolling window is the row-count-consistent construction that
    maps the declared 498 monthly observations to 474 usable observations.
    Jump indicators are deliberately not invented here: the first information
    experiment must declare its multiplier before calling
    :func:`add_gpr_jump_indicator`.
    """

    if volatility_window <= 1:
        raise DataContractError("GPR volatility window must exceed one month")
    _require_columns(
        monthly,
        ("month", "gpr_threat", "gpr_act"),
        name="GPR monthly panel",
    )
    result = monthly.copy()
    for column in ("gpr_threat", "gpr_act"):
        difference = f"{column}_diff"
        volatility = f"{column}_diff_volatility_{volatility_window}m"
        result[difference] = result[column].diff()
        result[volatility] = result[difference].rolling(
            volatility_window, min_periods=volatility_window
        ).std(ddof=1)
    feature_columns = [
        "gpr_threat_diff",
        "gpr_act_diff",
        f"gpr_threat_diff_volatility_{volatility_window}m",
        f"gpr_act_diff_volatility_{volatility_window}m",
    ]
    return result.dropna(subset=feature_columns).reset_index(drop=True)


def add_gpr_jump_indicator(
    features: pd.DataFrame, *, volatility_window: int, jump_sigma: float
) -> pd.DataFrame:
    """Add declared, auditable threat/act jump flags for the later HMM experiment."""

    if jump_sigma <= 0 or not np.isfinite(jump_sigma):
        raise DataContractError("The jump multiplier must be finite and positive")
    result = features.copy()
    for column in ("gpr_threat", "gpr_act"):
        difference = f"{column}_diff"
        volatility = f"{column}_diff_volatility_{volatility_window}m"
        _require_columns(result, (difference, volatility), name="GPR continuous features")
        result[f"{column}_jump"] = (
            result[difference].abs() > jump_sigma * result[volatility]
        ).astype("int8")
    return result

