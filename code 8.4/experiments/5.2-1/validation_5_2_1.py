"""Independent output recomputation and Stage-1 reporting for Experiment 5.2.1.

This module deliberately reads the generated CSV files rather than calling the
counterfactual, HMM, or event-construction implementations.  It therefore
provides a separate arithmetic and contract check of the formal output layer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(
    rows: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    observed: Any,
    expected: Any,
    detail: str,
) -> None:
    rows.append(
        {
            "check_id": check_id,
            "passed": bool(passed),
            "observed": json.dumps(observed, sort_keys=True, default=str),
            "expected": json.dumps(expected, sort_keys=True, default=str),
            "detail": detail,
        }
    )


def independent_recalculation(
    output_dir: Path, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Recalculate all central Stage-1 identities from formal CSV files."""

    tolerance = float(config["acceptance"]["numeric_tolerance"])
    rows: list[dict[str, Any]] = []
    audit = pd.read_csv(output_dir / "data_audit.csv")
    predictions = pd.read_csv(
        output_dir / "counterfactual_rolling_origin_predictions.csv",
        parse_dates=["origin_date", "target_date"],
    )
    summary = pd.read_csv(output_dir / "counterfactual_rolling_origin_summary.csv")
    selection = pd.read_csv(output_dir / "counterfactual_model_selection.csv")
    residuals = pd.read_csv(output_dir / "counterfactual_residual_library.csv")
    release = pd.read_csv(
        output_dir / "release_clock.csv",
        parse_dates=[
            "decision_week",
            "source_observation_month",
            "release_date",
            "decision_cutoff",
            "readiness_maturity_date",
        ],
    )
    event = pd.read_csv(
        output_dir / "historical_information_event_path.csv",
        parse_dates=["week", "release_date", "decision_cutoff"],
    )
    replay = output_dir / "historical_event_replay.csv"
    manifest = pd.read_csv(output_dir / "hmm_parameter_manifest.csv")

    _record(
        rows,
        "raw_hashes_dimensions_dates_missingness_reconciled",
        bool(audit["audit_status"].eq("pass").all())
        and bool(audit["sha256"].eq(audit["expected_sha256"]).all())
        and bool(audit["used_field_missing_cells"].eq(0).all()),
        {
            "sources": len(audit),
            "hash_mismatches": int((audit["sha256"] != audit["expected_sha256"]).sum()),
            "used_missing_cells": int(audit["used_field_missing_cells"].sum()),
        },
        {"hash_mismatches": 0, "used_missing_cells": 0},
        "Raw-file hashes, dimensions, date ranges, duplicates, and used-field missingness are recorded in data_audit.csv.",
    )

    monday = event["week"].dt.weekday.eq(0)
    expected_weeks = pd.date_range(
        config["data"]["event_start_week"],
        config["data"]["event_end_week"],
        freq="W-MON",
    )
    _record(
        rows,
        "event_calendar_independent_recalculation",
        len(event) == int(config["data"]["expected_event_weeks"])
        and bool(monday.all())
        and event["week"].reset_index(drop=True).equals(pd.Series(expected_weeks)),
        {"rows": len(event), "all_monday": bool(monday.all())},
        {"rows": 21, "all_monday": True},
        "The formal interface is exactly the 21 Monday weeks from 2026-02-23 through 2026-07-13.",
    )

    origin_counts = (
        predictions.loc[predictions["forecast_horizon"].eq(1)]
        .groupby("model")["origin_date"]
        .nunique()
        .to_dict()
    )
    _record(
        rows,
        "rolling_origins_independent_recalculation",
        bool(origin_counts) and all(value == 216 for value in origin_counts.values()),
        origin_counts,
        {model: 216 for model in config["counterfactual"]["candidate_models"]},
        "Every candidate uses all 216 feasible one-step weekly origins; no origin thinning is present.",
    )

    metric_deltas: dict[str, float] = {"wape": 0.0, "mae": 0.0, "bias": 0.0}
    count_mismatches = 0
    for row in summary.itertuples(index=False):
        model_rows = predictions.loc[predictions["model"].eq(row.model)]
        maximum = model_rows.groupby("origin_date")["available_horizon_at_origin"].max()
        origins = maximum.index[maximum.ge(int(row.evaluation_horizon_weeks))]
        subset = model_rows.loc[
            model_rows["origin_date"].isin(origins)
            & model_rows["forecast_horizon"].le(int(row.evaluation_horizon_weeks))
        ]
        error = subset["predicted_value"] - subset["observed_value"]
        values = {
            "wape": float(error.abs().sum() / subset["observed_value"].abs().sum()),
            "mae": float(error.abs().mean()),
            "bias": float(error.mean()),
        }
        for metric, value in values.items():
            metric_deltas[metric] = max(metric_deltas[metric], abs(value - float(getattr(row, metric))))
        count_mismatches += int(len(origins) != int(row.origin_count))
        count_mismatches += int(len(subset) != int(row.point_count))
    metric_ok = (
        metric_deltas["wape"] <= tolerance
        and metric_deltas["mae"] <= 1e-6
        and metric_deltas["bias"] <= 1e-6
        and count_mismatches == 0
    )
    _record(
        rows,
        "counterfactual_metrics_independently_recomputed",
        metric_ok,
        {**metric_deltas, "count_mismatches": count_mismatches},
        {"wape": tolerance, "mae": 1e-6, "bias": 1e-6, "count_mismatches": 0},
        "WAPE, MAE, Bias, origin counts, and point counts are recomputed directly from the prediction CSV.",
    )

    primary = summary.loc[
        summary["evaluation_horizon_weeks"].eq(
            int(config["counterfactual"]["primary_selection_horizon_weeks"])
        )
    ].copy()
    best_wape = float(primary["wape"].min())
    eligible = primary.loc[
        primary["wape"].le(
            best_wape
            * (1.0 + float(config["counterfactual"]["relative_wape_tie_tolerance"]))
            + np.finfo(float).eps
        )
    ].copy()
    dependence = selection.set_index("model")["one_step_residual_dependence"]
    eligible["absolute_bias"] = eligible["bias"].abs()
    eligible["dependence"] = eligible["model"].map(dependence)
    independently_selected = str(
        eligible.sort_values(["absolute_bias", "dependence", "model"]).iloc[0]["model"]
    )
    formally_selected = str(selection.loc[selection["selected"].astype(bool), "model"].iloc[0])
    _record(
        rows,
        "counterfactual_selection_rule_independently_replayed",
        independently_selected == formally_selected,
        independently_selected,
        formally_selected,
        "The frozen 21-week WAPE rule and declared tie breaks are replayed without event-period tuning.",
    )

    selected_one_step = predictions.loc[
        predictions["model"].eq(formally_selected)
        & predictions["forecast_horizon"].eq(1)
    ].sort_values("target_date")
    residuals_sorted = residuals.sort_values("residual_date")
    residual_ok = (
        len(residuals_sorted) == 216
        and residuals_sorted["residual_date"].astype(str).tolist()
        == selected_one_step["target_date"].dt.strftime("%Y-%m-%d").tolist()
        and np.allclose(
            residuals_sorted["residual"].to_numpy(float),
            selected_one_step["residual"].to_numpy(float),
            rtol=0.0,
            atol=tolerance,
        )
        and bool(pd.to_datetime(residuals_sorted["residual_date"]).le("2026-02-16").all())
    )
    _record(
        rows,
        "residual_library_independently_matched",
        residual_ok,
        {"rows": len(residuals_sorted), "selected_model": formally_selected},
        {"rows": 216, "event_free": True},
        "The residual library is exactly the selected model's unique event-free one-step residual series.",
    )

    parameters = manifest.set_index("parameter")["value"].to_dict()
    _record(
        rows,
        "hmm_split_and_semantics_independently_checked",
        str(parameters.get("training_rows")) == "456"
        and str(parameters.get("heldout_rows")) == "18"
        and not manifest.astype(str).apply(lambda column: column.str.contains("closure label", case=False)).any().any(),
        {
            "training_rows": parameters.get("training_rows"),
            "heldout_rows": parameters.get("heldout_rows"),
        },
        {"training_rows": 456, "heldout_rows": 18, "closure_label_used": False},
        "The two-state HMM uses the frozen chronological split and is interpreted only as a geopolitical-risk state model.",
    )

    source_period = release["source_observation_month"].dt.to_period("M")
    maturity_period = release["readiness_maturity_date"].dt.to_period("M")
    month_steps = np.asarray(
        [m.ordinal - s.ordinal for s, m in zip(source_period, maturity_period)], dtype=int
    )
    release_ok = (
        bool((release["release_date"] <= release["decision_cutoff"]).all())
        and np.array_equal(month_steps, release["monthly_transitions_to_maturity"].to_numpy(int))
        and bool(release["weekly_transition_matrix_applications"].eq(0).all())
    )
    _record(
        rows,
        "release_clock_and_monthly_propagation_independently_checked",
        release_ok,
        {
            "late_releases": int((release["release_date"] > release["decision_cutoff"]).sum()),
            "month_step_mismatches": int(
                (month_steps != release["monthly_transitions_to_maturity"].to_numpy(int)).sum()
            ),
            "weekly_matrix_applications": int(release["weekly_transition_matrix_applications"].sum()),
        },
        {"late_releases": 0, "month_step_mismatches": 0, "weekly_matrix_applications": 0},
        "The source-month-to-maturity power is a calendar-month difference; the monthly matrix is never applied weekly.",
    )

    blocked_identity = (
        (1.0 - event["serviceability"]) * event["estimated_no_disruption_activity"]
    )
    converted = (
        event["network_exposure_reference"]
        * event["blocked_activity_proxy"]
        / event["model_unit_tonnes"]
    )
    event_ok = (
        bool(event["serviceability"].between(0.0, 1.0).all())
        and bool(event["blocked_activity_proxy"].ge(0.0).all())
        and np.allclose(blocked_identity, event["blocked_activity_proxy"], rtol=0.0, atol=1e-6)
        and np.allclose(converted, event["model_blocked_units"], rtol=0.0, atol=1e-9)
    )
    _record(
        rows,
        "event_identities_independently_recomputed",
        event_ok,
        {
            "max_blocked_identity_error": float(np.max(np.abs(blocked_identity - event["blocked_activity_proxy"]))),
            "max_unit_error": float(np.max(np.abs(converted - event["model_blocked_units"]))),
        },
        {"serviceability_in_0_1": True, "blocked_nonnegative": True},
        "Serviceability, positive shortfall, and U_Q conversion are recomputed from the formal interface.",
    )

    unique_ok = _sha256(replay) == _sha256(output_dir / "historical_information_event_path.csv")
    source_ok = (
        set(event["risk_information_source"]) == {"released_hmm_filter"}
        and set(event["event_path_source"]) == {"formula_derived_5.2.1"}
    )
    _record(
        rows,
        "unique_downstream_interface_and_no_manual_ramp",
        unique_ok and source_ok,
        {
            "replay_sha256": _sha256(replay),
            "interface_sha256": _sha256(output_dir / "historical_information_event_path.csv"),
            "risk_sources": sorted(event["risk_information_source"].unique()),
        },
        {"identical_interfaces": True, "risk_source": ["released_hmm_filter"]},
        "The replay and unique interface are byte-identical and contain no artificial risk ramp.",
    )
    return pd.DataFrame(rows)


def figure_data_tables(
    summary: pd.DataFrame,
    acf: pd.DataFrame,
    selection: pd.DataFrame,
    density_summary: pd.DataFrame,
    filtered: pd.DataFrame,
    interface: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    selected = selection.set_index("model")["selected"].to_dict()
    panel_a = summary.copy()
    panel_a.insert(0, "panel", "A_WAPE_B_BIAS")
    panel_a["selected_model"] = panel_a["model"].map(selected).fillna(False)
    panel_c = acf.copy()
    panel_c.insert(0, "panel", "C_ONE_STEP_RESIDUAL_ACF")
    figure_a = pd.concat([panel_a, panel_c], ignore_index=True, sort=False)

    hmm_a = density_summary.copy()
    hmm_a.insert(0, "panel", "A_HELDOUT_LPD")
    hmm_b = filtered.copy()
    hmm_b.insert(0, "panel", "B_FILTERED_RISK_STATE")
    figure_b = pd.concat([hmm_a, hmm_b], ignore_index=True, sort=False)

    figure_c = interface.copy()
    figure_c.insert(0, "panel", "A_EVENT_INPUT_B_RELEASE_CLOCK")
    return {
        "figure_5_2_1a_data.csv": figure_a,
        "figure_5_2_1b_data.csv": figure_b,
        "figure_5_2_1c_data.csv": figure_c,
    }


def write_stage_reports(
    output_dir: Path,
    *,
    status: str,
    selected_model: str,
    baseline_comparison: Mapping[str, Any] | None = None,
) -> None:
    summary = pd.read_csv(output_dir / "counterfactual_rolling_origin_summary.csv")
    selection = pd.read_csv(output_dir / "counterfactual_model_selection.csv")
    density = pd.read_csv(output_dir / "heldout_density_summary.csv")
    event = pd.read_csv(output_dir / "historical_information_event_path.csv")
    checks = pd.read_csv(output_dir / "independent_recalculation_checks.csv")
    selected_row = selection.loc[selection["selected"].astype(bool)].iloc[0]
    pivot = density.pivot(index="horizon_months", columns="forecast_model", values="mean_log_predictive_density")
    event_last = event.iloc[-1]
    replication = "pending baseline comparison"
    if baseline_comparison is not None:
        replication = str(baseline_comparison.get("summary", baseline_comparison))
    result_lines = [
        "# 5.2.1 Results and Figure Analysis",
        "",
        f"## Acceptance status: {status.upper()}",
        "",
        "This rerun validates the frozen data, event-free counterfactual, released geopolitical-risk state information, and formula-derived historical event input. It does not train, rank, or evaluate any policy.",
        "",
        "## Reproduction result",
        "",
        f"- Scientific-output comparison with the pre-rerun baseline: {replication}.",
        "- This is not a claim of full byte-for-byte reproduction when the count above is below the comparison total. Any such difference is retained in baseline_reproduction_audit.csv; the frozen residual library, HMM/release outputs, and unique downstream interface must still reproduce exactly.",
        f"- Independent arithmetic and contract checks: {int(checks['passed'].sum())}/{len(checks)} passed.",
        "- The unique downstream interface is fail-closed and hash-frozen in run_manifest.json.",
        "",
        "## Figure 5.2.1a: Counterfactual predictive validity",
        "",
        f"The frozen rule selects `{selected_model}`. Its 21-week cumulative-path WAPE is {float(selected_row['wape']):.4f}, MAE is {float(selected_row['mae']):,.1f} proxy tonnes, and Bias is {float(selected_row['bias']):,.1f} proxy tonnes. All three candidates use 216 feasible weekly one-step origins; shorter horizons remain diagnostics and do not alter selection.",
        "",
        "## Figure 5.2.1b: Released HMM validity",
        "",
        "The two-state HMM is a geopolitical-risk regime model, not a Hormuz closure classifier. Held-out mean log predictive densities are reported below without hiding weaker horizons:",
        "",
        "| Horizon (months) | HMM transition | Unconditional | Persistence |",
        "|---:|---:|---:|---:|",
    ]
    for horizon, row in pivot.iterrows():
        result_lines.append(
            f"| {int(horizon)} | {row['hmm_transition']:.4f} | {row['unconditional']:.4f} | {row['persistence']:.4f} |"
        )
    result_lines.extend(
        [
            "",
            "The monthly transition matrix is powered by calendar-month distance from the released source month to readiness maturity; it is never applied once per week.",
            "",
            "## Figure 5.2.1c: Event input and release clock",
            "",
            f"The interface contains 21 complete Monday weeks. On 2026-07-13, formula-derived serviceability is {float(event_last['serviceability']):.6f}, estimated blocked activity is {float(event_last['blocked_activity_proxy']):,.1f} AIS-proxy tonnes, and converted blocked mass is {float(event_last['model_blocked_units']):,.3f} model units.",
            "",
            "Blocked activity is an estimated positive shortfall, not observed diversion. The HMM belief and lead-aligned forecast are geopolitical-risk state probabilities, not closure probabilities or closure-date forecasts.",
            "",
            "## Evidence boundary",
            "",
            "The output is admissible only as the unique frozen input to later experiments. It provides no evidence about policy superiority, readiness economic value, historical committed share, gateway expansion, or reclosure boundaries.",
        ]
    )
    (output_dir / "FIGURE_AND_RESULTS_ANALYSIS.md").write_text(
        "\n".join(result_lines) + "\n", encoding="utf-8"
    )

    parameter_lines = [
        "# 5.1 Parameter and Metric Additions for the 5.2.1 Rerun",
        "",
        "DATA SOURCE SECTION: NO CHANGE",
        "",
        "DATA TABLE: NO CHANGE",
        "",
        "PARAMETER AND METRIC ADDITIONS: NO ADDITION",
        "",
        "The rerun uses the already frozen 5.1 data sources, calendar, units, counterfactual specifications, HMM settings, release-clock rule, and acceptance tolerances. No data source, parameter, metric, formula, threshold, or model-selection rule was added or changed.",
    ]
    (output_dir / "5_1_PARAMETER_AND_METRIC_ADDITIONS.md").write_text(
        "\n".join(parameter_lines) + "\n", encoding="utf-8"
    )


def verify_manifest_hashes(output_dir: Path) -> dict[str, Any]:
    """Verify every non-circular output record after atomic publication."""

    with (output_dir / "run_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    mismatches: list[str] = []
    for record in manifest["outputs"]:
        path = output_dir / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            mismatches.append(record["path"])
    return {
        "passed": not mismatches,
        "checked_outputs": len(manifest["outputs"]),
        "mismatches": mismatches,
    }
