"""Single reproducible command for Experiment 5.2.1."""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


EXPERIMENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = EXPERIMENT_DIR.parents[1]
SRC_ROOT = CODE_ROOT / "src"
for path in (str(SRC_ROOT), str(CODE_ROOT), str(EXPERIMENT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.data.construction import add_gpr_jump_indicator
from experiments.data.prepare import build_shared_data

from counterfactual import (
    fit_forecast_model,
    model_specification_table,
    one_step_residual_acf,
    run_rolling_origins,
    select_counterfactual_model,
    summarise_rolling_predictions,
)
from event_input import build_frozen_information_event_path, construct_historical_event
from hmm_validity import (
    build_release_clock,
    calendar_month_transitions,
    filter_feature_history,
    fit_frozen_hmm,
    heldout_density_scores,
    hmm_parameter_manifest,
)
from interface import load_historical_path
from reporting import (
    build_data_audit,
    build_output_manifest,
    experiment_input_register,
    parameter_registry,
    plot_counterfactual_validity,
    plot_event_and_release,
    plot_hmm_validity,
    sha256,
    write_acceptance_report,
    write_csv,
    write_json,
)
from validation_5_2_1 import (
    figure_data_tables,
    independent_recalculation,
    verify_manifest_hashes,
    write_stage_reports,
)


OUTPUT_DIR = CODE_ROOT / "output" / "5.2.1_data_event_information_validity"
CONFIG_PATH = EXPERIMENT_DIR / "config_5_2_1.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    observed: Any,
    expected: Any,
    *,
    detail: str,
    blocking: bool = True,
) -> None:
    checks.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "detail": detail,
            "blocking": blocking,
        }
    )


def _build_checks(
    *,
    config: dict[str, Any],
    data_audit: pd.DataFrame,
    weekly_complete: pd.DataFrame,
    rolling_predictions: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    residual_library: pd.DataFrame,
    selection: pd.DataFrame,
    hmm_fit,
    hmm_features: pd.DataFrame,
    heldout_scores: pd.DataFrame,
    release_clock: pd.DataFrame,
    event: pd.DataFrame,
    interface: pd.DataFrame,
    registry: pd.DataFrame,
    output_dir: Path,
    figure_sources: dict[str, list[str]],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    cutoff = pd.Timestamp(config["data"]["counterfactual_selection_cutoff"])
    event_start = pd.Timestamp(config["data"]["event_start_week"])
    event_end = pd.Timestamp(config["data"]["event_end_week"])
    tolerance = float(config["acceptance"]["numeric_tolerance"])

    _add_check(
        checks,
        "used_input_fields_complete",
        bool(data_audit["used_field_missing_cells"].eq(0).all()),
        int(data_audit["used_field_missing_cells"].sum()),
        0,
        detail="Unused GPR wide-table nulls are reported separately and are not treated as HMM input missingness.",
    )
    _add_check(
        checks,
        "frozen_input_hashes_and_grain",
        bool(data_audit["audit_status"].eq("pass").all()),
        data_audit["audit_status"].value_counts().to_dict(),
        {"pass": len(data_audit)},
        detail="Every raw source must match its frozen SHA256 and intended grain.",
    )
    event_weeks = weekly_complete.loc[
        pd.to_datetime(weekly_complete["week_start"]).between(event_start, event_end)
    ]
    _add_check(
        checks,
        "event_window_21_complete_monday_weeks",
        len(event_weeks) == int(config["data"]["expected_event_weeks"])
        and bool(event_weeks["is_complete_week"].all())
        and bool(pd.to_datetime(event_weeks["week_start"]).dt.weekday.eq(0).all()),
        len(event_weeks),
        int(config["data"]["expected_event_weeks"]),
        detail="Event input must span exactly 2026-02-23 to 2026-07-13 at complete Monday weeks.",
    )
    _add_check(
        checks,
        "counterfactual_event_leakage_absent",
        bool(pd.to_datetime(rolling_predictions["target_date"]).le(cutoff).all())
        and bool(rolling_summary["event_observations_used"].eq(0).all())
        and bool(event["counterfactual_event_observations_used_in_fit"].eq(False).all()),
        str(pd.to_datetime(rolling_predictions["target_date"]).max().date()),
        str(cutoff.date()),
        detail="All model/hyperparameter validation targets end before the event starts.",
    )
    pre_event_rows = weekly_complete.loc[
        pd.to_datetime(weekly_complete["week_start"]).le(cutoff)
    ]
    expected_one_step_origins = len(pre_event_rows) - int(
        config["counterfactual"]["minimum_training_weeks"]
    )
    observed_one_step_origins = int(
        rolling_predictions.loc[
            rolling_predictions["model"].eq(config["counterfactual"]["candidate_models"][0])
            & rolling_predictions["forecast_horizon"].eq(1),
            "origin_date",
        ].nunique()
    )
    _add_check(
        checks,
        "all_feasible_weekly_origins_used",
        observed_one_step_origins == expected_one_step_origins,
        observed_one_step_origins,
        expected_one_step_origins,
        detail="No arbitrary 13-week thinning is allowed.",
    )
    _add_check(
        checks,
        "counterfactual_selection_rule_prefrozen",
        bool(config["acceptance"]["selection_rule_locked_before_run"])
        and bool(selection["selection_rule_locked_before_run"].all()),
        config["counterfactual"]["primary_selection_metric"],
        "cumulative_path_WAPE at 21 weeks with declared tie breaks",
        detail="The rule is read from the frozen configuration rather than changed after results.",
    )
    _add_check(
        checks,
        "residual_library_event_free_unique_one_step",
        bool(pd.to_datetime(residual_library["residual_date"]).le(cutoff).all())
        and bool(residual_library["forecast_horizon"].eq(1).all())
        and not bool(residual_library["residual_date"].duplicated().any()),
        {
            "rows": len(residual_library),
            "latest": str(pd.to_datetime(residual_library["residual_date"]).max().date()),
            "duplicates": int(residual_library["residual_date"].duplicated().sum()),
        },
        "unique one-step residuals ending no later than 2026-02-16",
        detail="Overlapping multi-step errors never enter the residual library or ACF.",
    )
    _add_check(
        checks,
        "hmm_training_and_heldout_split",
        hmm_fit.training_rows == 456 and hmm_fit.heldout_rows == 18,
        {"training": hmm_fit.training_rows, "held_out": hmm_fit.heldout_rows},
        {"training": 456, "held_out": 18},
        detail="Training ends at the 2024 calendar boundary; 2025-01 to 2026-06 is chronological held-out.",
    )
    _add_check(
        checks,
        "hmm_em_converged",
        bool(hmm_fit.converged),
        {
            "iterations": len(hmm_fit.likelihood_history),
            "last_change": abs(
                hmm_fit.likelihood_history[-1] - hmm_fit.likelihood_history[-2]
            )
            if len(hmm_fit.likelihood_history) >= 2
            else None,
        },
        f"absolute log-likelihood change <= {config['hmm']['em_convergence_tolerance']}",
        detail="The selected deterministic initialisation must satisfy the frozen numerical criterion.",
    )
    forbidden = tuple(config["acceptance"]["forbidden_hmm_feature_tokens"])
    feature_text = "|".join(config["hmm"]["emission_features"]).lower()
    _add_check(
        checks,
        "hmm_has_no_closure_or_event_label",
        not any(token.lower() in feature_text for token in forbidden),
        config["hmm"]["emission_features"],
        "threat/act levels, differences, volatilities, and declared jumps only",
        detail="The HMM is a geopolitical-risk state model, not a closure classifier.",
    )
    _add_check(
        checks,
        "hmm_feature_missingness",
        int(hmm_features[config["hmm"]["emission_features"]].isna().sum().sum()) == 0,
        int(hmm_features[config["hmm"]["emission_features"]].isna().sum().sum()),
        0,
        detail="Completeness is measured only over actual HMM inputs.",
    )
    _add_check(
        checks,
        "heldout_parameters_never_refit",
        bool(heldout_scores["hmm_parameters_refit"].eq(False).all()),
        heldout_scores["hmm_parameters_refit"].value_counts().to_dict(),
        {False: len(heldout_scores)},
        detail="Negative held-out comparisons cannot trigger sample or parameter changes.",
    )
    _add_check(
        checks,
        "release_date_not_after_decision",
        bool(release_clock["timing_valid"].all())
        and bool(
            pd.to_datetime(release_clock["release_date"]).le(
                pd.to_datetime(release_clock["decision_cutoff"])
            ).all()
        ),
        int((~release_clock["timing_valid"]).sum()),
        0,
        detail="This is a blocking released-information contract.",
    )
    expected_transitions = [
        calendar_month_transitions(source, maturity)
        for source, maturity in zip(
            pd.to_datetime(release_clock["source_observation_month"]),
            pd.to_datetime(release_clock["readiness_maturity_date"]),
        )
    ]
    _add_check(
        checks,
        "monthly_transition_matrix_not_applied_weekly",
        bool(
            np.array_equal(
                release_clock["monthly_transitions_to_maturity"].to_numpy(dtype=int),
                np.asarray(expected_transitions, dtype=int),
            )
        )
        and bool(release_clock["weekly_transition_matrix_applications"].eq(0).all()),
        {
            "maximum_monthly_transitions": int(
                release_clock["monthly_transitions_to_maturity"].max()
            ),
            "weekly_applications": int(
                release_clock["weekly_transition_matrix_applications"].sum()
            ),
        },
        "calendar-month difference with zero weekly applications",
        detail="P is powered by actual monthly transitions from source month to maturity date.",
    )
    _add_check(
        checks,
        "serviceability_closed_unit_interval",
        bool(event["serviceability"].between(0.0, 1.0).all()),
        [float(event["serviceability"].min()), float(event["serviceability"].max())],
        "[0, 1]",
        detail="Serviceability is min(1, observed/counterfactual), never hand-set.",
    )
    _add_check(
        checks,
        "blocked_flow_nonnegative",
        bool(event["blocked_activity_proxy"].ge(-tolerance).all()),
        float(event["blocked_activity_proxy"].min()),
        ">= 0",
        detail="Blocked activity is the positive no-disruption shortfall.",
    )
    _add_check(
        checks,
        "blocked_flow_identity",
        float(event["blocked_identity_residual"].abs().max()) <= tolerance,
        float(event["blocked_identity_residual"].abs().max()),
        f"<= {tolerance}",
        detail="q_blocked=(1-serviceability)*counterfactual=max(counterfactual-observed,0).",
    )
    unit_tolerance = float(config["acceptance"]["model_unit_tolerance"])
    _add_check(
        checks,
        "model_unit_conversion",
        float(event["model_unit_conversion_residual"].abs().max()) <= unit_tolerance
        and bool(event["model_unit_tonnes"].eq(config["data"]["model_unit_tonnes"]).all()),
        float(event["model_unit_conversion_residual"].abs().max()),
        f"<= {unit_tolerance} with U_Q={config['data']['model_unit_tonnes']}",
        detail="Units remain metric tonnes of AIS proxy; they are not TEU or observed diversion.",
    )
    _add_check(
        checks,
        "no_artificial_risk_ramp",
        set(interface["risk_information_source"]) == {"released_hmm_filter"}
        and set(interface["event_path_source"]) == {"formula_derived_5.2.1"},
        {
            "risk_sources": sorted(interface["risk_information_source"].unique()),
            "event_sources": sorted(interface["event_path_source"].unique()),
        },
        "released_hmm_filter plus formula_derived_5.2.1 only",
        detail="Observed blocked share never replaces the released HMM belief.",
    )
    _add_check(
        checks,
        "committed_share_absent",
        "chi" not in "|".join(interface.columns).lower()
        and not bool(interface["contains_committed_share"].astype(bool).any()),
        bool(interface["contains_committed_share"].astype(bool).any()),
        False,
        detail="Committed and decision-eligible splits belong to later experiments.",
    )
    registered = set(registry["parameter"])
    required_parameters = set(config["acceptance"]["required_parameter_names"])
    _add_check(
        checks,
        "required_parameters_registered",
        required_parameters.issubset(registered),
        sorted(required_parameters - registered),
        [],
        detail="Numerical and timing parameters cannot exist only in code.",
    )
    dpi_expected = int(config["figures"]["dpi"])
    figure_status = {}
    for figure_name, sources in figure_sources.items():
        if Path(figure_name).suffix.lower() != ".png":
            continue
        path = output_dir / figure_name
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
        figure_status[figure_name] = {
            "exists": path.exists(),
            "dpi": [round(float(value)) for value in dpi],
            "source_tables": sources,
        }
    _add_check(
        checks,
        "figures_reproducible_300dpi_and_mapped",
        all(
            status["exists"]
            and min(status["dpi"]) >= dpi_expected - 1
            and bool(status["source_tables"])
            for status in figure_status.values()
        ),
        figure_status,
        f"three code-generated figures at {dpi_expected} dpi with declared source tables",
        detail="No figure is manually edited or detached from its result tables.",
    )
    return checks


def main() -> int:
    started = time.perf_counter()
    config = _load_json(CONFIG_PATH)
    if config["scope"]["policy_training_or_comparison"]:
        raise RuntimeError("5.2.1 cannot train or compare policies")
    data_root = (CODE_ROOT / config["data"]["data_root"]).resolve()
    if data_root != (CODE_ROOT / "experiments" / "data").resolve():
        raise RuntimeError("5.2.1 may read only the frozen code 8.4 experiments/data layer")
    final_output_dir = OUTPUT_DIR.resolve()
    staging_dir = (OUTPUT_DIR.parent / ".5.2.1_data_event_information_validity.staging").resolve()
    backup_dir = (OUTPUT_DIR.parent / ".5.2.1_data_event_information_validity.backup").resolve()
    expected_parent = OUTPUT_DIR.parent.resolve()
    if staging_dir.parent != expected_parent or backup_dir.parent != expected_parent:
        raise RuntimeError("Refusing to stage outside the formal output parent")
    baseline_hashes = {
        path.name: sha256(path)
        for path in final_output_dir.iterdir()
        if final_output_dir.exists()
        and path.is_file()
        and path.suffix.lower() in {".csv", ".png"}
    } if final_output_dir.exists() else {}
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    output_dir = staging_dir

    build_shared_data(data_root)
    source_manifest = _load_json(data_root / "manifests" / "datasets.json")
    data_audit = build_data_audit(CODE_ROOT, data_root, source_manifest)
    input_register = experiment_input_register(data_audit)

    weekly = pd.read_csv(
        data_root / "processed/portwatch/hormuz_chokepoint_weekly.csv",
        parse_dates=["week_start"],
    )
    weekly_complete = weekly.loc[weekly["is_complete_week"].astype(bool)].copy()
    weekly_complete = weekly_complete.rename(
        columns={config["data"]["activity_field"]: "observed_activity"}
    )
    selection_cutoff = pd.Timestamp(config["data"]["counterfactual_selection_cutoff"])
    counterfactual_input = weekly_complete.loc[
        weekly_complete["week_start"].le(selection_cutoff),
        ["week_start", "observed_activity"],
    ]
    predictions = run_rolling_origins(counterfactual_input, config["counterfactual"])
    summary = summarise_rolling_predictions(
        predictions, config["counterfactual"]["forecast_horizons_weeks"]
    )
    acf = one_step_residual_acf(
        predictions, maximum_lag=int(config["counterfactual"]["acf_max_lag"])
    )
    selected_model, selection = select_counterfactual_model(
        summary,
        acf,
        primary_horizon=int(config["counterfactual"]["primary_selection_horizon_weeks"]),
        relative_wape_tolerance=float(
            config["counterfactual"]["relative_wape_tie_tolerance"]
        ),
    )
    final_fit = fit_forecast_model(
        selected_model,
        counterfactual_input["observed_activity"].to_numpy(dtype=float),
        int(config["counterfactual"]["maximum_forecast_horizon_weeks"]),
        config["counterfactual"],
    )
    selection["final_pre_event_parameters"] = json.dumps(
        final_fit.parameters, sort_keys=True, separators=(",", ":")
    )
    residual_library = predictions.loc[
        predictions["model"].eq(selected_model)
        & predictions["forecast_horizon"].eq(1),
        [
            "target_date",
            "origin_date",
            "forecast_horizon",
            "observed_value",
            "predicted_value",
            "residual",
            "model",
            "training_cutoff",
        ],
    ].rename(
        columns={"target_date": "residual_date", "model": "selected_model"}
    )
    specifications = model_specification_table(config["counterfactual"])

    exposure = pd.read_csv(
        data_root / "processed/anchors/network_exposure_reference.csv"
    )
    exposure_value = float(exposure.loc[0, "reference_network_exposure"])
    event, event_parameters = construct_historical_event(
        weekly_complete[["week_start", "observed_activity"]],
        selected_model=selected_model,
        counterfactual_config=config["counterfactual"],
        selection_cutoff=selection_cutoff,
        event_start=pd.Timestamp(config["data"]["event_start_week"]),
        event_end=pd.Timestamp(config["data"]["event_end_week"]),
        network_exposure_reference=exposure_value,
        model_unit_tonnes=float(config["data"]["model_unit_tonnes"]),
    )
    selection["event_forecast_parameters"] = json.dumps(
        event_parameters, sort_keys=True, separators=(",", ":")
    )

    gpr_continuous = pd.read_csv(
        data_root / "processed/gpr/gpr_continuous_features.csv",
        parse_dates=["month"],
    )
    hmm_features = add_gpr_jump_indicator(
        gpr_continuous,
        volatility_window=int(config["hmm"]["volatility_window_months"]),
        jump_sigma=float(config["hmm"]["jump_sigma"]),
    )
    hmm_fit = fit_frozen_hmm(hmm_features, config["hmm"])
    filtered, standardised_observations = filter_feature_history(
        hmm_features, hmm_fit, config["hmm"]
    )
    filtered_beliefs = filtered[
        ["filtered_state_0_probability", "filtered_state_1_probability"]
    ].to_numpy(dtype=float)
    density_scores, density_summary = heldout_density_scores(
        hmm_features,
        standardised_observations,
        filtered_beliefs,
        hmm_fit,
        config["hmm"],
    )
    release_clock = build_release_clock(
        weekly_complete["week_start"], filtered, hmm_fit, config["hmm"]
    )
    historical_interface = build_frozen_information_event_path(event, release_clock)
    hmm_manifest = hmm_parameter_manifest(hmm_fit, config["hmm"])
    registry = parameter_registry(config, network_exposure_reference=exposure_value)

    tables = {
        "experiment_input_register.csv": input_register,
        "data_audit.csv": data_audit,
        "counterfactual_model_specifications.csv": specifications,
        "counterfactual_rolling_origin_predictions.csv": predictions,
        "counterfactual_rolling_origin_summary.csv": summary,
        "counterfactual_model_selection.csv": selection,
        "counterfactual_residual_library.csv": residual_library,
        "counterfactual_one_step_residual_acf.csv": acf,
        "hmm_parameter_manifest.csv": hmm_manifest,
        "heldout_density_scores.csv": density_scores,
        "heldout_density_summary.csv": density_summary,
        "released_hmm_filter.csv": filtered,
        "release_clock.csv": release_clock,
        "historical_event_replay.csv": historical_interface,
        "historical_information_event_path.csv": historical_interface,
        "parameter_registry_5_2_1.csv": registry,
    }
    for filename, frame in tables.items():
        write_csv(output_dir / filename, frame)

    figure_data = figure_data_tables(
        summary, acf, selection, density_summary, filtered, historical_interface
    )
    for filename, frame in figure_data.items():
        write_csv(output_dir / filename, frame)
    tables.update(figure_data)

    png_figure_sources = {
        "figure_5_2_1a_counterfactual_predictive_validity.png": [
            "figure_5_2_1a_data.csv",
            "counterfactual_rolling_origin_summary.csv",
            "counterfactual_one_step_residual_acf.csv",
            "counterfactual_model_selection.csv",
        ],
        "figure_5_2_1b_released_hmm_validity.png": [
            "figure_5_2_1b_data.csv",
            "heldout_density_summary.csv",
            "released_hmm_filter.csv",
        ],
        "figure_5_2_1c_event_input_release_clock.png": [
            "figure_5_2_1c_data.csv",
            "historical_information_event_path.csv",
            "release_clock.csv",
        ],
    }
    figure_sources = dict(png_figure_sources)
    for png_name, sources in png_figure_sources.items():
        figure_sources[str(Path(png_name).with_suffix(".pdf"))] = list(sources)
    dpi = int(config["figures"]["dpi"])
    for suffix in (".png", ".pdf"):
        plot_counterfactual_validity(
            summary,
            acf,
            selection,
            output_dir / f"figure_5_2_1a_counterfactual_predictive_validity{suffix}",
            dpi=dpi,
        )
        plot_hmm_validity(
            density_summary,
            filtered,
            output_dir / f"figure_5_2_1b_released_hmm_validity{suffix}",
            dpi=dpi,
        )
        plot_event_and_release(
            historical_interface,
            output_dir / f"figure_5_2_1c_event_input_release_clock{suffix}",
            dpi=dpi,
        )

    independent_checks = independent_recalculation(output_dir, config)
    write_csv(output_dir / "independent_recalculation_checks.csv", independent_checks)
    tables["independent_recalculation_checks.csv"] = independent_checks

    baseline_rows = []
    for filename, old_hash in sorted(baseline_hashes.items()):
        candidate = output_dir / filename
        if not candidate.is_file():
            continue
        new_hash = sha256(candidate)
        baseline_rows.append(
            {
                "filename": filename,
                "pre_rerun_sha256": old_hash,
                "rerun_sha256": new_hash,
                "byte_identical": old_hash == new_hash,
            }
        )
    baseline_audit = pd.DataFrame(baseline_rows)
    write_csv(output_dir / "baseline_reproduction_audit.csv", baseline_audit)
    tables["baseline_reproduction_audit.csv"] = baseline_audit
    critical_reproduction_files = {
        "data_audit.csv",
        "counterfactual_residual_library.csv",
        "heldout_density_summary.csv",
        "historical_event_replay.csv",
        "historical_information_event_path.csv",
        "hmm_parameter_manifest.csv",
        "release_clock.csv",
        "released_hmm_filter.csv",
        "parameter_registry_5_2_1.csv",
    }
    critical_baseline = baseline_audit.loc[
        baseline_audit["filename"].isin(critical_reproduction_files)
    ]

    checks = _build_checks(
        config=config,
        data_audit=data_audit,
        weekly_complete=weekly_complete,
        rolling_predictions=predictions,
        rolling_summary=summary,
        residual_library=residual_library,
        selection=selection,
        hmm_fit=hmm_fit,
        hmm_features=hmm_features,
        heldout_scores=density_scores,
        release_clock=release_clock,
        event=event,
        interface=historical_interface,
        registry=registry,
        output_dir=output_dir,
        figure_sources=figure_sources,
    )
    _add_check(
        checks,
        "independent_recalculation_matches_formal_csv",
        bool(independent_checks["passed"].all()),
        {
            "passed": int(independent_checks["passed"].sum()),
            "total": len(independent_checks),
        },
        {"passed": len(independent_checks), "total": len(independent_checks)},
        detail="A separate output-layer implementation recomputes metrics, timing, identities, and the unique interface.",
    )
    _add_check(
        checks,
        "pre_rerun_artifact_byte_identity_diagnostic",
        bool(baseline_audit["byte_identical"].all()) if len(baseline_audit) else True,
        {
            "byte_identical": int(baseline_audit["byte_identical"].sum()) if len(baseline_audit) else 0,
            "compared": len(baseline_audit),
        },
        {"byte_identical": len(baseline_audit), "compared": len(baseline_audit)},
        detail="Nonblocking byte-level diagnostic: regenerated optimizer diagnostics and rendered PNG bytes may expose platform-level numerical/rendering drift even when the frozen downstream interface is unchanged.",
        blocking=False,
    )
    _add_check(
        checks,
        "critical_frozen_inputs_and_downstream_outputs_reproduced",
        len(critical_baseline) == len(critical_reproduction_files)
        and bool(critical_baseline["byte_identical"].all()),
        {
            "byte_identical": int(critical_baseline["byte_identical"].sum()),
            "compared": len(critical_baseline),
        },
        {
            "byte_identical": len(critical_reproduction_files),
            "compared": len(critical_reproduction_files),
        },
        detail="Frozen audits, the selected-model residual library, HMM/release outputs, parameter registry, and unique historical interface must remain byte-identical.",
    )

    required_outputs = set(tables) | set(figure_sources)
    _add_check(
        checks,
        "all_required_pre_manifest_outputs_created",
        all((output_dir / filename).exists() for filename in required_outputs),
        sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        sorted(required_outputs),
        detail="Every requested CSV and all three code-generated figures must exist.",
    )
    manuscript_target = float(
        source_manifest["derived_reference_targets"]["reclosure_event_marker"][
            "serviceability"
        ]
    )

    def write_acceptance_and_report(active_checks: list[dict[str, Any]]) -> str:
        blocking_failures = [
            check for check in active_checks if check["blocking"] and not check["passed"]
        ]
        status = "complete" if not blocking_failures else "blocked"
        diagnostics = {
            "hmm_benchmark_superiority_is_not_required": True,
            "selected_counterfactual_need_not_win_every_short_horizon": True,
            "residual_dependence_is_reported_not_suppressed": True,
            "formula_derived_event_serviceability": float(
                historical_interface.iloc[-1]["serviceability"]
            ),
            "current_manuscript_target_serviceability": manuscript_target,
        }
        write_json(
            output_dir / "acceptance_5_2_1.json",
            {
                "schema_version": 1,
                "experiment_id": config["experiment_id"],
                "status": status,
                "blocking_failures": [check["id"] for check in blocking_failures],
                "checks": active_checks,
                "nonblocking_evidence_rules": diagnostics,
                "policy_training_or_comparison_performed": False,
            },
        )
        write_acceptance_report(
            output_dir / "ACCEPTANCE_REPORT_5_2_1.md",
            checks=active_checks,
            selected_model=selected_model,
            counterfactual_summary=summary,
            selection=selection,
            acf=acf,
            density_summary=density_summary,
            event_interface=historical_interface,
            manuscript_serviceability_target=manuscript_target,
        )
        return status

    status = write_acceptance_and_report(checks)
    input_paths = [
        CONFIG_PATH,
        data_root / "manifests" / "datasets.json",
        data_root / "manifests" / "build_manifest.json",
        data_root / "raw/portwatch/hormuz_chokepoint_daily.csv",
        data_root / "raw/portwatch/gateway_ports_daily.csv",
        data_root / "raw/gpr/data_gpr_export.xls",
        data_root / "raw/anchors/gateway_official_capacity.csv",
        data_root / "raw/anchors/reclosure_event_advisories.csv",
        data_root / "processed/portwatch/hormuz_chokepoint_weekly.csv",
        data_root / "processed/gpr/gpr_continuous_features.csv",
        data_root / "processed/anchors/network_exposure_reference.csv",
    ]
    allowed_input_root = data_root.resolve()
    disallowed_inputs = [
        str(path)
        for path in input_paths
        if path.resolve() != CONFIG_PATH.resolve()
        and allowed_input_root not in path.resolve().parents
    ]
    _add_check(
        checks,
        "input_scope_excludes_policy_and_old_chapter5_results",
        not disallowed_inputs,
        disallowed_inputs,
        [],
        detail="The formal input allowlist contains only the frozen 5.2.1 configuration and experiments/data files.",
    )

    def write_manifest(active_status: str) -> None:
        manifest = build_output_manifest(
            output_dir,
            code_root=CODE_ROOT,
            experiment_dir=EXPERIMENT_DIR,
            input_paths=input_paths,
            figure_sources=figure_sources,
            acceptance_status=active_status,
        )
        write_json(output_dir / "run_manifest.json", manifest)

    write_manifest(status)
    interface_valid = True
    interface_error = ""
    try:
        loaded = load_historical_path(output_dir)
        interface_valid = len(loaded) == int(config["data"]["expected_event_weeks"])
    except Exception as exc:  # exact failure is retained in acceptance evidence
        interface_valid = False
        interface_error = f"{type(exc).__name__}: {exc}"
    _add_check(
        checks,
        "downstream_interface_hash_and_contract",
        interface_valid,
        "valid" if interface_valid else interface_error,
        "hash-verified 21-week released-information path",
        detail="Later experiments must use the frozen loader rather than reconstruct a risk path.",
    )
    _add_check(
        checks,
        "run_manifest_created_with_figure_sources",
        (output_dir / "run_manifest.json").exists()
        and set(figure_sources)
        == set(_load_json(output_dir / "run_manifest.json")["figure_source_mapping"]),
        sorted(figure_sources),
        sorted(figure_sources),
        detail="Each PNG and vector PDF is mapped to its source and figure-data CSV files.",
    )
    status = write_acceptance_and_report(checks)
    baseline_summary = {
        "summary": (
            f"{int(baseline_audit['byte_identical'].sum())}/{len(baseline_audit)} "
            "pre-existing scientific CSV/PNG artifacts are byte-identical"
        ) if len(baseline_audit) else "no pre-rerun scientific baseline was available"
    }
    write_stage_reports(
        output_dir,
        status=status,
        selected_model=selected_model,
        baseline_comparison=baseline_summary,
    )
    runtime_record = {
        "experiment_id": config["experiment_id"],
        "command": "python experiments/5.2-1/run_5_2_1.py",
        "python_version": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds_before_manifest": time.perf_counter() - started,
        "random_seeds": "none; deterministic HMM initialisations are enumerated in the parameter manifest",
        "policy_outputs_read": False,
        "old_chapter5_results_read": False,
        "publication_mode": "staging_then_atomic_directory_replace",
    }
    write_json(output_dir / "run_environment.json", runtime_record)
    write_manifest(status)

    prepublish_hash_audit = verify_manifest_hashes(output_dir)
    if not prepublish_hash_audit["passed"]:
        raise RuntimeError(
            "Pre-publication manifest verification failed: "
            + ", ".join(prepublish_hash_audit["mismatches"])
        )

    if status != "complete":
        print(
            json.dumps(
                {
                    "experiment": config["experiment_id"],
                    "status": status,
                    "staging_directory": str(output_dir),
                    "formal_output_preserved": str(final_output_dir),
                    "blocking_failures": [
                        check["id"]
                        for check in checks
                        if check["blocking"] and not check["passed"]
                    ],
                },
                indent=2,
            )
        )
        return 1

    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if final_output_dir.exists():
        final_output_dir.rename(backup_dir)
    try:
        output_dir.rename(final_output_dir)
    except Exception:
        if backup_dir.exists() and not final_output_dir.exists():
            backup_dir.rename(final_output_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    public_figure_dir = EXPERIMENT_DIR / "figures"
    public_figure_dir.mkdir(parents=True, exist_ok=True)
    for filename in sorted(set(figure_sources) | set(figure_data)):
        shutil.copy2(final_output_dir / filename, public_figure_dir / filename)
    report_dir = CODE_ROOT.parent / "report - 8.4" / "5.2.1"
    report_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("FIGURE_AND_RESULTS_ANALYSIS.md", "5_1_PARAMETER_AND_METRIC_ADDITIONS.md"):
        shutil.copy2(final_output_dir / filename, report_dir / filename)

    postpublish_hash_audit = verify_manifest_hashes(final_output_dir)
    if not postpublish_hash_audit["passed"]:
        raise RuntimeError(
            "Post-publication manifest verification failed: "
            + ", ".join(postpublish_hash_audit["mismatches"])
        )

    print(
        json.dumps(
            {
                "experiment": config["experiment_id"],
                "status": status,
                "selected_counterfactual": selected_model,
                "output_directory": str(final_output_dir),
                "manifest_outputs_verified": postpublish_hash_audit["checked_outputs"],
                "blocking_failures": [
                    check["id"]
                    for check in checks
                    if check["blocking"] and not check["passed"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
