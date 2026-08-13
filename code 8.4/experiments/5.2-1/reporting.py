"""Audits, parameter registry, figures, manifests, and acceptance reporting."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from figure_style import (  # noqa: E402
    TEXT_WIDTH,
    apply_publication_style,
    panel_title,
    save_figure,
)


PALETTE = {
    "harmonic_ridge": "#2468A2",
    "damped_local_trend": "#D97925",
    "seasonal_naive": "#7A8B3A",
    "hmm_transition": "#2468A2",
    "unconditional": "#D97925",
    "persistence": "#7A8B3A",
    "observed": "#1F2933",
    "counterfactual": "#2468A2",
    "blocked": "#D97925",
    "current_risk": "#2468A2",
    "lead_risk": "#D97925",
}
MODEL_LABELS = {
    "harmonic_ridge": "Harmonic ridge",
    "damped_local_trend": "Damped local trend",
    "seasonal_naive": "Seasonal naive",
    "hmm_transition": "HMM transition",
    "unconditional": "Unconditional",
    "persistence": "Filtered persistence",
}
LINE_STYLES = {
    "harmonic_ridge": "-",
    "damped_local_trend": "--",
    "seasonal_naive": ":",
    "hmm_transition": "-",
    "unconditional": "--",
    "persistence": ":",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%d")


def _audit_row(
    *,
    code_root: Path,
    path: Path,
    source_category: str,
    frame: pd.DataFrame,
    retained: pd.DataFrame,
    used_fields: list[str],
    date_field: str,
    grain_keys: list[str],
    freeze_date: str,
    retrieval_date: str,
    expected_hash: str,
    complete_week_status: Mapping[str, Any] | str,
    evidence_boundary: str,
) -> dict[str, Any]:
    if date_field.endswith("year"):
        dates = pd.to_datetime(
            retained[date_field].astype("Int64").astype(str), format="%Y", errors="coerce"
        )
    else:
        dates = pd.to_datetime(retained[date_field], errors="coerce")
    used_missing = int(retained[used_fields].isna().sum().sum())
    unused_fields = [column for column in retained.columns if column not in used_fields]
    unused_missing = int(retained[unused_fields].isna().sum().sum()) if unused_fields else 0
    duplicate_grain = int(retained.duplicated(grain_keys).sum())
    actual_hash = sha256(path)
    status = (
        "pass"
        if actual_hash.lower() == expected_hash.lower()
        and used_missing == 0
        and duplicate_grain == 0
        and int(dates.isna().sum()) == 0
        else "fail"
    )
    return {
        "source_category": source_category,
        "relative_path": path.relative_to(code_root).as_posix(),
        "file_size_bytes": path.stat().st_size,
        "sha256": actual_hash,
        "expected_sha256": expected_hash.lower(),
        "raw_rows": len(frame),
        "raw_columns": len(frame.columns),
        "retained_rows": len(retained),
        "retained_columns": len(retained.columns),
        "date_field": date_field,
        "date_start": dates.min(),
        "date_end": dates.max(),
        "used_fields": "|".join(used_fields),
        "used_field_missing_cells": used_missing,
        "unused_raw_field_missing_cells": unused_missing,
        "exact_duplicate_rows": int(retained.duplicated().sum()),
        "grain_keys": "|".join(grain_keys),
        "duplicate_grain_rows": duplicate_grain,
        "complete_week_status": json.dumps(complete_week_status, sort_keys=True)
        if isinstance(complete_week_status, Mapping)
        else complete_week_status,
        "project_freeze_date": freeze_date,
        "retrieval_date": retrieval_date,
        "evidence_boundary": evidence_boundary,
        "audit_status": status,
    }


def build_data_audit(
    code_root: Path, data_root: Path, source_manifest: Mapping[str, Any]
) -> pd.DataFrame:
    items = {item["id"]: item for item in source_manifest["datasets"]}
    freeze = "2026-08-03"
    rows: list[dict[str, Any]] = []

    choke_path = data_root / "raw/portwatch/hormuz_chokepoint_daily.csv"
    choke = pd.read_csv(choke_path)
    choke_weekly = pd.read_csv(
        data_root / "processed/portwatch/hormuz_chokepoint_weekly.csv"
    )
    item = items["portwatch_hormuz_daily"]
    rows.append(
        _audit_row(
            code_root=code_root,
            path=choke_path,
            source_category="PortWatch chokepoint observations",
            frame=choke,
            retained=choke,
            used_fields=["date", "portid", "portname", "capacity_container"],
            date_field="date",
            grain_keys=["portid", "date"],
            freeze_date=freeze,
            retrieval_date=item["frozen_retrieval_date"],
            expected_hash=item["sha256"],
            complete_week_status={
                "weekly_rows": len(choke_weekly),
                "complete_weeks": int(choke_weekly["is_complete_week"].sum()),
                "event_complete_weeks": int(
                    (
                        choke_weekly["is_complete_week"].astype(bool)
                        & pd.to_datetime(choke_weekly["week_start"]).between(
                            "2026-02-23", "2026-07-13"
                        )
                    ).sum()
                ),
            },
            evidence_boundary=item["use_boundary"],
        )
    )

    gateway_path = data_root / "raw/portwatch/gateway_ports_daily.csv"
    gateway = pd.read_csv(gateway_path)
    gateway_weekly = pd.read_csv(
        data_root / "processed/portwatch/gateway_ports_weekly.csv"
    )
    item = items["portwatch_gateway_ports_daily"]
    rows.append(
        _audit_row(
            code_root=code_root,
            path=gateway_path,
            source_category="PortWatch gateway observations",
            frame=gateway,
            retained=gateway,
            used_fields=[
                "date",
                "portid",
                "portname",
                "import_container",
                "export_container",
            ],
            date_field="date",
            grain_keys=["portid", "date"],
            freeze_date=freeze,
            retrieval_date=item["frozen_retrieval_date"],
            expected_hash=item["sha256"],
            complete_week_status={
                "port_weeks": len(gateway_weekly),
                "complete_port_weeks": int(
                    gateway_weekly["is_complete_week"].astype(bool).sum()
                ),
                "ports": int(gateway_weekly["portname"].nunique()),
            },
            evidence_boundary=item["use_boundary"],
        )
    )

    gpr_path = data_root / "raw/gpr/data_gpr_export.xls"
    gpr_wide = pd.read_excel(gpr_path, sheet_name="Sheet1")
    gpr_wide["month"] = pd.to_datetime(gpr_wide["month"], errors="coerce")
    gpr_retained = gpr_wide.loc[
        gpr_wide["month"].between("1985-01-01", "2026-06-01")
    ].copy()
    item = items["official_gpr_workbook"]
    rows.append(
        _audit_row(
            code_root=code_root,
            path=gpr_path,
            source_category="official GPR data",
            frame=gpr_wide,
            retained=gpr_retained,
            used_fields=["month", "GPRT", "GPRA"],
            date_field="month",
            grain_keys=["month"],
            freeze_date=freeze,
            retrieval_date="not_recorded_in_frozen_source_manifest",
            expected_hash=item["sha256"],
            complete_week_status="not_applicable_monthly_data",
            evidence_boundary=(
                item["use_boundary"]
                + "; nulls in unused wide-table fields are not HMM input missingness"
            ),
        )
    )

    capacity_path = data_root / "raw/anchors/gateway_official_capacity.csv"
    capacity = pd.read_csv(capacity_path)
    item = items["gateway_official_capacity"]
    rows.append(
        _audit_row(
            code_root=code_root,
            path=capacity_path,
            source_category="official gateway allocation anchors",
            frame=capacity,
            retained=capacity,
            used_fields=[
                "gateway",
                "annual_container_capacity_teu",
                "capacity_year",
                "source_organisation",
                "source_url",
            ],
            date_field="capacity_year",
            grain_keys=["gateway"],
            freeze_date=freeze,
            retrieval_date="not_recorded_in_frozen_source_manifest",
            expected_hash=item["sha256"],
            complete_week_status="not_applicable_annual_anchor",
            evidence_boundary=item["use_boundary"],
        )
    )

    event_path = data_root / "raw/anchors/reclosure_event_advisories.csv"
    advisories_raw = pd.read_csv(event_path)
    advisories = advisories_raw.assign(
        event_record_date=pd.to_datetime(advisories_raw["reopening_report_date"])
    )
    item = items["reclosure_event_advisories"]
    rows.append(
        _audit_row(
            code_root=code_root,
            path=event_path,
            source_category="public event advisories",
            frame=advisories_raw,
            retained=advisories,
            used_fields=[
                "reopening_report_date",
                "reclosure_report_date",
                "reopening_source",
                "reclosure_source",
                "event_record_date",
            ],
            date_field="event_record_date",
            grain_keys=["reopening_report_date", "reclosure_report_date"],
            freeze_date=freeze,
            retrieval_date="not_recorded_in_frozen_source_manifest",
            expected_hash=item["sha256"],
            complete_week_status="not_applicable_event_coordinates",
            evidence_boundary=item["use_boundary"],
        )
    )
    return pd.DataFrame(rows)


def experiment_input_register(data_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    uses = {
        "PortWatch chokepoint observations": "weekly observed Hormuz container carrying-capacity proxy and event shortfall",
        "PortWatch gateway observations": "audit the frozen gateway panel; gateway-scale alignment is inherited from the 5.1 processed anchor",
        "official GPR data": "threat/act levels, differences, rolling volatility, and declared jump flags for the HMM",
        "official gateway allocation anchors": "provenance audit only; no committed share enters this experiment",
        "public event advisories": "reported reopening and renewed-closure dates; no event duration completion is imputed",
    }
    for row in data_audit.itertuples(index=False):
        rows.append(
            {
                "source_category": row.source_category,
                "relative_path": row.relative_path,
                "date_start": row.date_start,
                "date_end": row.date_end,
                "used_variables": row.used_fields,
                "experiment_use": uses[row.source_category],
                "evidence_boundary": row.evidence_boundary,
                "project_freeze_date": row.project_freeze_date,
                "retrieval_date": row.retrieval_date,
            }
        )
    rows.append(
        {
            "source_category": "designed network inputs",
            "relative_path": "not_loaded_in_5.2.1",
            "date_start": "not_applicable",
            "date_end": "not_applicable",
            "used_variables": "none",
            "experiment_use": "no policy, capacity architecture, route, commitment, expansion, or reclosure design is evaluated",
            "evidence_boundary": "network design belongs to later experiments; only the scalar 5.1 activity-scale alignment enters unit conversion",
            "project_freeze_date": "2026-08-03",
            "retrieval_date": "not_applicable_designed_input",
        }
    )
    return pd.DataFrame(rows)


def parameter_registry(
    config: Mapping[str, Any], *, network_exposure_reference: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def recurse(layer: str, prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                recurse(layer, f"{prefix}.{key}" if prefix else key, item)
            return
        parameter = prefix.split(".")[-1]
        rows.append(
            {
                "layer": layer,
                "parameter": parameter,
                "parameter_path": prefix,
                "value": json.dumps(value, sort_keys=True)
                if isinstance(value, (list, tuple))
                else value,
                "value_type": type(value).__name__,
                "source": "config_5_2_1.json",
                "rationale_or_status": "frozen before result generation",
                "frozen_before_run": True,
            }
        )

    for layer in ("data", "counterfactual", "hmm", "figures", "acceptance"):
        recurse(layer, layer, config[layer])
    rows.append(
        {
            "layer": "data_derived_anchor",
            "parameter": "network_exposure_reference",
            "parameter_path": "data_derived_anchor.network_exposure_reference",
            "value": network_exposure_reference,
            "value_type": "float",
            "source": "experiments/data/processed/anchors/network_exposure_reference.csv",
            "rationale_or_status": "5.1 activity-scale alignment; rounds to 0.0263 and is not observed diversion",
            "frozen_before_run": True,
        }
    )
    return pd.DataFrame(rows)


def _set_plot_style() -> None:
    apply_publication_style()
    plt.rcParams.update({"axes.grid": True})


def _save_figure(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    if path.suffix.lower() == ".pdf":
        frozen_time = datetime(2026, 8, 3, tzinfo=timezone.utc)
        metadata = {
            "Creator": "TRE 5.2.1",
            "CreationDate": frozen_time,
            "ModDate": frozen_time,
        }
    else:
        metadata = {"Software": "TRE 5.2.1"}
    save_figure(fig, path, dpi=dpi, metadata=metadata)


def plot_counterfactual_validity(
    summary: pd.DataFrame,
    acf: pd.DataFrame,
    selection: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    _set_plot_style()
    selected = selection.loc[selection["selected"], "model"].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 3.05))
    for model in sorted(summary["model"].unique()):
        subset = summary.loc[summary["model"].eq(model)].sort_values(
            "evaluation_horizon_weeks"
        )
        label = MODEL_LABELS[model] + (" (selected)" if model == selected else "")
        linewidth = 2.2 if model == selected else 1.5
        axes[0].plot(
            subset["evaluation_horizon_weeks"],
            100 * subset["wape"],
            marker="o",
            linestyle=LINE_STYLES[model],
            color=PALETTE[model],
            linewidth=linewidth,
            label=label,
        )
        axes[1].plot(
            subset["evaluation_horizon_weeks"],
            subset["normalised_bias_percent"],
            marker="o",
            linestyle=LINE_STYLES[model],
            color=PALETTE[model],
            linewidth=linewidth,
            label=label,
        )
        residual = acf.loc[acf["model"].eq(model)].sort_values("lag_weeks")
        axes[2].plot(
            residual["lag_weeks"],
            residual["autocorrelation"],
            marker="o",
            markersize=3,
            linestyle=LINE_STYLES[model],
            color=PALETTE[model],
            linewidth=linewidth,
            label=label,
        )
    panel_title(axes[0], "A", "Cumulative forecast error")
    axes[0].set_xlabel("Forecast horizon (weeks)")
    axes[0].set_ylabel("WAPE (%)")
    axes[0].set_xticks(sorted(summary["evaluation_horizon_weeks"].unique()))
    panel_title(axes[1], "B", "Normalised prediction bias")
    axes[1].set_xlabel("Forecast horizon (weeks)")
    axes[1].set_ylabel("Bias / mean observed activity (%)")
    axes[1].axhline(0, color="#4B5563", linewidth=0.9)
    axes[1].set_xticks(sorted(summary["evaluation_horizon_weeks"].unique()))
    confidence = float(acf["confidence_upper"].iloc[0])
    axes[2].axhline(0, color="#4B5563", linewidth=0.9)
    axes[2].axhline(confidence, color="#6B7280", linestyle="--", linewidth=0.9)
    axes[2].axhline(-confidence, color="#6B7280", linestyle="--", linewidth=0.9)
    panel_title(axes[2], "C", "Residual autocorrelation")
    axes[2].set_xlabel("Lag (weeks)")
    axes[2].set_ylabel("Autocorrelation")
    axes[2].set_xticks(sorted(acf["lag_weeks"].unique()))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.1)
    _save_figure(fig, path, dpi=dpi)
    plt.close(fig)


def plot_hmm_validity(
    density_summary: pd.DataFrame,
    filtered: pd.DataFrame,
    path: Path,
    *,
    dpi: int,
) -> None:
    _set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.15))
    for model in ("hmm_transition", "unconditional", "persistence"):
        subset = density_summary.loc[
            density_summary["forecast_model"].eq(model)
        ].sort_values("horizon_months")
        axes[0].plot(
            subset["horizon_months"],
            subset["mean_log_predictive_density"],
            marker="o",
            linestyle=LINE_STYLES[model],
            color=PALETTE[model],
            linewidth=2 if model == "hmm_transition" else 1.5,
            label=MODEL_LABELS[model],
        )
    panel_title(axes[0], "A", "Held out predictive density")
    axes[0].set_xlabel("Forecast horizon (months)")
    axes[0].set_ylabel("Mean log predictive density")
    axes[0].set_xticks([1, 2, 3])
    axes[0].legend(loc="best")

    timeline = filtered.loc[
        pd.to_datetime(filtered["observation_month"]).ge("2018-01-01")
    ].copy()
    axes[1].plot(
        pd.to_datetime(timeline["observation_month"]),
        timeline["filtered_high_risk_probability"],
        color=PALETTE["hmm_transition"],
        linewidth=1.6,
        label="Filtered high-risk state probability",
    )
    heldout_start = pd.Timestamp("2025-01-01")
    axes[1].axvspan(
        heldout_start,
        pd.to_datetime(timeline["observation_month"]).max(),
        color="#E8EDF3",
        alpha=0.75,
        label="Chronological held-out period",
    )
    axes[1].axvline(heldout_start, color="#4B5563", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1)
    panel_title(axes[1], "B", "Filtered high risk state belief")
    axes[1].set_xlabel("Observation month")
    axes[1].set_ylabel("Risk state probability")
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].legend(loc="upper left")
    fig.tight_layout(w_pad=1.2)
    _save_figure(fig, path, dpi=dpi)
    plt.close(fig)


def plot_event_and_release(
    interface: pd.DataFrame, path: Path, *, dpi: int
) -> None:
    _set_plot_style()
    frame = interface.sort_values("week").copy()
    weeks = pd.to_datetime(frame["week"])
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.15))
    axes[0].plot(
        weeks,
        frame["estimated_no_disruption_activity"] / 1e6,
        color=PALETTE["counterfactual"],
        linewidth=2,
        label="Estimated no-disruption activity",
    )
    axes[0].plot(
        weeks,
        frame["observed_activity"] / 1e6,
        color=PALETTE["observed"],
        linestyle="--",
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Observed activity",
    )
    axes[0].plot(
        weeks,
        frame["blocked_activity_proxy"] / 1e6,
        color=PALETTE["blocked"],
        linestyle=":",
        linewidth=2,
        label="Estimated blocked activity proxy",
    )
    panel_title(axes[0], "A", "Historical event input")
    axes[0].set_xlabel("Decision week")
    axes[0].set_ylabel("AIS activity proxy  million tonnes")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[0].legend(loc="best")

    axes[1].plot(
        weeks,
        frame["filtered_high_risk_probability"],
        color=PALETTE["current_risk"],
        marker="o",
        markersize=3,
        linewidth=1.7,
        label="Released current risk belief",
    )
    axes[1].plot(
        weeks,
        frame["lead_time_high_risk_probability"],
        color=PALETTE["lead_risk"],
        linestyle="--",
        marker="s",
        markersize=3,
        linewidth=1.7,
        label="Readiness-maturity risk forecast",
    )
    source_change = pd.to_datetime(frame["source_observation_month"]).ne(
        pd.to_datetime(frame["source_observation_month"]).shift()
    )
    for index, week in enumerate(weeks[source_change]):
        axes[1].axvline(
            week,
            color="#AEB8C2",
            linewidth=0.7,
            zorder=0,
            label="New source month" if index == 0 else None,
        )
    axes[1].set_ylim(0, 1)
    panel_title(axes[1], "B", "Released belief and lead forecast")
    axes[1].set_xlabel("Decision week")
    axes[1].set_ylabel("Risk state probability")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[1].legend(loc="best")
    fig.tight_layout(w_pad=1.2)
    _save_figure(fig, path, dpi=dpi)
    plt.close(fig)


def build_output_manifest(
    output_dir: Path,
    *,
    code_root: Path,
    experiment_dir: Path,
    input_paths: Iterable[Path],
    figure_sources: Mapping[str, list[str]],
    acceptance_status: str,
) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.name == "run_manifest.json" or not path.is_file():
            continue
        record: dict[str, Any] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            record.update({"rows": len(frame), "columns": len(frame.columns)})
        elif path.suffix.lower() == ".png":
            with Image.open(path) as image:
                record.update(
                    {
                        "width_pixels": image.width,
                        "height_pixels": image.height,
                        "dpi": [float(value) for value in image.info.get("dpi", (0, 0))],
                        "source_tables": figure_sources[path.name],
                    }
                )
        outputs.append(record)
    code_files = sorted(
        [path for path in experiment_dir.iterdir() if path.suffix in {".py", ".json", ".md"}],
        key=lambda item: item.name,
    )
    return {
        "schema_version": 1,
        "experiment_id": "5.2.1_data_event_information_validity",
        "acceptance_status": acceptance_status,
        "policy_training_or_comparison_performed": False,
        "input_files": [
            {
                "path": path.relative_to(code_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in input_paths
        ],
        "experiment_code": [
            {
                "path": path.relative_to(code_root).as_posix(),
                "sha256": sha256(path),
            }
            for path in code_files
        ],
        "outputs": outputs,
        "figure_source_mapping": figure_sources,
        "downstream_interface": "historical_information_event_path.csv",
        "downstream_loader": "experiments/5.2-1/interface.py::load_historical_path",
    }


def write_acceptance_report(
    path: Path,
    *,
    checks: list[dict[str, Any]],
    selected_model: str,
    counterfactual_summary: pd.DataFrame,
    selection: pd.DataFrame,
    acf: pd.DataFrame,
    density_summary: pd.DataFrame,
    event_interface: pd.DataFrame,
    manuscript_serviceability_target: float,
) -> None:
    blocking_failures = [check for check in checks if check["blocking"] and not check["passed"]]
    status = "COMPLETE" if not blocking_failures else "BLOCKED"
    lines = [
        "# 5.2.1 Acceptance Report",
        "",
        f"## Overall status: {status}",
        "",
        "This report concerns data, counterfactual predictive validity, released geopolitical-risk information, and formula-derived event inputs only. It contains no policy training or comparison.",
        "",
        "## Blocking acceptance checks",
        "",
        "| Check | Result | Observed | Expected |",
        "|---|---:|---|---|",
    ]
    for check in checks:
        if check["blocking"]:
            lines.append(
                f"| {check['id']} | {'PASS' if check['passed'] else 'FAIL'} | {check['observed']} | {check['expected']} |"
            )
    lines.extend(
        [
            "",
            "## Counterfactual evidence",
            "",
            f"The frozen selection rule selected **{MODEL_LABELS[selected_model]}** using 21-week cumulative path WAPE, the predeclared relative tie tolerance, absolute bias, and unique one-step residual dependence in that order.",
            "",
        ]
    )
    primary = selection.sort_values("wape")
    lines.append("| Model | 21-week WAPE | Bias | One-step dependence | Selected |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {MODEL_LABELS[row.model]} | {row.wape:.4f} | {row.bias:.2f} | {row.one_step_residual_dependence:.3f} | {'yes' if row.selected else 'no'} |"
        )
    short = counterfactual_summary.loc[
        counterfactual_summary["evaluation_horizon_weeks"].lt(21)
    ]
    short_winners = (
        short.sort_values("wape").groupby("evaluation_horizon_weeks").first()["model"].to_dict()
    )
    not_best = [h for h, model in short_winners.items() if model != selected_model]
    selected_acf = acf.loc[acf["model"].eq(selected_model)]
    significant_lags = selected_acf.loc[
        selected_acf["autocorrelation"].abs() > selected_acf["confidence_upper"].abs(),
        "lag_weeks",
    ].astype(int).tolist()

    lines.extend(
        [
            "",
            "## HMM held-out evidence",
            "",
            "Mean log predictive density is reported without suppressing horizons where a benchmark performs better.",
            "",
            "| Horizon | HMM transition | Unconditional | Persistence |",
            "|---:|---:|---:|---:|",
        ]
    )
    pivot = density_summary.pivot(
        index="horizon_months",
        columns="forecast_model",
        values="mean_log_predictive_density",
    )
    for horizon, row in pivot.iterrows():
        lines.append(
            f"| {horizon} | {row['hmm_transition']:.4f} | {row['unconditional']:.4f} | {row['persistence']:.4f} |"
        )
    benchmark_better = []
    for horizon, row in pivot.iterrows():
        for benchmark in ("unconditional", "persistence"):
            if row[benchmark] > row["hmm_transition"]:
                benchmark_better.append(f"{benchmark} at {horizon} month(s)")

    final_serviceability = float(event_interface.iloc[-1]["serviceability"])
    selected_primary = selection.loc[selection["model"].eq(selected_model)].iloc[0]
    selected_normalised_bias = float(
        counterfactual_summary.loc[
            counterfactual_summary["model"].eq(selected_model)
            & counterfactual_summary["evaluation_horizon_weeks"].eq(21),
            "normalised_bias_percent",
        ].iloc[0]
    )
    sharp_event_beliefs = int(
        (
            event_interface["filtered_high_risk_probability"].le(0.01)
            | event_interface["filtered_high_risk_probability"].ge(0.99)
        ).sum()
    )
    lead_min = float(event_interface["lead_time_high_risk_probability"].min())
    lead_max = float(event_interface["lead_time_high_risk_probability"].max())
    lines.extend(
        [
            "",
            "## Negative, weak, or uncertain findings",
            "",
            f"- Counterfactual: the selected model was not the lowest-WAPE model at these shorter horizons: {not_best if not_best else 'none'}. This does not overturn the frozen 21-week rule.",
            f"- Counterfactual bias: the selected 21-week path has raw Bias {selected_primary.bias:.2f} metric tonnes and normalised Bias {selected_normalised_bias:.2f}%. The positive bias can increase estimated event shortfall and must remain visible in interpretation.",
            f"- Residual dependence: selected-model one-step ACF exceeded the approximate 95% bounds at lags {significant_lags if significant_lags else 'none'}. Any remaining dependence must be retained in downstream resampling design.",
            f"- HMM comparison: benchmarks with higher held-out LPD were {benchmark_better if benchmark_better else 'none'}. A weaker HMM horizon is evidence, not a code failure.",
            f"- HMM sharpness: {sharp_event_beliefs} of {len(event_interface)} event-week current beliefs lie at or beyond 0.01/0.99, while the eight-week lead forecast ranges from {lead_min:.3f} to {lead_max:.3f}. This sharp regime separation is an uncertainty and must not be presented as calibrated closure probability.",
            f"- Event reference: the formula-derived 2026-07-13 serviceability is {final_serviceability:.6f}; the prior manuscript target is {manuscript_serviceability_target:.6f}. Any difference is a counterfactual-reconstruction result and was not manually overwritten.",
            "- Interpretation: filtered state probabilities and lead-time forecasts describe geopolitical-risk regimes. They are not closure probabilities, closure labels, or closure dates.",
            "- Provenance caveat: exact external retrieval dates were recorded for PortWatch. Other frozen public files have hash-verified project freeze dates but no independently recorded retrieval date in the source manifest.",
            "",
            "## Evidence boundary and next gate",
            "",
            "If every blocking check above passes, the frozen historical information/event interface is admissible for 5.2.2. This acceptance does not establish policy superiority, readiness value, a historical committed share, or any structural boundary.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
