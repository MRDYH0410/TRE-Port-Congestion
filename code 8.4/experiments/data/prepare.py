"""Build and validate the shared Section 5.1 data layer.

Run from the repository root with::

    python -m experiments.data.prepare
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .construction import (
    EVENT_END_WEEK,
    EVENT_START_WEEK,
    GATEWAY_NAME_MAP,
    build_gpr_continuous_features,
    build_gpr_monthly,
    build_gateway_reference_scales,
    build_portwatch_weekly,
    committed_itinerary_shares,
    network_exposure_reference,
)
from .quality import AuditReport, audit_frame


DATA_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = DATA_ROOT / "manifests" / "datasets.json"

CHOKEPOINT_ACTIVITY_COLUMNS = (
    "n_container",
    "n_dry_bulk",
    "n_general_cargo",
    "n_roro",
    "n_tanker",
    "n_cargo",
    "n_total",
    "capacity_container",
    "capacity_dry_bulk",
    "capacity_general_cargo",
    "capacity_roro",
    "capacity_tanker",
    "capacity_cargo",
    "capacity",
)
PORT_ACTIVITY_COLUMNS = (
    "portcalls_container",
    "portcalls_dry_bulk",
    "portcalls_general_cargo",
    "portcalls_roro",
    "portcalls_tanker",
    "portcalls_cargo",
    "portcalls",
    "import_container",
    "import_dry_bulk",
    "import_general_cargo",
    "import_roro",
    "import_tanker",
    "import_cargo",
    "import",
    "export_container",
    "export_dry_bulk",
    "export_general_cargo",
    "export_roro",
    "export_tanker",
    "export_cargo",
    "export",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")


def _load_manifest(root: Path) -> dict[str, Any]:
    with (root / "manifests" / "datasets.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def _audit_raw_files(root: Path, report: AuditReport, manifest: dict[str, Any]) -> None:
    for item in manifest["datasets"]:
        path = root / item["path"]
        report.add(
            f"raw.{item['id']}.exists",
            observed=path.exists(),
            expected=True,
        )
        if path.exists():
            report.add(
                f"raw.{item['id']}.sha256",
                observed=sha256(path),
                expected=item["sha256"].lower(),
            )


def _audit_weekly_panels(
    report: AuditReport,
    chokepoint_weekly: pd.DataFrame,
    gateway_weekly: pd.DataFrame,
) -> None:
    report.add("chokepoint_weekly.rows", observed=len(chokepoint_weekly), expected=395)
    report.add(
        "chokepoint_weekly.complete_rows",
        observed=int(chokepoint_weekly["is_complete_week"].sum()),
        expected=393,
    )
    per_port_rows = gateway_weekly.groupby("portname").size().to_dict()
    report.add(
        "gateway_weekly.rows_per_port",
        observed=per_port_rows,
        expected={name: 395 for name in sorted(GATEWAY_NAME_MAP)},
    )
    per_port_complete = (
        gateway_weekly.groupby("portname")["is_complete_week"].sum().astype(int).to_dict()
    )
    report.add(
        "gateway_weekly.complete_rows_per_port",
        observed=per_port_complete,
        expected={name: 393 for name in sorted(GATEWAY_NAME_MAP)},
    )
    report.add("gateway_weekly.total_rows", observed=len(gateway_weekly), expected=1_580)

    complete_chokepoint = set(
        chokepoint_weekly.loc[chokepoint_weekly["is_complete_week"], "week_start"]
    )
    complete_by_port = gateway_weekly.loc[gateway_weekly["is_complete_week"]].groupby(
        "portname"
    )["week_start"].apply(set)
    common = complete_chokepoint.copy()
    for values in complete_by_port:
        common.intersection_update(values)
    common_sorted = sorted(common)
    report.add("common_complete_window.weeks", observed=len(common_sorted), expected=393)
    report.add(
        "common_complete_window.start",
        observed=str(common_sorted[0].date()),
        expected="2019-01-07",
    )
    report.add(
        "common_complete_window.end",
        observed=str(common_sorted[-1].date()),
        expected="2026-07-13",
    )
    event_weeks = [
        week
        for week in common_sorted
        if EVENT_START_WEEK <= week <= EVENT_END_WEEK
    ]
    report.add("event_replay.complete_weeks", observed=len(event_weeks), expected=21)

    activity = gateway_weekly.assign(
        gateway=gateway_weekly["portname"].map(GATEWAY_NAME_MAP),
        container_activity=(
            gateway_weekly["import_container"] + gateway_weekly["export_container"]
        ),
    )
    zero_counts = (
        activity.assign(is_zero=activity["container_activity"].eq(0))
        .groupby("gateway")["is_zero"]
        .sum()
        .astype(int)
        .to_dict()
    )
    report.add(
        "gateway_weekly.zero_container_activity_weeks",
        observed=zero_counts,
        expected={"Fujairah": 321, "Jebel Ali": 2, "Khor Fakkan": 146, "Sohar": 0},
        detail="Zeros are retained; no positive floor is applied.",
    )


def build_shared_data(root: Path = DATA_ROOT) -> AuditReport:
    """Validate raw inputs, reconstruct shared processed data, and write evidence."""

    root = root.resolve()
    manifest = _load_manifest(root)
    build_manifest_path = root / "manifests" / "build_manifest.json"
    _write_json(
        build_manifest_path,
        {
            "schema_version": 1,
            "status": "validation_in_progress",
            "all_data_quality_checks_passed": False,
            "source_manifest": "manifests/datasets.json",
        },
    )
    report = AuditReport()
    _audit_raw_files(root, report, manifest)

    chokepoint_daily = pd.read_csv(root / "raw/portwatch/hormuz_chokepoint_daily.csv")
    gateway_daily = pd.read_csv(root / "raw/portwatch/gateway_ports_daily.csv")
    audit_frame(
        report,
        chokepoint_daily,
        name="chokepoint_daily",
        expected_rows=2_761,
        expected_columns=21,
        key_columns=("portid",),
        date_column="date",
        expected_start="2019-01-01",
        expected_end="2026-07-23",
    )
    audit_frame(
        report,
        gateway_daily,
        name="gateway_daily",
        expected_rows=11_048,
        expected_columns=30,
        key_columns=("portid",),
        date_column="date",
        expected_start="2019-01-01",
        expected_end="2026-07-24",
    )

    chokepoint_weekly = build_portwatch_weekly(
        chokepoint_daily,
        location_columns=("portid", "portname"),
        activity_columns=CHOKEPOINT_ACTIVITY_COLUMNS,
    )
    gateway_weekly = build_portwatch_weekly(
        gateway_daily,
        location_columns=("portid", "portname", "country", "ISO3"),
        activity_columns=PORT_ACTIVITY_COLUMNS,
    )
    _audit_weekly_panels(report, chokepoint_weekly, gateway_weekly)

    gpr_workbook = pd.read_excel(root / "raw/gpr/data_gpr_export.xls", sheet_name="Sheet1")
    gpr_monthly = build_gpr_monthly(gpr_workbook)
    gpr_features = build_gpr_continuous_features(gpr_monthly, volatility_window=24)
    expected_months = pd.period_range("1985-01", "2026-06", freq="M")
    observed_months = pd.PeriodIndex(gpr_monthly["month"], freq="M")
    report.add("gpr_monthly.rows", observed=len(gpr_monthly), expected=498)
    report.add(
        "gpr_monthly.complete_cells",
        observed=int(gpr_monthly.isna().sum().sum()),
        expected=0,
    )
    report.add(
        "gpr_monthly.consecutive_months",
        observed=observed_months.equals(expected_months),
        expected=True,
    )
    report.add("gpr_continuous_features.usable_rows", observed=len(gpr_features), expected=474)

    gateway_scales = build_gateway_reference_scales(gateway_weekly)
    scale_observed = {
        row.gateway: round(float(row.activity_scale_model_units), 3)
        for row in gateway_scales.itertuples()
    }
    report.add(
        "gateway_reference_scales.rounded_3dp",
        observed=scale_observed,
        expected={"Khor Fakkan": 30.796, "Fujairah": 1.500, "Sohar": 70.001},
    )
    exposure = network_exposure_reference(gateway_scales, chokepoint_weekly)
    report.add(
        "network_exposure_reference.rounded_4dp",
        observed=round(float(exposure.loc[0, "reference_network_exposure"]), 4),
        expected=0.0263,
    )

    official_capacity = pd.read_csv(root / "raw/anchors/gateway_official_capacity.csv")
    committed = committed_itinerary_shares(official_capacity)
    shares_observed = {
        row.gateway: round(float(row.committed_itinerary_share), 4)
        for row in committed.itertuples()
    }
    report.add(
        "committed_itinerary_shares.rounded_4dp",
        observed=shares_observed,
        expected={"Khor Fakkan": 0.5952, "Fujairah": 0.1071, "Sohar": 0.2976},
    )

    advisories = pd.read_csv(
        root / "raw/anchors/reclosure_event_advisories.csv",
        parse_dates=["reopening_report_date", "reclosure_report_date"],
    )
    target = manifest["derived_reference_targets"]["reclosure_event_marker"]
    duration = int(
        (advisories.loc[0, "reclosure_report_date"] - advisories.loc[0, "reopening_report_date"]).days
    )
    event = advisories.assign(
        reopening_duration_days=duration,
        reopening_duration_weeks=duration / 7.0,
        serviceability_week=pd.Timestamp(target["serviceability_week"]),
        serviceability=float(target["serviceability"]),
        reclosure_severity=float(target["reclosure_severity"]),
        reclosure_duration_status=target["reclosure_duration_status"],
        severity_evidence_status=target["evidence_status"],
    )
    report.add("event_advisories.rows", observed=len(advisories), expected=1)
    report.add("event_marker.reopening_duration_days", observed=duration, expected=23)
    report.add(
        "event_marker.duration_weeks",
        observed=round(float(event.loc[0, "reopening_duration_weeks"]), 2),
        expected=3.29,
    )
    report.add(
        "event_marker.serviceability_plus_severity",
        observed=round(
            float(event.loc[0, "serviceability"] + event.loc[0, "reclosure_severity"]),
            12,
        ),
        expected=1.0,
    )
    report.add(
        "event_marker.duration_status",
        observed=event.loc[0, "reclosure_duration_status"],
        expected="right_censored_at_frozen_vintage",
    )

    quality_path = root / "manifests" / "data_quality_report.json"
    _write_json(quality_path, report.as_dict())
    report.require_passed()

    outputs: dict[str, pd.DataFrame] = {
        "processed/portwatch/hormuz_chokepoint_weekly.csv": chokepoint_weekly,
        "processed/portwatch/gateway_ports_weekly.csv": gateway_weekly,
        "processed/gpr/gpr_monthly.csv": gpr_monthly,
        "processed/gpr/gpr_continuous_features.csv": gpr_features,
        "processed/anchors/gateway_reference_scales.csv": gateway_scales,
        "processed/anchors/network_exposure_reference.csv": exposure,
        "processed/anchors/committed_itinerary_reference.csv": committed,
        "processed/anchors/reclosure_event_marker.csv": event,
    }
    for relative, frame in outputs.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, date_format="%Y-%m-%d")

    build_manifest = {
        "schema_version": 1,
        "status": "complete",
        "source_manifest": "manifests/datasets.json",
        "all_data_quality_checks_passed": report.passed,
        "construction": {
            "weekly_calendar": "Monday-based",
            "zero_activity_policy": "retain",
            "event_start_week": str(EVENT_START_WEEK.date()),
            "event_end_week": str(EVENT_END_WEEK.date()),
            "gpr_volatility_window_months": 24,
            "gpr_jump_indicator": "deferred until the first information experiment declares jump_sigma",
        },
        "outputs": [
            {
                "path": relative,
                "rows": len(frame),
                "columns": len(frame.columns),
                "sha256": sha256(root / relative),
            }
            for relative, frame in outputs.items()
        ],
    }
    _write_json(build_manifest_path, build_manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Root containing raw/, processed/, and manifests/.",
    )
    args = parser.parse_args()
    report = build_shared_data(args.data_root)
    print(f"Section 5.1 data build passed {len(report.checks)} checks.")


if __name__ == "__main__":
    main()
