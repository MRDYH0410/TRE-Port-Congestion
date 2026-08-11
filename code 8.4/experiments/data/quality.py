"""Inspectable data-quality contracts for the frozen Section 5.1 inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    """Raised when a frozen input no longer satisfies its declared contract."""


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    observed: Any
    expected: Any
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass
class AuditReport:
    checks: list[AuditCheck] = field(default_factory=list)

    def add(
        self,
        name: str,
        *,
        observed: Any,
        expected: Any,
        passed: bool | None = None,
        detail: str = "",
    ) -> None:
        outcome = observed == expected if passed is None else bool(passed)
        self.checks.append(AuditCheck(name, outcome, observed, expected, detail))

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def require_passed(self) -> None:
        failures = [check for check in self.checks if not check.passed]
        if failures:
            summary = "; ".join(
                f"{check.name}: observed={check.observed!r}, expected={check.expected!r}"
                for check in failures
            )
            raise DataContractError(summary)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "check_count": len(self.checks),
            "failed_count": sum(not check.passed for check in self.checks),
            "checks": [check.as_dict() for check in self.checks],
        }


def audit_frame(
    report: AuditReport,
    frame: pd.DataFrame,
    *,
    name: str,
    expected_rows: int,
    expected_columns: int,
    key_columns: Iterable[str],
    date_column: str,
    expected_start: str,
    expected_end: str,
) -> None:
    """Audit grain, shape, completeness, uniqueness, and temporal coverage."""

    keys = tuple(key_columns)
    missing_required = sorted(set(keys + (date_column,)) - set(frame.columns))
    report.add(
        f"{name}.required_columns",
        observed=missing_required,
        expected=[],
    )
    if missing_required:
        return

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    report.add(f"{name}.rows", observed=len(frame), expected=expected_rows)
    report.add(f"{name}.columns", observed=len(frame.columns), expected=expected_columns)
    report.add(
        f"{name}.missing_cells",
        observed=int(frame.isna().sum().sum()),
        expected=0,
    )
    report.add(
        f"{name}.invalid_dates",
        observed=int(dates.isna().sum()),
        expected=0,
    )
    report.add(
        f"{name}.duplicate_grain_rows",
        observed=int(frame.duplicated(list(keys) + [date_column]).sum()),
        expected=0,
    )
    report.add(
        f"{name}.date_start",
        observed=str(dates.min().date()),
        expected=expected_start,
    )
    report.add(
        f"{name}.date_end",
        observed=str(dates.max().date()),
        expected=expected_end,
    )

